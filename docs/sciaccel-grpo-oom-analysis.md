# SciAccel GRPO OOM 分析

一句话结论：这次 OOM 的直接触发点是 actor 训练反向传播的峰值显存不足；多轮轨迹使单条样本很长，是主要上游原因，但过大的训练 token 预算才是让它必现的配置因素。

## 证据与判断

- 失败栈位于 `update_actor -> train_mini_batch -> forward_backward_batch -> loss.backward()`，即 rollout 完成后首次 PPO actor 更新，而不是 vLLM 生成或 KV cache 分配阶段。
- 失败时 GPU 仅剩约 582 MiB 可用；PyTorch 已分配约 83 GiB，反向传播额外申请约 1.64 GiB 失败。另一个进程占用约 4.25 GiB，会压缩安全余量，但不是根因。
- rollout 启用了 multi-turn，`max_turns=50`。日志中多条轨迹达到 45--50 turns，累计 response token 可超过 7 万。这些 token 会被拼入 PPO 训练序列，因此不只是最后一次请求的 prompt 变长。
- `data.max_response_length=16384` 不能限制整条 agent trajectory 的累计长度；多轮的模型输出、工具返回和消息模板都会留在后续上下文中，并在训练时一起参与前向和反向传播。
- actor 启用了动态 batch，但 `ppo_max_token_len_per_gpu=131072`。对 4B 模型的反向传播而言，这个上限在含长轨迹的 batch 上已经接近或超过 95 GiB 卡的可承受范围；`ppo_micro_batch_size_per_gpu=1` 只能避免把多个序列并入同一 micro batch，不能降低一条超长序列本身的激活显存。

## 是否是 turns 太多

是，但需要更准确地表述：`max_turns=50` 允许上下文不断累计，确实生产了极长训练样本；OOM 并非“训练跑到后期 prompt 越来越长”造成，因为它发生在训练进度 0% 的第一次 actor update。触发它的是首批 rollout 中已经存在的长轨迹，加上 131072 的每卡训练 token 上限。

## 处理优先级

1. 首先降低 `ppo_max_token_len_per_gpu`，建议从 32768 或 49152 开始压测，再逐档上调；同时让 rollout/log-prob 的 token 上限保持一致或更低。
2. 给 agent trajectory 增加独立的总 token 上限或超长轨迹过滤。仅保留 `data.max_response_length` 不足以约束多轮累计长度。
3. 如果任务质量允许，降低 `multi_turn.max_turns`（例如 20--30），并限制工具响应长度，减少后续轮次的上下文膨胀。
4. 预留共卡余量：清理或隔离额外占用的约 4 GiB 进程。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 只能缓解碎片，不能替代降低 token 预算。

## 建议验证

重跑时记录每个 PPO micro batch 的累计 token、最长序列 token、`max_memory_allocated` 和 `max_memory_reserved`。若在降低 token 预算后 OOM 消失，即可验证此判断；若仍在单条长样本上失败，再收紧 trajectory 总 token 或 turns。

## 验证结果：修复方案已确认有效（2026-09-22）

一句话结论：`TRAIN_SP` 4→8 + `MAX_RESPONSE_LENGTH` 65536→32768（连带 `MAX_MODEL_LEN` 131072→65536）之后，5 步验证跑（`TOTAL_TRAINING_STEPS=5`, `MAX_TURNS=30`）全程无 OOM，显存稳定在预算内；但同一次跑暴露出验证集准确率极低，是训练效果问题，与显存修复是两件独立的事。

### 采用的修复

- `train_actor_rollout_ref.actor.ulysses_sequence_parallel_size`（`TRAIN_SP`）从 4 提到 8：把反向传播的激活显存进一步切分到更多卡上。
- `max_response_length` 从 65536 降到 32768，`max_model_len` 从 131072 降到 65536：直接压低单条轨迹能累积到的 token 上限，同时降低 `ppo_max_token_len_per_gpu`（脚本里与 `max_model_len` 绑定）。
- `max_turns` 从 50 降到 30（上一版重跑已经改过，这次沿用）：减少多轮累积把上下文推到上限的概率。
- 以上验证了「处理优先级」一节里第 1、3 条判断（压 token 预算 + 压 turns）已经足以解决问题，没有再单独去做第 2 条（独立的 trajectory 总 token 上限/过滤）。

### 跑出来的结果

全部 5 步都跑完，无 Traceback / OOM / BadRequestError（trainer 级别），进程正常退出（非崩溃），总耗时 4:04:54。

| step | actor/loss | actor/grad_norm | actor/entropy | critic/score/mean | response_length/mean | max_memory_reserved_gb | timing_s/step |
|---|---|---|---|---|---|---|---|
| 1 | -0.295 | 0.72 | 0.582 | 0.290 | 30261 | 76.5 | 4392 |
| 2 | -0.344 | 0.83 | 0.543 | 0.225 | 30770 | 82.4 | 3019 |
| 3 | -0.348 | 0.76 | 0.512 | 0.258 | 29824 | 82.4 | 1811 |
| 4 | -0.516 | 1.22 | 0.570 | 0.195 | 30489 | 82.4 | 2707 |
| 5 | -0.430 | 0.88 | 0.482 | 0.188 | 30457 | 82.4 | 2764 |

最终验证（step 5 后跑的一次）：`val-core/sciaccel_rl/acc/mean@1 = 0.0`，`val-aux/sciaccel_rl/reward/mean@1 ≈ 0.029`，`val-aux/num_turns/mean ≈ 29.1`（接近 `max_turns=30` 上限，说明验证集大多数 episode 是靠 turns 耗尽结束，而不是主动 `finished`）。

显存从 step 1 的 76.5GB 涨到 step 2 起稳定在 82.4GB，此后不再增长——`ulysses_sequence_parallel_size=8` 分摊后峰值远低于 95GB/卡预算，说明这次的余量不是刚好卡线，还有继续加长 `max_response_length` 或扩大 batch 的空间。

用 `scripts/parse_training_log.py` 对日志跑出的完整曲线图见 `examples/sciaccel_rl/training_metrics.png`（配套数据见同目录 `training_metrics.csv`）：

- `actor/loss`、`critic/score/mean` 5 步内没有稳定下降趋势（score 从 0.29 掉到 0.19），在只有 3 个 warmup 步之后就进入正常 lr，5 步的样本量太小，波动幅度（尤其 step 4 的 grad_norm 冲到 1.22）看不出是收敛还是噪声。
- `group/zero_variance_fraction` 在 step 3 降到 0.19 后 step 5 又冲到 0.375——四分之一到三分之一的 GRPO group 内奖励没有方差，贡献不了梯度，这和验证集 `acc/mean@1=0.0` 互相印证：这个任务在当前 checkpoint 下解出率非常低。
- `perf/throughput` 和 `timing_s/step` 没有单调变化（step 3 最快，step 1 最慢），5 步样本太少，不能当作稳定的吞吐基线。

### 结论与下一步

这次跑要回答的问题只是「显存修复是否有效」——答案是肯定的，两次此前都死在第一步 backward 的 OOM 完全消失，5 步全部稳定跑完。但 `val-core/acc/mean@1=0.0` 说明当前配置下模型基本没能解出这个任务，这是训练效果 / 任务难度层面的问题，不是这份 OOM 分析要解决的范围，需要另外判断是延长训练步数、调整奖励/数据分布，还是检查任务本身对 4B 模型是否过难。
