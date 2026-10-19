# PSRL 分组采样、DAPO 与训练损失

> **核心结论**：解释同一 prompt 的分组采样、DAPO 改动、熵与 KL 项，以及多轮 response_mask、奖励标签和 rollout correction。

## 目录

- [6.0b Group Rollout 的工程实现：uid / parent_id 与磁盘上的重复 prompt](#60b-group-rollout-的工程实现uid--parent_id-与磁盘上的重复-prompt)
- [6.0c DAPO 在 GRPO 基础上的四个改进（附 PSRL 源码）](#60c-dapo-在-grpo-基础上的四个改进附-psrl-源码)
- [6.0d 熵奖励 `H` 与 KL 惩罚 `D_KL(π‖π_ref)`（附 PSRL/verl 源码）](#60d-熵奖励-h-与-kl-惩罚-d_klππ_ref附-psrlverl-源码)
- [6.0e 输入输出的 label 设计与 `rollout_correction` 修正（附 PSRL 源码，对照 `fsdp_qwen2.5_3b_laps.sh`）](#60e-输入输出的-label-设计与-rollout_correction-修正附-psrl-源码对照-fsdp_qwen25_3b_lapssh)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。原文整理日期为 2026-09-10，后续追加内容的日期以各节记载为准；保留章节编号以便对照旧引用。

[返回专题索引](../vllm-and-training-parallelism.md)

阅读顺序：本篇承接[PSRL 训推协作与 PPO、GRPO 基础](rl-training-workflow.md)，其中给出了轨迹处理步骤与优势估计公式。

## 6.0b Group Rollout 的工程实现：uid / parent_id 与磁盘上的重复 prompt

**现象**：打开 `psrl_demo/logs/.../trajectories/v{version}/` 会发现大量文件名是连续整数（如 `80.txt 81.txt 82.txt 83.txt`），内容里的 prompt 完全一样，只有生成结果不同——这正是前篇 [§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 提到的 GRPO"对同一个 prompt 一次采样 G 条 response"在系统层面的落地方式，不是 bug。

**怎么实现的（PSRL 源码）**：

1. **fan-out 成 `rollout_n` 个独立请求**：`entry_infos_to_kv_batch_meta`（`psrl/workers/agent_loop/manager.py:931-953`）里，每个 prompt 有一个 `prompt_id`，按 `rollout_n`（配置项 `gen_actor_rollout_ref.rollout.n`，也可以走 `redundant_rollout.redundant_rollout_n`）展开成 `rollout_n` 份请求，分别打上：
   - `uid = prompt_id * rollout_n + request_idx`（每条 response 的唯一 id，也是轨迹文件名的来源）
   - `parent_id = prompt_id`（同一组内所有 response 共享，用于后面按组归一化）
   这 `rollout_n` 个请求是**各自独立、异步调度**的生成任务，而不是一次 vLLM 调用里 `n=4` 的批量采样——所以它们可能被不同的 vLLM 副本接住、在不同时刻完成，磁盘上不一定连续写入（这也是为什么同一次快照里，有的组已经凑齐 4 条，大多数组还只有 1 条：generation 还在飞）。
   - `uid` 最终原样传进 `_dump_trajectory_text`（`psrl/workers/agent_loop/loops/base_agent_loop.py:1144-1158`），变成轨迹文件名 `{uid}.txt`——这就是我们在目录里直接看到 `80/81/82/83.txt` 四个文件、prompt 一模一样的原因。

2. **为什么必须是"同一个 prompt 采多条"**：GRPO 不训 critic，用组内 reward 的均值/方差直接当 baseline（[§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 已详述公式）。`compute_grpo_outcome_advantage` 在 `psrl/trainer/ppo/utils.py:181-192` 按 `data.non_tensor_batch["parent_id"]` 分组算 advantage——没有并行采出的这组"重复 prompt"，GRPO 就没有基线可减，advantage 无从算起。这也解释了为什么 DAPO/GRPO 类算法天然需要"并行采样同一个问题 G 次"，而 PPO+critic 不需要（critic 给单条轨迹就能估值）。

3. **DAPO 的 dynamic sampling 理论上会反过来丢掉某些组**：`DynamicSamplingFilterProcessor`（`psrl/utils/post_processor/buffer_post_process/filter.py:31-95`）按 `pid = uid // rollout_n` 把同组样本的 reward 收集起来算方差，`std == 0`（一组里全对或全错，组内没有优劣之分，GRPO advantage 恒为 0，没有梯度信号）的组会被整体丢弃、不进训练 buffer。**但要注意**：这份 `staleness_3` demo 跑的日志里 `group_post_process.enable=False` 且 `buffer_post_process.enable=False`（下面 [§6.0c](#60c-dapo-在-grpo-基础上的四个改进附-psrl-源码) 有原文引用），也就是说这个过滤器在这次跑的实验里实际是关着的——磁盘上看到的重复 prompt 组，不管组内是否有区分度都会原样进入训练。DAPO 四个改进点的完整讲解和是否启用的核对见 [§6.0c](#60c-dapo-在-grpo-基础上的四个改进附-psrl-源码)。

| 概念 | 对应字段/代码 | 作用 |
|---|---|---|
| 组标识 | `parent_id`（= `prompt_id`） | GRPO advantage 按组归一化的分组键 |
| 组内序号 | `request_idx`（0 ~ rollout_n-1） | 拼出 `uid`，标识组内第几条采样 |
| 轨迹文件名 | `uid = prompt_id * rollout_n + request_idx` | `TrajectoryWriter` 直接用作 `{uid}.txt` |
| 组大小 | `rollout_n`（`gen_actor_rollout_ref.rollout.n`） | 一个 prompt 并行采多少条 response（GRPO 的 "G"） |
| 组过滤 | `DynamicSamplingFilterProcessor` | reward 方差为 0 的组整体丢弃（DAPO dynamic sampling） |

## 6.0c DAPO 在 GRPO 基础上的四个改进（附 PSRL 源码）

**一句话结论**：DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）不是一种新的 advantage 估计方式——这次 demo 用的 `algorithm.adv_estimator=grpo`，advantage 仍然是 [§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 那套组内归一化公式（`compute_grpo_outcome_advantage`）。DAPO 是在 GRPO 之上打的四个训练稳定性/效率补丁，四个补丁相互独立、可单独开关：**Clip-Higher、Dynamic Sampling、Token-level Policy Gradient Loss、Overlong Reward Shaping**。下面逐个讲原理、配 PSRL/verl 源码，并核对这次 `DAPO-Qwen2.5-3B-fsdp-singlenode-demo-staleness_3` 跑的实际配置里哪些真正开着。

**1. Clip-Higher（非对称 clip range）**

- 问题：标准 PPO/GRPO 用对称 clip range `[1-ε, 1+ε]`。GRPO 里 advantage 是标量、贴给整条 response 的每个 token；当一条概率本来就低的 response（模型不常走的稀有正确解法）拿到正 advantage 时，对称的小 ε 上限把它的更新幅度也摁住了，模型难以提升这类"稀有但正确"路径的概率——长期训练下容易**熵坍缩**：只在已经高概率的路径上反复加强，多样性消失。
- 做法：把上下界分开，`clip_ratio_high > clip_ratio_low`，给正 advantage、低概率 token 更大的上探空间，下界保持较紧防止 KL 过度发散。
- 源码（`third_party/verl/verl/trainer/ppo/core_algos.py:1279-1369`，`compute_policy_loss_vanilla`）：
  ```python
  clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
  clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
  ...
  pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
  clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
  ```
- 本次 demo 实际配置（`examples/dapo_trainer/qwen2.5_3b_fsdp.sh:47-48,133-134`，日志同步确认）：`clip_ratio_low=0.2`，`clip_ratio_high=0.28`——正 advantage 时 ratio 最多涨到 1.28，负 advantage 时最多跌到 0.8，非对称就体现在这里。
- 顺带一提 **Dual-Clip**（`clip_ratio_c=10.0`，同一脚本第 135 行）：这是另一篇论文的技巧（arXiv:1912.09729），不是 DAPO 本身提出的，但和 Clip-Higher 用在同一个 loss 函数里。当 advantage 为负、ratio 很小时，无 clip 的 `-ratio·A` 会变成一个很大的正 loss，反而鼓励把这个该被抑制的动作概率抬回去；`pg_losses3 = -advantages * clip_ratio_c` 给负 advantage 场景的 loss 加了个下限，防止这种情况把训练打飞：
  ```python
  pg_losses3 = -advantages * clip_ratio_c
  clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
  pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
  ```

**2. Dynamic Sampling（过滤零方差组）**

- 问题：GRPO advantage 公式的分母是组内 reward 的 std。一组 `rollout_n` 条 response 如果全对或全错，std=0，这组所有 token 的 advantage 恒为 0——这批生成和这批梯度全部浪费。
- 做法：按组算 reward 方差，方差为 0 的组整体丢弃，不断从数据集里补采新 prompt，直到凑够 batch size 个"组内有区分度"的组再进训练。
- 源码（`psrl/utils/post_processor/buffer_post_process/filter.py:31-95`，`DynamicSamplingFilterProcessor`）：
  ```python
  pid = uid // self.rollout_n
  pid2metric_vals[pid].append(metric_val)
  ...
  kept_pids = {pid for pid, std in pid2metric_std.items() if std > 0 or len(pid2metric_vals[pid]) == 1}
  kept_traj_idxs = [idx for idx, pid in enumerate(pid_list) if pid in kept_pids]
  ```
- **这次 demo 的真实情况（核对结论）**：运行日志 `DAPO-Qwen2.5-3B-fsdp-singlenode-demo-staleness_3.log` 里
  ```
  group_post_process: {'enable': False, ...}    # 第 994 行
  buffer_post_process: {'enable': False, 'name': None}   # 第 938 行
  ```
  这个过滤器在这次实验里其实**没有真正跑**——尽管启动脚本第 94-95 行写了 `psrl.group_post_process.enable=False psrl.group_post_process.name=dynamic_sampling_filter`（`name` 指定了但 `enable=False`，等于埋了配置没打开开关）。所以严格说，这次 demo 实际用来"补数据"的机制是同一份日志里 `enable=True` 的 `redundant_rollout`（`alg_rollout_n=4`、`redundant_rollout_n=4`、`alg_global_batch_size=16`）——多采一批冗余样本作为缓冲，但并不等价于"检测到零方差组就丢弃重采"的 Dynamic Sampling 语义，两者是 PSRL 里两条不同的补偿机制，这次跑的是前者、没跑后者。

**3. Token-level Policy Gradient Loss（token-mean 聚合）**

- 问题：GRPO 论文原始 loss 是先对每条 response 内部的 token 取平均，再对 batch 内的 response 取平均（"先条内、后条间"）。一条 4000 token 的长 response 和一条 20 token 的短 response 对总 loss 贡献的权重相同——长 response 里单个 token 的梯度被自己那几千个 token 稀释了，等价于长而正确的推理链被系统性欠训练。
- 做法：把 batch 内所有 response 的 token loss 直接相加，除以 batch 内总 token 数，每个 token 无论属于长/短 response 都拿到同量级的梯度权重。
- 源码（`core_algos.py:1138-1197`，`agg_loss`，`token-mean` 分支）：
  ```python
  if loss_agg_mode == "token-mean":
      loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
  ```
  对比 GRPO 论文原始的 `seq-mean-token-mean` 分支（同函数里另一个分支，先条内平均再条间平均）。
- 本次 demo 配置（脚本第 55、146 行）：`loss_agg_mode="token-mean"`——走的正是 DAPO 这条路径。

**4. Overlong Reward Shaping（超长响应软惩罚）**

- 问题：训练设了 `max_response_length` 硬截断。一条推理本身正确、只是啰嗦超长被截断的 response，如果直接按"没有正常 stop"判 0 分，等于给了它和"推理错误"完全一样的惩罚——这个惩罚跟对错无关，纯粹是长度噪声，会干扰训练信号。
- 做法：不一刀切，超过某个缓冲阈值后按超出量线性给一个小的负分，只有真正顶到硬上限才扣满。
- 源码（`psrl/workers/reward/reward_loop/dapo.py:98-107`，`DAPORewardManager.run_single`）：
  ```python
  overlong_buffer_len = self.overlong_buffer_cfg.len
  expected_len = self.max_resp_len - overlong_buffer_len
  exceed_len = valid_response_length - expected_len
  overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
  reward += overlong_reward
  ```
- 本次 demo 配置（脚本第 50、52-54 行）：`max_response_length=4096`，`overlong_buffer_len=2048`，`overlong_penalty_factor=1.0`——超过 2048 token 开始线性扣分，到 4096 token 扣满 1.0 分，留了 2048 token 的缓冲区而不是在 2048 直接判死刑。日志确认 `overlong_buffer_cfg.enable=True`（第 1127 行），这一项在本次 demo 里是**真正启用**的。

**小结对比表（承接 [§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 的 PPO vs GRPO）**

| | PPO | GRPO | DAPO |
|---|---|---|---|
| advantage 估计 | critic（GAE，逐 token） | 组内归一化 reward（标量广播） | 同 GRPO，不变 |
| clip range | 对称 `[1-ε,1+ε]` | 对称 `[1-ε,1+ε]` | 非对称 `high>low`（Clip-Higher）+ 可选 dual-clip 下限 |
| 组/样本过滤 | 无 | 无 | 可选：reward 方差为 0 的组整体丢弃、补采新 prompt（本次 demo **关闭**，实际靠 `redundant_rollout` 补冗余样本） |
| loss 聚合 | 逐 token | 通常 sample-level（条内先平均） | token-level：整个 global batch 的 token 一起平均（本次 demo **开启**） |
| 超长惩罚 | 视实现而定，常见硬截断 0 分 | 同左 | 软惩罚，缓冲区内线性扣分（本次 demo **开启**） |

一句话记忆：DAPO = GRPO 的 advantage 公式不变 + 四个"防止训练因工程细节而失真/失效"的补丁；四个补丁可独立开关，本次 demo 实际开了 Clip-Higher（含 dual-clip）、Token-level Loss、Overlong Reward Shaping 三项，Dynamic Sampling 那一项虽然写进了脚本参数但 `enable=False`，并未真正生效。

## 6.0d 熵奖励 `H` 与 KL 惩罚 `D_KL(π‖π_ref)`（附 PSRL/verl 源码）

**一句话结论**：完整的 actor loss 是 `policy_loss = pg_loss − entropy_coeff·H + kl_loss_coef·D_KL(π‖π_ref)`（后一项仅当 `use_kl_loss=True` 时加入）；`H` 是策略对每个 token 输出分布的**信息熵**，鼓励探索、防止过早收敛到确定性输出（熵坍缩）；`D_KL` 是训练策略与冻结参考策略之间的**KL 散度**，防止训练策略跑得离参考策略太远。这两项在 `fsdp_qwen2.5_3b_laps.sh` 里都被**关闭**了（`entropy_coeff=0`、`use_kl_loss=False`），是 DAPO 类配置的常见做法。

**loss 组装的原始代码**（`third_party/verl/verl/workers/utils/losses.py:174-217`）：

```python
policy_loss_fn = get_policy_loss_fn(loss_mode)
pg_loss, pg_metrics = policy_loss_fn(...)
policy_loss = pg_loss

# entropy bonus（减号：熵越大，loss 越小，鼓励高熵）
entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
entropy_coeff = config.entropy_coeff
policy_loss -= entropy_coeff * entropy_loss

# KL 惩罚（加号：与参考策略越远，loss 越大）
if config.use_kl_loss:
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    policy_loss += kl_loss * config.kl_loss_coef
```

这和用户问题里写的公式 $L = L_{PG} - \lambda_H \cdot H + \lambda_{KL} \cdot D_{KL}(\pi \Vert \pi_{ref})$ 完全对应：`entropy_coeff` 就是 $\lambda_H$，`kl_loss_coef` 就是 $\lambda_{KL}$。

**`H`：逐 token 的分类熵（Shannon entropy）**

来源：`compute_entropy_loss`，`third_party/verl/verl/trainer/ppo/core_algos.py:2073`：

```python
def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)
    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)
    """
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss
```

数学上，对每个 token 位置，策略在词表 $V$ 上给出一个分布 $\pi(\cdot \mid s)$，该位置的熵是标准 Shannon entropy：

$$H(\pi(\cdot \mid s)) = -\sum_{v \in V} \pi(v \mid s) \log \pi(v \mid s)$$

再按 `loss_agg_mode`（这里是 `token-mean`）对 response 里所有 token 取平均，得到标量 $H$。熵越大说明模型对下一个 token 越"不确定"（分布越平），熵越小说明模型越"武断"（趋近 one-hot）。把 $-\lambda_H \cdot H$ 加进 loss（即在 loss 里做**减法**）等价于在梯度上鼓励熵变大，是防止 RL 训练早期就把策略压成几乎确定性输出（"熵坍缩"，在 [§6.0c](#60c-dapo-在-grpo-基础上的四个改进附-psrl-源码) Clip-Higher 的动机部分也提到过这个词）的标准正则手段。这次 laps 训练把 `entropy_coeff` 设成 `0`，即完全关闭这个正则项——常见于 DAPO 类配置，因为 Clip-Higher 本身已经缓解了熵坍缩问题，不再需要熵奖励兜底。

**`D_KL(π‖π_ref)`：KL 散度惩罚，5 种估计量**

来源：`kl_penalty_forward`，`third_party/verl/verl/trainer/ppo/core_algos.py:2160`（`kl_penalty` 在 2132 行是它的 wrapper，处理 `"+"` 后缀的直通梯度技巧）：

```python
def kl_penalty_forward(logprob, ref_logprob, kl_penalty) -> torch.FloatTensor:
    """Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html"""
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == "full":
        raise NotImplementedError  # 需要完整 vocab 的 logits，未实现
    raise NotImplementedError
```

这里的 `logprob`/`ref_logprob` 都是**当前采样到的 token** 在训练策略 / 参考策略下的 log 概率（标量，逐 token），不是完整分布。这是因为精确计算 $D_{KL}(\pi \Vert \pi_{ref}) = \sum_v \pi(v)\log\frac{\pi(v)}{\pi_{ref}(v)}$ 需要两边完整的 vocab 分布，代价很大（`"full"` 分支就是这个精确版本，未实现）；实践中改用**蒙特卡洛估计**——只用实际采样到的那个 token 来估计期望，牺牲精确度换取效率：

| 估计量 | 公式（$r = \log\pi - \log\pi_{ref}$） | 性质 |
|---|---|---|
| `k1`/`kl`（naive） | $r$ | 期望上等于真实 KL（当从 $\pi$ 采样时），但方差大、可能为负 |
| `abs` | $\lvert r \rvert$ | 简单粗暴，无严格统计意义 |
| `k2`/`mse` | $\tfrac{1}{2}r^2$ | 恒非负、方差低，梯度是"正确"的（详见下方梯度说明），但期望值有偏 |
| `k3`/`low_var_kl` | 令 $\delta=-r$，clamp 到 $[-20,20]$ 后取 $e^{\delta}-\delta-1$，再 clamp 到 $[-10,10]$ | Schulman 提出的无偏、低方差估计（http://joschu.net/blog/kl-approx.html），实践中最常用 |
| `full` | 精确 $\sum_v \pi(v)\log(\pi(v)/\pi_{ref}(v))$ | 未实现，需要完整 vocab 分布 |

`k3` 的构造直觉：$k_1$（naive）的问题是虽然 $\mathbb{E}[r]$ 等于真实 KL，但单样本估计的方差很高，容易出现负值（KL 本应非负）；$k_3$ 用 $e^{-r}-(-r)-1$（本质是 $D_{KL}$ 与其一阶近似之差的形式）在**保持期望不变**的同时大幅降低方差，且恒为非负值，是三者中最"物理直觉正确"的版本。下面单独展开 `k3` 的推导。

#### `k3` 估计量的数学推导：控制变量法（control variate）

来源：Schulman, "Approximating KL Divergence"（http://joschu.net/blog/kl-approx.html）。为避免和上面表格里 $r=\log\pi-\log\pi_{ref}$（对数比）混淆，这里换一个符号：记**真正的重要性比值**（不取对数）为

$$\rho(x) = \frac{\pi_{ref}(x)}{\pi(x)}$$

对应代码：`kl = ref_logprob - logprob` $= \log\pi_{ref}-\log\pi = \log\rho$，`ratio = exp(kl)` $=\rho$。样本 $x$ 是训练时从当前策略 $\pi$ 采样出来的（rollout 引擎按 $\pi$ 生成轨迹），要估计的目标是

$$D_{KL}(\pi\|\pi_{ref}) = \mathbb{E}_{x\sim\pi}\Big[\log\frac{\pi(x)}{\pi_{ref}(x)}\Big] = \mathbb{E}_{x\sim\pi}[-\log\rho(x)]$$

**第一步：$k_1$ 就是这个定义式本身的单样本蒙特卡洛估计。**

$$k_1(x) = -\log\rho(x) = \log\pi(x)-\log\pi_{ref}(x) = \texttt{logprob} - \texttt{ref\_logprob}$$

它的期望精确等于 $D_{KL}(\pi\|\pi_{ref})$（这只是重要性采样恒等式的直接展开，不需要任何近似），但对**单个** $x$，$-\log\rho(x)$ 完全可能是负数（比如这一步恰好采到了一个 $\pi_{ref}$ 比 $\pi$ 给出更高概率的 token），而真实 KL 散度作为期望值必然 $\ge 0$——问题出在"单样本估计"和"期望"之间的方差,不是期望本身错了。

**第二步：找一个零均值的"控制变量"去抵消方差,同时不改变期望。**

控制变量法的原理：若 $Z$ 是任意满足 $\mathbb{E}_{x\sim\pi}[Z(x)]=0$ 的随机变量，那么对任何无偏估计量 $k$，$k+Z$ 仍然无偏（期望不变），但方差可能更低（如果 $Z$ 和 $k$ 的波动相关）。这里最自然的候选是 $\rho(x)-1$ 本身，因为重要性采样有一个恒等式：

$$\mathbb{E}_{x\sim\pi}[\rho(x)-1] = \sum_x \pi(x)\cdot\frac{\pi_{ref}(x)}{\pi(x)} - 1 = \sum_x \pi_{ref}(x) - 1 = 1-1 = 0$$

（倒数第二步用到 $\pi_{ref}$ 是一个归一化的概率分布，$\sum_x\pi_{ref}(x)=1$）。于是定义

$$k_3(x) := k_1(x) + \big(\rho(x)-1\big) = -\log\rho(x) + \rho(x) - 1 = \big(\rho(x)-1\big) - \log\rho(x)$$

对应代码 `kld = ratio - kl - 1`（`ratio` $=\rho$，`kl` $=\log\rho$），逐项对上。因为加的是零均值项，$k_3$ 仍然精确无偏：

$$\mathbb{E}_{x\sim\pi}[k_3(x)] = \mathbb{E}_{x\sim\pi}[k_1(x)] + 0 = D_{KL}(\pi\|\pi_{ref})$$

**第三步：为什么 $k_3$ 恒非负。** 令 $f(\rho)=\rho-1-\log\rho$（$\rho>0$）。求导：$f'(\rho)=1-1/\rho$，在 $\rho=1$ 处为零；$f''(\rho)=1/\rho^2>0$，说明 $\rho=1$ 是全局最小值点，且 $f(1)=0$。所以

$$f(\rho) = \rho-1-\log\rho \ge 0,\quad \forall \rho>0,\ \text{等号成立当且仅当 } \rho=1$$

这正是常见不等式 $\log x \le x-1$ 的等价形式。因此 $k_3(x)=f(\rho(x))\ge 0$ **对每一个单样本都成立**，不像 $k_1$ 只在期望意义下非负——这也是它被叫 "low_var_kl" 的另一层原因：不但方差低，数值本身也和"散度应当非负"这个物理约束一致。

**第四步：为什么方差更低——在 $\pi\approx\pi_{ref}$ 附近做泰勒展开。** RL 微调场景下每一步的策略更新幅度通常很小，$\pi$ 和 $\pi_{ref}$ 差别不大，即 $\rho(x)\approx 1$。令 $\varepsilon=\rho-1$（小量），用 $\log(1+\varepsilon)=\varepsilon-\frac{\varepsilon^2}{2}+\frac{\varepsilon^3}{3}-\cdots$ 展开：

$$k_1 = -\log\rho = -\varepsilon+\frac{\varepsilon^2}{2}-\cdots \quad\Rightarrow\quad k_1 = O(\varepsilon)\ \text{（一阶量）}$$

$$k_3 = \varepsilon-\log(1+\varepsilon) = \varepsilon-\Big(\varepsilon-\frac{\varepsilon^2}{2}+\frac{\varepsilon^3}{3}-\cdots\Big) = \frac{\varepsilon^2}{2}-\frac{\varepsilon^3}{3}+\cdots \quad\Rightarrow\quad k_3 = O(\varepsilon^2)\ \text{（二阶量）}$$

而真实的 $D_{KL}(\pi\|\pi_{ref})$ 本身，在两个分布接近时也是关于扰动的**二阶小量**（这是 KL 散度的标准性质：一阶项恒为零，因为 $\pi=\pi_{ref}$ 处 KL 取到最小值 0 且是光滑的最小值点，Hessian 对应到 Fisher 信息量级）。也就是说：

- $k_1$ 是一阶量，但要估计的目标是二阶小量——单样本估计里包含了一个数量级远大于信号本身的"一阶噪声"，方差主要来自这个被抵消掉的一阶项在不同样本间的剧烈波动。
- $k_3$ 直接就是二阶量，和目标量本身同阶，样本间的波动幅度和信号幅度是同一量级，相对噪声小得多。

这就是控制变量 $(\rho-1)$ 的作用本质：它精确抵消掉了 $k_1$ 展开式里方差的主要来源——那个一阶项 $-\varepsilon$（因为 $(\rho-1)-\log\rho$ 的一阶项 $\varepsilon-\varepsilon=0$，只留下二阶开始的项），而不改变期望。

**和 $k_2$ 的关系**：$k_2=\tfrac12(\log\rho)^2\approx\tfrac12\varepsilon^2$，泰勒展开的leading order 和 $k_3$ 完全一样（都是 $\varepsilon^2/2$），这就是为什么两者方差表现接近、且都恒非负（$k_2$ 是平方，显然非负）。但 $k_2$ 的无偏性没有 $k_3$ 这样精确的代数证明——它只是在 $\rho\approx1$ 的展开阶数上和真实 KL"对得上"，全局范围内 $\mathbb{E}_{x\sim\pi}[(\log\rho)^2]/2$ 并不精确等于 $D_{KL}(\pi\|\pi_{ref})$；而 $k_3$ 无论 $\rho$ 离 1 多远，第二步的控制变量论证都严格成立，是**在整个定义域上精确无偏、且同时保有低方差和非负性**的估计量，这也是 Schulman 推荐把它作为默认选择、verl 把它设成 `low_var_kl`/`k3` 这个名字的原因。

代码里两次 `clamp`（`kl` 先 clamp 到 $[-20,20]$ 再 `exp`，`kld` 最后 clamp 到 $[-10,10]$）纯粹是数值稳定性的工程考虑——$\rho=e^{\log\rho}$ 在 $\log\rho$ 很大时会数值溢出或梯度爆炸，和上面的数学推导本身无关，理论上 $k_3$ 不需要下界保护（已经证明恒 $\ge0$），上界保护则是为了防止某个异常 token 的极端概率比把整个 batch 的梯度带偏。

`kl_penalty`（2132 行，带 `"+"` 后缀时的直通梯度技巧）：

```python
def kl_penalty(logprob, ref_logprob, kl_penalty) -> torch.FloatTensor:
    base_kl_penalty = kl_penalty[:-1] if kl_penalty.endswith("+") else kl_penalty
    forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score
    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()
```

比如配置成 `"k3+"`：前向数值上用的是 `k3`（无偏、低方差，数值本身更准），但反向传播时梯度用的是 `k2`（`0.5*r^2`）的梯度——因为 `k2` 的梯度形式在数学上被证明是"正确"的（无偏梯度估计），而 `k1`/`k3` 数值上更准但梯度方向有偏。`backward_score - backward_score.detach() + forward_score.detach()` 是标准的直通估计（straight-through estimator）写法：前向传播时 `backward_score - backward_score.detach() = 0`，所以数值上等于 `forward_score`；反向传播时 `forward_score.detach()` 不产生梯度，梯度全部来自 `backward_score`。这样就把"数值用 k3，梯度用 k2"缝合到了一次计算里。

**这次训练里的实际取值**：`fsdp_qwen2.5_3b_laps.sh` 里 `use_kl_loss=False`、`kl_loss_coef=0.0`、`algorithm.use_kl_in_reward=False`——KL 惩罚在 loss 和 reward 两条路径上都被关掉了。这也是 DAPO 论文的典型选择：GRPO/DAPO 本身用组内相对 reward 做 advantage，没有像 PPO 那样一个容易被过度优化的 critic，训练更依赖 clip 机制（Clip-Higher）而非 KL 约束来控制更新幅度，所以经常直接关闭 KL 惩罚，省掉维护参考模型前向计算的开销。

## 6.0e 输入输出的 label 设计与 `rollout_correction` 修正（附 PSRL 源码，对照 `fsdp_qwen2.5_3b_laps.sh`）

**一句话结论**：训练用的"标签"不是 token id 本身（那是自回归交叉熵天然自带的），而是一条 `response_mask`——它决定哪些 token 参与 loss；reward 是一个标量，只贴在 response 的最后一个 token 上；这次脚本没有走 [§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 说的 bypass 捷径，而是真正重算 `old_log_probs`，并且额外打开了 `algorithm.rollout_correction.rollout_is=token` 这层修正——用重算出来的 `old_log_probs` 和 vLLM 生成时的 `rollout_log_probs` 的差距，反过来给 pg_loss 加一个逐 token 的重要性权重。

**1. 多轮 agentic rollout 怎么变成 `(prompt_ids, response_ids, response_mask)`**

代码在 `psrl/utils/tito/training_data.py::build_training_data()`（42–232 行）。核心逻辑：`prompt_ids` 取第一轮请求时的 prompt token；从第一轮往后，凡是**模型自己生成**的 token 打 `response_mask=1`，凡是**插进来的工具/环境输出**（下一轮 prompt 里比上一轮 response 多出来的那一段）打 `response_mask=0`：

```python
# tool/环境 token（两轮之间插进来的部分）
if cursor > 0 and prompt_len > cursor:
    env_ids = accumulated_token_ids[cursor:prompt_len]
    all_response_ids.extend(env_ids)
    all_response_mask.extend([0] * len(env_ids))      # 不参与 loss

# 模型自己生成的 token
all_response_ids.extend(output_ids)
all_response_mask.extend([1] * len(output_ids))       # 参与 loss
```

对应到这次 `fsdp_qwen2.5_3b_laps.sh` 的 `gen_actor_rollout_ref.rollout.multi_turn.max_turns=50`：一条轨迹可能有 50 轮工具调用，`response_ids` 是"模型输出"和"工具输出"交替拼接的一条长序列，但 loss 只在 `response_mask=1`（模型自己吐出来的 token）的位置计算——这就是 [§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 步骤⑧"只在 response_mask=1 的位置求和"背后的具体构造方式。

**2. reward 怎么变成一个 token 级张量：只贴最后一个 token**

`DAPORewardManager.run_single`（`psrl/workers/reward/reward_loop/dapo.py:42-98`）算出标量 `reward_score` 后，`psrl/workers/reward/reward_manager.py::normalize_reward()`（348-355 行）把它写回 `rm_scores`：

```python
rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
rm_scores[:, -1] = torch.as_tensor(reward_scores, ...)   # 只有最后一个 token 非零
data["rm_scores"] = rm_scores
```

即整条 response 只有最后一个有效 token 拿到非零 reward，其余位置全是 0——这是 GRPO/DAPO 这类 outcome-reward 场景的标准做法，`compute_grpo_outcome_advantage`（[§6.0b](#60b-group-rollout-的工程实现uid--parent_id-与磁盘上的重复-prompt)）里 `scores = token_level_rewards.sum(dim=-1)` 也印证了"reward 只集中在最后一个位置，sum 等价于取那一个值"。

**3. `old_log_probs` 这次走的是哪条路**

`ray_trainer.py::_compute_old_log_prob()`（2973–3049 行）用 `algorithm.rollout_correction.bypass_mode` 做二选一：

```python
if bypass_recomputing_logprobs:              # bypass 模式：直接拿 vLLM 的 log prob
    data["old_log_probs"] = data.pop("rollout_log_probs")
    policy_loss_config["loss_mode"] = "bypass_mode"
else:                                          # 默认：训练侧重新做一次 teacher-forcing 前向
    output = self.actor_wg.compute_log_prob(batch)   # "recompute log_prob"
    data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
```

`fsdp_qwen2.5_3b_laps.sh` 全脚本没有设置 `algorithm.rollout_correction.bypass_mode`（只设了 `psrl.log_prob.enable_rollout_engine_log_prob=True`，这只是让 vLLM **额外汇报**自己的 log prob、填充 `rollout_log_probs` 字段，不等于启用 bypass），所以走的是默认的**重算路径**：Megatron/FSDP 对整条 `(prompt_ids, response_ids)` 做一次并行前向，重新算出每个 `response_mask=1` 位置的 log prob 当 `old_log_probs`——这和之前训练日志里看到的 `"recompute log_prob"` 阶段名完全对应。

**4. `rollout_correction`：拿 vLLM 的 log prob 和重算的 log prob 做逐 token 重要性修正**

正因为①重算出的 `old_log_probs` 和②vLLM 采样时的 `rollout_log_probs` 是两套引擎、两个时间点算出来的（[§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例) 已经讲过为什么两者会有数值差），`rollout_is=token` 这条路径把这个差距本身当作一次额外的重要性采样修正，而不是简单信任重算结果。核心公式（`third_party/verl/verl/trainer/ppo/rollout_corr_helper.py::compute_rollout_correction_weights()`，520-594 行）：

$$
w_t=\min\!\Big(\exp\big(\operatorname{clip}(\log\pi_{\text{old}}(a_t)-\log\pi_{\text{rollout}}(a_t),\,-B,\,B)\big),\ \tau\Big),\qquad \tau=\texttt{rollout\_is\_threshold}=2.0
$$

```python
raw_rollout_is_weights = torch.exp(torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND))
raw_rollout_is_weights = raw_rollout_is_weights * response_mask
rollout_is_weights = raw_rollout_is_weights.clamp(max=rollout_is_threshold_upper)   # 这里 =2.0
```

这个逐 token 权重 `w_t` 被直接乘进 [§6.0c](#60c-dapo-在-grpo-基础上的四个改进附-psrl-源码) 的 clip 目标（`core_algos.py:1363-1364`）：`pg_losses = pg_losses * rollout_is_weights`，再走 `agg_loss`（token-mean）聚合。直觉：如果某个 token 位置 vLLM 采样时的概率和训练侧重算的概率差得很远（两套引擎数值不一致，或者这批数据消费时权重已经漂移了几步），这个 token 对最终 pg_loss 的贡献会被按比例调低（封顶在 `τ=2.0`，防止个别 token 的权重把梯度带飞）——是在 [§6.0](rl-training-workflow.md#60-同时在线具体怎么理解以-psrl-为例)"为什么不能直接复用 rollout_log_probs"那段讨论的数值差距/staleness 问题上，追加的一层运行时补偿，而不是替代重算。

---

**回到最初的问题**：`fsdp_qwen2.5_3b_laps.sh` 这次训练里，输入是一条 `(prompt_ids, response_ids)`，"标签"体现为两层——`response_mask`（决定哪些 token 参与 loss，把工具/环境输出的 token 排除在外）和最后一个 token 上的标量 `reward`（决定这条 response 的好坏，经 GRPO 组内归一化变成逐 token 均匀广播的 `advantage`）；loss 是 `pg_loss(clip, ×rollout_is_weights) − entropy_coeff·H + kl_loss_coef·D_KL`（这次 `entropy_coeff=0`、`use_kl_loss=False`，只剩第一项），用 `token-mean` 聚合成标量后 `backward()`。

**对照自学的源码顺序**（数据/rollout → loss/优化器）：

1. `psrl/utils/tito/training_data.py` — 一条多轮轨迹如何变成 `(prompt_ids, response_ids, response_mask)`
2. `psrl/workers/gen/utils.py`（`TokenOutput.as_dict`） — vLLM 输出（`responses`、`rollout_log_probs`）怎么打包
3. `psrl/workers/reward/reward_loop/dapo.py` + `psrl/workers/reward/reward_manager.py`（`normalize_reward`） — reward 打分与 `rm_scores` 组装
4. `psrl/trainer/ppo/ray_trainer.py` — `_compute_old_log_prob` / `_compute_ref_log_prob` / `_compute_advantage`（把上面几步串起来，`rollout_correction` 也在这里触发）
5. `psrl/trainer/ppo/strategies/full_batch.py`（配合 `strategies/base.py` 的 `STAGE_META`） — 每一步真正的 phase 顺序
6. `third_party/verl/verl/trainer/ppo/core_algos.py` — GRPO advantage、PPO clip loss、`agg_loss`
7. `third_party/verl/verl/workers/utils/losses.py`（`ppo_loss`） — 最终标量 loss 组装
8. `third_party/verl/verl/trainer/ppo/rollout_corr_helper.py` — `rollout_is` 权重计算
