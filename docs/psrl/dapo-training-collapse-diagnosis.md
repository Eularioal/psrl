# DAPO 训练曲线诊断：熵崩溃 → 长度打满 → Reward 归零的连锁崩溃

## 一句话结论

`dapo-qwen2.5-3b-fsdp.png`（`psrl/scripts/`）展示的是一次 DAPO + Qwen2.5-3B 在 GSM8K 上的 RLVR 训练：前 ~110 步健康（验证准确率涨到 ~0.85），step ~115 起发生**不可逆崩溃**——熵崩溃、`response_length` 打满 max、reward 钉死在 -1、`group/zero_variance_fraction` 冲到 1、梯度范数爆炸、验证准确率砸到 ~0，且此后未再恢复，后 85 步（约 40% 训练量）基本是无效算力。

## 训练过程时间线

| 阶段 | step 区间 | 现象 |
|---|---|---|
| 健康期 | 0–45 | `entropy` 0.3→0.1 缓降，`critic/score` 0.2→0.9，验证准确率 0.18→0.85，`response_length` 稳定 ~300 |
| 预警性崩溃（自愈） | ~50–60 | `loss` 冲到 1.0，`response_length` 短暂窜到 1150，`critic/score` 砸到 0.1，验证准确率跌到 0.1；随后自行恢复 |
| 主崩溃（不可逆） | ~115 起 | `grad_norm` 飙到 150+（后续破千），`response_length` 打满 4096，`critic/score`/`rewards` 砸到 -1.0，`zero_variance_fraction`→1.0，验证准确率→~0 |
| 失败的反弹尝试 | ~155–170 | `entropy` 异常飙升到 0.6–0.75（远超健康期水平，为"胡言乱语"式高熵而非健康探索），验证准确率昙花一现到 0.43，随后再次坍缩 |
| 死寂期 | 175–200 | 所有指标钉死：`entropy`≈0、`response_length`≈4096、`reward`≈-1、验证准确率≈0 |

## 逐项问题：证据 → 判断依据 → 原理 → 危害

### 1. 熵崩溃（entropy collapse）
- **证据**：`actor/entropy` 在 step 20–100 持续走低（0.4→0.05），step 180 后跌到 0 不再变化。
- **判断依据**：entropy $H=-\sum_t p(t)\log p(t)$ 衡量每个 token 位置输出分布的发散程度，趋近 0 即策略输出近似 one-hot 确定性预测。
- **原理**：纯 on-policy RL 微调若无足够强的熵正则/KL 约束，会把概率质量单调集中到历史高奖励 token 序列上。
- **危害**：熵低 → 同一 prompt 的 `n` 条 rollout 高度雷同 → 组内奖励方差趋于 0（见问题 2）→ 梯度信号消失 → 策略丧失自我修正能力，为硬崩溃埋雷。

### 2. `group/zero_variance_fraction` 飙升到 1.0
- **证据**：指标直接统计"同一 prompt 的 n 条 rollout 中奖励方差为 0（全对或全错）"的组占比，step 120–155 与 175–200 几乎恒为 1。
- **原理**：GRPO/DAPO 的 advantage 为 $\hat A_i=(r_i-\text{mean}(r))/\text{std}(r)$，std=0 时该组 advantage 无效。DAPO 设计了 dynamic sampling 过滤退化组并重采（参见 [ppo-grpo-dapo-sample-requirements.md](ppo-grpo-dapo-sample-requirements.md)）。
- **危害**：大量 rollout 算力被浪费在零方差样本上；仅存的"有方差"组承担了不成比例的梯度权重，噪声被放大，对应此时 `grad_norm` 的剧烈毛刺。

### 3. Reward 精确钉死在 -1.0
- **证据**：`critic/score/mean`、`critic/rewards/mean` 在 step 115 后几乎恒为精确的 -1.0（非围绕 -1 的噪声波动）。
- **判断依据**：精确钉在常数值（而非噪声分布）通常意味着触发了奖励函数的硬编码惩罚分支（如"格式非法/未在 max length 内给出可解析答案"），而非"答案错误"（后者一般给 0 且随对错波动）。
- **危害**：全 batch 拿到同一 reward 下限 → 组内方差再次归 0 → 与问题 2 形成正反馈死循环，无任何梯度线索能指回"给出简短正确答案"。

### 4. `response_length` 打满 max（4096）不回落
- **证据**：从 ~300 瞬间跳到硬顶 ~4096，此后长期钉死（非自然波动到较大值）。
- **原理**：熵崩溃 + 剧烈梯度更新后，策略可能失去"该输出 EOS 时输出 EOS"的校准能力，生成重复/跑题/不连贯文本直到被 `max_response_length` 硬切断——RLHF/RLVR 中经典的 "EOS collapse" 故障模式。
- **危害**：
  - 每步耗时翻两三倍（`timing_s/step` 从 15–50s 跳到 90–270s）；
  - 被截断的回复几乎必然触发问题 3 的格式惩罚，构成死亡螺旋；
  - on-policy RL 自身梯度很难走出该局部陷阱，通常需要外部干预（回滚 checkpoint、降 LR、奖励整形）。

### 5. `grad_norm` / `actor/loss` 同步剧烈脉冲（最高上千）
- **证据**：三次尖峰（~55、~115、~160-170）与上述所有异常窗口精确对齐。
- **原理**：clipped surrogate 目标（PPO/GRPO/DAPO 核心）在新旧策略差异过大（重要性比率越界）时，未被裁剪方向的极端 token 仍可能贡献巨大梯度；叠加问题 2 中"仅少数组存活"的情况，幸存样本承担远超比例的梯度权重，导致范数量级跳升。
- **危害**：巨大且方向未必正确的参数更新造成灾难性遗忘，解释了每次 `grad_norm` 尖峰后验证准确率的断崖下跌。

### 6. GSM8K 验证准确率不可逆崩溃（0.85 → ~0）
- **证据**：崩溃时间点与上述所有异常信号高度吻合；崩溃后（除 step ~160 附近昙花一现的 0.43 又跌回 0）再无恢复迹象直到 step 200。
- **原理**：这是前述所有"训练态"异常在任务表现上的最终确认。step 160 附近的反弹伴随 entropy 反常飙升到 0.6–0.75（远超健康期 0.3–0.4），说明并非恢复连贯生成，而是从"过度确定性的退化输出"甩到"接近随机的胡言乱语"，是在不稳定平衡点附近震荡而非真正收敛回稳态。
- **危害**：step 115 之后约 85 步（40%+ 训练量）算力基本无效甚至有害；若不检测直接使用最终 checkpoint，会得到完全退化的模型。

## 建议

这套连锁反应正是 DAPO 论文设计初衷要规避的问题（Clip-Higher 防熵崩溃、Dynamic Sampling 过滤零方差组、超长回复软惩罚而非硬截断惩罚、token-level loss）。出现该现象大概率是相关机制未生效或阈值不够激进：

1. 给 `actor/entropy`、`group/zero_variance_fraction`、`response_length/mean` 设自动报警/回滚阈值，在第一次小崩溃（如本例 step ~55）即触发；
2. 检查超长回复是否用的是硬 -1 惩罚而非 soft length penalty（overlong reward shaping）；
3. 确认 dynamic sampling 是否真正过滤+补采零方差组，而不只是记录 `zero_variance_fraction` 却未作为控制信号。

## 附录：图中各项指标在 PSRL 里是怎么算出来的（代码溯源）

PSRL 是 [verl](https://github.com/volcengine/verl) 的 fork：vendor 了一份打过 patch 的 verl（`third_party/verl/verl`），自己的 `psrl/trainer/ppo/ray_trainer.py` 重写了调度层（换成 TransferQueue 异步 pipeline），但**所有指标的实际数学计算都直接复用 verl 的 `core_algos.py` / `metric_utils.py` / `workers/utils/losses.py`**，只有 `group/zero_variance_fraction` 是 PSRL 自己加的。下文路径未加前缀的默认在 vendored verl 下，显式标注 `psrl:` 的在 `psrl/` 包内。

### `actor/entropy`

$$H(\pi(\cdot\mid s_t)) = -\sum_{v} p_v \log p_v$$

`utils/torch_functional.py::entropy_from_logits` 用 logsumexp 技巧数值稳定地实现：

```python
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
```

图中数值取自**旧策略 $\pi_{\text{old}}$ 的 log-prob 计算阶段**（`psrl: ray_trainer.py::_compute_old_log_prob`），按 token 聚合：

```python
entropy_agg = agg_loss(loss_mat=data.batch["entropy"], loss_mask=data.batch["response_mask"],
                        loss_agg_mode=actor_config.loss_agg_mode)
old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
```

### `actor/loss`

`workers/utils/losses.py::ppo_loss`：

$$\mathcal{L} = \mathcal{L}_{\text{PG}} - \beta_{\text{ent}} \cdot H \;(+\, \lambda_{\text{KL}}\cdot D_{KL}(\pi\|\pi_{\text{ref}})\text{，若启用 KL loss})$$

裁剪代理目标是 DAPO 的非对称 dual-clip 版本（`trainer/ppo/core_algos.py::compute_policy_loss_vanilla`）：

$$r_t=\exp(\log\pi-\log\pi_{\text{old}}),\quad
\mathcal{L}_{\text{PG}}=-\mathbb{E}\Big[\min\big(r_t\hat A_t,\ \text{clip}(r_t,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\hat A_t\big)\Big]$$

```python
ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -20, 20))
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
pg_losses3 = -advantages * clip_ratio_c        # dual-clip 下界
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

step 50/115/160 三次 `actor/loss` 尖峰，是崩溃瞬间 $\hat A_t$（见下）与 ratio 同时异常导致的。

### `actor/grad_norm`

FSDP engine `optimizer_step`：

```python
grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
if not torch.isfinite(grad_norm):
    self.optimizer.zero_grad()   # 非有限值直接跳过这次更新
else:
    self.optimizer.step()
```

即 $\lVert g\rVert_2=\sqrt{\sum_i g_i^2}$（裁剪前的原始范数），三次尖峰与 `actor/loss` 尖峰精确对齐——图上能看到远超裁剪阈值的量级，说明裁剪只限制了更新步长，没能阻止参数被带偏方向。

### `critic/score/mean` vs `critic/rewards/mean`

`trainer/ppo/metric_utils.py::compute_data_metrics`：

```python
sequence_score  = batch.batch["token_level_scores"].sum(-1)   # 原始 reward 函数输出
sequence_reward = batch.batch["token_level_rewards"].sum(-1)  # 真正喂给 advantage 计算的量
```

关系由 `trainer/ppo/ray_trainer.py::apply_kl_penalty` 建立：

$$\text{token\_level\_rewards} = \text{token\_level\_scores} - \beta \cdot D_{KL}(\pi_{\text{old}}\|\pi_{\text{ref}})$$

若 `use_kl_in_reward=False`（DAPO/GRPO 常见配置，KL 约束走 loss 而非 reward shaping）两者直接相等——这次训练图上二者曲线几乎完全重合，说明本次跑的就是这种配置。

**展开：`token_level_scores` 怎么来的、KL 项怎么算、以及和 GRPO 组内优势的先后关系**

`token_level_scores` 这个名字有误导性：GSM8K 的 `compute_score` 只返回一个**标量**（0/1，整条回复对/错），并不是逐 token 打分。PSRL 把这个标量**散射到回复最后一个有效 token 位置**，其余位置全 0（`psrl/workers/reward/reward_manager.py:637-640`）：

```python
rm_score_tensor = torch.zeros(response_len, dtype=torch.float32)
rm_score_tensor[-1] = float(f.get("reward_score", 0.0))
f["rm_scores"] = rm_score_tensor
```

再由 `psrl/trainer/ppo/ray_trainer.py:2890` 改名成 `token_level_scores`：`data.batch["token_level_scores"] = data.batch["rm_scores"]`。

$D_{KL}(\pi_{\text{old}}\|\pi_{\text{ref}})$ 不是解析式，是逐 token 的采样估计量（`trainer/ppo/core_algos.py::kl_penalty`，设 $l=\log\pi_{\text{old}}(a\mid s)$、$l_{\text{ref}}=\log\pi_{\text{ref}}(a\mid s)$ 为实际采样到的那个 token 的 log-prob）：

| 估计量 | 公式 | 特点 |
|---|---|---|
| `k1` | $k_1=l-l_{\text{ref}}$ | 朴素 log-ratio，无偏但方差大 |
| `k2`/`mse` | $k_2=\tfrac12(l-l_{\text{ref}})^2$ | 梯度性质好 |
| `k3`（低方差） | $d=\text{clip}(l_{\text{ref}}-l,\pm20)$，$k_3=\text{clip}(e^{d}-d-1,\pm10)$ | Schulman 低方差无偏估计，最常用 |

系数 $\beta$（`kl_ctrl.value`）可以是固定值，也可以自适应（`core_algos.py::AdaptiveKLController`，向 `target_kl` 方向乘性调节）：

```python
proportional_error = np.clip(current_kl / target_kl - 1, -0.2, 0.2)
mult = 1 + proportional_error * n_steps / horizon
self.value *= mult
```

完整 `apply_kl_penalty`（`trainer/ppo/ray_trainer.py:78-117`）：

```python
kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
kld = kld * response_mask
beta = kl_ctrl.value
token_level_rewards = token_level_scores - beta * kld
kl_ctrl.update(current_kl=..., n_steps=batch_size)
```

**这一步和 GRPO 组内优势是流水线里两个独立、先后发生的步骤，不是二选一**：

$$
\underbrace{\text{token\_level\_rewards}_t=\text{token\_level\_scores}_t-\beta D_{KL,t}}_{\text{①逐 token，本节公式}}
\;\Rightarrow\;
\underbrace{r_i=\sum_t\text{token\_level\_rewards}_{i,t}}_{\text{②求和成序列级标量}}
\;\Rightarrow\;
\underbrace{\hat A_i=\dfrac{r_i-\text{mean}(r_{1..n})}{\text{std}(r_{1..n})+\epsilon}}_{\text{③GRPO 组内相对优势}}
$$

①决定 $r_i$ 是否混入 KL 惩罚，③完全不关心 $r_i$ 从哪来、只对②的标量做组内标准化——两者正交。本次训练 `use_kl_in_reward=False`，①实际被跳过（`token_level_rewards = token_level_scores`，`psrl/ray_trainer.py:2895-2903` 的 else 分支），KL 约束改走 actor loss 里单独的 `kl_loss` 项；③ GRPO 组内优势始终生效，与①有没有开无关。

### `response_length/mean`

```python
response_length = attention_mask[:, -response_length:].sum(-1)   # 非 padding 的真实 token 数
"response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
```

verl 同批指标里还有 `response_length/clip_ratio`（图中未画），可以精确量化"被截断比例"，比只看均值更能确认 EOS collapse。

### `group/zero_variance_fraction`（PSRL 自加指标）+ GRPO advantage

`psrl: psrl/trainer/ppo/utils.py::_compute_termination_metrics`：

```python
groups: dict[str, list[float]] = defaultdict(list)
for parent_id, score in zip(parent_ids, scores, strict=True):
    groups[parent_id].append(score)
degenerate = sum(1 for g in groups.values() if len(g) > 1 and float(np.std(g)) < 1e-6)
metrics["group/zero_variance_fraction"] = degenerate / len(groups) if groups else 0.0
```

按 `parent_id`（同一 prompt 的 rollout group）分组，统计组内 reward 标准差为 0（全对或全错）的组占比，直接对应 GRPO advantage（`trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`）：

$$\hat A_i=\frac{r_i-\text{mean}(r_1,\dots,r_n)}{\text{std}(r_1,\dots,r_n)+\epsilon}$$

```python
id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
id2std[idx]  = torch.std(torch.stack(id2score[idx]))
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

`std→0` 时 $\hat A_i\to 0$，该组白算一遍前向+反向。图中该指标在 step 120 后钉在 1.0：熵崩溃使同一 prompt 的 n 条 rollout 高度雷同，天然全对或全错，与熵崩溃直接因果相连。

### `perf/mfu/actor`

$$\text{MFU} = \frac{\text{实际估算 FLOPs}}{\text{理论峰值 FLOPs} \times \text{world\_size}}$$

```python
estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)
final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
```

图中几次崩溃窗口的骤降毛刺，是 `response_length` 骤变导致 batch 内 token 数分布突变、短暂打乱估算/实际吞吐比例所致。

### `perf/throughput` 与 `timing_s/step`

```python
"perf/throughput": total_num_tokens / (time * n_gpus),   # tokens/s/GPU
f"timing_s/{name}": timing_raw[name]                       # 该计时段墙钟秒数，"step" 为全步耗时
```

`timing_s/step` 从 15–50s 跳到 90–270s，是 `response_length` 打满 4096 后 decode 步数暴涨的直接代价；`perf/throughput` 图里的高台阶同理——长度打满后单步处理 token 总量变大，throughput 数值"看起来"升高，但这是被拉长的无效生成撑起来的，不代表训练效率提升。

### `val-core/openai/gsm8k/acc/mean@1`

reward 函数（见 [rlvr-explained.md](rlvr-explained.md)，`verl/utils/reward_score/gsm8k.py`）返回 0/1 标量，PSRL 验证 reward loop 把它塞进 `acc` 字段：

```python
# psrl: psrl/workers/reward/reward_loop/naive.py
if isinstance(result, dict):
    score = result["score"]
else:
    score = result
    reward_extra_info["acc"] = score
```

`psrl: ray_trainer.py::_val_metrics_update` 调用 verl `process_validation_metrics`（bootstrap 聚合 `mean@N`/`std@N`/`best@N`/`maj@N`），按规则命名：

```python
core_var = "acc" if "acc" in var2metric2val else "reward"
metric_sec = "val-core" if (var_name == core_var and metric_name.startswith(("mean","maj","best")) and f"@{n_max}" in metric_name) else "val-aux"
pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
```

`data_source="openai/gsm8k"`、验证时每 prompt 采 1 条（`n_max=1`）→ 拼出 `val-core/openai/gsm8k/acc/mean@1`，即验证集 0/1 exact-match 的均值——图右下角那条曲线，0.85 崩到 0，是前面所有训练态异常在任务表现上的最终读数。

## 补充：MFU 公式详解（对照 verl 源码）

`### perf/mfu/actor` 一节给出的公式

$$\text{MFU} = \frac{\text{实际估算 FLOPs}}{\text{理论峰值 FLOPs} \times \text{world\_size}}$$

容易被误读成"分子分母是原始计算量（次数）"，但对照源码（`verl/utils/flops_counter.py`，`psrl` vendor 路径 `${PSRL_WORKSPACE}/third_party/verl/verl/utils/flops_counter.py`）可以确认分子分母其实都是**速率（FLOPs/s）**：

```python
# 每个模型专属的 _estimate_xxx_flops()：
dense_N_flops = 6 * dense_N * tokens_sum          # 前向+反向总 FLOPs（6N 近似，Kaplan/PaLM 惯例）
attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
flops_all_token = dense_N_flops + attn_qkv_flops  # 这一步(delta_time)的模型前反向总计算量
flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12   # 除以耗时 → TFLOPs/s

# get_device_flops() 默认单位也是 T，返回单卡理论峰值 TFLOPs/s
promised_flops = get_device_flops()

# 调用处 final_metrics["mfu"] = estimated_flops / promised_flops / world_size
```

关键点：
- `estimated_flops` 传入的 `global_token_num` 是**全局**（所有 rank 加总）的 token 数，所以它是整个集群这一步"实际达到的 FLOPs/s"，不是单卡的。
- `promised_flops` 是**单卡**理论峰值 FLOPs/s（`_DEVICE_FLOPS` 表按 GPU 型号查表，比如 H20 是 148e12）。
- 所以必须再除以 `world_size`，把分母换算成"全集群理论峰值"，才能和分子（全集群实际吞吐）对齐，结果是 0~1 之间、代表"平均每卡利用率"的量。

标准定义可以写成：

$$\text{MFU} = \frac{\text{全局(所有卡)实际达到的 FLOPs/s}}{\text{单卡理论峰值 FLOPs/s} \times \text{world\_size}}$$

对应到用户常记的简化版"模型前反向计算量 / 理论 FLOPs 峰值"：这个说法本身方向没错，只是漏了两层——(1) 分子分母要是**速率**（除以耗时），量纲才对得上；(2) 多卡场景分母要乘 `world_size`，否则数值会远大于 1，没有意义。

## 补充：几处代码里没展开讲的数值技巧详解

正文和上面附录里几段代码（`entropy_from_logits`、ratio 的 `clamp`、dual-clip 的三段 `pg_losses`、k3 KL 估计量）只贴了代码没解释"为什么这么写"，这里逐一展开推导，方便对照阅读源码。

### 1. `entropy_from_logits` 里的 logsumexp 技巧

```python
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
```

为什么不按熵的定义直接写 `entropy = -(pd * pd.log()).sum(-1)`？两个数值坑：
- `softmax` 对小概率类别算出的 $p_v$ 接近 0，再 `.log()` 会下溢到 `-inf`，`0 * -inf = nan`；
- 如果自己手写 `log_softmax = logits - logits.exp().sum(-1).log()`，某个 token 的 logit 只要偏大（几十量级很常见），`logits.exp()` 直接 `inf`。

推导思路：先把 $\log p_v$ 用 logits 表达，再代入熵定义消掉显式的 `log(概率)`。因为

$$p_v=\frac{e^{z_v}}{\sum_u e^{z_u}}\ \Rightarrow\ \log p_v=z_v-\log\sum_u e^{z_u}=z_v-\text{logsumexp}(z)$$

代入 $H=-\sum_v p_v\log p_v$：

$$H=-\sum_v p_v\big(z_v-\text{logsumexp}(z)\big)=\text{logsumexp}(z)\underbrace{\sum_v p_v}_{=1}-\sum_v p_v z_v=\text{logsumexp}(z)-\sum_v p_v z_v$$

这正对应代码 `torch.logsumexp(logits,-1) - (pd*logits).sum(-1)`——**全程只对 logits 做一次 `logsumexp`（内部自带"减最大值"的稳定实现：$\text{logsumexp}(z)=\max_v z_v+\log\sum_v e^{z_v-\max_v z_v}$，指数项永远 $\le1$ 不会 overflow）和一次 `softmax`（同样是减最大值实现），完全绕开了对一个可能极小的概率取 `log` 这一步**，这就是"logsumexp 技巧"数值稳定的本质：把"先算概率再取 log"改写成"只用原始 logits 表达"，避免概率趋于 0 或 logits 过大时的下溢/上溢。

### 1.5 $p_v=\pi(\cdot\mid s_t)$ 这个分布到底是怎么来的：vLLM 生成 vs 训练引擎重算

**生成阶段（vLLM，自回归 decode）**：对位置 $t$，输入当前序列 $s_t=(\text{prompt},y_1,\dots,y_{t-1})$，模型最后一层输出、过完最终 LayerNorm/RMSNorm 后的 **hidden state** $h_t\in\mathbb R^d$（注意不是 attention 里的 Query 向量——Q 是每层 self-attention 内部算 $QK^\top$ 用的中间量，跟 LM head 无关）过 LM head 投影成 logits，再 softmax：

$$z_t=W_{\text{lm}}h_t\ (\text{常与 embedding 权重共享})\qquad p_v=\pi(v\mid s_t)=\text{softmax}(z_t/\tau)_v$$

vLLM 用带 KV cache 的增量 decode，在这个分布上采样出 $y_t$，拼回序列滚到 $t+1$——这一步确实是"prompt 塞进 vLLM decode 出来的"。

**但算 `actor/entropy`/`old_log_prob` 用的 $p_v$，不是直接复用 vLLM 采样时的概率，而是训练引擎（FSDP/Megatron）事后重算的**（`psrl: ray_trainer.py::_compute_old_log_prob` → `verl/workers/engine/fsdp/transformer_impl.py`）：

```python
logits = output.logits                          # (bsz, response_length, vocab_size)：整条(prompt+response)序列一次并行前向的输出
temperature = output_args["temperature"]         # 与 rollout 采样时的温度对齐
logits.div_(temperature.clamp(min=1e-8))
entropy = verl_F.entropy_from_logits(logits)     # 即上面 H=-Σ p_v log p_v 那条公式
```

这里是把 rollout 阶段已生成好的**完整序列一次性喂进训练模型**（teacher forcing，并行前向，不用 KV cache 逐步生成），对每个位置同时算出 $p_v$。理论上应该等于第一步 vLLM 采样时用的分布（同一份权重、同一温度），但 vLLM 作为专用推理引擎可能用不同 attention kernel/精度/chunked prefill，跟训练引擎数值上会有细微漂移。为了让 PPO/GRPO ratio $r_t=\pi(a_t)/\pi_{\text{old}}(a_t)$ 里的 $\pi_{\text{old}}$ 和后面要更新的 $\pi$ 走同一套计算路径，verl/psrl 默认用训练引擎整段重算，而不是直接用 vLLM decode 时的概率；代码里也留了 `bypass_mode` 分支可以直接复用 vLLM 的 `rollout_log_probs` 省掉这次前向，但那样需要额外的 rollout correction 机制去修正训练/推理数值差异。

### 2. `ratio = exp(clamp(log_prob - old_log_prob, -20, 20))`

重要性采样比值 $r_t=\pi(a_t)/\pi_{\text{old}}(a_t)=\exp(\log\pi-\log\pi_{\text{old}})$。训练不稳定（比如本文档描述的熵崩溃/梯度尖峰期）时新旧策略在某些 token 上可能差出十几到几十个 nat，不钳制的话 `exp` 直接 `inf`/反传出 `nan`，会直接冲垮这一步更新。先把指数的输入钳到 $\pm20$：$e^{20}\approx4.85\times10^8$ 已经远超任何有意义的 ratio 取值，钳制之后再 `exp`，保证结果永远是有限值，只是精确度在这种极端 token 上打了折扣（反正这种极端情况本来梯度信号也不可靠）。

### 3. Dual-Clip PPO 的三段 `pg_losses`（对应问题 5 的梯度尖峰）

```python
pg_losses1 = -advantages * ratio                                           # 原始 -r·A
pg_losses2 = -advantages * clamp(ratio, 1-cliprange_low, 1+cliprange_high)  # 标准 PPO 裁剪
clip_pg_losses1 = max(pg_losses1, pg_losses2)                              # 标准 PPO：取更保守(更大)的 loss

pg_losses3 = -advantages * clip_ratio_c        # clip_ratio_c 默认 3.0，dual-clip 下界
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)

pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

前两行是标准 PPO clip：$\max(-rA,\ -\text{clip}(r,1-\epsilon_l,1+\epsilon_h)A)$（这是 loss，带负号；PPO 论文里对 objective 取 min，等价于对 loss 取 max）。记 $|A|=-A$、$\ell=1-\epsilon_l$、$h=1+\epsilon_h$，逐段代入看 $\hat A_t<0$（该 token 该被抑制）时两端各自发生什么：

- **$r\to0$（策略几乎不再选这个 token）**：此时 $r<\ell$，`clip` 把 `pg_losses2` 钳成常数 $|A|\ell$；而 `pg_losses1`$=|A|r\to0$。由于 $r<\ell\Rightarrow|A|r<|A|\ell$，`max(pg_losses1, pg_losses2)` 取到的是**常数地板 $|A|\ell$**，不是趋于 0 的 `pg_losses1`。举例 $A=-1,\epsilon_l=0.2\Rightarrow\ell=0.8$，$r=0.01$：`pg_losses1`$=0.01$，`pg_losses2`$=0.8$，`max`$=0.8$——标准 clip 在这一侧本来就有保护，loss **不会**塌到 0，也就没有"无惩罚/梯度消失"的问题。
- **$r\to\infty$（该被抑制的 token 概率反而被冲得比旧策略更高）**：此时 $r>h$，`pg_losses2` 钳成常数 $|A|h$，而 `pg_losses1`$=|A|r$ 随 $r$ 无界增长，一旦 $r>h$ 就超过这个常数，`max` 选中的是**无界的 `pg_losses1`**。同样例子取 $r=5$：`pg_losses1`$=5$，`pg_losses2`$=1.2$，`max`$=5$，且 $r$ 越大这个值越大，没有上限——标准 PPO clip 完全没有管住"负 advantage + ratio 爆炸"这一侧，单个 token 就能贡献任意大的 loss/梯度。

（$\hat A_t>0$ 侧对称地看：$r\to\infty$ 时 `pg_losses1`$=-Ar\to-\infty$，会被 `max` 天然压成 `pg_losses2` 的常数上限，标准 clip 已经够用；不需要 dual-clip。）

Dual-Clip PPO（[Ye et al. 2020, MOBA 游戏强化学习](https://arxiv.org/pdf/1912.09729)）堵的正是上面 $\hat A_t<0,\ r\to\infty$ 这个无界口子，不是 $r\to0$ 那一侧。加一个常数上限 `clip_ratio_c`（要求 >1，默认 3.0）：

$$\text{pg\_losses3}=-A\cdot c=|A|c\ (\text{常数})\qquad \text{clip\_pg\_losses2}=\min(\text{pg\_losses3},\ \text{clip\_pg\_losses1})$$

只在 `advantages<0` 时用这个 `min` 顶住 `clip_pg_losses1`：接上面的例子 $c=3$，`pg_losses3`$=3$，`clip_pg_losses2 = min(3, 5) = 3`——把原本会随 $r$ 无限冲高的 5 死死按在 3 这个天花板下。`pg_clipfrac_lower` 统计的就是"撞到这个上界保护"的 token 占比，可以用来判断训练是否频繁触发。对应到问题 5：熵崩溃期个别 token 的 ratio 很容易被推到极端值，若恰好是负 advantage token，没有 dual-clip 这一刀就是无界的梯度贡献，与观测到的 `grad_norm` 破千尖峰直接对应。

**追问 1：单看一个 token，$\hat A_t<0$ 时梯度方向不是应该推 $r$ 往小走吗，为什么 $r$ 会往大走？**

**第一步：为什么 loss $=-Ar$ 对参数的梯度是 $-Ar\cdot\nabla_\theta\log\pi(a_t\mid s_t)$？**

关键前提：$\pi_{\text{old}}(a\mid s)$ 是 rollout 时用旧参数算好、存下来的常数，对当前更新的 $\theta$ 求导时不随 $\theta$ 变化。把 $r$ 写成 $r(\theta)=\exp(\log\pi_\theta(a|s)-\log\pi_{\text{old}}(a|s))$，链式法则求导（$\log\pi_{\text{old}}$ 那一项对 $\theta$ 的导数是 0）：

$$\nabla_\theta r(\theta)=\exp(\log\pi_\theta-\log\pi_{\text{old}})\cdot\nabla_\theta\log\pi_\theta(a|s)=r(\theta)\cdot\nabla_\theta\log\pi_\theta(a|s)$$

$A$ 也是这一步的常数（rollout 后算好的），所以 $\nabla_\theta(-Ar)=-A\cdot\nabla_\theta r=-Ar\cdot\nabla_\theta\log\pi_\theta(a_t|s_t)$——精确等式，不是近似。

**第二步：softmax 对 logit 的梯度 $\partial\log\pi(a)/\partial z_v=\mathbb 1[v=a]-\pi(v)$ 怎么来的**

用第 1 节的恒等式 $\log\pi(a)=z_a-\text{logsumexp}(z)$ 对 $z_v$ 求偏导：$\partial z_a/\partial z_v=\mathbb 1[v=a]$（显然），$\partial\,\text{logsumexp}(z)/\partial z_v=e^{z_v}/\sum_u e^{z_u}=\pi(v)$（对 $z_v$ 求导只有分子里 $u=v$ 那一项有贡献）。两者相减即得该恒等式。再链式法则接到 LM head 权重 $z_v=w_v\cdot h$（$w_v$ 是词表第 $v$ 行权重，只影响 $z_v$ 自己）：

$$\frac{\partial\log\pi(a)}{\partial w_v}=\big(\mathbb 1[v=a]-\pi(v)\big)h$$

**第三步：不做二分类简化，用真实的 3-token softmax + 2 维隐藏状态重算一遍**

词表 $\{A,B,C\}$，隐藏维度 $d=2$，权重 $w_A,w_B,w_C\in\mathbb R^2$ 全初始化为 $\mathbf0$（此时对任意 $h$ 都有 $\pi(A)=\pi(B)=\pi(C)=1/3$，方便算干净数字）。batch 里两条无关序列都在某位置采样到了 $A$：

| | 隐藏状态 $h$ | advantage |
|---|---|---|
| 位置①（当前在看的坏样本） | $h_1=(1.0,\ 0.0)$ | $A_1=-1$ |
| 位置②（batch 里另一好样本） | $h_2=(0.9,\ 0.1)$（跟①方向接近，$h_1\cdot h_2=0.9$） | $A_2=+2$ |

取各自的**第一次**梯度步（$\theta=\theta_{\text{old}}\Rightarrow r=1$），此时第一步公式精确退化为 $\nabla_\theta\ell_i=-A_i\nabla_\theta\log\pi(a|s_i)$。由第二步公式，$\partial\log\pi(A|h)/\partial w_A=(1-\tfrac13)h=\tfrac23h$，所以：

$$\frac{\partial\ell_1}{\partial w_A}=-A_1\cdot\tfrac23h_1=-(-1)\cdot\tfrac23(1.0,0.0)=(0.667,\ 0)$$
$$\frac{\partial\ell_2}{\partial w_A}=-A_2\cdot\tfrac23h_2=-(2)\cdot\tfrac23(0.9,0.1)=(-1.2,\ -0.133)$$

单看位置①：梯度 $(0.667,0)$，梯度下降 $w_A\leftarrow w_A-\eta(0.667,0)$ 会让 $w_A$ 第一维变小 → $z_{A,1}=w_A\cdot h_1$ 变小 → $\pi(A|h_1)$ 下降——局部方向没错，确实是"压低 $A$"。但 $w_A$ 是共享参数，一次更新用的是两个位置梯度之和：

$$\frac{\partial\ell_{\text{total}}}{\partial w_A}=(0.667,0)+(-1.2,-0.133)=(-0.533,\ -0.133)$$

$$w_A\leftarrow w_A-\eta(-0.533,-0.133)=w_A+\eta(0.533,\ 0.133)$$

$w_A$ 第一维净增大——位置②幅度 $1.2$ 压过了位置①的 $0.667$。代回位置①验证：

$$z_{A,1}^{\text{new}}=w_A^{\text{new}}\cdot h_1=z_{A,1}^{\text{old}}+\eta(0.533,0.133)\cdot(1.0,0.0)=z_{A,1}^{\text{old}}+0.533\eta$$

$z_{A,1}$ 增大 $\Rightarrow\pi(A|h_1)$ 增大 $\Rightarrow r_{t_1}>1$——位置①本该被压低的 token，纯粹因为跟位置②抢同一个 $w_A$、位置②信号更强，被"搭便车"带着往上走了。这个污染幅度直接取决于 $h_1\cdot h_2=0.9$ 这个相似度：隐藏状态越相关，位置②的更新方向在 $h_1$ 方向上的投影越大，污染越强——**熵崩溃恰好会让不同序列的隐藏状态更相似**（输出趋同），例子里 $h_1,h_2$ 故意取得接近就是体现这点。这正是 Dual-Clip 论文在实践中观察到的现象，也对应本文档 `grad_norm` 尖峰与 `zero_variance_fraction` 飙升同时出现的时间窗：熵崩溃（问题 1）→ 隐藏状态跨序列高度相关 → 共享参数梯度互相污染 → 负 advantage token 的 $r$ 被别的正 advantage 序列带飞 → 撞上 dual-clip 要堵的 $r\to\infty$。参数更新从来不是"干净"地按每个 token 的 advantage 符号走的，是所有 token 共享参数下的聚合结果。

**追问 2：那为什么不对称地也堵一下 $\hat A_t>0,\ r\to0$ 这个方向？**

这个方向确实是真实的梯度消失：`pg_losses1`$=-Ar\to0$ 且被 `max` 选中，意味着 loss 和梯度（$-Ar\cdot\nabla\log\pi$，$r\to0$ 把梯度也一起压没）一起趋于 0。但它不是 clip 类机制能修的问题：

- clip（`max`/`min`）只能把值钳到某个常数上/下限，解决的是"值/梯度过大"，无法在梯度已经消失的地方凭空造出梯度——需要完全不同的机制（如反传时单独给 ratio 设下限、或加熵正则防止概率一开始就掉到接近 0），不是加一层 clip 能做到的。
- 后果也不对称：$\hat A_t<0,\ r\to\infty$ 是单个 token 就能贡献无界暴涨的梯度，直接对应 `grad_norm` 破千的瞬时尖峰；$\hat A_t>0,\ r\to0$ 是梯度平滑趋于 0，副作用是"逐渐放弃对一个好动作的探索"，是缓慢退化而非剧烈失控，不会单独制造观测到的那种瞬时尖峰。这类问题实践中通常靠熵正则、KL/trust-region 约束（让 $r$ 一开始就不容易漂太远）、较少的 `ppo_epochs` 缓解，而不是靠 clip。

### 4. `low_var_kl` / `k3` KL 估计量哪来的 + `k3+` 的 straight-through 技巧

```python
kl = torch.clamp(ref_logprob - logprob, -20, 20)   # d = log(π_ref/π)，先钳制防溢出
ratio = torch.exp(kl)                               # ratio = π_ref/π
kld = torch.clamp(ratio - kl - 1, -10, 10)          # k3 = ratio - d - 1
```

来自 Schulman《[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)》：设 $r=\pi_{\text{ref}}(a)/\pi(a)$，$a$ 采样自 $\pi$（即代码里实际发生的采样过程），构造 $f(r)=r-\log r-1$。因为 $f$ 是凸函数、$f(1)=0$、$f(r)\ge0$，可以证明 $\mathbb E_{a\sim\pi}[f(r)]=D_{KL}(\pi\|\pi_{\text{ref}})$——这就是 k3 估计量的来源。相比朴素的 $k_1=\log\pi-\log\pi_{\text{ref}}$（有正有负，方差大），k3 恒非负、更贴近 KL 本身非负的性质，方差明显更小，故称"低方差无偏估计"。同样是先 clamp 差值再 `exp`，算完再 clamp 一次结果，双重防溢出。

`kl_penalty` 里还有一层正文没提的 `"k3+"` 分支（`kl_penalty="k3+"` 时触发）：

```python
backward_score = 0.5 * (logprob - ref_logprob).square()      # 这是 k2
return backward_score - backward_score.detach() + forward_score.detach()
```

这是"straight-through"写法：`backward_score - backward_score.detach()` 前向数值恒为 0，所以前向返回值 = `forward_score.detach()`（即 k3 的数值）；但反向传播时 `.detach()` 的项不产生梯度，唯一贡献梯度的是没 detach 的 `backward_score`（k2），所以反传梯度等于 k2 的梯度。代码注释里说得很直接：k1/k3 的**期望值**等于真实 KL，但它们的**期望梯度**不等于 KL 的梯度方向；k2 恰好反过来——梯度方向对，但数值本身有偏。于是"日志/监控用 k3 的数值，参数更新用 k2 的梯度"，两者都要就用这个 trick 拼起来。

### 5. `AdaptiveKLController` 的乘性反馈更新

```python
proportional_error = np.clip(current_kl / target_kl - 1, -0.2, 0.2)
mult = 1 + proportional_error * n_steps / horizon
self.value *= mult
```

这是个比例反馈控制器（只有 P 项，没有 I/D）：`current_kl` 比 `target_kl` 大 → `proportional_error>0` → `mult>1` → 下一步 KL 惩罚系数 $\beta$（即前面 `apply_kl_penalty` 公式里的 $\beta$）被放大，加重惩罚把 KL 拉回目标；`current_kl` 偏小则反向减小 $\beta$，放松约束让策略多探索。`±0.2` 限制单步"误差比例"的幅度，`n_steps/horizon` 把调整量摊到一个时间窗口里——两者共同作用是防止 $\beta$ 因为单步的 KL 抖动就大幅震荡。

## 相关文档

- [ppo-grpo-dapo-sample-requirements.md](ppo-grpo-dapo-sample-requirements.md) — PPO/GRPO/DAPO 的 advantage 计算与样本组织方式
- [rlvr-explained.md](rlvr-explained.md) — GSM8K reward 函数与 RLVR 概念
- [psrl-run-guide.md](psrl-run-guide.md) — psrl 项目训练运行指南

## 复现：2 节点 FSDP 部署下同一套连锁崩溃再次出现（`dapo_fsdp_2nodes.png`）

**一句话结论**：把训练从单节点换到 2 节点 FSDP 部署（`psrl/scripts/dapo_fsdp_2nodes.png`），`actor_lr=1e-6` 的 warmup 曲线、崩溃发生的 step 区间、`response_length` 打满的硬顶（4096）、失败反弹期验证准确率精确弹回 ~0.43 这几个细节都和正文分析的单节点版本几乎逐点对应——这说明前面六条问题诊断针对的是**训练配置/算法动力学本身的系统性缺陷**，不是单节点部署下的偶发硬件/数值问题，换节点数不会绕开它。

**对照时间线**（"单节点" = 正文表格，"2 节点" = 本节新增）：

| 阶段 | 单节点 step | 2 节点 step | 现象是否一致 |
|---|---|---|---|
| 健康期 | 0–45 | 0–45 | 一致：`entropy` 缓降、验证准确率爬到 0.83–0.85、`response_length` 稳定在几百 |
| 预警性崩溃（自愈） | ~50–60 | ~50–60 | 一致：`actor/loss` 冲高、验证准确率跌到 ~0.1，随后自行恢复 |
| 主崩溃（不可逆） | ~115 起 | ~110–120 起 | 基本一致：`response_length` 打满 4096、`critic/score`/`rewards` 砸到 -1.0、`zero_variance_fraction`→1.0；2 节点版本在 ~110–120 之间多了一段 `actor/loss` 先冲到 0.5 的过渡波动，主崩溃的 `grad_norm` 尖峰没有单节点版本那么陡（本次最大尖峰出现在后面的反弹窗口） |
| 失败的反弹尝试 | ~155–170 | ~155–175 | 一致：`entropy` 异常飙升到 ~0.75，验证准确率昙花一现地弹到 **~0.43**（与正文数值几乎精确重合），随后再次坍缩；本次 `grad_norm` 在这个窗口冲到全程最高点（~1300） |
| 死寂期 | 175–200 | 175–200 | 一致：`entropy`≈0、`response_length` 钉死在 4096、`reward`≈-1、验证准确率≈0 |

**支持"同一套失败机制"的额外证据**：
- `perf/mfu/actor` 在两次跑里都稳定在 ~0.6–0.64（这是按卡归一化的效率指标，不随节点数变化），只有 `perf/throughput`/`timing_s/step` 的锯齿幅度因为 GPU 总数不同而不同——排除了"是某台机器数值不稳"这类基础设施解释。
- `actor/lr` 的 warmup 形状与目标值（`1e-6`）相同，说明两次跑用的是同一套超参，崩溃不是某次跑碰巧调错学习率导致的。

**结论对正文建议的强化**：既然换节点数依然在几乎相同的 step 复现整条连锁（熵崩溃 → 零方差组占满 → reward 钉死 → 长度打满 → 验证准确率归零），"建议"一节里提到的自动回滚阈值就应该设在第一次自愈型崩溃出现的 step 附近（本例两次跑都在 ~50–60），而不是等到主崩溃后再干预——因为这个窗口本身就是可复现的早期预警信号，不依赖具体的部署拓扑。
