# 训练排障：DAPO退化诊断

这次历史训练在约 step 115 后明显退化：回答长度长期达到 4096-token 上限，奖励接近 -1，组内同分比例升高，验证准确率降至接近 0，并持续到报告末尾 step 200。应以这些联合信号判断训练质量，而不只看进程是否存活。

原记录基于 `dapo-qwen2.5-3b-fsdp.png` 与 `dapo_fsdp_2nodes.png` 的曲线读取。图片及原始逐步日志未在当前仓库找到，以下近似数字沿用历史记录；2026-11-16 只整理解释与证据范围，没有复现实验或确认修复。

## 目录

- [这次训练发生了什么](#这次训练发生了什么)
- [把几个指标串起来理解](#把几个指标串起来理解)
- [下一次怎样排查与停止](#下一次怎样排查与停止)
- [两节点记录提供了什么对照](#两节点记录提供了什么对照)
- [进一步理解指标和公式](#进一步理解指标和公式)
- [训练曲线指标：奖励、长度、梯度和效率怎样计算](#训练曲线指标奖励长度梯度和效率怎样计算)
  - [历史指标实现与计算](#历史指标实现与计算)
  - [MFU：计算量、耗时与设备数怎样对齐](#mfu计算量耗时与设备数怎样对齐)

## 这次训练发生了什么

模型为 Qwen2.5-3B，任务为 GSM8K 数学题，使用 DAPO 类强化学习配置。DAPO 基于同题多次回答的相对奖励训练，并调整裁剪、采样、长度反馈与损失聚合。

<!-- figure:training-health-signals -->
这张图沿同一个场景解释：训练进程还在运行，怎样判断它有没有拿到有效学习信号？

[![训练健康：进程存活之外，还要看样本和更新信号（1/2）](../_static/img/document-figures/training-health-signals-screen.png)](../_static/img/document-figures/training-health-signals.png)

[![训练健康：进程存活之外，还要看样本和更新信号（2/2）](../_static/img/document-figures/training-health-signals-screen-02.png)](../_static/img/document-figures/training-health-signals.png)

一次更新先有采样数据，再形成奖励与优势，最后执行前向、反向和优化器调用。 指标沿数据到更新的因果链读取。零梯度有多种原因，模型效果改善需要另设验收。 指标类别示意，指示灯不是实测健康状态，也没有为本次文档改图启动训练。 [查看矢量图](../_static/img/document-figures/training-health-signals.svg)。
<!-- /figure:training-health-signals -->

| 阶段 | step | 历史曲线中的现象 |
|---|---|---|
| 前期改善 | 0–45 | 熵约 0.3→0.1，平均分约 0.2→0.9，验证准确率约 0.18→0.85，回答约 300 token |
| 短时退化后恢复 | 约 50–60 | loss 约 1.0，回答约 1150 token，平均分及验证准确率短时降到约 0.1 |
| 主要退化窗口 | 约 115 起 | 梯度范数 150+，回答达到 4096，奖励接近 -1，组内同分比例接近 1 |
| 短时反弹 | 约 155–170 | 熵升到约 0.6–0.75，验证准确率短时到约 0.43，随后回落 |
| 持续低表现 | 175–200 | 熵接近 0、长度接近 4096、奖励接近 -1、验证准确率接近 0 |

“持续到观察末尾”描述这份记录的结果。进一步恢复能力与具体根因需要针对性证据，历史曲线尚未回答这些问题。

## 把几个指标串起来理解

**熵下降要与答案和正确率联合判断。** 本次关注它与超长回答、同分组及验证下降是否同步。低熵可能来自稳定的正确输出，也可能来自重复；短时熵反弹同样需要检查回答质量。

**组内同分比例表示相对比较还能否区分回答。** 教学例子：同题 4 条回答奖励 `[1,1,0,0]`，减均值后有正有负；若变成 `[-1,-1,-1,-1]`，减均值后全为 0，这一奖励驱动的策略梯度项失去组内区分。熵、KL 等其他项是否还有梯度，要看实际配置。

**回答打满上限说明终止行为值得检查。** 从约 300 token 变为持续 4096 token，可能是重复、难题、更长推理或结束标记异常。应抽取真实回答、结束原因和有效 token 标记，确认是哪一种。

**奖励接近固定 -1，应沿评分代码追踪来源。** 它可能来自格式、截断或长度惩罚等分支。曲线只展示最终数字，真实原因需对照原始答案与评分器返回值。把“回答超长”和“奖励变差”连在一起，是待验证线索。

**梯度范数尖峰与验证下降说明更新值得暂停检查。** 尖峰可能来自概率比、优势分布、数值或聚合方式。裁剪能限制部分更新方向，实际参数移动仍是全部 token 共享参数后的结果。

这些现象可以相互放大：回答趋同使组内比较减弱，长度增加提高采样成本，反馈变化又影响下一次更新。现有证据支持联合退化的描述，具体先后因果需要样本和训练张量核对。

## 下一次怎样排查与停止

1. 先保存首次明显退化前后的日志、配置和 checkpoint，以及少量真实回答与结束原因。
2. 沿实际奖励键追踪分数，区分任务错误、长度惩罚和基础设施异常。
3. 核对有效 token、组内奖励差异、概率比和梯度。检查动态采样是否实际开启，补采应有时间与次数上限。
4. 按预先约定的质量与资源阈值停止本次运行，选择一个假设做小范围对照；需要恢复时从已保存的健康点开始。

例如目标只是确认退化是否由长度惩罚触发，可以先对保存的同一批回答离线重算评分，比较各分支结果。只有离线证据无法回答的部分，再设计有界训练。具体公式与配置说明见[采样与损失](02-强化学习_PPO与GRPO及DAPO.md#4-dapo四项机制怎样配合)。

## 两节点记录提供了什么对照

历史两节点 FSDP 曲线也出现相近退化窗口，学习率记录为 `1e-6`：

| 阶段 | 单节点 step | 2 节点 step | 现象是否一致 |
|---|---|---|---|
| 前期表现改善 | 0–45 | 0–45 | 一致：`entropy` 缓降、验证准确率爬到 0.83–0.85、`response_length` 稳定在几百 |
| 短时退化后恢复 | ~50–60 | ~50–60 | 一致：`actor/loss` 冲高、验证准确率跌到 ~0.1，随后自行恢复 |
| 主要退化窗口 | ~115 起 | ~110–120 起 | 基本一致：`response_length` 打满 4096、`critic/score`/`rewards` 砸到 -1.0、`zero_variance_fraction`→1.0；2 节点版本在 ~110–120 之间多了一段 `actor/loss` 先冲到 0.5 的过渡波动，主崩溃的 `grad_norm` 尖峰没有单节点版本那么陡（本次最大尖峰出现在后面的反弹窗口） |
| 短时反弹 | ~155–170 | ~155–175 | 一致：`entropy` 异常飙升到 ~0.75，验证准确率昙花一现地弹到 **~0.43**（与正文数值几乎精确重合），随后再次坍缩；本次 `grad_norm` 在这个窗口冲到全程最高点（~1300） |
| 持续低表现阶段 | 175–200 | 175–200 | 一致：`entropy`≈0、`response_length` 钉死在 4096、`reward`≈-1、验证准确率≈0 |


两份记录增强了“这种训练设置下存在可重复退化模式”的线索。MFU 相近或节点数变化，无法单独排除硬件、数值或配置差异；确认某个修复有效仍需保留版本和受控对照。

## 进一步理解指标和公式

指标的含义、历史计算和 MFU 口径在[下节](#训练曲线指标奖励长度梯度和效率怎样计算)连续阅读。PPO、GRPO、DAPO 的共同原理已集中到[完整教程](02-强化学习_PPO与GRPO及DAPO.md)，其中第 6、7 节包含熵、KL、logsumexp、Dual-Clip 和共享参数手算。理解算法时可以直接从教程开始。

## 训练曲线指标：奖励、长度、梯度和效率怎样计算


### 历史指标实现与计算

本节保留原分析所用 veRL/PSRL 代码摘录与指标定义，外部依赖的文件位置、字段和行号按当时版本理解。当前 PSRL 通过外部 veRL 依赖复用部分算法与指标，实现范围需核对实际导入来源；本节不提供当前安装版本的逐行保证。
#### `actor/entropy`

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

#### `actor/loss`

`workers/utils/losses.py::ppo_loss`：

$$\mathcal{L} = \mathcal{L}_{\text{PG}} - \beta_{\text{ent}} \cdot H \;(+\, \lambda_{\text{KL}}\cdot D_{KL}(\pi\|\pi_{\text{ref}})\text{，若启用 KL loss})$$

这个指标汇总策略梯度和实际启用的熵、KL 项。非对称裁剪、Dual-Clip 的分段公式与手算见[原理教程](02-强化学习_PPO与GRPO及DAPO.md#dual-clip-怎样保护负优势分支)。

step 50/115/160 三次 `actor/loss` 尖峰，与训练退化窗口重合；定位具体触发项还需要 token 级概率比、优势与梯度证据。

#### `actor/grad_norm`

FSDP engine `optimizer_step`：

```python
grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
if not torch.isfinite(grad_norm):
    self.optimizer.zero_grad()   # 非有限值直接跳过这次更新
else:
    self.optimizer.step()
```

即 $\lVert g\rVert_2=\sqrt{\sum_i g_i^2}$（裁剪前的原始范数），三次尖峰与 `actor/loss` 尖峰精确对齐——图上能看到远超裁剪阈值的量级，说明原始梯度在这些步骤较大。梯度裁剪限制送入优化器的范数；参数变化方向及其对回答的影响，需要结合完整更新检查。

#### `critic/score/mean` vs `critic/rewards/mean`

`trainer/ppo/metric_utils.py::compute_data_metrics`：

```python
sequence_score  = batch.batch["token_level_scores"].sum(-1)   # 原始 reward 函数输出
sequence_reward = batch.batch["token_level_rewards"].sum(-1)  # 真正喂给 advantage 计算的量
```

关系由 `trainer/ppo/ray_trainer.py::apply_kl_penalty` 建立：

$$\text{token\_level\_rewards} = \text{token\_level\_scores} - \beta \cdot D_{KL}(\pi_{\text{old}}\|\pi_{\text{ref}})$$

若 `use_kl_in_reward=False`（DAPO/GRPO 常见配置，KL 约束走 loss 而非 reward shaping）两者直接相等——这次训练图上二者曲线几乎完全重合，与历史记录中关闭 reward KL 的配置相符；单靠曲线重合仍需结合配置解释。

**展开：`token_level_scores` 怎么来的、KL 项怎么算、以及和 GRPO 组内优势的先后关系**

`token_level_scores` 这个名字有误导性：GSM8K 的 `compute_score` 只返回一个**标量**（0/1，整条回复对/错），并不是逐 token 打分。历史摘录把这个标量放在 `response_len` 张量的最后一个位置，其余位置全 0；该位置是否为有效响应还要结合掩码确认（`psrl/workers/reward/reward_manager.py:637-640`）：

```python
rm_score_tensor = torch.zeros(response_len, dtype=torch.float32)
rm_score_tensor[-1] = float(f.get("reward_score", 0.0))
f["rm_scores"] = rm_score_tensor
```

再由 `psrl/trainer/ppo/ray_trainer.py:2890` 改名成 `token_level_scores`：`data.batch["token_level_scores"] = data.batch["rm_scores"]`。

KL 项通常由已采 token 的对数概率估计，并按 response_mask 作用于有效位置。若在奖励中加入 KL，先得到 token_level_rewards，再进行组内优势估计。估计量、采样分布条件及控制系数的推导见[原理教程第 6 节](02-强化学习_PPO与GRPO及DAPO.md#6-熵与-kl探索和参考模型约束)。

#### `response_length/mean`

```python
response_length = attention_mask[:, -response_length:].sum(-1)   # 非 padding 的真实 token 数
"response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
```

verl 同批指标里还有 `response_length/clip_ratio`（图中未画），统计达到长度上限的样本比例。触顶样本的真实结束原因，还要结合 EOS（结束标记）和截断记录判断。

#### `group/zero_variance_fraction`（PSRL 自加指标）+ GRPO advantage

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

同组奖励完全相同时，每项奖励减均值都为 0，因此这部分组相对策略优势为 0；其他损失项仍需单独检查。历史记录中该指标在 step 120 后接近 1.0，与熵下降同时出现。回答相似度和题目难度如何造成同分，需要抽样回答进一步定位。

#### `perf/mfu/actor`

$$\text{MFU} = \frac{\text{实际估算 FLOPs}}{\text{理论峰值 FLOPs} \times \text{world\_size}}$$

```python
estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)
final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
```

历史曲线的下降与长度变化同时出现。MFU 同时依赖估算计算量、耗时和硬件峰值口径，定位原因需拆分训练计算、通信及等待。

#### `perf/throughput` 与 `timing_s/step`

```python
"perf/throughput": total_num_tokens / (time * n_gpus),   # tokens/s/GPU
f"timing_s/{name}": timing_raw[name]                       # 该计时段墙钟秒数，"step" 为全步耗时
```

`timing_s/step` 从 15–50s 跳到 90–270s，与 `response_length` 触顶 4096 的窗口重合，更多生成步骤是应重点检查的耗时来源；`perf/throughput` 图里的高台阶同理——长度打满后单步处理 token 总量变大，throughput 数值"看起来"升高，但这是被拉长的无效生成撑起来的，不代表训练效率提升。

#### `val-core/openai/gsm8k/acc/mean@1`

reward 函数（见[可验证奖励](02-强化学习_PPO与GRPO及DAPO.md#rlvr-与-rlhf奖励从哪里来)，`verl/utils/reward_score/gsm8k.py`）返回 0/1 标量，PSRL 验证 reward loop 把它塞进 `acc` 字段：

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



### MFU：计算量、耗时与设备数怎样对齐

[指标专题](#perfmfuactor)给出的公式

$$\text{MFU} = \frac{\text{实际估算 FLOPs}}{\text{理论峰值 FLOPs} \times \text{world\_size}}$$

容易被误读成"分子分母是原始计算量（次数）"，但对照源码（历史 veRL 的 `utils/flops_counter.py`，当时位于 vendor 依赖；现仓库使用外部依赖）可以确认分子分母其实都是**速率（FLOPs/s）**：

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
- 所以必须再除以 `world_size`，把分母换算成"全集群理论峰值"，才能和分子（全集群实际吞吐）对齐，得到相同口径下的模型计算效率比例。实际估算受 FLOPs 公式、峰值表和计时影响，需先核对这些条件。

标准定义可以写成：

$$\text{MFU} = \frac{\text{全局(所有卡)实际达到的 FLOPs/s}}{\text{单卡理论峰值 FLOPs/s} \times \text{world\_size}}$$

对应到用户常记的简化版"模型前反向计算量 / 理论 FLOPs 峰值"：这个说法本身方向没错，只是漏了两层——(1) 分子分母要是**速率**（除以耗时），量纲才对得上；(2) 多卡场景分母要乘 `world_size`，否则数值会远大于 1，没有意义。
