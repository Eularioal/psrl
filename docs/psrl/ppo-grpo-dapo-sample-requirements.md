# PPO / GRPO / DAPO：一个更新 step 需要多少样本

**核心结论**：三者的核心区别在于 **advantage 怎么算**，这决定了"凑齐多少样本才能做一次更新"：PPO 用 critic 给每条独立轨迹算 baseline，理论上 1 条轨迹就能更新；GRPO/DAPO 没有 critic，baseline 是组内统计量，必须凑齐一个 prompt 的完整 `n` 条 response（一个 group）才能算 advantage。

## 目录

- [两种"step"的区分](#两种step的区分)
- [对比表](#对比表)
- [关于"GRPO 一个 prompt rollout 出 n 个 response 得到 n 个 advantage"](#关于grpo-一个-prompt-rollout-出-n-个-response-得到-n-个-advantage)
- [与 psrl 配置的对应](#与-psrl-配置的对应)
- [延伸阅读](#延伸阅读)

## 两种"step"的区分

日志里的 `training/global_step` 指**一整轮**：rollout 一批 prompt → 算 advantage → 用这批数据做若干次梯度更新 → global_step+1。这一轮内部按 `ppo_mini_batch_size` 切成若干 mini-batch，每个 mini-batch 对应一次真正的梯度下降（可能还会重复 `ppo_epochs` 遍）。下面讨论的是"凑够一轮 rollout 需要多少样本"，不是单次梯度下降需要多少样本。

## 对比表

| 算法 | Rollout 的最小可用单元 | Advantage 怎么算 | 凑够一轮需要多少样本 |
|---|---|---|---|
| **PPO** | 1 条 trajectory（1 个 prompt + 1 个 response） | 用 critic（value 网络）估计 $V(s_t)$，配合 GAE($\lambda$) 逐 token 算 advantage：<br>$$\hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$<br>不依赖同一 prompt 的其他 response | `train_batch_size`(prompt 数) $\times$ `rollout.n`（**理论上 n=1 即可**，n>1 只是为了降方差，不是算法硬要求） |
| **GRPO** | 1 个 **group** = 1 个 prompt 的 `n` 条 response（$n\geq2$，常见 4~16） | 没有 critic，用组内统计量做 baseline：<br>$$\hat{A}_i = \frac{r_i - \text{mean}(r_1,\dots,r_n)}{\text{std}(r_1,\dots,r_n)}$$<br>必须等这个 prompt 的 n 条全部生成完才能算，缺一条都不行 | `train_batch_size`(prompt/group 数) $\times$ `n` |
| **DAPO** | 同 GRPO，1 group = 1 prompt $\times$ n response | 组内相对 baseline 计算方式跟 GRPO 一样，但加了 **dynamic sampling**：如果一组 n 条 response 的 reward 全相同（方差为 0，即全对或全错），这组 advantage 全为 0，等于白采，会被过滤掉并重新采样补足 | 名义上同 GRPO：`train_batch_size` $\times$ `n`；但因为要过滤退化组再补采，实际消耗的 rollout 数会 $\geq$ 这个值（对应指标 `group/zero_variance_fraction`，量化这个浪费比例） |

## 关于"GRPO 一个 prompt rollout 出 n 个 response 得到 n 个 advantage"

这个理解是对的，补充两点：

1. n 个 advantage 不是独立算的，是**同一个 group 内互相依赖**——每个 $\hat{A}_i$ 都要用到全组的 mean/std，所以这 n 条必须打包在一起，不能拆散成单条样本去更新。这也是为什么很多实现里会校验"chunk 必须能被 `rollout_n` 整除"，本质是保证 group 不被切散。
2. DAPO 在此基础上多了一层"这个 group 是否有效"的判断（方差是否为 0），本质上是 GRPO + 过滤机制，不是重新定义了 advantage 公式。DAPO 论文里另外的改动——非对称 clip（`clip_ratio_low`/`clip_ratio_high`）、超长回复惩罚、token-level loss 聚合——影响的是 loss 怎么聚合，不影响"需要多少样本"这个问题。

## 与 psrl 配置的对应

结合 [批量配置的含义](psrl-run-guide/monitoring-and-batches.md#train_prompt_bsz-对应哪个配置是什么意思)：`train_prompt_bsz`（对应 `data.train_batch_size`）就是上表里的"group 数"，`n_resp_per_prompt`（对应 `rollout.n`）就是"n"，两者相乘就是一轮实际用到的样本总数——GRPO/DAPO 场景下这两个数缺一不可，PPO 场景下 `rollout.n` 理论上可以设成 1。

## 延伸阅读

- GRPO 原始论文：DeepSeekMath（组内相对 baseline，去掉 critic）
- DAPO 原始论文：字节 DAPO（decoupled clip + dynamic sampling + overlong reward shaping + token-level loss）
- [生成长尾](psrl-run-guide/ray-and-rollout.md#为什么-rl-rollout-会有生成耗时不均long-tail-latency)：为什么 RL rollout 会有长尾延迟（与 dynamic sampling / redundant rollout 相关）
