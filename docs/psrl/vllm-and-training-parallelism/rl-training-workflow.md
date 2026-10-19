# PSRL 训推协作与 PPO、GRPO 基础

> **核心结论**：串联轨迹生成、概率重算、优势估计、参数更新与权重同步，并保留 PPO/GRPO 的逐步推导和长尾分析。

## 目录

- [六、RL 后训练场景（PSRL）](#六rl-后训练场景psrl)
  - [6.0 "同时在线"具体怎么理解（以 PSRL 为例）](#60-同时在线具体怎么理解以-psrl-为例)
  - [6.1 权重同步与 Resharding](#61-权重同步与-resharding)
  - [6.1b PP+TP 二维网格：训练端 shard 到底是什么](#61b-pptp-二维网格训练端-shard-到底是什么)
  - [6.2 Rollout 长尾](#62-rollout-长尾)
  - [6.3 资源分配的两难](#63-资源分配的两难)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。原文整理日期为 2026-09-10，后续追加内容的日期以各节记载为准；保留章节编号以便对照旧引用。

[返回专题索引](../vllm-and-training-parallelism.md)

## 六、RL 后训练场景（PSRL）

RLHF / Agentic RL 的特殊之处在于：**推理引擎（vLLM）和训练引擎（Megatron/FSDP）同时在线**，这带来两个纯粹的系统问题。

### 6.0 "同时在线"具体怎么理解（以 PSRL 为例）

**紧耦合式为什么做不到同时在线**：朴素 PPO 实现里生成和训练共用同一批 GPU、同一进程——跑完一整批 rollout 才能切到训练模式，训练完一步才能切回生成模式，是严格的同步屏障（分时复用），vLLM 和 Megatron 从未真正并发跑在 GPU 上。

**PSRL 的做法**：把训练和生成拆成两组完全独立的 GPU 集群（如 node0/1 跑 vLLM 生成、node2/3 跑 Megatron/FSDP 训练），中间架一个 **Parameter Server（PS）** 做异步权重同步桥梁，两边互不共享显存/进程，物理上并行运行。

**权重怎么流动（Push/Pull 异步协议）**：

- **Push**（训练侧，每个梯度步后）：各 train rank 按自己已有的切分（FSDP shard / Megatron TP-PP shard）直接把权重分片流式写入 PS Worker，不需要先 all-gather 出完整权重。
- **Pull**（生成侧，sync 策略触发时）：vLLM 各 rank 从同节点的 PS Worker 读取自己所需分片（同机 CUDA IPC/共享内存，跨机 RDMA），读完更新本地版本号继续生成。
- **锁语义**：single-writer（push 独占）/ multi-reader（pull 可并发）。push 短且稀疏（每步一次），稳态下生成基本不被 PS 拖慢。

**真正让并发成立的是 Staleness Buffer**：用 Reserve → Occupy → Consume 协议解耦"训练需要凑够一批数据"和"生成必须用最新权重"这两件事——训练允许消费略旧策略版本产出的轨迹（bounded staleness），因此 vLLM 不必等最新权重就能持续生成，Megatron 也不必等所有 vLLM 实例都追上最新版本就能开始下一步梯度更新。

**时序上的重叠**：生成侧的 `dispatch → generate → reward → occupy` 和训练侧的 `buffer ready → pull weights → train → push new weights → sync` 是并发执行的两条流水线——任意时刻，vLLM 可能正用第 k 版权重生成，Megatron 可能正用第 k-1 版攒够的 buffer 做梯度更新，PS 上可能正有一次 push/pull 在传输——这才是"同时在线"的字面含义：两套系统各自持续运转，靠 PS + staleness buffer 两层异步机制解耦，而不是轮流占用同一批卡。

**例子：一条 trajectory 具体怎么被训练**（对照 `psrl/psrl/trainer/ppo/ray_trainer.py` 的真实字段名）：

Prompt："1+1等于几？" → Response（vLLM 自回归生成）："1+1等于2"

```
① 生成（vLLM，权重版本 v_k）
   逐 token 自回归采样：'1' '+' '1' '等' '于' '2'
   每生成一个 token，vLLM 记录采样时的 log prob
   → 这份 log prob 叫 rollout_log_probs（"生成引擎自己算出来的"）

② Reward（规则校验器）
   校验最终答案 "2" 正确 → reward = 1.0，写入 rm_scores

③ Buffer occupy
   (prompt, response, rollout_log_probs, reward, version_tag=v_k) 存入 staleness buffer

④ 攒够一批 → 触发训练
   此时 Megatron 权重可能已经从 v_k 走到了 v_{k+m}（staleness 允许的滞后）

⑤ old_log_probs 重新计算（Megatron，teacher forcing）
   把同样的 (prompt+response) token 序列一次性喂给当前训练权重，
   在 ONE 次并行前向里算出每个位置的 log prob
   → 这是 decoupled 模式下的 old_log_probs（稳定的 π_old 锚点，详见下文）

⑥ ref_log_prob（冻结的参考模型，同样 teacher forcing 算一遍）
   用于 KL 惩罚项，防止策略跑偏太远

⑦ Advantage：组内归一化 reward（GRPO）或 GAE（PPO+critic）→ advantages（公式见下文）

⑧ Loss & 更新
   ratio·advantage 的 PPO-clip 目标（公式见下文）+ KL(π_θ, ref) 惩罚，
   只在 response_mask=1 的位置求和（prompt token 不参与 loss）
   backward → optimizer.step() → 权重变成 v_{k+1} → push 给 PS
```

**怎么理解④"此时 Megatron 权重可能已经从 $v_k$ 走到了 $v_{k+m}$"**：Megatron 不等某一条具体轨迹，只等"哪个 buffer 先攒满"。如果"1+1等于几"这条轨迹恰好是一条长轨迹（[§6.2](#62-rollout-长尾) 提到的"有的 30 轮还在调工具"），它生成、算 reward、occupy 进自己 buffer 槽位的这段时间里，其他并发的、更短的 rollout 早就把别的 buffer 填满了，一个接一个触发 Megatron 训练：

```
t=0s  版本 v_k     R1 开始生成"1+1等于几"（长轨迹，还在跑）
t=1s  Batch_A 攒满（跟本条无关）→ Megatron 训一步 → v_{k+1}
t=2s  Batch_B 攒满 → 训一步 → v_{k+2}
t=3s  Batch_C 攒满 → 训一步 → v_{k+3}
...
t=8s  "1+1等于几"终于生成完，occupy 进 Batch_D，Batch_D 攒满 → 触发训练
      → 但 Megatron 现在已经是 v_{k+3} 了（m=3）
```

即 `m` 不是"这条轨迹自己等了多久"，而是"它生成期间系统靠其他更快的轨迹推进了多少步训练"——轨迹越长，等它被消费时训练往往跑得越远。如果强行等这条慢轨迹，就退回 [§6.2](#62-rollout-长尾) 说的"同步式 PPO 里训练端要等最慢的那条轨迹"，PSRL 靠不等它换来异步加速，代价就是这批数据被消费时用的是"落后当前训练版本 m 步"的旧策略产出。`m` 也不是无限的：staleness controller（`psrl/workers/ps/staleness_controller.py`）给它设了配置好的上限，超过阈值就阻塞/丢弃或反向节流该 rollout 去 pull 最新权重，即架构文档里说的"保证有界的 off-policy 误差"。

⑧ 的 ratio 与 clip 目标：

$$
\text{ratio}_t=\exp\!\Big(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\text{old}}(a_t\mid s_t)\Big)
$$

$$
L^{\text{CLIP}}=\min\Big(\text{ratio}_t\cdot A_t,\ \operatorname{clip}(\text{ratio}_t,\,1-\epsilon,\,1+\epsilon)\cdot A_t\Big)
$$

**什么是"复用 rollout log probs"，为什么不能直接复用**：`old_log_probs` 理论上应该是"训练即将更新、偏离的那个旧策略"的 log prob，有两个可能来源：

$$
\text{old\_log\_probs}^{\text{bypass}}=\log\pi_{v_k}(a_t\mid s_t)\quad(\text{vLLM 采样那一刻算出来的，即 rollout\_log\_probs})
$$

$$
\text{old\_log\_probs}^{\text{decoupled}}=\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)\quad(\text{Megatron 对这批数据重新算一次，只算一次，作为锚点})
$$

"复用"就是 PSRL 的 bypass 模式：不让 Megatron 重新做一次 teacher-forcing 前向，直接把 vLLM 生成时已经算好的 `rollout_log_probs` 塞进 `old_log_probs`，省一次完整前向。不能放心直接复用的原因有两层：

1. **引擎数值不一致（即使权重完全相同）**：vLLM 和 Megatron 是两套独立计算栈——不同 attention kernel（vLLM 用 PagedAttention，按 block_table 分块累加，见 [§1.3](inference-mechanisms.md#13-pagedattention把-kv-cache-当虚拟内存管) 的在线 softmax；Megatron 训练前向是连续显存的稠密 attention）、不同精度/量化、不同并行切分。浮点加法不满足结合律，求和顺序不同结果就有细微差异——同一组权重、同一段 token，vLLM 和 Megatron 算出的 `log π(token)` 数学上该相等，数值上不相等。这正是 [§6.1](#61-权重同步与-resharding) "两种切法通常不一致" 在数值层面的体现。
2. **语义/时间不一致（PPO 理论要求稳定锚点）**：PPO 的 trust-region 保证要求一批数据做多次 mini-batch 内层更新时，`ratio_t` 分母必须固定不变。`rollout_log_probs` 是生成阶段、版本 $v_k$ 时算的，到这批数据真正进训练时 Megatron 权重可能已经是 $v_{k+m}$——直接拿它当分母，`ratio_t` 里混进的不只是"真实参数移动量"，还有"引擎数值偏差 + 额外版本滞后"，且这个偏差会被 `exp(·)` 指数放大，导致 PPO-clip 裁剪阈值失真、梯度产生系统性偏置。

decoupled 模式因此引入 `π_old`：这批数据进训练时 Megatron 只对它做一次前向、结果固定住，作为本批数据所有 mini-batch 内层更新共用的锚点——ratio 的"旧"和"新"从此来自同一套引擎、同一套数值实现，只反映纯粹的参数移动。`bypass_mode` 是用正确性换一次前向的性能优化，在引擎数值差异小、staleness 步数少时经验上通常可接受。

关键观察：⑤⑥ 这两步的前向计算方式，和 pretraining 的前向计算方式完全一样——都是"整条序列已知、一次并行前向、teacher forcing"，也就是 [§1.1](inference-mechanisms.md#11-prefill-与-decode-的根本差异) 里说的 **prefill**。区别只在于：pretraining 直接拿这个前向的 logits 去和真实下一个 token 做交叉熵；RLHF 是拿它去和 vLLM 生成时的 log prob 做比值修正。

**为什么 pretraining 没有这些问题**：根本原因是 pretraining 的 token 不是模型自己生成的，训练全过程只用得到"teacher forcing 前向"这一种计算模式，从不需要自回归解码。

```
Pretraining：
  磁盘语料（固定、不依赖当前权重）
    → Megatron 一次并行前向算出每个位置的 logits（就是一次巨大的 "prefill"）
    → 直接和语料里真实的下一个 token 算交叉熵
    → backward → optimizer.step()
  全程只有一套引擎、一份权重，永远是"当前权重版本"在算 loss

RLHF/Agentic RL：
  token 必须由模型自己"采样"出来（response 提前不存在，要现算）
    → 自回归采样天然需要 KV cache + 逐 token 解码 → 只有专门的推理引擎（vLLM）
       能把这个做到吞吐够用（continuous batching、PagedAttention，见 §1.3/§1.4）
    → 但推理引擎和训练引擎是两套并行切分、两套数值实现的进程/权重副本
    → 采样时用的权重版本，和训练时用来更新的权重版本，必然存在时间差
      （push/pull 需要时间，且刻意异步以获得吞吐，见上文）
    → 于是产生：off-policy 差距（需要 importance ratio 修正）
              权重切分不一致（需要 resharding，§6.1）
              数值不一致（需要重算 old_log_probs 而不能直接信任 rollout_log_probs）
```

一句话总结：pretraining 的"生成"是语料自带的，训练只需要一次前向（prefill 形态）；RLHF 的"生成"是模型自己现采的，必须先跑一个独立的解码引擎，这个引擎和训练引擎之间的权重同步缝隙，就是本节这些问题的唯一根源。

**PPO vs GRPO：advantage 怎么算的对比**（承接上面⑦步"组内归一化 reward（GRPO）或 GAE（PPO+critic）"）

两者的 clipped surrogate 更新规则本身完全相同（都是上面⑧的 `ratio·A` 公式），区别全部在 `A_t`（advantage）的来源：

- **PPO**：额外训练一个 **critic（价值网络 V_φ)**，预测"从当前前缀继续生成，期望还能拿多少 reward"，再用 GAE 把稀疏的终局 reward 分摊到每个 token：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

  代价：critic 要从任意中间前缀回归期望终局 reward，本身难学；且 actor+critic+reference(+reward model) 最多要同时维护 4 份模型。

**GAE 怎么把稀疏终局 reward 分摊到每个 token（原理展开）**：

GAE 不是把 reward 物理切成很多份分给每个 token，而是用 critic 对每个中间前缀的价值估计 `V(s_t)`，把"这一步让最终成功的概率提升了多少"变成逐步可算的信号，再指数加权累加。把 `k` 步 TD 残差展开：

$$
A_t^{(k)} = \delta_t + \gamma\delta_{t+1} + \cdots + \gamma^{k-1}\delta_{t+k-1}
$$

GAE 是这些估计的指数加权平均：

$$
A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l}
$$

`λ=0` 时只信任 critic 的单步局部判断（低方差、critic 不准则有偏）；`λ=1` 时完全信任真实采样到的终局 reward（无偏、方差大）。

**`δ_t` 到底是什么**：`δ_t` 叫 **TD 残差**（Temporal-Difference residual），是 GAE 求和里的基本单元：

$$
\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{这一步之后：实际 reward + 对未来的预期}} - \underbrace{V(s_t)}_{\text{这一步之前的预期}}
$$

拆开看：$r_t$ 是这一步实际拿到的 reward（LLM RLHF 里通常除了最后一个 token 都是 0）；$V(s_t)$ 是 critic 在**这一步发生前**对"这条前缀最终能拿多少 reward"的预测；$V(s_{t+1})$ 是生成 $a_t$ 之后，critic 对**新前缀**的预测。所以 $\delta_t$ 是一个"意外程度"信号：$\delta_t>0$ 说明这个 token 让 critic 觉得"更接近成功了"（该被鼓励），$\delta_t<0$ 说明变差了（该被抑制），$\delta_t\approx0$ 说明结果跟 critic 早就料到的一样，没有额外信息。把 GAE 求和截断到只留一项就是 $A_t\approx\delta_t$（对应 $\lambda=0$）；critic 自身的训练目标也是让 $V(s_t)$ 逼近 $r_t+\gamma V(s_{t+1})$，即让 $\delta_t\to0$——critic 训得越准，$\delta_t$ 才越能反映"真正的意外"而非"critic 没学好"。

**稀疏 reward 只在最后一个 token $T$ 非零时**（$r_t=0,\ t<T$），只有 $\delta_T=R-V(s_T)$ 含有真实 reward，其余 $\delta_t=V(s_{t+1})-V(s_t)$ 只是 critic 自己对相邻前缀的价值判断之差。GAE 的求和把 $\delta_T$ 以 $(\gamma\lambda)^{T-t}$ 的权重"传"给每个更早的 $t$。令 $\gamma=1,\ \lambda=1$，这个无穷和望远镜消项：

$$
A_t=\sum_{l=0}^{T-t}\delta_{t+l}=\big[V(s_{t+1}){-}V(s_t)\big]+\cdots+\big[R{-}V(s_T)\big]=R-V(s_t)
$$

即终局 reward 被完整传给每个 token，只是每个 token 的"起点" `V(s_t)` 不同，拿到的 advantage 也不同——这就是逐 token 的 credit assignment。

**数值例子**（接上文"1+1等于2"，`R=1`，critic 给出的价值估计 `V(s_t)` 依次为 `0.3,0.3,0.4,0.4,0.6,0.6,0.9`）：

```
δ_0=0  δ_1=0.1  δ_2=0  δ_3=0.2  δ_4=0  δ_5=0.3  δ_6=R-V(s_6)=0.1

λ=1 时 A_t = R - V(s_t)：
A_0=0.7  A_1=0.7  A_2=0.6  A_3=0.6  A_4=0.4  A_5=0.4  A_6=0.1
```

开局第一个 token（critic 只有 30% 把握时生成）拿到 advantage 0.7——很大的鼓励，因为它把"不确定"带向了"确定"；最后一个 token（critic 已 90% 确信时生成）只拿到 0.1——它没提供太多新信息。同一个 `R=1` 按"每个 token 实际让成功概率提升了多少"被不均匀地分配。$\lambda<1$ 时远处的 $\delta_T$ 被 $(\gamma\lambda)^{T-t}$ 打折，advantage 更依赖 critic 的局部判断而非那一次采样的真实 `R`，是用偏差换方差。

对比 GRPO：GRPO 给同一条 response 里每个 token 贴完全相同的标量，不区分"这个 token 有没有让结果变好"；GAE 借助 critic 实现真正的逐 token credit assignment，代价是 critic 必须先训准，否则错误的 `V(s_t)` 估计也会被这套传播机制带进每个 token 的梯度里。

- **GRPO**：不训 critic，对同一个 prompt 一次采样 **G 条** response，直接用组内 reward 的均值/方差做基线：

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
$$

  这个标量**均匀贴到整条 response 的每个 token 上**。少一个模型，但把额外成本转嫁到生成侧（一个 prompt 要采 G 条而不是 1 条）。

**数值例子**（接上文"1+1等于几"）：vLLM 对同一 prompt 采 G=4 条 response：

```
response1: "1+1等于2"        reward=1.0（对）
response2: "1+1等于3"        reward=0.0（错）
response3: "等于2"            reward=0.5（对但格式差）
response4: "1+1等于2，因为…"  reward=1.0（对，啰嗦）

mean=0.625, std≈0.415
A_1≈+0.90（贴给 response1 每个 token）
A_2≈-1.51
A_3≈-0.30
A_4≈+0.90
```

**对比表**：

| | PPO | GRPO |
|---|---|---|
| 需要 critic | 是 | 否 |
| advantage 粒度 | 逐 token（GAE，靠 critic bootstrap） | 整条 response 共享一个标量 |
| 额外显存/模型数 | actor+critic+ref(+RM) 最多 4 份 | actor+ref(+RM)，少一份 |
| 对 rollout 吞吐要求 | 一个 prompt 采 1 条 | 一个 prompt 采 G 条（8/16 常见），生成侧压力更大 |
| 适合场景 | reward 密集、可逐步评估 | reward 只在终局给出（RLVR：数学/代码对错），当前主流 Agentic RL 首选 |

GRPO advantage 粒度粗、在长/多轮轨迹里被稀释得更严重，呼应 [§6.3](#63-资源分配的两难) 提到的多轮长尾问题；但 RLVR 场景 reward 天然是终局稀疏信号，critic 很难学出有意义的中间价值估计，而 PSRL 这类系统本就把生成做成了专门优化吞吐的独立引擎（[§1.4](inference-mechanisms.md#14-continuous-batchingiteration-level-scheduling) continuous batching），一次多采 G 条的额外生成成本相对可控——这是当前主流 RLVR/Agentic RL 框架（DeepSeek-R1 等）选择 GRPO 而非经典 PPO 的核心原因。

**On-policy vs Off-policy：核心区别与代表算法**（回扣前面 staleness 的讨论）

区分点只有一个：生成数据的策略（行为策略 `μ`）和正在被优化的策略（目标策略 `π`）是不是同一个。`μ=π` 是 on-policy——策略一变旧数据就作废；`μ≠π` 是 off-policy——可以用任何来源的数据优化当前策略。

能"借用" off-policy 数据的数学工具是重要性采样：

$$
\mathbb{E}_{a\sim\pi}\big[f(a)\big] = \mathbb{E}_{a\sim\mu}\left[\frac{\pi(a\mid s)}{\mu(a\mid s)}\,f(a)\right]
$$

这个比值 `π(a|s)/μ(a|s)` 正是前面反复出现的 `ratio_t`——PPO 的 `ratio_t` 本质就是一次重要性采样修正，`clip` 是在"借用旧数据"和"控制比值方差爆炸"之间的工程妥协。另一条不需要重要性采样的路是估计的量本身与行为策略无关，比如 Q-learning 估计的 `Q*(s,a)` 只依赖真实环境动力学，不依赖当年谁选的动作，因此可以放心复用任意旧策略的数据。

**为什么 `ratio_t` 就是重要性采样（推导展开）**：PPO 真正想优化的目标定义在当前策略 `π_θ` 的采样分布下：

$$
J(\theta) = \mathbb{E}_{a\sim\pi_\theta}\big[A(s,a)\big]
$$

但每次梯度更新时手头的数据是已经生成好的、来自旧策略 `π_old` 的样本（PSRL 里可能是 vLLM 用 `v_k` 生成的 rollout，也可能是同一 batch 内 mini-batch 更新到中途的 `π_old` 锚点）——这正是"想对 `π` 求期望，手上只有 `μ` 的样本"的标准重要性采样场景。把上面的重要性采样公式代入 `π=π_θ`、`μ=π_old`、`f(a)=A(s,a)`：

$$
\mathbb{E}_{a\sim\pi_\theta}\big[A(s,a)\big] = \mathbb{E}_{a\sim\pi_{\text{old}}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\text{old}}(a\mid s)}\,A(s,a)\right] = \mathbb{E}_{a\sim\pi_{\text{old}}}\big[\text{ratio}_t\cdot A_t\big]
$$

`ratio_t` 就是这个重要性权重本身，不是碰巧长得像，是公式代入后的直接结果。两个需要注意的细节：① 严格的重要性采样要修正整条轨迹的分布，但 PPO/TRPO 只对"给定 `s` 选哪个 `a`"这一步做修正，状态访问分布仍然沿用 `π_old` 采出来的那批（近似当成没变）——这个近似只有 `θ` 离 `θ_old` 不远时才成立，是"trust region"这个名字的来源；② 重要性采样天生高方差，`π` 和 `μ` 差得越远，某些样本的权重越容易爆炸，PPO 不像 TRPO 用精确 KL 约束控制它，而是直接把 `ratio_t` 夹到 `[1-ε, 1+ε]`——`clip` 就是这个"防止 IS 权重失控"的安全阀。

**展开①：为什么说"整条轨迹的分布"，PPO 只修正了哪一部分**：一条轨迹 `τ=(s_0,a_0,...,s_T,a_T)` 在策略 `π` 下的联合概率是

$$
p_\pi(\tau) = \rho_0(s_0)\prod_{t=0}^{T}\pi(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t)
$$

`P` 是环境转移动力学，`π_θ` 和 `π_old` 共享同一个环境所以会在比值里约掉；真正严格的轨迹级重要性权重是**所有时间步单步比值的连乘积**：

$$
w(\tau)=\frac{p_{\pi_\theta}(\tau)}{p_{\pi_{\text{old}}}(\tau)}=\prod_{t=0}^{T}\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
$$

而 PPO/TRPO 实际用的 `ratio_t` 只是这个连乘积里的**单独一项**，并且状态 `s_t` 直接取 `π_old` 真实走过的那个前缀，当成 `π_θ` 也会走到这里——完全没有乘上 t 之前的比值。差别在于：状态 `s_t`（LLM 场景里就是前缀 `y_{<t}`）本身是前面所有动作的结果，如果真的换 `π_θ` 去生成，它在更早的位置就可能选到不同的 token，走到的前缀分布 `d^{π_θ}(s)` 一般不等于 `d^{π_old}(s)`——这才是"整条轨迹分布"没被修正的部分。

敢这样近似有两个理由。理论上，Kakade & Langford 的性能差异恒等式给出精确形式 `η(π_θ)-η(π_old) = (1/(1-γ))·𝔼_{s~d^{π_θ},a~π_θ}[A^{π_old}(s,a)]`，但右边需要 `π_θ` 的状态样本，还没跑出来无法计算；把 `d^{π_θ}` 换成手头已有的 `d^{π_old}` 构造出可计算的 surrogate 目标，再对内层 `a~π_θ` 做单步重要性采样才得到 `ratio_t·A_t`，而这个替换引入的误差被证明是有界的、正比于 `π_θ` 和 `π_old` 之间的 KL 散度——`θ` 离 `θ_old` 越近，近似越准，这就是 trust region / clip 存在的理论根据。实践上，`∏_t ratio_t` 是几十上百项连乘，序列一长这个乘积几乎必然坍缩到 0 或者爆炸，方差远比单步 `ratio_t` 更失控，没法直接用。

落到 PSRL：轨迹的状态 `s_t` 就是到当前为止的前缀，PPO 更新时这个前缀是 vLLM 用 `π_old`（版本 `v_k`）已经生成好、写死的序列，训练时不会用 `π_θ` 重新采样一条前缀出来对比——损失问的是"给定这个固定前缀，`π_θ` 对同一个 token 的打分是多少"，而不是"`π_θ` 自己生成会不会走到完全不同的前缀"。PSRL 的 staleness bound（限制 `v_k→v_{k+m}` 的版本差距）和 PPO 的 `clip(ε)` 本质上是同一件事：都在限制"`π_θ` 已经漂移到让『前缀分布不变』这个近似失真"的程度。

代表算法：

| 类别 | 代表算法 | 数据复用方式 |
|---|---|---|
| 纯 on-policy 策略梯度 | REINFORCE, A2C/A3C | 每批数据只用一次 |
| 信任域/近似 on-policy | **TRPO, PPO, GRPO** | 同批数据允许多次 mini-batch 更新，靠 `ratio+clip` 容忍漂移 |
| 纯 off-policy 价值迭代 | Q-learning, DQN | 靠贝尔曼最优性方程，replay buffer 里任意旧策略数据都能用 |
| off-policy actor-critic | DDPG, TD3, SAC | critic 走 Q-learning 式 off-policy 更新，actor 借 critic 梯度优化 |
| 全离线 | Offline RL, LLM 对齐里的 **DPO** | 完全不做在线采样，从头到尾用静态数据集训练 |

PPO/GRPO 理论上是 on-policy——目标函数定义在 `π_θ` 自己的采样分布下，但同一批数据要做多轮 mini-batch 更新，`θ` 每挪一点数据就老一点，所以需要 `ratio_t=π_θ/π_old` 和 clip：**PPO/GRPO 不是纯 on-policy，而是"容忍一个被 clip 主动限制住的小范围 off-policy 漂移"**。上文 $v_k\to v_{k+m}$ 的版本漂移是同一种机制在系统层面被放大：生成数据的策略和拿去训练的策略之间的差距，从"mini-batch 内层几次更新的挪动量"变成了"好几个完整训练步的挪动量"，量级更大但用的修正工具（`ratio+clip`、重算 `old_log_probs`）是同一套；staleness controller 的上限就是在限制这个"系统级 off-policy 程度"不超出 clip 机制还能兜住的范围。

反例参照：LLM 对齐里常拿来对比的 **DPO** 完全离线——直接在固定偏好数据集上做监督式对比 loss，不做在线采样、不需要 reward model、不需要 rollout 引擎，天然没有 on/off-policy 匹配问题，也就不需要 `ratio`、`clip`、staleness bound、PS 权重同步，这是它工程上比 PPO/GRPO 简单得多的根本原因。

### 6.1 权重同步与 Resharding

训练端权重按 FSDP 或 Megatron 的方式切分，推理端按 vLLM 的 TP 方式切分，**两种切法通常不一致**。每次策略更新都需要：

1. 从训练端各 rank 聚合出完整（或按目标切法重排的）权重
2. 传输到所有 rollout 副本
3. 按 vLLM 的切分方式装载

**成本**：70B bf16 = 140 GB。若跨节点走单口 50 GB/s 的朴素 broadcast，一个副本就要 **2.8 秒**；rollout 副本多了，同步开销会直接吃掉训练时间。

常见优化：

- **分桶流水**：边转换边发送，让 reshard 计算与网络传输重叠
- **Tree / Ring broadcast** 替代朴素 one-to-many
- **同机零拷贝**：CUDA IPC 直接共享显存句柄
- **RDMA 直传**：绕过 CPU 内存拷贝
- **参数服务器 + 版本滞后容忍**：把「每步严格同步」放松为「允许样本使用略陈旧的策略」，用算法侧的 off-policy 容忍（重要性采样修正）换系统侧的吞吐。这是 PSRL 的核心设计之一

### 6.1b PP+TP 二维网格：训练端 shard 到底是什么

回应一个常见疑问：`6.0` 里提到的"Megatron TP-PP shard"（train rank 直接 push 给 PS 的那份权重），具体组织方式是不是"先按 stage 切层，每个 stage 再对拿到的那部分层做 TP 切分"？——是的，两个切分方向正交叠加：

- **PP 切分方向**：沿"层索引"轴切。80 层模型、`PP=4`，stage0 拿 layer0-19、stage1 拿 layer20-39……每个 stage **整块拥有**这些层的**完整参数矩阵**（这一步还没有 TP 介入）。
- **TP 切分方向**：在每个 stage 内部，对它拥有的那些层，每个权重矩阵再按 [§3.1](training-parallelism.md#31-张量并行让两次矩阵乘只通信一次) 的 column/row parallel 切成 `t` 份，分给该 stage 内的 `t` 张卡。

所以完整的 GPU 排布是一个二维网格，每张卡由 `(pp_rank, tp_rank)` 唯一确定（叠 DP 则是三维）。`(pp_rank=i, tp_rank=j)` 这张卡上实际持有的权重 = "stage i 拥有的那些层" ∩ "这些层每个权重矩阵按列/行切出的第 j 份"。一个具体的权重张量（比如某层的 QKV 投影矩阵）**只属于一个 PP stage**（哪个 stage 拥有那一层），但在这个 stage 内部被横着切成 `t` 片分布在 TP 组的 `t` 张卡上——PP 决定"归哪个 stage 管"，TP 决定"这一层内部怎么再切细"，两者是正交的两根轴，不冲突、不重叠。这跟 [§5.3 标准排布图](training-parallelism.md#53-标准排布) 是同一件事的两个角度：那张图画的是"外层 DP → PP → TP"的并行度嵌套关系，这里说的是"权重张量本身"在这个嵌套结构下具体落在哪张卡的哪一小块显存里。

**为什么 push 权重时不用先 all-gather**：正因为每个 train rank 天生就是这个二维网格里固定的一格，它手上的权重本来就是"某几层 × 某个列/行切片"这样一份现成、边界清晰的 shard，可以直接原样流式发给 PS Worker（`6.0` 提到的 Push 协议），不需要先把整层权重拼回完整再重新切。真正的开销转移到了 PS/拉取侧：vLLM 推理端的切法（通常 TP 度不同，且推理端一般不用 PP）跟训练端这套 `(pp_rank, tp_rank)` 网格边界对不齐，这正是 [§6.1](#61-权重同步与-resharding) 需要 resharding 的根源。

### 6.2 Rollout 长尾

Agent 多轮交互场景下，单条轨迹的生成长度方差极大（有的 2 轮结束，有的 30 轮还在调工具）。同步式 PPO 中，训练端必须等**最慢的那条轨迹**完成。

注意区分两个层次：

- **continuous batching 解决的是 GPU 内部的空转**：某条序列结束后立刻让新请求补位
- **它解决不了「训练端整体在等生成端」**：这是框架级的同步屏障

所以需要框架层的异步解耦：rollout、reward 计算、训练三条流水线独立推进，通过队列（PSRL 的 TransferQueue）传递样本，用版本号管理 off-policy 程度。PSRL 论文报告的 2.68× 提升，主要来自消除这类等待，而非单卡 kernel 更快。

### 6.3 资源分配的两难

- **共置（colocate）**：训练和推理共享同一批 GPU，分时复用。显存需要在两者间腾挪（训练时 offload 推理的 KV cache，反之亦然），切换有开销，但资源利用率高、权重同步可走 NVLink 甚至 IPC
- **分离（disaggregate）**：训练和推理各占独立 GPU 池，无切换开销、可独立扩缩容，但权重同步要跨网络，且两边负载不匹配时有一方闲置

异步框架通常倾向分离 + 版本容忍，因为异步本身就要求两边独立推进。

---
