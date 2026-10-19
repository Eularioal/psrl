# 训练并行、通信量与带宽分析

> **核心结论**：按切分对象理解训练并行，再用每卡通信量和互联带宽判断并行策略应如何排布。

## 目录

- [二、训练侧的并行维度](#二训练侧的并行维度)
  - [显存构成（理解为什么要切）](#显存构成理解为什么要切)
- [三、Megatron 的具体切分方式](#三megatron-的具体切分方式)
  - [3.1 张量并行：让两次矩阵乘只通信一次](#31-张量并行让两次矩阵乘只通信一次)
  - [3.2 Sequence Parallelism（Megatron-LM 第二篇论文）](#32-sequence-parallelismmegatron-lm-第二篇论文)
  - [3.3 Context Parallelism（长上下文）](#33-context-parallelism长上下文)
  - [3.4 Pipeline Parallelism 的调度](#34-pipeline-parallelism-的调度)
- [四、通信量定量分析](#四通信量定量分析)
  - [4.1 集合通信原语的每卡流量](#41-集合通信原语的每卡流量)
  - [4.2 TP 通信量（最关键）](#42-tp-通信量最关键)
  - [4.3 PP 通信量](#43-pp-通信量)
  - [4.4 DP 通信量](#44-dp-通信量)
  - [4.5 ZeRO / FSDP 通信量](#45-zero--fsdp-通信量)
  - [4.5b 一个 FSDP shard 里装的是什么？跨层一致吗](#45b-一个-fsdp-shard-里装的是什么跨层一致吗)
  - [4.6 MoE / EP 通信量](#46-moe--ep-通信量)
- [五、带宽的硬约束与策略排布](#五带宽的硬约束与策略排布)
  - [5.1 GPU 显存带宽（决定推理 decode 上限）](#51-gpu-显存带宽决定推理-decode-上限)
  - [5.2 互联带宽（双向聚合，每 GPU）](#52-互联带宽双向聚合每-gpu)
  - [5.3 标准排布](#53-标准排布)
  - [5.4 MFU 估算](#54-mfu-估算)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。原文整理日期为 2026-09-10，后续追加内容的日期以各节记载为准；保留章节编号以便对照旧引用。

[返回专题索引](../vllm-and-training-parallelism.md)

## 二、训练侧的并行维度

| 维度 | 切分对象 | 主要通信原语 | 通信是否在关键路径 | 典型放置 |
|---|---|---|---|---|
| **DP** 数据并行 | batch 维 | 梯度 All-Reduce | 否（可与反向重叠） | 最外层，跨节点 |
| **TP** 张量并行 | 权重矩阵（层**内**） | 激活 All-Reduce | **是** | **节点内 NVLink** |
| **PP** 流水并行 | 层（层**间**） | 边界激活 P2P | 部分（有 bubble） | 跨节点 |
| **SP** 序列并行 | LayerNorm/Dropout 区的序列维 | Reduce-Scatter + All-Gather | 是 | 跟随 TP |
| **CP** 上下文并行 | attention 的序列维 | Ring P2P 或 All-Gather(KV) | 是 | 长上下文时启用 |
| **EP** 专家并行 | MoE experts | All-to-All | 是 | MoE 模型 |
| **ZeRO/FSDP** | 优化器态/梯度/参数 | Reduce-Scatter + All-Gather | 部分 | 替代或叠加 DP |

三个维度的组合称为 **3D 并行**，实际大规模训练常是 4D/5D（再叠 SP、CP、EP）。

### 显存构成（理解为什么要切）

以参数量 P、bf16 混合精度 + Adam 为例，每个参数需要：

| 项 | 字节/参数 |
|---|---|
| bf16 参数 | 2 |
| bf16 梯度 | 2 |
| fp32 参数副本（master weight） | 4 |
| fp32 Adam 一阶动量 | 4 |
| fp32 Adam 二阶动量 | 4 |
| **合计** | **16** |

70B 模型 → **1.12 TB** 静态显存，远超单卡 80 GB。再加上激活显存（与 `b·s·h·L` 成正比），必须切分。

- **TP** 切的是「参数 + 梯度 + 优化器态 + 激活」全部
- **PP** 切的是「参数 + 梯度 + 优化器态」，激活按 stage 分布
- **ZeRO-1/2/3** 分别切「优化器态 / +梯度 / +参数」
- **SP** 只切激活（LayerNorm/Dropout 区）

---

## 三、Megatron 的具体切分方式

### 3.1 张量并行：让两次矩阵乘只通信一次

核心技巧是 **column parallel 后接 row parallel**，中间的逐元素操作无需通信。

**MLP 部分**（`h → 4h → h`）：

```
第一个线性层  Y = X · A      A 按【列】切:  A = [A₁ | A₂ | ... | A_t]
              → 每卡得到 Y_i = X · A_i，无需通信
GeLU(Y_i)     → 逐元素，无需通信（这是列切的关键理由）
第二个线性层  Z = Y · B      B 按【行】切:  B = [B₁; B₂; ...; B_t]
              → 每卡得到部分和 Z_i = Y_i · B_i
              → 【All-Reduce】求和得到完整 Z
```

如果第一层按行切，GeLU 之前就得先通信一次——列切正是为了避开这一点。

**Attention 部分**：

```
QKV 投影      column parallel → 等价于【按 attention head 切分】
              每卡负责 n_heads/t 个 head，head 之间的 attention 计算天然独立
输出投影 O    row parallel → 【All-Reduce】
```

**结论**：每个 transformer layer，**前向 2 次 All-Reduce，反向 2 次 All-Reduce**。Megatron 论文中记作共轭算子对 f / g：前向 f 为恒等、g 为 all-reduce；反向恰好相反。

### 3.2 Sequence Parallelism（Megatron-LM 第二篇论文）

**动机**：TP 切不到 LayerNorm、Dropout、残差这些区域，它们的激活在每张卡上是完整复制的，成为显存瓶颈。

**做法**：在这些区域沿**序列维**切分，每卡只持有 `s/t` 的序列。于是 TP 区与 SP 区的边界需要转换：

```
SP 区 → TP 区：All-Gather（把序列拼回完整）
TP 区 → SP 区：Reduce-Scatter（原来的 All-Reduce 降级）
```

关键性质：**`All-Reduce = Reduce-Scatter + All-Gather`**，所以

- 通信**总量完全不变**
- 但 LayerNorm/Dropout 区的激活显存降到 `1/t`

这是纯收益的优化，现代 Megatron 配置基本默认开启（`--sequence-parallel`，需与 TP 同时使用）。

### 3.3 Context Parallelism（长上下文）

当序列长度到 32k、128k 甚至 1M，attention 的 `O(s²)` 计算和激活成为主要瓶颈。CP 沿序列维切分 attention 本身：

- **Ring Attention**：KV 块在环上轮转，每卡依次与本地 Q 计算，用 P2P 通信与计算重叠
- **All-Gather KV**：直接聚合完整 KV（通信量小但显存占用高，适合 GQA 下 KV 很小的情况）
- 因果 mask 导致负载不均，需要 **zigzag / striped** 等分片方式做负载均衡

### 3.4 Pipeline Parallelism 的调度

**朴素 GPipe**：所有 microbatch 前向做完再做反向 → 气泡比 `(p-1)/m`，且要缓存 m 份激活。

**1F1B（PipeDream-Flush）**：稳定阶段交替执行一次前向一次反向 → 气泡比仍是 `(p-1)/m`，但**峰值激活显存降到约 p 份**（而非 m 份）。Megatron 的默认调度。

**Interleaved 1F1B（Virtual Pipeline）**：每张卡持有 v 段**不连续**的层（如卡 0 持有 layer 0-3 和 layer 16-19）→ 气泡比降到 `(p-1)/(m·v)`，代价是 P2P 通信次数变为 v 倍。

**气泡比的实际含义**：p=8、m=8 时气泡 = 7/8 = 87.5%（灾难）；m=64 时 = 11%；再开 v=4 → 2.7%。所以 **PP 必须配足够多的 microbatch**，这也意味着 global batch size 不能太小。

---

## 四、通信量定量分析

### 4.1 集合通信原语的每卡流量

设消息大小为 D 字节，参与设备数 n（Ring 算法）：

| 原语 | 每卡收发字节数 |
|---|---|
| All-Reduce | `2(n-1)/n · D` |
| Reduce-Scatter | `(n-1)/n · D` |
| All-Gather | `(n-1)/n · D` |
| Broadcast (ring/tree) | ≈ `D` |
| All-to-All | `(n-1)/n · D` |

n 较大时 `2(n-1)/n → 2`，即 All-Reduce 约等于 **2 倍消息大小**。

**NCCL 的 busbw 概念**：`algbw = D / t`，`busbw = algbw × 2(n-1)/n`。busbw 才是可以直接和硬件链路带宽比较的量。

**为什么需要这个区分**：

`algbw`（algorithm bandwidth）把集合通信当成"一次性搬 D 字节"来算速度，但 Ring 算法里数据是像接力棒一样绕环逐跳转发累加的，物理链路上实际传输的字节数远大于逻辑消息大小 D。以 Ring All-Reduce 为例，它由 Reduce-Scatter（n-1 步）+ All-Gather（n-1 步）两段组成，数据切成 n 份，每步每卡收发 `D/n`：

```
Reduce-Scatter 阶段：(n-1) 步 × D/n = (n-1)/n · D
All-Gather    阶段：(n-1) 步 × D/n = (n-1)/n · D
────────────────────────────────────────────
合计每卡实际收发量  = 2(n-1)/n · D
```

`busbw` 就是拿这个"实际物理流量"除以时间：`busbw = (2(n-1)/n · D) / t = algbw × 2(n-1)/n`。不同原语系数不同（对应上面表格）：Reduce-Scatter / All-Gather / All-to-All 只是"半程"，系数 `(n-1)/n`；Broadcast 近似只走一遍，系数≈1。

**物理意义**：`busbw` 衡量的是"每条物理链路上平均实际流过多少字节/秒"，量纲上和硬件带宽规格（NVLink/IB 的额定值）是同一件事，可以直接算链路利用率 `busbw / 硬件额定带宽`。如果只看裸的 `algbw` 去跟 NVLink 900GB/s 比，会严重低估链路已经跑满的程度——因为 `algbw` 没有反映 ring 算法让同一份数据在链路上被多次搬运的事实。[§4.2](#42-tp-通信量最关键) 算例里「ring 系数 × D」算出的「实际每卡收发字节」正是这个道理的具体应用，若漏乘该系数会严重低估通信耗时。

### 4.2 TP 通信量（最关键）

单次 All-Reduce 的消息大小（bf16）：

```
D = b · s · h · 2   [bytes]
```

其中 `b·s` 是一个 microbatch 的 token 总数。一个训练步（前向+反向）的 TP 通信总量：

```
TP_comm_per_gpu = 4 × 2(t-1)/t × (b·s·h·2) × L
                   ↑         ↑                ↑
              fwd2+bwd2   ring系数         层数
```

#### 完整算例

**配置**：70B 模型（h=8192, L=80），TP=8，microbatch = 4096 token，bf16，H100

```
D                  = 4096 × 8192 × 2                 = 67.1 MB
ring 系数           = 2 × (8-1)/8                     = 1.75
单次 AllReduce/卡   = 1.75 × 67.1 MB                  = 117.4 MB
每层 fwd+bwd (4次)  = 117.4 × 4                       = 469.8 MB
整步 80 层          = 469.8 MB × 80                   = 37.6 GB / GPU / step
```

**同一步的计算量**：

```
FLOPs = 6 · P · T = 6 × 70e9 × 4096              = 1.72 PFLOP
每卡（TP=8 均分）                                  = 215 TFLOP
H100 实测有效算力（约 40% MFU）≈ 400 TFLOP/s
计算耗时                                           ≈ 537 ms
```

**不同互联下的 TP 通信耗时**：

| 互联 | 有效 busbw | 通信耗时 | 占计算时间比 | 可用性 |
|---|---|---|---|---|
| NVLink 4（H100 节点内） | ~380 GB/s | **99 ms** | 18% | ✅ 可用 |
| NVLink 3（A100 节点内） | ~240 GB/s | 157 ms | 29% | ✅ 可用 |
| PCIe 5.0 ×16 | ~55 GB/s | 684 ms | 127% | ❌ |
| InfiniBand NDR 400G（单口） | ~45 GB/s | **836 ms** | 156% | ❌ |

**两个关键结论**：

1. **TP 域绝不能跨出 NVLink**——跨节点后通信时间超过计算时间，扩展效率跌破 50%
2. **TP 通信无法隐藏**——它在关键路径上（`AllReduce` 的结果是下一个算子的输入），不像 DP 梯度可以和反向计算重叠。只有部分实现能做细粒度的 GEMM-通信重叠（如 TransformerEngine 的 `--tp-comm-overlap`）

### 4.3 PP 通信量

每个 microbatch 在每个 stage 边界只传一次激活：

```
PP_comm_per_boundary = b · s · h · 2  [bytes]   （前向一次，反向一次）
```

同样配置下 = **67 MB**，比 TP 的 37.6 GB **小约 500 倍**。

这就是并行策略排布的根本依据：**通信重的维度（TP）放在带宽高的域内（NVLink），通信轻的维度（PP）放到带宽低的域（IB）**。

注意：若同时开 SP，边界激活也是切分的，PP 通信量还会再除以 t。

### 4.4 DP 通信量

梯度 All-Reduce：

```
DP_comm_per_gpu = 2(d-1)/d × P_bytes
```

70B 模型 bf16 梯度 = 140 GB → d 较大时每卡约 **245 GB**。数字很大，但：

- **每个 global step 只做一次**，而一个 global step 包含 m 个 microbatch 的计算
- **可与反向计算充分重叠**（按 bucket 分桶，某层反向算完就立即发起该桶的 all-reduce）
- 实际配 TP/PP 后，DP 组内每卡的参数分片已经小很多

所以 DP 通常不是瓶颈，前提是有足够的 IB 带宽（8×400G 节点级 ~400 GB/s）且重叠做得好。

### 4.5 ZeRO / FSDP 通信量

| 级别 | 切分内容 | 每步通信量 | 相对 DDP |
|---|---|---|---|
| DDP | 无 | All-Reduce = `2P` | 1.0× |
| ZeRO-1 | 优化器态 | RS(`P`) + AG(`P`) = `2P` | 1.0× |
| ZeRO-2 | + 梯度 | RS(`P`) + AG(`P`) = `2P` | 1.0× |
| **ZeRO-3 / FSDP** | + 参数 | 前向 AG(`P`) + 反向 AG(`P`) + RS(`P`) = **`3P`** | **1.5×** |

ZeRO-3/FSDP 多出的 50% 通信来自：参数本身也被切开了，前向和反向都要临时 All-Gather 回完整权重。

**HSDP（Hybrid Sharded Data Parallel）**：节点内 shard（走 NVLink，承担 AG/RS 的高频通信）、节点间 replicate（走 IB，只做一次梯度 All-Reduce）。用显存换带宽，是跨节点 FSDP 的标准做法。

**动机**：纯 FSDP-3 把 N 张卡当成一个 shard 组时，AG/RS 是逐层触发的阻塞式通信（和 [§4.2](#42-tp-通信量最关键) 里 TP 通信"在关键路径上、无法隐藏"是同一类问题）。一旦 shard 组跨节点，这些通信就要走 IB，被慢速链路拖垮。

**结构**：把 N 张卡排成 2D device mesh `(replicate 维度 r, shard 维度 s)`，`N = r×s`，通常 s = 单节点 GPU 数（如 8）：

```
节点0（NVLink domain）: GPU0..7  ──┐
节点1（NVLink domain）: GPU8..15 ──┼─ 同一 shard rank 之间做 replicate（IB）
节点2（NVLink domain）: ...       ──┘
      └── 组内 s 张卡做 shard（NVLink）
```

- **组内（shard 维度）**：标准 FSDP-3 的 AG/RS，全部发生在节点内，走 NVLink。
- **组间（replicate 维度）**：每个 shard 组各自持有完整的"1/s 参数"，组间只对同一分片做一次梯度 All-Reduce——和 DDP 一样是一次性、可与反向计算重叠的通信，且数据量已经缩小到本地分片 `P/s`（而不是完整 `P`）。

**每卡通信量**：

```
组内（NVLink，高频、阻塞）：前向 AG + 反向 AG + 反向 RS  ≈ 3 · (P/s)
组间（IB，仅一次，可重叠）：梯度 All-Reduce            ≈ 2(r-1)/r · (P/s)
```

高频阻塞的部分被关在快链路里，慢链路只承担一次性、可重叠、且数据量已缩小的那部分。

**代价**：纯 FSDP-3（N 卡一个 shard 组）给每卡 `1/N` 的参数/梯度/优化器态显存节省；HSDP 只有 `1/s`，比纯 FSDP-3 多用约 `r` 倍显存——这就是"用显存换带宽"的具体含义。是 PyTorch FSDP2 的 `HYBRID_SHARD` 策略，也是 [§5.3](#53-标准排布)"TP 不出 NVLink、DP/FSDP 走 IB"同一设计哲学在 FSDP 内部的再应用。

### 4.5b 一个 FSDP shard 里装的是什么？跨层一致吗

**一个 shard = 参数分片 + 对应梯度分片 + 对应优化器态分片**，三者边界严格对齐（by construction，不是碰巧）：

- **优化器态分片**：优化器（如 Adam 的 m/v）从来只见过、只创建本地那一份参数分片，所以优化器态天然和参数分片同形状、同分布
- **梯度分片**：反向传播时先把当前 FSDP unit 的完整参数 All-Gather 回来、算出**完整梯度**，算完立刻 Reduce-Scatter，切回和参数一样的分片方式——所以梯度分片=参数分片"同位置"那一份

**不包括激活值**：激活不是 FSDP 主动切分的对象，是不同 DP rank 天然吃不同数据 batch 导致的差异，跟 shard 机制无关。

**跨层布局是否一致，取决于 FSDP1 还是 FSDP2：**

| | FSDP1（经典 `FlatParameter`） | FSDP2（per-parameter，基于 DTensor） |
|---|---|---|
| 切分粒度 | 按 wrap policy 划的 unit（通常每个 transformer layer 一个 unit），把 unit 内所有参数 **flatten 成一个 1D 向量**，padding 后切成 N 个等长连续 chunk | 不 flatten，每个参数张量单独视为 DTensor，沿指定维度（通常 dim 0）切成 N 份 |
| rank r 拿到什么 | 该 unit flatten 向量的第 r 段（逐字节连续切分，可能把单个参数横跨在两个 rank 的边界上） | 每个参数 `[r·d/N : (r+1)·d/N]` 这一段连续行 |
| 跨层是否一致 | 若各层结构完全同构（形状、参数顺序一致），flatten 后总长度相同 → 切分边界落在相同相对偏移，规律上一致；但具体某参数是否被横切在边界上，取决于该 unit 内参数排列顺序和总长度，与层无关 | 若各层同名参数形状相同（标准同构 transformer stack），**天然严格一致**：每一层 rank r 永远拿到结构上同一相对位置的那一段 |
| 备注 | PyTorch 原始 FSDP 实现 | 更"干净"，更容易和 TP 的切分组合（DTensor 可以描述"先按 TP 切一刀，再按 FSDP 切一刀"） |

### 4.6 MoE / EP 通信量

每个 MoE 层前向需要两次 All-to-All：

```
dispatch：把 token 按路由结果发给对应 expert 所在卡
combine： 把 expert 输出送回原 token 所在卡
```

单次量级 ≈ `b · s · h · topk · 2 × (e-1)/e` 字节。反向再来两次。

All-to-All 是对网络拓扑最敏感的原语（每对节点都有流量，容易打满交换机），是 MoE 大规模训练的头号系统难题。常见缓解手段：

- **分层 All-to-All**（先节点内聚合再跨节点）
- **通信-计算重叠**（DeepEP 等）
- **路由负载均衡损失**（避免 expert 热点）

---

## 五、带宽的硬约束与策略排布

### 5.1 GPU 显存带宽（决定推理 decode 上限）

| GPU | HBM 容量/带宽 | BF16 算力(dense) | Ridge point |
|---|---|---|---|
| A100 80GB | 80 GB / 2.0 TB/s | 312 TFLOPS | 156 |
| H100 SXM | 80 GB / 3.35 TB/s | 989 TFLOPS | 295 |
| H200 | 141 GB / 4.8 TB/s | 989 TFLOPS | 206 |
| B200 | 192 GB / ~8 TB/s | ~2250 TFLOPS | 281 |

注：厂商标称的 BF16 算力常含 2:1 结构化稀疏（数字翻倍），上表取 dense 值。

### 5.2 互联带宽（双向聚合，每 GPU）

| 互联 | 带宽 | 域 |
|---|---|---|
| PCIe 4.0 ×16 | 64 GB/s | 节点内 |
| PCIe 5.0 ×16 | 128 GB/s | 节点内 |
| NVLink 3（A100） | 600 GB/s | 节点内 8 卡 |
| NVLink 4（H100） | 900 GB/s | 节点内 8 卡 |
| NVLink 5（B200/GB200） | 1800 GB/s | NVL72 可达 72 卡 |
| InfiniBand HDR 200G | 25 GB/s | 单口 |
| InfiniBand NDR 400G | 50 GB/s | 单口 |
| 8 口 NDR 节点 | ~400 GB/s | 节点级 |

**关键事实：节点内互联比跨节点快约一个数量级**（900 vs 50 GB/s 单口）。所有并行策略的排布规则都源自这条鸿沟。

GB200 NVL72 把 NVLink 域扩展到 72 卡，正是为了把「NVLink 域」这个稀缺资源做大，让 TP/EP 能开得更宽。

### 5.3 标准排布

```
┌─ DP / FSDP ──── 最外层，跨节点 IB，通信可与反向重叠
│  ┌─ PP ──────── 跨节点，相邻 stage 尽量物理相邻，通信轻但有 bubble
│  │  ┌─ CP ───── 节点内或近邻，长上下文才启用
│  │  │  ┌─ TP ── 严格节点内 NVLink，通信最重且在关键路径
│  │  │  │  └─ SP ── 跟随 TP，零额外通信，省激活显存
```

**配置经验法则**：

1. **TP**：先用 TP 撑到「单层权重 + 激活」能放下，一般 `TP ≤ 8`（不超出 NVLink 域）。模型 hidden size 小的时候 TP 效率会下降（消息太小、通信延迟占比高）
2. **PP**：再用 PP 撑到整个模型能放下。`PP` 的选择要保证 `microbatch 数 m ≥ 4·p`，否则气泡吃掉收益
3. **DP/FSDP**：剩下的卡全给 DP 提吞吐
4. **SP**：只要开了 TP 就开
5. **CP**：序列 > 16k 时考虑
6. **重计算（activation checkpointing）**：显存还不够时才开，代价约 30% 额外前向计算；优先用 selective recompute（只重算 attention 部分）

### 5.4 MFU 估算

```
MFU = (6 · P · T_tokens) / (t_step · N_gpu · Peak_FLOPS)
```

（含 attention 项时为 `6·P·T + 12·L·h·T²`，长上下文下第二项不可忽略）

实践参考值：稠密模型 40~55% 是良好水平，MoE 由于 All-to-All 通常 25~40%。

---
