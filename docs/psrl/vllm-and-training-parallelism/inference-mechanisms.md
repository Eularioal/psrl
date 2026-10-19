# vLLM 推理机制：缓存、调度与合批

> **核心结论**：从 prefill/decode 的计算差异出发，理解 KV 缓存、PagedAttention、前缀复用、连续合批和选择性合批；原有算例与性能边界保留。

## 目录

- [一、vLLM 的核心机制](#一vllm-的核心机制)
  - [1.1 Prefill 与 Decode 的根本差异](#11-prefill-与-decode-的根本差异)
  - [1.2 KV Cache 的显存账](#12-kv-cache-的显存账)
  - [1.3 PagedAttention：把 KV cache 当虚拟内存管](#13-pagedattention把-kv-cache-当虚拟内存管)
  - [1.3b Prefix Caching：跨请求复用 KV block 的机制](#13b-prefix-caching跨请求复用-kv-block-的机制)
  - [1.4 Continuous Batching（iteration-level scheduling）](#14-continuous-batchingiteration-level-scheduling)
  - [1.5 Chunked Prefill](#15-chunked-prefill)
  - [1.6 Prefill/Decode 分离（Disaggregation）](#16-prefilldecode-分离disaggregation)
  - [1.7 其余关键优化](#17-其余关键优化)
  - [1.8 vLLM 中的并行](#18-vllm-中的并行)
  - [1.9 Linear Attention / SSM 层的 cache 管理：什么时候需要多存一份 checkpoint](#19-linear-attention--ssm-层的-cache-管理什么时候需要多存一份-checkpoint)
- [Selective batching：投影按 token 合批，attention 按请求隔离](#selective-batching投影按-token-合批attention-按请求隔离)
  - [自建算例：两个 decode 请求与一个 prefill 请求](#自建算例两个-decode-请求与一个-prefill-请求)
  - [投影：沿 token 维拼接，共用权重矩阵](#投影沿-token-维拼接共用权重矩阵)
  - [Attention：保留请求边界，分别访问变长 KV](#attention保留请求边界分别访问变长-kv)
  - [性能边界与现代实现](#性能边界与现代实现)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。原文整理日期为 2026-09-10，后续追加内容的日期以各节记载为准；保留章节编号以便对照旧引用。

[返回专题索引](../vllm-and-training-parallelism.md)

## 一、vLLM 的核心机制

### 1.1 Prefill 与 Decode 的根本差异

LLM 自回归推理分两阶段，计算特性完全相反：

| | Prefill（处理 prompt） | Decode（逐 token 生成） |
|---|---|---|
| 一次前向处理 token 数 | T（整个 prompt） | 1（每条序列） |
| 计算量 | `2·P·T` | `2·P·B`（B = batch size） |
| 权重读取量 | 全部权重读一遍 | 全部权重读一遍 |
| 算术强度 | 高 | ≈ B FLOP/byte |
| 瓶颈 | **compute-bound** | **memory-bandwidth-bound** |
| 优化目标 | TTFT（首 token 延迟） | TPOT / TBT（token 间延迟）、吞吐 |

**Roofline 分析**：decode 阶段的 GEMM，权重为 bf16（2 byte/元素），每个元素对每条序列贡献 2 FLOP（一次乘一次加），所以

```
算术强度 = (2 × B × N_weights) / (2 × N_weights) = B  [FLOP/byte]
```

H100 SXM 的 ridge point：

```
989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/byte
```

即 **batch 必须堆到约 300 才能让 Tensor Core 饱和**。batch = 1 时算力利用率不足 1%，GPU 几乎全部时间在从 HBM 搬运权重。

**数值感受**：70B 模型 bf16 权重 140 GB，8 卡 TP 后每卡 17.5 GB，H100 读一遍需 `17.5 / 3.35e3` ≈ **5.2 ms**。这就是单步 decode 的理论下界，对应约 190 step/s；不管 batch 是 1 还是 256，这个时间几乎不变——**batching 是纯赚**。

**推导：为什么 prefill 计算量是 `2·P·T`**

对一个形状 `(dᵢₙ, dₒᵤₜ)` 的权重矩阵 `W`（参数量 `p = dᵢₙ·dₒᵤₜ`），prefill 把长度 T 的整段序列打包成矩阵 `X`（`T × dᵢₙ`）做一次 GEMM `Y = X·W`：`Y` 每个元素是长度 `dᵢₙ` 的内积，`2·dᵢₙ` FLOPs；`Y` 共 `T·dₒᵤₜ` 个元素，所以这一次矩阵乘法总 FLOPs = `2·dᵢₙ·dₒᵤₜ·T = 2·p·T`。模型的矩阵乘法层（QKVO 投影 + MLP 两个投影）参数量之和就是总参数量 `P`（其余 LayerNorm/bias 等是低阶项，忽略），把各矩阵的 `2·pᵢ·T` 加起来：

```
Σᵢ (2·pᵢ·T) = 2·T·Σᵢ pᵢ = 2·P·T
```

直觉：P 个参数、每个都要在 T 个 token 位置上各做一次乘加，天然是 `P×T` 量级——decode 阶段单步的 `2·P·B` 是同一个公式，只是把"T 个 token 一次处理"换成了"B 条序列各自 1 个新 token"。也正因为 prefill 把 T 个 token 打包成了矩阵而不是逐个的向量，GEMM 能喂满 Tensor Core（算术强度高），这是 prefill compute-bound、decode（矩阵-向量乘法，退化到带宽瓶颈）memory-bound 的根源。

完整公式 `2·P·T + 4·L·h·T²`（见 [§7](experiments-and-reference.md#七符号表与常用公式速查)）里的平方项来自 attention score 本身（`QKᵀ` 和 `softmax·V`），不经过权重矩阵：每层 `QKᵀ` 是 `(T,h)×(h,T)`，代价 `2T²h`；`softmax·V` 同理再 `2T²h`；L 层相加得 `4LhT²`。T ≪ h 时可忽略，长上下文（T 接近/超过 h）时才显著，这也是长上下文要上 Context Parallelism（[§3.3](training-parallelism.md#33-context-parallelism长上下文)）的原因之一。

因此：**推理系统的第一性目标不是让单次计算更快，而是把 batch 堆大**。而唯一的障碍是 KV cache 占用的显存。

### 1.2 KV Cache 的显存账

每个 token 的 KV cache 占用：

```
2 (K和V) × L(层数) × n_kv_heads × d_head × dtype_bytes
```

| 模型 | L | n_kv_heads | d_head | bf16 每 token |
|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 128 KiB |
| Llama-3-70B | 80 | 8 | 128 | **320 KiB** |
| Qwen2.5-72B | 80 | 8 | 128 | 320 KiB |

**换算过程**(bf16 每元素 2 字节,代入上面的公式):

```
Llama-3-8B :  2 × 32 × 8  × 128 × 2 = 131072 字节 = 128 KiB
Llama-3-70B:  2 × 80 × 8  × 128 × 2 = 327680 字节 = 320 KiB
              ↑K和V  ↑层数 ↑n_kv_heads×d_head ↑dtype_bytes
```

70B 模型下，一条 128k 上下文的序列 KV 就要 `320 KiB × 131072` ≈ **40 GB**。这解释了三件事：

- 为什么 **GQA**（Grouped Query Attention）成为标配：`n_kv_heads` 从 64 降到 8，KV 直接省 8 倍
- 为什么有 **MLA**（DeepSeek 的低秩 KV 压缩）、**KV 量化**（FP8/INT8）
- 为什么 KV cache 的管理效率直接决定了能开多大 batch

### 1.3 PagedAttention：把 KV cache 当虚拟内存管

**问题**：传统实现给每个请求按 `max_seq_len` 预留一整块**连续**显存。

- **内部碎片**：请求实际只生成 100 token，却占了 4096 token 的坑 → 浪费 97%
- **外部碎片**：不同请求预留块大小不一，释放后剩余显存拼不出足够大的连续区间
- **无法共享**：同一 prompt 的 n 个采样各存一份完全相同的前缀 KV

实测传统方案的有效利用率常常只有 20~40%。

**方案**：借用操作系统分页机制。

- KV cache 物理上切成**固定大小的 block**（默认 16 token/block），统一放进一个全局 block pool
- 每个序列维护一张 **block table**：逻辑块号 → 物理块号
- attention kernel 改写成按 block table 做**间接寻址**，不要求物理连续
- 需要新空间时按块分配，序列结束时按块回收

**整体框架图解**（跟 OS 虚拟内存/页表是同一套思路：序列"看到"的 KV 序列逻辑连续，物理显存可以分散在任何地方，靠 block table 做地址转换）：

```
传统方案（问题）：每个请求独占一段"量身预留"的连续显存
┌────────────────────────────────────────────────┐
│ 请求A: [K0][K1][K2][K3][ ][ ][ ][ ][ ][ ][ ][ ]  │ 预留16格，实际用4个 → 12格空占（内部碎片）
├────────────────────────────────────────────────┤
│ 请求B: [K0][K1][K2][K3][K4][K5][ ][ ][ ][ ][ ][ ]│ 预留16格，实际用6个 → 10格空占
└────────────────────────────────────────────────┘
   这一段专属于对应请求，哪怕空着，也不能借给别的请求用


PagedAttention（方案）：显存切成统一大小的 block，扔进一个全局池子里混着用

  全局 Block Pool（物理显存，谁都能申请/归还）：
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │ ...
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
    ↑给A     空闲  ↑给A    空闲   ↑给B          ↑给B

  请求A 的 block table（逻辑块号→物理块号）：      请求B 的 block table：
    逻辑块0 ──► B0                                  逻辑块0 ──► B2
    逻辑块1 ──► B3                                  逻辑块1 ──► B5
   （只按实际 token 数申请了 2 个 block，                逻辑块2 ──► B8
    用多少申请多少，不再"预留"整段坑位）              （3 个 block，按需动态增长）

  attention kernel 算某个 query token 时：
     query
       │  查自己的 block_table
       ▼
     跳到 B0 → 取出这 16 个 token 的 K/V
     跳到 B3 → 取出这 16 个 token 的 K/V     （物理上 B0、B3 根本不相邻，也不用相邻）
       │
       ▼
     拼成完整 KV 序列，正常做 attention 计算
```

看图理解三件事：① **间接寻址**——kernel 通过 block table 做"逻辑块号→物理块号"转换再取数，跟"虚拟地址→页表→物理地址"同一思路；② **按需增长**——block table 按实际生成的 token 数动态申请 block，不像传统方案一次性按 `max_seq_len` 预留到底；③ **物理块任意分散、任意复用**——正因为所有 block 大小统一，A、B 归还的 block 下一个新请求都能直接拿来用，不挑大小不挑位置，这是碎片率能压到 <4% 的根本原因。下面「块内 slot」「COW」的图解是这张整体框架的细节展开。

**Attention kernel 具体怎么用 block_table 算出一个 token 的输出**：核心是"在线（流式）softmax"——不先把所有 score 算完存好，而是逐 block 处理、边算边更新三个运行量 `m`(running max)、`l`(running 归一化和)、`acc`(running 加权输出)，这是 FlashAttention 的技巧，PagedAttention 只是把"逐块读 K/V"换成了"按 block_table 间接寻址"。

例：`block_size=4`，序列已 cache 10 个 token（位置0~9），现在要为位置10的新 token 算 attention，`block_table=[B5, B2, B9]`（B9 只有前3个slot有效）：

```
Query token @ 位置10，query 向量 q（现场由该 token 过 Wq 投影算出，不查表、不进KV cache）

初始：running max m=-∞，running 归一化和 l=0，running 加权输出 acc=0

┌── 第1步：逻辑块0→物理块B5（位置0~3）──────────────────┐
│ for s in slot0..3:                                     │
│   K_s = key_cache[B5][s]      ← 从B5这块物理显存取       │
│   score = dot(q, K_s)·scale                             │
│   m_new = max(m, score)                                 │
│   l   = l·exp(m-m_new) + exp(score-m_new)               │
│   acc = acc·exp(m-m_new) + exp(score-m_new)·V_s          │
│   m   = m_new                                           │
└────────────────────────────────────────────────────────┘
                    ▼ m/l/acc 已"吸收"位置0-3的信息，不必保留原始score
┌── 第2步：逻辑块1→物理块B2（位置4~7）── 同样更新，在上一步基础上累加 ─┐
└────────────────────────────────────────────────────────┘
                    ▼
┌── 第3步：逻辑块2→物理块B9（位置8~10，只处理3个有效slot）── 同样更新 ─┐
└────────────────────────────────────────────────────────┘
                    ▼
            最终输出 = acc / l   ← 这个 token 在这个 head 上的 attention 输出
```

几个关键点：① query 向量是当前 token 现场投影算出来的，K/V 才是查表取的；② 用 `m,l,acc` 三个常数大小的量做在线更新，显存开销跟序列长度无关，长上下文下这一点很关键；③ block_table 只改变"去哪取 K/V 这一步"，算法本身不变——这也是"标准 attention kernel 不认识 block id+slot 这种间接寻址，PagedAttention 必须专门写一版 kernel"的原因；④ 多头场景对每个 query head 各自重复这套累加（GQA 下多个 query head 共享同一份 K/V），所有 head 输出拼起来再过输出投影 `O`（TP 场景下 row-parallel + all-reduce，见 [§3.1](training-parallelism.md#31-张量并行让两次矩阵乘只通信一次)）；⑤ prefill 时 T 个 query token 都要各算一遍，且因果 mask 下每个 query token i 只扫 `block_table` 里覆盖位置 ≤ i 的 block，落在同一 block 内的部分需要块内再做一次因果 mask。

**在线 softmax（running average）的数学原理**：上面 `m,l,acc` 的更新公式不是近似（不是 EMA 那种老数据权重递减的近似统计量），而是标准 softmax 加权平均的精确流式写法。

*减 max 是为了数值稳定性*：softmax 加权平均本是 `Σᵢexp(scoreᵢ)·Vᵢ / Σᵢexp(scoreᵢ)`，但 `scoreᵢ` 一大 `exp` 就溢出。分子分母同乘 `exp(-M)`（`M` 是全局最大值）结果不变，但指数项都 ≤1（`exp(0)=1` 最大），不会溢出——这是标准 log-sum-exp 技巧，代价是要先扫一遍求出 `M`。

*怎么边处理边换基准*：只用一条恒等式 `exp(a-c) = exp(a-b)·exp(b-c)`——减去哪个基准值可以随时换，只要把已经按旧基准算好的量乘上"旧基准→新基准"的修正系数。设处理完前 k 个 score 后维护 `mₖ=max(score₁..scoreₖ)`、`lₖ=Σᵢexp(scoreᵢ-mₖ)`、`accₖ=Σᵢexp(scoreᵢ-mₖ)·Vᵢ`（三者绑定同一基准 `mₖ`）。来了新的 `scoreₖ₊₁`，新基准 `mₖ₊₁=max(mₖ,scoreₖ₊₁)`，旧的 `lₖ` 换算到新基准：

```
Σᵢ₌₁ᵏ exp(scoreᵢ-mₖ₊₁) = Σᵢ₌₁ᵏ exp(scoreᵢ-mₖ)·exp(mₖ-mₖ₊₁) = lₖ·exp(mₖ-mₖ₊₁)
```

加上新元素按新基准的贡献 `exp(scoreₖ₊₁-mₖ₊₁)`，就得到跟上一节完全对应的更新式：

```
lₖ₊₁   = lₖ  ·exp(mₖ-mₖ₊₁) + exp(scoreₖ₊₁-mₖ₊₁)
accₖ₊₁ = accₖ·exp(mₖ-mₖ₊₁) + exp(scoreₖ₊₁-mₖ₊₁)·Vₖ₊₁
```

修正系数 `exp(mₖ-mₖ₊₁)` 因为 `mₖ₊₁≥mₖ` 永远落在 `(0,1]`，只会缩小旧值，不会指数爆炸。处理到最后一个元素时 `mₙ` 恰好就是全局最大值 `M`，所以 `accₙ/lₙ` 跟"先扫一遍求 M、再扫一遍算 softmax"的两遍版本结果逐位相等——是精确算法，不是近似。

*可合并、可并行*：两组各自算好的 `(m₁,l₁,acc₁)`、`(m₂,l₂,acc₂)` 可以直接合并，不用回到原始数据重算：

```
m   = max(m₁,m₂)
l   = l₁·exp(m₁-m) + l₂·exp(m₂-m)
acc = acc₁·exp(m₁-m) + acc₂·exp(m₂-m)
```

这个合并操作满足结合律和交换律，所以 block 可以并行各自算完再两两归并（树形 reduction），结果和"一次性算全部 softmax"完全一致——这是 FlashAttention / vLLM 能把 KV 切块并行处理的数学基础；前面"逐 block 顺序更新"只是这个通用合并公式在 `l₂,acc₂` 退化成单个新元素时的特例。

**什么是"池化管理"**：一句话结论——把资源预先切成规格统一的小份、丢进一个共享的"池子"里，需要时按份取用、用完按份归还，而不是按需临时分配一整块"量身定制"大小的资源、用完再销毁。核心三要素：

1. **规格化**：所有单元大小完全一致（这里是"一个 block = 16 个 token 的 K/V"）
2. **集中存放**：所有单元放进同一个池子（全局空闲块列表），不归属某个请求"私有"
3. **借还而非造销**：谁需要就从池子里取一份，用完还回池子供下一个使用者复用——单元本身不被创建/销毁，只是反复"借出/收回"

**为什么必须"固定大小"才能池化**：如果单元大小可以各不相同（像传统实现里每个请求按自己的 `max_seq_len` 预留一块），释放后剩下的空闲空间形状/大小参差不齐，很可能凑不出下一个请求需要的尺寸——这就是上面说的"外部碎片"。只有所有单元**大小整齐划一**，任何一个空闲单元才能满足**任意**使用者的需求，池子里的资源才能真正"万能互换、随取随用"。这也是操作系统虚拟内存用固定大小"页"而不是任意大小"段"的同一个设计思路。放到 PagedAttention 里：序列需要新空间就从 block pool 领一个空闲 block 登记进自己的 block table；序列结束就把 block 还回空闲列表给下一个序列用；因为每个 block 规格相同、彼此可互换，池子里的空闲 block 数几乎就等于真正闲置的显存量，不会出现"有空闲显存但形状不匹配用不上"的外部碎片。更泛化地看，这和数据库连接池、线程池、内存分配器的对象池是同一套模式：统一规格 + 共享池 + 借还复用，换取不必为每次请求单独创建/销毁资源的开销，以及资源可在不同使用者间灵活复用的灵活性。

**收益**：

1. **碎片率 <4%**（内部碎片最多一个 block，外部碎片消失），等显存下 batch 大数倍
2. **Copy-on-Write 共享**：parallel sampling（`n>1`）、beam search 的公共前缀只存一份物理块，引用计数管理；只有当某个分支要写入时才复制该块

   **block 的物理结构**：每层 transformer 各自维护**独立的一整套**`key_cache`/`value_cache` 张量，
   形状 `[num_blocks, block_size, n_kv_heads, d_head]`——第0维（`num_blocks`）是"块级"下标，
   第1维（`block_size`）才是前面说的"块内slot"，两者是不同粒度的索引，别混。

   "block id=7"指的是**第0维**上下标为7的那一整块预留区域（形状`[block_size, n_kv_heads, d_head]`，
   能装16个token，不是单个数字）。关键是**每一层的张量都各自独立**：
   ```
   layer0.key_cache[7]  ──► 物理显存区域 A（独立一块）
   layer1.key_cache[7]  ──► 物理显存区域 B（跟A完全不同的另一块显存）
        ...
   layer31.key_cache[7] ──► 物理显存区域 ??（第32块）
   ```
   同一个编号7，在32层里各自指向一块不同的物理显存，但**编号统一、生命周期同步**——block manager
   只维护**一份**按id记的`ref_count`表，一次COW/释放的判断，同步作用到全部32层，不会出现某层已私有化、
   另一层还共享的数据错乱。如果每层各自编号，就要维护32份账本，复杂度乘以32倍。按1.2节数字，
   Llama-3-8B 每 token 128 KiB、`block_size=16` → 一个 block 占 **2 MiB**（这2MiB分摊在32层各自的
   key/value 张量里，不是一块连续显存）。

   **为什么 ref_count 只记一份、COW/释放判断必须同步到全部层（详细展开）**：关键约束在于
   `block_table` 每个逻辑位置**只存一个 id**，这一个 id 被同时拿去索引全部 32 层的张量
   （`layer0.key_cache[7]`、`layer1.key_cache[7]`……`layer31.key_cache[7]`）。也就是说同一个序列
   的同一段 token，在 32 层里必然用同一个 id——`block_table` 的设计根本不允许"某层用 id=7、
   另一层用别的 id"这种情况存在。既然"这个 id 现在被几个序列共享"这件事对 32 层来说永远是
   同一个答案，`ref_count` 按层分别记 32 份就是纯冗余（32 份数字永远相等）。

   反证一下不同步会怎样：假设 32 层各自独立判断要不要 COW——layer0 判断 `ref_count[7]=2>1`
   触发 COW，分配新块22，seq0 这一层改用 block22；但 layer1 因为记账不同步，判断还显示1，
   没有触发 COW，继续写共享块7。这样 seq0 在 layer0 的这个 token 数据存在 block22（私有），
   在 layer1 却还是写进了共享块7——而 `seq0.block_table` 这个位置只能填一个数字，填22还是7都
   会丢失另一层的数据，或者把 seq1 在 layer1 的数据意外覆盖污染。这就是"数据错乱"的具体样子。

   所以不是"选择"把判断同步到全层，而是 `block_table` 单数字的设计本身就强制"共享/私有"状态
   必须对全部 32 层保持一致：COW 触发条件（`ref_count>1?`）只判断一次，判断结果同时应用到
   全部 32 层的物理拷贝——即"1 次逻辑判断 → 64 次(32层×K/V)实际数据搬运"，前面"对每层 layer
   复制"那段讲的正是这 64 次搬运。真按层分别记 32 份账本，为了不出现上面的分裂状态，写入时
   还是要把 32 份强行同步更新，等于多做 32 倍记账却什么问题都没多解决。

   **块内 slot 是什么、为什么需要它**：

   - *背景*：block 把 token 按固定数量（`block_size`）分组管理，分配/回收/COW 都以 block 为最小单位。
     但一个 block 塞了 16 个 token，必须知道某个具体 token 存在这 16 个位置里的第几个——这就是
     "块内 slot"。
   - *怎么做*：一个 block 在物理存储上是一段连续区域 `[slot0][slot1]...[slot15]`，每个 slot 能装一个
     token 完整的 K（或 V）向量。给定 token 在序列里的**全局位置**，两次整数运算即可定位：
     ```
     逻辑块号  = 全局位置 // block_size   (决定去 block_table 里查第几个物理块)
     块内 slot = 全局位置 %  block_size   (决定在该物理块的 16 格里存第几格)
     ```
     例：`block_size=16`，prompt 第8个 token（全局位置7）→ 逻辑块号`7//16=0`、slot`7%16=7`（对应前面
     例子里的"slot0~7"）；decode 第1个新 token（全局位置8）→ 逻辑块号`8//16=0`、slot`8%16=8`（对应
     "写入 slot8"）；生成到全局位置17 → 逻辑块号`17//16=1`，翻到下一个逻辑块，若无对应物理块则新分配。
   - *为什么这样设计*：① O(1) 直接定位，除法+取余立刻算出物理块+格子，不用查找/搜索，跟操作系统
     "虚拟地址→页号+页内偏移"是同一思路；② 固定大小才能让 block 池化管理、任意物理块可分给任意序列的
     任意逻辑块位置，避免变长分配带来的碎片问题；③ block 内部 16 个 slot 物理连续，attention kernel
     可以用简单地址算术（`base + slot × 每slot字节数`）批量取数，这也是 PagedAttention 需要专门写
     一版 kernel 的原因——标准 attention kernel 假设 K/V 整体连续，不认识"块id+slot"这种间接寻址。

   **block table**：每条序列自己维护数组 `block_table[逻辑块号] = 物理块id`。attention kernel
   按 block_table 依次去每层的 `key_cache[物理块id][slot]` 取数，物理块不要求显存连续。

   **图解**（token → 逻辑块/slot → block table → 物理块 → COW 全过程）：

   ```
   图1: token 怎么落到"逻辑块 + 块内slot"

   全局位置:   0    1    2    3    4    5    6    7    8    9  ...  16   17
   token:      介   绍   一   下   红   烧   肉   的   锅  (新生成) ... (下一块第0个)
               └──────────────── 逻辑块 0 ────────────────┘   └─ 逻辑块 1 ─┘
   块内slot:   0    1    2    3    4    5    6    7    8   ...        0    1
                                                     ↑
                                         decode 第1步新 token 落在这里

   图2: block table——逻辑块号 → 物理块 id

   seq0.block_table:
     逻辑块0  ──►  物理块 id = 7
     逻辑块1  ──►  物理块 id = 50  (生成到第17个token才会分配)

   图3: 一个物理块内部（某一层的 key_cache），block_size=16

   物理块 id=7
   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬─────┬─────┬─────┐
   │slot0│slot1│slot2│slot3│slot4│slot5│slot6│slot7│slot8│slot9│ ... │slot15│
   │ 介 │ 绍 │ 一 │ 下 │ 红 │ 烧 │ 肉 │ 的 │ 空 │ 空 │     │ 空  │
   └────┴────┴────┴────┴────┴────┴────┴────┴────┴─────┴─────┴─────┘
   每个 slot 存 1 个 token 的 K 向量(n_kv_heads×d_head 个数)；V 另存一份同结构。
   "物理块 id=7"这个编号在 32 层模型里各自都有一份存储（layer0~31 的 key/value_cache[7]），
   32 份共用同一个 ref_count。

   **一个 slot 里存的是这个 token 在全部 kv-head 上的数据，不是单个 head**：形状里最内两维
   `n_kv_heads×d_head` 合在一起就是"一个 slot"的内容，`n_kv_heads` 个 head 的 K 向量紧挨着
   存放（V 向量在 value_cache 里同结构存放）。以 Llama-3-8B（`n_kv_heads=8`）为例，一个 slot
   （某层 key_cache，某 block 第 s 个位置，对应某个 token）存的 K 向量，按内存先后顺序摊开：

   `[head0 的 d_head 个数][head1 的 d_head 个数][head2 的 d_head 个数][head3 的 d_head 个数][head4 的 d_head 个数][head5 的 d_head 个数][head6 的 d_head 个数][head7 的 d_head 个数]`

   8 段首尾紧挨着存放，合计 `n_kv_heads×d_head` 个数字就是这一个 slot 的全部内容。

   这样设计是因为 attention kernel 处理一个 token 时要一次性取到它在**所有 head** 上的 K/V，
   紧挨着存放可以一次连续内存读取取全，不用为每个 head 单独跳址——跟"block 内 16 个 slot 物理
   连续、可批量地址算术取数"是同一个优化思路，只是粒度从"token 之间"下沉到"head 之间"。

   注意这里是 `n_kv_heads` 不是 `n_heads`：GQA 下 `n_kv_heads < n_heads`（Llama-3-8B 是
   `n_heads=32、n_kv_heads=8`，4 个 query head 共享同一组 kv-head），slot 里实际只存 8 份，
   MHA（无 GQA）才是 `n_kv_heads=n_heads`、每个 query head 各自独立一份。代入数字校验：
   一个 slot 的 K = `8×128×2字节=2KiB`，V 同样 2KiB，单层单 token 共 4KiB，乘 32 层 = 128KiB，
   正好对上 [1.2](#12-kv-cache-的显存账) 表格里 Llama-3-8B 每 token 128 KiB 这个数字。

   图4: COW 全过程(3 条序列并行采样)

   【Prefill 后】3 条序列共享同一个物理块
     seq0.block_table=[7] ─┐
     seq1.block_table=[7] ─┼──►  物理块7 [介绍一下红烧肉的做法|空 空 空...]   ref_count=3
     seq2.block_table=[7] ─┘

   【Decode: seq0 写"锅"进 slot8】ref_count=3>1 → 触发COW
                                        ┌──► 物理块22(新) [...|锅|空...]  ref_count=1
     seq0.block_table  [7] ──改为──►[22]┘
     物理块7.ref_count:  3 ──减1──► 2

   【Decode: seq1 写"先"进 slot8】ref_count=2>1 → 再触发COW
                                        ┌──► 物理块23(新) [...|先|空...]  ref_count=1
     seq1.block_table  [7] ──改为──►[23]┘
     物理块7.ref_count:  2 ──减1──► 1

   【Decode: seq2 写"准"进 slot8】ref_count=1 → 不用拷贝!直接原地写
     seq2.block_table  [7] ──保持──►[7]   物理块7 [...|准|空...]  ref_count=1(独占)

   【最终】seq0──►私有块22，seq1──►私有块23，seq2──►私有块7(原地复用，零拷贝)
   → 3 个共享者，只发生了 2 次真正的显存拷贝
   ```

   **具体例子**（`block_size=16`，prompt "介绍一下红烧肉的做法" 8 token 下标0~7 全落在逻辑块0，
   `n=3` 并行采样）：

   - Prefill：物理块 `id=7` 存 8 个 token（slot0~7），3 条序列 `block_table` 都是 `[7]`，`ref_count(7)=3`
   - Decode 第1步（新 token 写入全局下标8，块内 slot8，仍在逻辑块0）：
     - `seq0` 调度：`block7.ref_count=3>1` → 触发 COW：分配 `block22`，拷贝 slot0~7，`block7.ref_count`→2，
       `seq0.block_table`改`[22]`，"锅"写进 block22 的 slot8
     - `seq1` 调度：`block_table`仍是`[7]`，查到`ref_count=2>1` → 再触发 COW：分配 `block23`，拷贝，
       `ref_count`→1，`seq1.block_table`改`[23]`，"先"写进 block23 的 slot8
     - `seq2` 调度：`block_table`仍是`[7]`，查到`ref_count=1`（只剩自己）→ **不拷贝**，直接把"准"写进
       block7 的 slot8，block7 变成它的私有块

   **三个易漏的点**：① COW 只在"该序列第一次要往共享块写入"时触发一次，拿到私有块后续写入不再检查；
   ② 谁先被调度谁先拷贝，最后一个免拷贝——所以 3 个共享者只拷贝了 2 次（= 共享者数 − 1），不是 3 次；
   ③ 写到全局下标16（翻到新逻辑块1的 slot0）时是"分配全新物理块"，不是 COW——新块 `ref_count` 从 1
   开始，天然没有共享，不涉及拷贝。

   **算法本体**（写入一个新 token 时触发，判断条件只有一个：目标物理块 `ref_count>1`？拷贝后再写 :
   直接写）：

   ```
   写入(序列 seq, 新 token 的 K/V):
       pos = seq.已生成token数
       L = pos // block_size        # 逻辑块号
       s = pos %  block_size        # 块内slot

       如果 L 超出 seq.block_table 长度:
           new_id = 分配器.取空闲块(); ref_count[new_id] = 1   # 全新块，天生不共享
           seq.block_table.append(new_id); 目标 = new_id
       否则:
           目标 = seq.block_table[L]
           如果 ref_count[目标] > 1:                            # 触发COW
               new_id = 分配器.取空闲块()
               对每层layer: 复制 key_cache/value_cache[目标] → [new_id]
               ref_count[目标] -= 1; ref_count[new_id] = 1
               seq.block_table[L] = new_id; 目标 = new_id
           # 否则 ref_count==1，已私有，直接写

       对每层layer: key_cache[目标][s], value_cache[目标][s] = 该层的 K/V
       seq.已生成token数 += 1
   ```

   配套两个算法：**fork**（parallel sampling/beam search 产生共享时，`child.block_table` 浅拷贝
   `parent.block_table`，共享块 `ref_count` 逐个 `+1`）；**free**（序列结束/被抢占时，`block_table`
   里每个块 `ref_count -1`，减到0才真正还回空闲池）。

   **"对每层layer复制"具体在做什么**：每层 transformer 的 key_cache/value_cache 是独立显存，layer0
   的 block7 和 layer1 的 block7 是内容完全不同的两份数据（各自权重算出来的），拷贝 layer0 的对
   layer1 毫无帮助。所以"拷贝一个block"实际拆成 `32层 × 2(K,V) = 64次`独立小拷贝：
   ```
   layer0:  key_cache[7]──copy──►key_cache[22] (32KiB)   value_cache[7]──copy──►value_cache[22] (32KiB)
   layer1:  同上...
      ...
   layer31: 同上...
   合计: 64次 × 32KiB = 2MiB
   ```
   每次小拷贝搬 `block_size×n_kv_heads×d_head×dtype_bytes`=`16×8×128×2`=32KiB（Llama-3-8B参数）；
   `64×32KiB=2MiB`正好等于前面算出的"一个block总大小"，可以互相校验。之所以必须逐层循环而不能一次
   搬完，是因为32层的数据物理上分布在32块互不相通的独立显存里，拷贝操作只能对"同一块连续显存"生效
   ——这是数据分布决定的，不是实现偷懒（真实引擎可能把32次搬运合并成一次kernel调用以减少启动开销，
   但要搬的"32份不同数据"这件事本身不变）。

   **可证明的性质**：k 条序列共享同一物理块、同轮都要追加 token 时，按调度顺序处理，每次拷贝让
   `ref_count` 严格减1，减到1就不再拷贝——所以 **k 个共享者恰好触发 `k-1` 次拷贝**，这是算法结构的
   必然结果，不是经验观察（`k=3` 时就是前面例子里的"2次拷贝"）。
3. **Prefix Caching**：system prompt、few-shot 示例、多轮对话历史的 KV 块可跨请求复用，直接跳过重复 prefill。vLLM V1 中默认开启，采用块级哈希索引（机制详见 [§1.3b](#13b-prefix-caching跨请求复用-kv-block-的机制)）

### 1.3b Prefix Caching：跨请求复用 KV block 的机制

以下基于本机 vLLM 开发版源码核实（`vllm/v1/core/kv_cache_utils.py` / `block_pool.py`），不是转述文档。

**一句话**：不同请求共享**完全相同的 token 前缀**时，直接复用已经算好的那部分 KV block，跳过对这段前缀重新做 prefill——省的是**计算**（避免重复的 compute-bound 前向），不是省显存。典型场景：固定 system prompt、多轮对话历史、few-shot 示例、Agent 框架里反复复用的工具说明——命中越长的共享前缀省得越多，这也是 [§6.2](rl-training-workflow.md#62-rollout-长尾) 长轨迹场景里 prefix caching 收益最大的原因。

**链式哈希**（`hash_block_tokens`）：给每个"满"的 block 算一个哈希，链式纳入上一个 block 的哈希：

```
block_hash = hash(parent_block_hash, 本 block 的 token_ids, extra_keys)
```

"链式"是关键——不是只 hash 本 block 自己的 token，而是把上一个 block 的 hash 也纳进来，保证 hash 值反映"从序列开头到这个 block 为止的完整前缀"，否则两个前缀不同、但某一段 token 碎片恰好相同的请求会被错误判定为可共享，拿到语义不对的 KV。`extra_keys`（`generate_block_hash_extra_keys`）还会纳入 LoRA adapter id、多模态输入哈希、prompt embeddings 哈希，以及只在第一个 block 加入的 `cache_salt`（多租户场景按请求加盐，防止跨租户猜前缀探测/命中别人的 cache）——因为同样的 token id 换一个 LoRA 或换一段图片输入，算出来的 K/V 数值是不同的，不能被当成同一份可复用 cache。

**命中判定**：新请求到达时，从第一个 block 开始逐个查全局哈希表（`BlockPool.get_cached_block`），一旦某个 block 未命中就停止——前面连续命中的部分直接复用（该 block `ref_count+1`，跟 [§1.3 COW](#13-pagedattention把-kv-cache-当虚拟内存管) 是同一套引用计数机制），真正需要跑 prefill 的只是从第一个未命中位置往后的部分。只有攒够 `block_size`（默认16）个 token 的"满" block 才会被登记进哈希表，正在增长的最后一个未满 block 不参与判定——命中粒度因此被截断到 block 边界。

**驱逐**：请求结束后 block 不立刻释放，进入 `free_block_queue` 排队；只要没被物理复用，cache 对后续任意请求依然有效。驱逐顺序是 **LRU**（最久未访问的 block 最先回收），同优先级下优先驱逐"链尾"（更长、更具体的那段前缀），保留更靠前、更通用的公共前缀，最大化未来命中概率。

**跟 COW 的区别**：COW 共享的是**同一时刻并发存活**的多个请求（parallel sampling/beam search 的公共前缀）；prefix caching 靠内容哈希索引，共享范围是**任意时间、任意请求**——产生该 cache 的请求早已结束也不影响后续任意新请求命中它，只要没被驱逐。

**收益与限制**：省的是 prefill 算力，不减少 decode 阶段的显存需求；要求计算确定性（相同 token+相同 LoRA/多模态等条件→相同 KV）；哈希算法默认 `sha256`（比 `xxhash` 慢但被认为能更好避免哈希碰撞带来的跨请求误命中/信息泄露风险）；vLLM V1 默认开启（`enable_prefix_caching=True`）。

### 1.4 Continuous Batching（iteration-level scheduling）

**静态 batching 的问题**：把 N 条请求打包成固定 batch，等这一批**全部**生成完才能整批返回、才能接纳下一批新请求。只要批里混入一条输出长度远超其他请求的"长尾"，其余请求生成完之后也必须**干等**它，占着 GPU slot 空转到底——这不是"效率低"，而是有请求提前完成但硬件资源不能被回收利用。这在生成长度方差大的场景（Agent 多轮交互、RL rollout，见 [6.2](rl-training-workflow.md#62-rollout-长尾)）里尤其致命，因为长短请求混合几乎是常态。

**数值感受**：4 条请求一批，1 条要生成 400 token，3 条只要 16 token。静态 batching 下，这一批要跑满 400 step 才能整批返回，3 条短请求各有 `400-16=384` 步在空占 GPU slot——这批的"有效工作"只占 `(400+16×3)/(400×4)` ≈ 30%。如果连续来几批、每批都恰好摊上一条长请求（实验 04（本地资料，未随仓库发布） 构造的正是这种"每个 chunk 必混一条长请求"的最坏情况），静态方案的总耗时会趋近于"块数 × 单条长请求耗时"，短请求完全被拖累。

**做法**：调度粒度从「一批请求」降到「一个 decode step」，**取消批次边界**这个概念。vLLM 的调度器（`Scheduler`）维护两个队列：

- **`waiting`**：已提交但还没开始处理的请求（可能还没做 prefill，或 prefill 做了一部分——见 [1.5 chunked prefill](#15-chunked-prefill)）
- **`running`**：当前正在参与 decode 的请求

每个 step 的调度循环大致是：

1. 检查 `running` 里有没有序列生成了 EOS 或到达长度上限 → **立刻从 running 移除**，释放它占用的 KV 块（[1.3](#13-pagedattention把-kv-cache-当虚拟内存管) 的按块回收）
2. 只要 KV cache 还有空闲块、且没超过 `max_num_seqs` / `max_num_batched_tokens`（token budget），就从 `waiting` 队头**立刻拉新请求补位**——先给它排一次（chunked）prefill，产出第一个 token 后自然并入下一个 decode step
3. 把这一 step 里 `running` 中所有序列的下一个 token 一次性组 batch，喂给模型做前向

**具体例子**：设 `max_num_seqs=3`，4 个请求 A（生成2个token）、B（生成5个token）、C（生成3个token）、D（较晚提交，生成2个token）：

| step | slot1 | slot2 | slot3 | 这一步调度器做的事（对应上面 1/2/3 步） |
|---|---|---|---|---|
| 1 | A 生成 tok1 | B 生成 tok1 | C 生成 tok1 | running 为空，从 waiting 拉 A/B/C 补满3个slot，一起做 prefill+生成第1个token |
| 2 | A 生成 tok2（=EOS） | B 生成 tok2 | C 生成 tok2 | 3条都还在running，继续生成 |
| 3 | **D 生成 tok1**（新补位） | B 生成 tok3 | C 生成 tok3（=EOS） | 先把上一步产生EOS的A**立刻移除**、释放KV block；发现waiting里有D且有空slot→**立刻拉D进来**排它的prefill，跟B/C的decode混进同一次前向 |
| 4 | D 生成 tok2（=EOS） | B 生成 tok4 | （空） | 先移除上一步EOS的C；waiting已空，没人可补，slot3只能空着——这是waiting队列本身空了导致的空闲，不是调度策略的问题 |
| 5 | （空） | B 生成 tok5（=EOS） | （空） | 先移除D；waiting仍空，只剩B一条在跑 |

D 在 **step4** 就已完成。对比静态 batching（A/B/C 打包一批提交，必须等全部完成才能整批返回、才能处理D）：

```
step1~5：A/B/C 一起跑满5步（batch长度由最慢的B决定，B要5个token）
          A 在 step2 已产出 EOS，但因为是"一批"一起过 kernel，仍占着 slot1 陪跑到 step5（占着GPU slot空转）
          C 在 step3 也已结束，同样陪跑到 step5
step6：批次整体返回后，D 才开始 prefill + 生成 tok1
step7：D 生成 tok2（=EOS），才真正完成
```

同样4个请求，静态方案里 D 要到 **step7** 才完成，比 continuous batching 晚3步——晚的这3步，正是 A、C 提前结束后"陪跑"占用的、本可以用来跑 D 的那部分 GPU 时间。

**为什么这样调度能让 GPU 不空转**：关键在于"检查完成→释放→补位"是以**一个 decode step 为粒度**做的（上表每一行都重新判断一次），而不是"以一整批为粒度"。谁先结束，谁的 slot 下一个 step 就能被别人接手，不用等同批其他成员也结束。结合 [§1.1](#一vllm-的核心机制) 的结论——decode 是显存带宽瓶颈，一次前向要把全部权重从 HBM 读一遍，这个时间几乎跟 batch 里塞了几条序列无关（"batching 是纯赚"），也就是说不管这一步塞3条还是1条，付出的 GPU 时间几乎一样。continuous batching 做的事，就是让每一次"反正都要付出"的前向尽量捎带处理更多条有效请求的 token，不让这份开销因为同批里有请求陪跑、或队列一时排不满而被浪费。静态 batching 的问题不是算得慢，而是在批次边界引入了**同步屏障**——必须等最慢的成员完成，提前结束的成员这段时间对系统整体贡献不了新产出；continuous batching 把资源回收/分配下沉到单个 step，取消了这个屏障，物理上不产生新计算，只是不让 GPU 在能干活时空转或做无效功。

因为空位是**逐条、逐 step**回收和填充的，不存在"等全批做完"这个同步点：GPU 上随时维持着尽量接近满载的 batch，[1.1](#一vllm-的核心机制) 里"decode 阶段 batch 越大越划算、而且几乎不要额外时间成本"的结论在这里被物理兑现——continuous batching 本质上就是把这份"近乎免费"的 batching 收益,通过调度层尽最大努力压榨出来的机制,不产生新的计算,只是不让 GPU 空转。

**和其他系统的"batching"要区分开**：Triton Inference Server 等推理服务常说的 dynamic batching，是"多等一小段时间，把短时间内到达的请求凑成一批再一起发进模型"——批次边界依然存在，只是形成批次的时机是动态的。vLLM 的 continuous batching 没有这层意义上的"批次"，调度粒度直接是单个 step，请求随时进、随时出。

**显存不足时的抢占（preemption）**：当 `waiting` 里有新请求要进来，但 KV cache 块池已经分不出足够的空闲块时，调度器要从 `running` 里**抢占**掉一部分序列腾地方（通常按到达顺序，先来的请求优先保留，后来的被抢占）。抢占后如何恢复，有两种策略：

- **Swap**：把被抢占序列已占用的 KV 块整块搬到 CPU 内存，等它重新被调度时再搬回显存,续着原来的位置继续 decode——不用重算,但要走 PCIe,且要在 CPU 侧预留对应内存
- **Recompute**：直接释放这些 KV 块（相当于清空），恢复时把"prompt + 该序列已经生成出的所有 token"重新整体当作一次 prefill 跑一遍,再继续 decode——不占额外内存,但要重新算一遍已经生成过的部分

vLLM 默认用 recompute，原因还是回到 [1.1](#一vllm-的核心机制) 的算力/带宽账：decode 阶段本来算力就有富余（算术强度只有 `B` FLOP/byte，远低于 H100 的 ridge point ≈295），重新做一次 prefill 这种 compute-bound 的操作,对本来就"喂不饱"的 Tensor Core 来说代价很小；而 swap 要走 PCIe 带宽（几十 GB/s 量级,远低于 HBM 的 3.35 TB/s),对于长序列(KV 大)反而更贵。只有当序列已生成的部分特别长、重算成本明显高于搬运成本时,swap 才更划算——这也是 vLLM 把两种策略都保留、可配置的原因。

### 1.5 Chunked Prefill

**问题**：一条 8k token 的 prefill 需要几十毫秒，期间所有正在 decode 的请求全部卡住，造成 TBT（token 间延迟）尖刺。这是「吞吐 vs 延迟」的经典冲突。

**做法**：把长 prefill 切成固定大小的 chunk（如 512 token），每个 step 混合调度「一部分 prefill token + 一批 decode token」，用统一的 token budget（`max_num_batched_tokens`）控制每步总量。

- 代价：单条 prefill 的完成时间变长（TTFT 略升）
- 收益：TBT 平稳、算力利用更均衡（prefill 的高算术强度正好补上 decode 的低算术强度）

**常见疑问：如果不 prefill 完，怎么开始 decode？**——答案是它确实不能：chunked prefill 从不会让一个序列在自己 prompt 还没处理完之前就开始生成 token。会混淆的地方在于，"一部分 prefill token + 一批 decode token"混合的是**不同序列之间**，不是同一个序列自己的 prefill 和 decode 混在一起。

设 `max_num_batched_tokens=512`，新请求 D（prompt 1500 token，切3个chunk）prefill 期间，B、C 早就过了自己的 prefill，正在逐 token decode：

```
step1: batch = [ D_chunk1(位置0~511) | B的decode token | C的decode token ]
        D_chunk1：只是在"建KV cache"——这512个token过一遍模型，K/V写进它的block，
        不对这些位置采样，不产生新token（D还没资格生成）
        B、C：各自正常采样出1个新token

step2: batch = [ D_chunk2(位置512~1023) | B的decode token | C的decode token ]
        D_chunk2 的 attention 因果地attend到"0~自己"：chunk1的KV已经在step1写进cache了，
        chunk2可以直接跨chunk往前attend（跟"attention kernel按block_table逐块取KV"同一套
        机制，只是query端是一整个chunk而不是1个token）。D这一步依然不采样

step3: batch = [ D_chunk3(位置1024~1499，最后一块) | B的decode token | C的decode token ]
        这是D prompt的最后一块——对最后一个位置(1499)过完模型后走LM head采样，
        产出D自己的第一个输出token！★ 就在这一刻，D 才从"prefilling"转正成"decoding"，
        加入running集合

step4: batch = [ D的decode token(第2个输出token) | B的decode token | C的decode token ]
        现在D也跟B、C一样，每个step只贡献1个token，正常参与decode
```

D 自己要等 3 个 step、把 1500 个 prompt token 全部处理完（KV cache 全部建好），才能在 step3 末尾产出第一个输出 token——generate 第2个 token 需要用第1个输出 token 做新的 query，而第1个输出 token 本身还没产生，所以逻辑上也不可能在自己没 prefill 完时就 decode。step1~3 里真正被"混合"进同一次 GPU 前向的，是 D（还没资格生成的 prefill token）和 B、C（早就能生成的 decode token）——它们只是恰好塞进了同一个 batch matrix，彼此逻辑上完全独立、互不依赖。

这也解释了 chunked prefill 怎么解决前面的 TBT 尖刺：不切 chunk 的话，D 的 1500 个 token 会在一个 step 里一次性处理完，这个巨大 step 跑完之前 B、C 完全没机会插进来跑它们的 decode——被迫等这一整段时间。切 chunk 之后，每个 step 都留出预算给 B、C 的 decode token 一起跑，D 要多花几个 step 完成自己的 prefill（TTFT 略升），但 B、C 的每个 decode step 都能按时执行。

**这不是简单的零和取舍**：容易把它理解成"偷 B、C 的计算时间分给 D 的 chunk换 D 的 TTFT"，但更准确的账要回到 roofline——纯 decode 的 step 是显存带宽瓶颈（算术强度只有 `B` FLOP/byte，远低于 H100 ridge point ≈295），Tensor Core 大部分时间是闲着的，GPU 时间被"读一遍权重"这个固定成本占满。这时候往同一个 step 里塞一段 compute-bound 的 prefill chunk，只要 chunk 的计算量还没把这一步的瓶颈从"读权重"切换到"算矩阵乘"，这部分算力基本是**白捡的**——不会明显拖慢 B、C 这一步。这正是"算力利用更均衡（prefill 的高算术强度正好补上 decode 的低算术强度）"的含义：不是牺牲 B、C 换 D，而是利用 B、C 本来就浪费掉的那部分 Tensor Core 空闲。只有 chunk 切得太大、计算量超过读权重本来要花的时间，瓶颈才真正切换到算力，B、C 的 TBT 才会被明显拖慢——`max_num_batched_tokens` 存在的意义就是把 chunk 大小控制在"基本白捡、不明显拖慢别人"的区间内，同时决定着 TTFT/TBT 之间具体的权衡曲线（chunk 越大→D 的 TTFT 越低但拖慢 B/C 的风险越高，反之越小）。

另需澄清：不是"一个 chunk 穿插一次 decode"这种交替节奏——continuous batching 没有"轮流"的概念，每一个 step 都是把当前所有 running 序列各自的 1 个 decode token，和这次要处理的 prefill chunk，一次性混合打进同一个 batch。

**"一次性混合打进同一个 batch"具体是怎么算的**：模型内部不同层对这件事的处理方式不一样。

- **Embedding、QKV/O 投影、FFN、LayerNorm**：逐 token 独立运算，跟 token 属于哪个序列无关，所以直接把这一 step 所有 token（decode token + chunk token）拼成一个扁平张量 `[总token数, hidden_size]` 一次性矩阵乘。例如 B、C 各1个decode token + D_chunk1的512个token，拼成 `[514, hidden_size]`，过这些层时完全不区分这514行分别来自谁——这里的"514"正好对应"2·P·T"公式里的 T，行数越多（有效batch越大）这几层效率越高。
- **Attention**：唯一的例外，每个 token 只能 attend 自己那条序列的 KV cache（通过自己的 `block_table`），不能跑到别的序列。所以 attention kernel 拿到 `[514,hidden]` 后，靠一份"每个 token 属于哪条序列"的元信息（`cu_seqlens`/`seq_start_loc`）把这514行分段处理：B的那1行只查B的block_table，D的512行只查D的——仍是一次kernel调用处理完整个batch，但内部按序列边界隔离计算，结果不会跨序列串在一起。
- **LM head + 采样（"分别decode"）**：只对"需要产出下一个token"的位置做——decode token这几行（B、C，每个都要）+ 如果这次chunk恰好是某序列prefill的最后一块，则该序列的最后一行（D只有在最后一块时才对第512行采样）。其余prefill chunk内部的行完全不采样，只用来建KV cache、把hidden state往下传，直接跳过词表投影这个开销不小的矩阵乘。

一句话：拼在一起过的是"跟token本身无关的通用计算层"（省算力、堆大有效batch）；attention靠元信息分段隔离（保证不同序列不串数据）；最后采样靠位置筛选（只对该产出token的那几行做）。

### 1.6 Prefill/Decode 分离（Disaggregation）

Chunked prefill 是在同一组 GPU 上做时间片混合，更彻底的做法是**空间分离**：

- **Prefill 池**：compute-bound，适合大 TP、追求算力
- **Decode 池**：memory-bound，适合大 batch、追求显存容量与带宽
- 两者之间通过 RDMA / NVLink / **NIXL** 传输 KV cache

优点是两个池可以独立选择并行度、独立扩缩容、互不干扰尾延迟。代价是引入 KV 传输开销（70B 模型 4k prompt 的 KV 约 1.3 GB），需要高速网络支撑。

### 1.7 其余关键优化

**CUDA Graph / Piecewise Graph**
decode step 的 kernel 数量多、单个执行时间极短（微秒级），CPU 侧的 launch 开销占比很高。把整个 step 录制成 CUDA Graph 一次性提交可消除该开销。难点在于 attention kernel 的形状随序列长度动态变化，无法入图——vLLM V1 采用 **piecewise CUDA graph**，把 attention 挖出来单独 eager 执行，其余部分（各种 GEMM、LayerNorm、激活）入图。

**Custom All-Reduce**
TP 场景下，decode 的 all-reduce 消息大小只有 `B × h × 2` 字节（例如 B=256, h=8192 → 4 MB）。这个尺度下 NCCL ring 算法的**延迟**（多跳同步）而非带宽成为瓶颈。vLLM 实现了基于 NVLink P2P 直写的 one-shot / two-shot all-reduce kernel，小消息下显著快于 NCCL。

**Speculative Decoding**
用小模型 / n-gram / EAGLE 头起草 k 个候选 token，大模型一次前向**并行验证**这 k 个位置，接受率高时一步出多个 token。由于 decode 本来就 memory-bound（算力有富余），多算 k 个 token 的额外时间几乎为零——这是用富余算力直接换延迟。

**V1 引擎架构**
EngineCore 独立进程运行，tokenize / detokenize / HTTP 处理与 GPU 执行循环解耦，避免 Python GIL 和 CPU 预处理阻塞 GPU 关键路径。

### 1.8 vLLM 中的并行

vLLM 支持 TP（层内切分，同训练）和 PP（层间切分）。要点：

- **TP 在 decode 中的通信占比很高**：每层 2 次 all-reduce，虽然消息小，但 decode step 本身只有几毫秒，通信容易占到 30~50%
- 所以推理的 TP 也**必须在 NVLink 域内**
- PP 在推理中主要用于单卡放不下的超大模型，会引入 pipeline bubble，对延迟不友好

**为什么不是训练那种完整的 5D 并行（TP/PP/DP/CP/EP 全上）**：不是 vLLM 不支持，而是训练 5D 里每一维解决的问题在自回归 decode 场景下有的消失了、有的换了个形态：

| 训练里的维度 | 训练中解决什么问题 | decode 阶段还需要吗 | vLLM 的对应形态 |
|---|---|---|---|
| **TP** | 单层矩阵乘太大，单卡放不下/算不动 | **需要**，约束不变：模型太大单卡装不下，且要保证低延迟 | 直接复用，限定在 NVLink 域内，每层 all-reduce |
| **PP** | 层数太多，配合 TP 摊开超大模型 | **部分需要**，用于单实例内放不下的超大模型跨节点扩展 | 支持，但每加一级 pipeline 多一跳延迟，bubble 对小 batch 更伤 |
| **DP** | 复制整个模型，不同卡吃不同 batch 分片，算完要把梯度 all-reduce 平均，因为优化器要用全局梯度更新**同一份**参数 | **"同步"这一步没有了**：推理没有反向传播、没有梯度，也没有"必须用全局信息更新同一份参数"这件事，每个请求互相独立 | 依然存在，但降级成**零通信的复制**：起 N 个独立 replica（即[PSRL 训推专题 §6](rl-training-workflow.md#六rl-后训练场景psrl) 提到的 PSRL `n_rollout_instances`），各吃各的请求，中间不通信，由 Router 做负载均衡——比训练 DP 更简单，因为训练 DP 还要 all-reduce 把大家拉回同一份参数，推理 DP 连这步都不需要 |
| **CP** | 单条序列太长，一次前向+反向放不进单卡，把序列维度切开并行算 | decode 阶段**用不上**：每步只生成 1 个新 token，没有"一条长序列需要在单步内切开并行"这件事，`s_t→s_{t+1}` 严格串行 | prefill 阶段有对应物（chunked prefill / 长 prompt 的序列并行），但那是 prefill 专属，不是常规意义的 CP |
| **EP** | MoE 把不同 expert 分到不同卡，token 路由到对应专家，all-to-all 通信 | **同样需要**，MoE 推理（DeepSeek-V3/Kimi K2 这类）一样要摊开专家 | 支持，且近期趋势是 attention 部分用 DP（各卡各算各的，避免 TP 通信开销）、expert 部分用 EP（分片+all-to-all）的混合模式——这正是推理场景下"多维并行"的新形态，只是维度组合跟训练 5D 不是同一套 |

本质区别：训练 5D 优化的是"一次前向+反向+梯度同步，尽量吃满集群算力做超大 global batch"；推理优化的是"每 token 延迟尽量低、吞吐尽量高，但没有反向传播、没有跨请求的全局同步需求"。CP 依赖"序列可切开、反向时再拼起来"——decode 没有反向、逐 token 串行，没有可切的对象；训练 DP 依赖"最终要把所有卡的梯度合并成同一份参数"——推理没有这个合并需求，直接退化成零通信的 replica 复制反而更高效（embarrassingly parallel）；TP/PP/EP 保留，是因为它们解决的还是"模型太大、单卡装不下/算不动"这同一个约束，训练推理都一样，所以直接复用。

### 1.9 Linear Attention / SSM 层的 cache 管理：什么时候需要多存一份 checkpoint

以下内容基于实际读取本机 vLLM 开发版源码（`0.22.1.dev0`，2026-07-06，位于 `psrl/third_party/vllm`），不是转述文档。

**为什么"存不存"在这里是个真问题**：标准 attention 的 KV cache 必须存每个 token 的 K/V，序列越长 cache 越大，没有选择空间（见 [§1.3](#13-pagedattention把-kv-cache-当虚拟内存管)）。但 Mamba / GDN（Gated DeltaNet）这类线性/门控 recurrence 的"状态"（`conv_state`+`ssm_state`，或线性 attention 的 KV 状态矩阵）是**固定大小**的——把任意长前缀压缩成一个定长向量/矩阵的有损递归。只要手里有"当前最新状态"就能继续生成，中间时刻的状态不存也不影响正确性，只在两种场景下才有价值：① 跨请求复用前缀（prefix caching）；② chunked prefill 跨 scheduler step 断点续算。于是"何时存"在 vLLM 里被做成一个显式策略：`CacheConfig.mamba_cache_mode`（`vllm/config/cache.py:132`），三档：

| 模式 | 触发条件 | 存的时机 | 用途 |
|---|---|---|---|
| **none** | prefix caching **关闭**时的默认值 | 从不额外存 checkpoint，每个请求只 1 个 slot，反复**原地覆盖** | 等价于完全不做 prefix cache |
| **all** | prefix caching **开启**，且模型的递归形式数学上"chunk 可分解"（见下） | 在每个 `block_size` 个 token 的**边界**（位置 `i·block_size`）额外存一份状态快照，写进跟标准 KV block table 同构的表里，用 block hash 索引 | 真正的跨请求 prefix cache 复用——后续请求命中同一前缀 block，直接从这份状态接着算 |
| **align** | prefix caching 开启但模型不支持"all"时自动降级；**强制要求开 chunked prefill** | 只在两类时刻存：① 每个 scheduler step 结束时最后一个 token 的位置；② 正好落在 `block_size` 整数倍位置的 token | 不给跨请求复用用（没进全局 block 池子），只是让 chunked prefill 能正确"断点续算" |

判定逻辑在 `vllm/model_executor/models/config.py:340-390`（`MambaModelConfig.verify_and_update_config`）。

**"chunk 可分解"是谁能上"all"档、谁只能降级到"align"的分界线**：声明支持 `SupportsMambaPrefixCaching` 协议的是纯 Mamba/Mamba2 家族（`mamba.py`/`mamba2.py`/`jamba.py`/`bamba.py`/`zamba2.py`/`nemotron_h.py`/`falcon_h1.py`/`granitemoehybrid.py`）；Qwen3-Next/Kimi/OLMo 用的 GDN、以及最朴素的 `linear_attn.py` backend 都没有声明，只能落到 `"align"`。原因是 Mamba2 的 SSD（structured state-space duality）chunked scan 算法保证：某位置的状态只依赖"到这个位置为止的输入"，跟怎么切 chunk 无关——不管一次性算完整条序列还是先算前 512 再算后面，`position=512` 处的状态永远一样。这个性质保证"提前存某个 block 边界的状态"可以放心当成可复用 checkpoint；GDN/纯线性 attention 目前还没做到这个保证，只能用更保守、不跨请求复用的"align"模式。

**具体写入发生在哪一步**：`mamba_mixer2.py` 里读布尔位 `is_mamba_cache_all = cache_config.mamba_cache_mode == "all"`（`mamba_mixer2.py:689`）。为 `True` 时，调用底层 SSD kernel（`mamba_chunk_scan_combined_varlen`）传 `return_intermediate_states=True`，让 kernel 在做 chunk scan 的同时把每个 block 边界的中间状态一并算出来，再按 `block_idx_first/last_scheduled_token` 逐块写进 `ssm_state`/`conv_state` 对应的 slot；为 `False`（align/none）时，kernel 只算并只写"这次前向结束时的最终状态"，不产生中间 checkpoint——省掉这部分显存和写带宽。这个代价差异也直接体现在 `MambaSpec.max_memory_usage_bytes`（`vllm/v1/kv_cache_interface.py:579`）：`"all"` 按 `max_model_len/block_size` 分配一大堆 block（接近标准 KV cache 的显存量级），`"align"` 只固定 2 份、`"none"` 只固定 1 份（均 + 推测解码所需的额外块）。

一句话：vLLM 把"何时存 SSM/线性 attention 状态"绑定在"要不要支持 prefix caching"这个开关上——不开就完全不存中间态（最省）；开了之后，能不能做到"块粒度、可跨请求复用"取决于模型的递归公式是否数学上 chunk-可分解（目前只有 Mamba2 系列做到），做不到的退化成仅支撑 chunked prefill 正确续算的轻量模式。

---

## Selective batching：投影按 token 合批，attention 按请求隔离

一句话结论：ORCA 将不同请求本轮待处理的 token 拼成一个二维矩阵，统一做 QKV、输出投影和其他逐 token 运算；在 attention 核心计算处按请求拆开，分别访问各自的 KV cache，随后重新合并。

这里的 attention 专指 `softmax(QKᵀ / √d_head + mask)V`，不包含 QKV 投影和输出投影。它选择的是哪些**算子**合批。Iteration-level scheduling 决定每一轮处理哪些请求，selective batching 则负责执行这批长度、阶段各不相同的请求。[ORCA 论文 §3、图 5](https://www.usenix.org/system/files/osdi22-yu.pdf)

### 自建算例：两个 decode 请求与一个 prefill 请求

设隐藏维度为 `d`，用标准 MHA 说明，忽略 bias；上下文长度包含本轮 token。

| 请求 | 阶段 | 本轮待处理 token 数 | Attention 使用的 KV 长度 |
|---|---|---:|---:|
| A | decode | 1 | 128 |
| B | decode | 1 | 2048 |
| C | prefill | 4 | 4 |

本轮输入拼成 `X = concat(X_A, X_B, X_C)`，形状为 `[6, d]`。历史 KV 只供 attention 使用，A、B 的历史 token 无需重新经过 QKV 投影或 MLP。

### 投影：沿 token 维拼接，共用权重矩阵

逐 token 的线性变换满足：

```text
concat(X_A, X_B, X_C) @ W
    = concat(X_A @ W, X_B @ W, X_C @ W)

QKV = X @ W_QKV
[6, d] @ [d, 3d] -> [6, 3d]     # 标准 MHA，Q/K/V 各宽 d
```

矩阵乘不会混合不同行，只变换每个 token 的特征，因此不需要把请求 padding 到相同长度。合成较大的 GEMM 能在多行计算间复用权重，提高硬件利用率。Attention 后的输出投影 `Y = O @ W_O` 同理；MLP、LayerNorm 和逐元素激活也可在 token 拼接后的布局上计算。QKV 融合为一次 GEMM 是常见实现方式，selective batching 的关键是跨请求的 token 合批。

### Attention：保留请求边界，分别访问变长 KV

按本轮 token 数 `[1, 1, 4]` 拆开投影结果，将新 K/V 加入各请求的 cache。对每个 head 分别计算：

```text
O_i = softmax(Q_i @ K_i^T / sqrt(d_head) + mask_i) @ V_i

A 的 score: [1, 128]    -> 输出 [1, d_head]
B 的 score: [1, 2048]   -> 输出 [1, d_head]
C 的 score: [4, 4]     -> 输出 [4, d_head]，使用 causal mask
```

两个乘数都是请求相关的数据，KV 长度又不同，无法像投影那样直接共享一个 `W`。直接把所有请求的 Q/K 拼起来相乘而不隔离边界，会让请求互相看到对方的 token。统一 padding 加 mask 可以保证正确性，但会引入额外工作。

ORCA 在此按请求分别执行 attention。各请求输出在拼接多头后仍为 `[本轮 token 数, d]`，于是能重新合成 `[6, d]`，继续统一做输出投影。

```text
拼接本轮 token [6,d]
  -> 合批 QKV 投影 [6,3d]
  -> 按请求拆开：A / B / C
  -> 各自 attention，访问各自 KV cache
  -> 合并输出 [6,d]
  -> 合批输出投影、MLP
```

### 性能边界与现代实现

Attention 核心没有 QKV/输出投影那样的共享可训练权重，因此跨请求合批没有同样的权重读取摊销收益；但它仍然可能从更少的 kernel launch 和更好的 GPU 并行利用率获益。不能据此说 attention 合批没有价值，也不能说 attention 必须永远串行执行。

变长 attention 实现可以在一次接口调用中处理多个长度不同的请求，利用序列边界保持隔离，例如 FlashAttention 的 varlen 接口。它保留了“计算不能跨请求混合”的数学约束，但不必照搬 ORCA 逐请求调用 attention 的执行方式。[FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)

这是推理执行方式的优化，不会自动缩短上下文或消除长轨迹的训练激活显存。
