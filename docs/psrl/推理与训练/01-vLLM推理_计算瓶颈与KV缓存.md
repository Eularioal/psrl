# vLLM推理：计算瓶颈与KV缓存

vLLM 是大模型生成服务的执行引擎。它一边保存每条请求的历史计算结果，一边把当前能继续计算的请求合到一轮执行中。理解它可以沿一条主线：先处理输入，再逐步生成；缓存避免重算，调度让空出来的位置接收新请求。

本页先解释机制和可手算例子，缓存内部结构的推导另放在[缓存管理详解](#kv-缓存管理详解块表写时复制与前缀复用)。2026-11-18 额外核对 PSRL 网关的 PD 配置和 vLLM 引擎参数，见 1.6 节。历史性能数字保留原条件。

[返回专题索引](README.md)。

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
- [KV 缓存管理详解：块表、写时复制与前缀复用](#kv-缓存管理详解块表写时复制与前缀复用)
  - [块号和槽位怎样定位一个 token](#块号和槽位怎样定位一个-token)
  - [三条分支怎样共享并各自续写](#三条分支怎样共享并各自续写)
  - [每层的数据怎样对应同一块号](#每层的数据怎样对应同一块号)
  - [分页注意力怎样读完所有历史](#分页注意力怎样读完所有历史)
  - [前缀哈希与缓存回收](#前缀哈希与缓存回收)

## 一、vLLM 的核心机制

### 1.1 Prefill 与 Decode 的根本差异

token 是模型处理文字的基本片段。生成一段回答通常有两个阶段：

| 阶段 | 做什么 | 例子 | 常看的指标 |
|---|---|---|---|
| Prefill，输入处理 | 一次处理已有输入，建立历史缓存 | 处理 2,000 个输入 token | 首 token 延迟，即多久开始回答 |
| Decode，逐步生成 | 根据已有内容生成下一个 token，并更新缓存 | 本轮每条请求各生成 1 个 token | 每 token 延迟与总吞吐 |

这里的数字是教学假设。第一轮 prefill 后得到第一个输出，接着 decode，直到结束标记或长度上限。

为什么合批有用？模型每层都要使用同一份权重。把多条请求本轮的 token 一起做矩阵乘，可以在更多计算间分摊读取权重的成本。小批量 decode 常受显存带宽限制；较大输入的 prefill 往往更依赖算力，实际瓶颈还与上下文、并行通信和算子实现有关。

用简化模型估算：若模型有 `P` 个参数，prefill 处理 `T` 个 token，线性层前向计算约为 `2PT` 次浮点运算；decode 一轮有 `B` 条请求，则约为 `2PB`。bf16 权重每参数 2 字节，若只计读取权重，算术强度约为 `B` 次运算/字节。这里暂忽略注意力、KV 读写和通信，只用来解释合批的方向。

原笔记的硬件算例采用 H100 SXM 标称 989 TFLOPS 与 3.35 TB/s，交点约为 `989/3.35≈295` 次运算/字节。这是简化硬件模型的交点，实际最佳批量需根据延迟和吞吐测量。70B 参数的 bf16 权重约 140 GB，8 卡均分后每卡 17.5 GB，仅理想读取时间约 `17.5/3350≈5.2 ms`；完整单步还包含计算与通信。

图中把公式、对象和调度用途连在同一个算例里，按编号阅读。

<!-- figure:roofline-capacity-decision -->
这张图沿同一个场景解释：为什么性能取两个上界的较小值？这个拐点如何帮助判断请求并发？

[![Roofline：从硬件上界走到并发调度（1/4）](../../_static/img/document-figures/roofline-capacity-decision-screen.png)](../../_static/img/document-figures/roofline-capacity-decision.png)

[![Roofline：从硬件上界走到并发调度（2/4）](../../_static/img/document-figures/roofline-capacity-decision-screen-02.png)](../../_static/img/document-figures/roofline-capacity-decision.png)

[![Roofline：从硬件上界走到并发调度（3/4）](../../_static/img/document-figures/roofline-capacity-decision-screen-03.png)](../../_static/img/document-figures/roofline-capacity-decision.png)

[![Roofline：从硬件上界走到并发调度（4/4）](../../_static/img/document-figures/roofline-capacity-decision-screen-04.png)](../../_static/img/document-figures/roofline-capacity-decision.png)

采用正文教学工作量：F=20万亿次、Q=1 TB、C=100万亿次/秒、W=2 TB/秒。计算至少0.2秒、搬运至少0.5秒，算术强度I=20，上界为40万亿次/秒；硬件交点I=50。图中的理想上界并非实测性能，也不直接给出某个模型的最佳请求数。 [查看矢量图](../../_static/img/document-figures/roofline-capacity-decision.svg)。
<!-- /figure:roofline-capacity-decision -->

**Roofline 通常研究什么？** 它用来判断某段 GPU 计算的性能主要受算力还是数据搬运限制，并估计离硬件上界还有多少优化空间。横轴是算术强度 `I=浮点运算次数/显存搬运字节数`，纵轴是实际每秒浮点运算数。设该精度和计算路径的算力上限为 `C`、显存带宽为 `W`，则性能上界为 `min(C, W×I)`。低算术强度时，搬数据决定上界，图上是一条斜线；强度提高到 `I=C/W` 后，算力决定上界，图上变成水平线，形状像屋顶。[NVIDIA Roofline 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)

**为什么相乘，再取最小值？** 设这一段程序需要完成 `F` 次浮点运算，从显存实际读写 `Q` 字节，总耗时为 `t`。计算和数据搬运都必须完成，所以同时满足 `t≥F/C` 与 `t≥Q/W`。即使它们能充分重叠，也至少要等较慢的那一项结束，因此：

```text
t ≥ max(F/C, Q/W)
实际性能 F/t ≤ min(C, F×W/Q) = min(C, W×I)
其中 I = F/Q
```

乘积的单位也吻合：`字节/秒 × 运算次数/字节 = 运算次数/秒`。`W×I` 表示按这段程序的数据复用程度，带宽最多能支撑多快的计算；`C` 表示计算硬件本身最多能算多快。两个约束必须同时满足，所以取较小者。原式描述理想上界，实际性能可以更低。计算与访存若不能充分重叠，还会付出串行等待和其他开销，不能直接用等号预测实际速度。

教学例子：一次工作需要 20 万亿次运算、搬运 1 TB 数据，GPU 算力上限为 100 万亿次/秒、带宽为 2 TB/秒。计算至少需 0.2 秒，搬运至少需 0.5 秒，整段至少需 0.5 秒，所以性能至多为 `20/0.5=40` 万亿次/秒。用 Roofline 算也是 `min(100, 2×20)=40`。这里的 1 TB 是整个执行期间累计搬运量，并非需要同时驻留 1 TB 显存；同一张量反复读写会重复计入。

下面是教学算例，数字不指代本机 H20。假设 GPU 有效算力为 100 TFLOP/s、带宽为 2 TB/s，模型有 30 亿参数、bf16 权重约 6 GB。只考虑一次解码的权重线性层，读取权重至少约 3 ms；一条请求的 6 GFLOP 运算约需 0.06 ms。合批为 32 条后，计算约需 1.92 ms，仍有机会被 3 ms 的读权重成本遮住；合批为 64 条后，计算约需 3.84 ms，理想模型开始受到算力约束。在这个极简例子中，批量变大前每条请求的单步延迟近似不变，吞吐随批量上升；之后单步变慢，吞吐收益减小。

[![经典 Roofline 上界与同一教学模型的合批延迟、吞吐曲线（1/4）](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab-screen.png)](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab.png)

[![经典 Roofline 上界与同一教学模型的合批延迟、吞吐曲线（2/4）](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab-screen-02.png)](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab.png)

[![经典 Roofline 上界与同一教学模型的合批延迟、吞吐曲线（3/4）](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab-screen-03.png)](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab.png)

[![经典 Roofline 上界与同一教学模型的合批延迟、吞吐曲线（4/4）](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab-screen-04.png)](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab.png)

<!-- explained-source:explained-roofline-and-batching-1ddd7ab -->
**读图结论：**min 的性能上限对应 max 的时间下界。真实上限还受 KV、上下文和共运行影响。 数字是教学假设，不是 H20 的固定 50 请求上限，一个请求步不是完成整条回答。 [完整解释图（SVG）](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab.svg) · [原始机制图或实验图](../../_static/img/literature-reviews/roofline-and-batching.png)。
<!-- /explained-source:explained-roofline-and-batching-1ddd7ab -->

[查看可放大的 SVG](../../_static/img/document-figures/explained-roofline-and-batching-1ddd7ab.svg)。左图横轴是每搬运一字节能做多少运算，右图横轴是同一轮 decode 中的请求数。两者在这个简化模型中相连：合批让同一次权重读取服务更多请求，算术强度随批量提高，直到计算时间超过读权重的时间。

沿右图手算一次：B 为 32 时，整轮至少需 `max(3, 0.06 × 32) = 3 ms`，每秒约完成 `32 / 0.003 = 10,667` 个请求步。B 为 64 时，整轮变为 3.84 ms，每秒约完成 16,667 个请求步。单步变慢了，吞吐仍提高。B 再增到 128，整轮需 7.68 ms，理想吞吐仍约 16,667，增加并发只让每条请求的下一 token 更晚返回。这里一个请求步是“一条请求前进一个 token”，不是完成整条回答。

图上的第 50 条请求来自 `3 / 0.06 = 50` 这个教学条件。它说明怎样形成拐点，实际调度上限还要看延迟目标、注意力开销与共运行任务，不能把这个数移植到 H20 或环境命令。对 Env 调度更有用的判断是：继续加任务后，**相同整批工作是否更早结束**，以及模型变慢会不会延迟后续工具命令的就绪。

实际推理由很多内核组成，注意力还会读每条请求各自的历史 KV，模型也有 CPU 调度、内核启动和通信开销。精确分析应选择匹配精度的 Tensor Core 或其他运算上界，并结合实测访存量与算子性能。整模型的“并发数 → token/s、首响应、每 token 延迟”曲线，是 Roofline 启发下的服务性能测量，横轴和单位与经典 Roofline 图不同。

因此要分别找两个位置：**延迟开始超过业务容忍值的并发数**，以及**继续加并发已几乎不能提高吞吐的位置**。两者经常不同。即使还没到吞吐平台，某个依赖模型回复才能执行环境命令的 agent，也可能已经因为响应变慢而拖延后续工作。

### 1.2 KV Cache 的显存账

注意力计算会使用历史 token 的 Key 和 Value 张量，简称 K/V。KV cache 把它们存下来，生成下一个 token 时直接读取。

对标准注意力模型，每 token 的缓存近似为：

```text
2（K 和 V）× 层数 × KV 头数 × 每头维度 × 每元素字节数
```

例如 Llama-3-70B 的教学计算使用 80 层、8 个 KV 头、每头 128 维和 bf16：

```text
2 × 80 × 8 × 128 × 2 = 327680 字节 = 320 KiB/token
320 KiB × 131072 token = 40 GiB
```

这是整模型的原始 KV 量，未计对齐、元数据或并行切分。长请求的缓存可与权重同样占用大量显存，因此并发越大、上下文越长，缓存预算越关键。

GQA（分组查询注意力）让多个查询头共用 K/V，减少 KV 头数；KV 量化用更少字节存一个元素；不同模型也可能采用压缩状态。这些机制改变上面的显存账，估算时应读实际模型配置。

### 1.3 PagedAttention：把 KV cache 当虚拟内存管

<!-- paper-figure:paper-vllm-fig06 -->
直接看vLLM论文的块表原图，能把逻辑连续和物理分散对应起来。

[![Efficient Memory Management for Large Language Model Serving with PagedAttention，图6：逻辑KV块通过块表映射到物理显存](../../_static/img/literature-reviews/paper-vllm-fig06-screen.png)](../../_static/img/literature-reviews/paper-vllm-fig06.png)

**原图来源：**Woosuk Kwon et al.，[原文](https://arxiv.org/abs/2309.06180v1)，图6，PDF第6页，arXiv 2309.06180v1。[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。仅截取图形区域与等比例显示，中文说明置于图外，访问于2026-11-23。

**怎样读这张图。** 左侧Logical KV blocks按请求的token顺序编号，中间Block Table记录每个逻辑块对应的物理块号和已填充数量，右侧Physical KV blocks则可以分散在显存池不同位置。箭头通过块表把两种编号连接起来，计算时按逻辑顺序读取对应物理数据。

**与本节的联系。** 分页改善的是缓存管理与可复用空间。图中物理块仍属于vLLM管理的池，逻辑上空着的槽位不会自动变成Env可以分配的显存。
<!-- /paper-figure:paper-vllm-fig06 -->


PagedAttention 将缓存切成固定大小的块（block），请求按需领取；每条请求用一张块表记录逻辑顺序与实际显存位置的对应关系。这借鉴了操作系统“逻辑地址连续、物理页可分散”的方法。

假设每块存 16 个 token，一条请求目前有 20 个 token，只需 2 块，第二块使用 4 个位置。若按照最大长度 128 预留，则需要 8 块。按需分配把尚未用到的空间留给其他请求。

```text
请求 A 的逻辑块：[0] [1]
                 ↓   ↓
显存中的物理块：[7] [2]     块表记住 0→7、1→2
```

执行注意力时，计算内核沿块表找到 K/V，按正确 token 顺序计算。请求增长时申请下一块，结束时归还。共享前缀还可以让多条请求引用同一物理块；需要改写共享的未满块时，先复制一份，称为 Copy-on-Write（写时复制）。

更详细的块号、引用计数和三分支例子见[缓存管理详解](#kv-缓存管理详解块表写时复制与前缀复用)。块大小依版本、后端和配置变化，这里的 16 用于教学。

### 1.3b Prefix Caching：跨请求复用 KV block 的机制

Prefix caching 即前缀缓存：新请求开头的内容与已有缓存一致时，直接复用这段计算结果，减少重复 prefill。它还可以让同时存活的请求共享前缀的物理缓存；保留待复用的缓存本身也占空间，由回收策略管理。

例如两个请求共享 1,024-token 的系统提示，第二个请求多出 100 token。若完整共享前缀命中，就主要补算新增部分。最终还需处理的 token 受块边界、模型后端和采样路径影响。

缓存通常给每块计算“前面所有块的摘要 + 本块 token + 影响模型结果的附加条件”的哈希。哈希是便于查找的内容摘要；把前块摘要串进来，保证匹配的是整个前缀。例如同样的一段文字出现在不同前文后，得到的状态可能不同，应分别计算。

前缀复用所需的具体模型支持与设置见 [vLLM 官方前缀缓存说明](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)。实现推导见[缓存管理详解](#前缀哈希与缓存回收)。

### 1.4 Continuous Batching（iteration-level scheduling）

<!-- chapter-figure:chapter-continuous-batching -->
连续合批按迭代补位，不必让一批里所有请求同时结束。

[![连续合批按迭代补位，不必让一批里所有请求同时结束](../../_static/img/document-figures/chapter-continuous-batching-screen.png)](../../_static/img/document-figures/chapter-continuous-batching.png)

格子只表示参与哪轮，不假设真实每轮都一样长。A结束后C可进入，具体每轮仍受token预算、KV和prefill安排约束。 [放大查看 SVG](../../_static/img/document-figures/chapter-continuous-batching.svg)。
<!-- /chapter-figure:chapter-continuous-batching -->


Continuous batching 即连续合批：每轮重新决定哪些请求参与计算，已结束请求及时离开，空位接收新请求。

先把“一个模型”和“一个请求”分开。一张卡上部署一个 `TP=1` 的普通 vLLM 实例时，通常有一份模型权重，同时保存多条请求各自的 KV 缓存。8 个请求同时解码，意味着把这 8 条序列本轮要处理的 token 合到模型计算中，并非复制 8 份权重，也不是启动 8 个彼此独立的模型进程。常规自回归生成中，同一请求的下一步依赖上一步结果，所以主要是在不同请求之间合批；推测解码是另一个需要单独分析的配置。

教学例子：A 已生成 10 个 token，B 已生成 30 个 token。一轮执行可以同时算出 A 的第 11 个与 B 的第 31 个，两个位置各读取自己的历史 KV，但在线性层使用同一份权重。张量并行 `TP=2` 则是把这份模型的矩阵分到两张卡上，共同完成这一轮；多副本则是另起一份模型分担其他请求。这三个维度不能只用“同时跑了几个 D”来计数。

假设最多同时处理 2 条请求，A、B、C 分别需要 2、5、2 轮生成：

| 轮次 | 位置 1 | 位置 2 | 变化 |
|---|---|---|---|
| 1 | A 第 1 轮 | B 第 1 轮 | A、B 开始 |
| 2 | A 第 2 轮，完成 | B 第 2 轮 | 释放 A 的位置 |
| 3 | C 第 1 轮 | B 第 3 轮 | C 补位 |
| 4 | C 第 2 轮，完成 | B 第 4 轮 | C 已完成 |
| 5 | 空闲 | B 第 5 轮，完成 | 全部结束 |

这是忽略 prefill 和调度成本的示例。固定批次若先等 A、B 全结束，再处理 C，则需要 7 轮；连续补位只需 5 轮。优势来自及时利用可用容量，实际每轮耗时会随批量与上下文变化。

调度同时受两类预算影响：`max_num_seqs` 控制并发序列数，`max_num_batched_tokens` 控制本轮处理 token 数。缓存不足时，部分请求需要等待或抢占后重算，具体行为依所用引擎版本。

### 1.5 Chunked Prefill

Chunked prefill 把长输入分成多个小段处理，让长请求的输入处理与其他请求的逐步生成穿插进行。

例如新请求 D 有 8,192-token 输入，已有 B、C 在生成。若本轮给 D 512 个输入 token，同时给 B、C 各 1 个生成 token，本轮共处理 514 个 token。D 的所有输入处理完后才开始采样输出，B、C 则在这些轮次继续前进。

这样可以缓和整段长输入一次占满执行时间造成的延迟波动。chunk 越大，长请求可能更快完成 prefill，但其他请求的每 token 延迟也可能增加；合理值要结合实际工作负载测量。

这一轮怎样同时执行 512 个输入 token 和 2 个生成 token，见下文 [Selective batching](#selective-batching投影按-token-合批attention-按请求隔离)。

### 1.6 Prefill/Decode 分离（Disaggregation）

<!-- paper-figure:paper-distserve-fig06 -->
DistServe原图展示真正的PD分离：两边都执行模型，阶段之间传输KV。

[![DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving，图6：P池、D池与跨池KV传输](../../_static/img/literature-reviews/paper-distserve-fig06-screen.png)](../../_static/img/literature-reviews/paper-distserve-fig06.png)

**原图来源：**Yinmin Zhong et al.，[原文](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)，图6，PDF第9页，OSDI 2024 published edition。版权归原作者，作为本文讨论对象节选。仅截取图形区域与等比例显示，中文说明置于图外，访问于2026-11-23。

**怎样读这张图。** 顶部Controller接收请求，左侧Prefill Instance处理输入，右侧Decoding Instance逐步生成，中间KV Cache Transfer连接两组实例。每组内部的GPU和Parallel Runtime表示阶段内部也可以做模型并行；这与把模型前后两半分别命名为P和D不同。

**与本节的联系。** 这张图是DistServe的设计参照。本节下面继续说明PSRL当前普通生成路径及PD配置边界；读外部系统的图时，要与本地已经启用哪些能力分开。
<!-- /paper-figure:paper-distserve-fig06 -->


普通 vLLM 生成实例负责 P 和 D 两个阶段，同一套权重用于两者。单条请求先做 P，再做 D；不同请求可以分别处于 P 或 D，并由引擎在迭代间穿插，开启 chunked prefill 时还可把部分 P token 与 D token 放进同一轮。这里的“同时服务”不要求两个独立进程或两套权重。

Prefill/decode 分离把两个阶段交给不同执行实例，分别按首响应速度与持续生成吞吐配置资源。P 和 D 都需要执行模型各层，常见部署是在两个执行组各自保留所需的完整模型副本或相应分片。它不是把前一半模型层当成 P、后一半模型层当成 D，那属于不同的切分问题。输入处理完成后，将 KV 缓存和请求状态传给 D 实例，再继续回答。

```text
当前 PSRL 的一般路径：
请求 → SMG 网关选择推理副本 → 同一副本执行 P → 该副本继续 D → 返回结果

真正的 PD 分离路径（尚未接入当前网关配置）：
请求 → P 池执行输入处理 → KV 与请求状态传输 → D 池逐步生成 → 返回结果
```

2026-11-18 核对当前代码，生成网关的 [`build_router_args`](../../../psrl/workers/gen/smg_adapter.py#L71) 明确设置 `pd_disaggregation=False`，`prefill=None`、`decode=None`；奖励模型网关也关闭 PD 分离。服务端的[引擎参数](../../../psrl/workers/gen/vllm_async_server.py#L350)把序列数、每轮 token 数、chunked prefill 与张量并行度交给普通生成实例。当前已有的跨副本 KV 迁移是请求重新路由时复用缓存的机制，不能据此认定已经建立 P 池与 D 池。这里核对的是本地代码路径，没有对另一个部署或自定义启动参数作推断。

例如 A 负责长输入处理，B 专门处理正在生成的请求。A 完成 20,000-token 前缀后传给 B，B 从对应状态继续。这样可减少长 prefill 对 B 的干扰，代价是缓存传输、协调和额外资源。是否划算取决于输入长度、传输带宽和负载。[NIXL](../04-高速传输_NIXL内存搬运机制.md)解释了大块内存的传输方式。

用本次 Qwen2.5-3B 配置做容量算例：36 层、2 个 KV 头、每头 128 维、bf16，则每 token 的原始 KV 为 `2×36×2×128×2=36 KiB`。一个 2,048-token 输入约需传 72 MiB 的 KV，32 个这样的请求约为 2.25 GiB。若教学假设有效传输带宽为 25 GiB/s，仅数据传输下界约 90 ms，完整转交还包括打包、同步、排队与接收端容量分配。真实值受缓存布局、复用、并行切分和实现影响，不能把链路标称带宽直接当作端到端速度。

PD 分离主要提供分别管理首 token 延迟和持续生成延迟的机会。vLLM 官方强调其功能本身不保证提高吞吐；特别是固定只有两张卡时，从两个 P+D 副本改成一张 P、一张 D，也减少了可同时承担 D 的卡数。应先与两个普通副本以及调优 chunked prefill 的方案比较，再判断收益。[官方 PD 分离说明](https://docs.vllm.ai/en/latest/features/disagg_prefill/)、[调优说明](https://docs.vllm.ai/en/latest/configuration/optimization/)

对 Env 调度，阶段分开后更容易分别测量 P 与 D 的计算曲线，但 D 常有大量 KV 读取，低矩阵算力使用率仍可能伴随高显存带宽压力。P 卡此刻没请求也不表示可以借给一个无法暂停的长命令。如何把阶段性能、后续请求和环境归还时间一起考虑，见[共享决策专题](../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#16-从计算上限到整批完成时间)。

### 1.7 其余关键优化

CUDA Graph 预先记录可复用的 GPU 执行图，减少 CPU 重复发起算子的开销。量化降低权重或缓存的存储量，精度和算子支持需核对实际配置。

Speculative decoding（推测解码）先起草多个候选 token，再由目标模型验证；接受多个候选时，一轮验证可得到多个输出。收益取决于接受率、草稿成本与目标模型负载。

引擎也可将网络处理、分词与 GPU 执行分到不同进程，减轻主执行循环的 CPU 干扰。这些优化通常需要共同衡量首响应、每 token 延迟和吞吐。

### 1.8 vLLM 中的并行

并行是把工作分给多张 GPU。选择维度取决于要解决的容量或速度问题：

| 方式 | 切分什么 | 适合解决什么 |
|---|---|---|
| TP，张量并行 | 一层中的矩阵与计算 | 单卡放不下模型，或需要共同执行大矩阵 |
| PP，流水线并行 | 模型层 | 将更多层分布到多卡，需关注阶段等待 |
| 多副本 / DP | 不同请求 | 多个模型实例分担服务流量 |
| CP，上下文并行 | 长上下文的状态及注意力工作 | 分摊长请求缓存与注意力计算 |
| EP，专家并行 | MoE 模型的不同专家 | 分摊专家参数和对应计算 |

Decode 每步通常只产生一个新 token，但这个 token 仍要访问很长的历史 K/V。因此，上下文状态与注意力工作仍可并行切分。当前 vLLM 官方文档已包含 decode context parallelism，覆盖 MLA 与 GQA 场景，见[上下文并行部署](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)。

TP 通信频繁，优先利用高速互联；跨节点是否合适要结合带宽、延迟和模型配置测量。多个生成副本之间通常无需像训练 DP 那样合并梯度，但路由、负载协调以及 PSRL 的权重更新仍有通信。

本节说明通用机制。PSRL 部署能启用哪些参数，要以安装的 vLLM 版本、模型及本地适配代码为准；训练侧各维度见[训练并行](02-训练并行_模型切分与通信带宽.md)。

### 1.9 Linear Attention / SSM 层的 cache 管理：什么时候需要多存一份 checkpoint

标准注意力通常保存每个历史 token 的 K/V；部分线性注意力、Mamba 等状态空间模型（SSM）则以固定大小的递归状态概括历史。继续当前请求只需最新状态，但要让另一个请求从较早前缀接着算，就需要保存该前缀处的状态快照，称为 checkpoint。

教学例子：原请求已处理 2,000 token，新请求只共享前 1,000 token。原请求的最新状态已包含后 1,000 token，若要复用前缀，就需找到位置 1,000 的快照，或从更早位置重算。

原记录核对的是本机 vLLM `0.22.1.dev0`（2026-07-06）中的 `mamba_cache_mode`：

| 历史模式 | 保存策略 | 典型用途 |
|---|---|---|
| `none` | 主要保留当前请求最新状态 | 继续当前请求 |
| `align` | 为该版本的对齐与分段执行保留状态 | 分段处理后续算 |
| `all` | 额外保存支持位置上的中间状态 | 可复用前缀快照 |

这些名称和具体内存份数属于当时实现。模型能否使用某档，取决于协议声明、内核能否导出正确状态及后端支持；旧记录根据支持列表推断“只有 Mamba2 在数学上能分块”的说法已移除。部署前应核对当前版本模型与缓存支持组合。

## Selective batching：投影按 token 合批，attention 按请求隔离

ORCA 论文中的 selective batching（选择性合批）解释了不同长度、不同阶段的请求怎样一起执行：逐 token 的投影合并计算，注意力保留各自请求边界。调度器决定本轮选谁，执行器负责正确地算这一批。[ORCA 论文 §3](https://www.usenix.org/system/files/osdi22-yu.pdf)

### 自建算例：两个 decode 请求与一个 prefill 请求

设隐藏维度为 `d`，使用标准多头注意力（MHA）作教学说明：

| 请求 | 阶段 | 本轮 token | 使用的历史长度，含本轮 |
|---|---|---:|---:|
| A | decode | 1 | 128 |
| B | decode | 1 | 2048 |
| C | prefill | 4 | 4 |

本轮输入拼成 `X = concat(X_A, X_B, X_C)`，形状 `[6,d]`。A、B 已缓存的历史 token 供注意力读取，本轮只投影新增 token。

### 投影：沿 token 维拼接，共用权重矩阵

<!-- chapter-figure:chapter-selective-batching -->
先把6个新token拼起来做投影，再按请求边界做attention。

[![先把6个新token拼起来做投影，再按请求边界做attention](../../_static/img/document-figures/chapter-selective-batching-screen.png)](../../_static/img/document-figures/chapter-selective-batching.png)

沿用本节A、B各1个decode token，C有4个prefill token的例子。共享的是投影权重的计算机会，A、B、C的历史和遮罩仍分开。 [放大查看 SVG](../../_static/img/document-figures/chapter-selective-batching.svg)。
<!-- /chapter-figure:chapter-selective-batching -->


```text
concat(X_A, X_B, X_C) @ W
    = concat(X_A @ W, X_B @ W, X_C @ W)

QKV = X @ W_QKV
[6,d] @ [d,3d] → [6,3d]
```

矩阵乘独立变换每一行，因此可以合并不同请求的 token，分摊使用同一权重的成本。输出投影、逐 token 的 MLP（前馈网络）等也可类似处理。

### Attention：保留请求边界，分别访问变长 KV

投影后按 `[1,1,4]` 分清请求边界，各自访问自己的 K/V：

```text
A：score 形状 [1,128]，只访问 A 的历史
B：score 形状 [1,2048]，只访问 B 的历史
C：score 形状 [4,4]，用因果 mask 限制对未来位置的访问
```

这里的 score 是注意力打分矩阵，mask 是屏蔽无效位置的标记。各请求算完后，输出仍可合并成 `[6,d]`，继续做投影和前馈计算。

因此，合批发生在可以逐token独立计算的部分；历史注意力仍按A、B、C分别读取。上图的左右两部分正对应这两种边界。

### 性能边界与现代实现

注意力访问的是各请求自己的 K/V，主要收益与逐 token 投影不同。现代变长内核可以在一次调用中执行多个请求，以请求边界保证隔离，并减少启动开销。[FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)

这解释了“请求数据保持隔离，执行仍可合批”。若想继续推导块表寻址、在线 softmax 和写时复制，读[缓存管理详解](#kv-缓存管理详解块表写时复制与前缀复用)；若关注训练激活为何占显存，读[训练并行](02-训练并行_模型切分与通信带宽.md)。

## KV 缓存管理详解：块表、写时复制与前缀复用

缓存管理需要同时回答三个问题：历史 K/V 存在哪里，多条请求能共享哪些内容，分支改写时怎样保持独立。本节沿一个三分支例子展开，接着解释注意力怎样读取分页数据。以下块大小、槽位与分支数字均为教学假设，实际布局与配置依后端而定。


### 块号和槽位怎样定位一个 token

假设每块存 4 个 token。一条请求有 6 个 token，块表为 `[7,2]`，表示逻辑块 0 位于物理块 7，逻辑块 1 位于物理块 2：

```text
逻辑位置：  0 1 2 3 | 4 5
物理位置：  块7      | 块2的槽位0、1
```

第 `i` 个 token 对应逻辑块 `i//4`、块内槽位 `i%4`。例如 `i=5` 时，查块表第 1 项得到物理块 2，读取其中槽位 1。这样逻辑上连续的历史，可以存放在分散的物理块中。

统一块大小让池中的空闲块可互换，按需申请、归还即可。多种大小的池同样能存在，但管理和匹配会更复杂；PagedAttention 的基本讲解采用固定规格。

### 三条分支怎样共享并各自续写

<!-- paper-figure:paper-vllm-fig08 -->
先看论文的两条采样分支怎样共享前缀，再看下面三条分支的教学例子。

[![Efficient Memory Management for Large Language Model Serving with PagedAttention，图8：共享前缀的引用计数与写时复制](../../_static/img/literature-reviews/paper-vllm-fig08-screen.png)](../../_static/img/literature-reviews/paper-vllm-fig08.png)

**原图来源：**Woosuk Kwon et al.，[原文](https://arxiv.org/abs/2309.06180v1)，图8，PDF第7页，arXiv 2309.06180v1。[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。仅截取图形区域与等比例显示，中文说明置于图外，访问于2026-11-23。

**怎样读这张图。** A1与A2的逻辑块共同指向相同物理前缀块，引用计数记录共享者数量。写到仍被共享的最后一块时，红色Copy-on-write箭头复制该块，让一条分支在自己的物理块上续写。已经写满且不再修改的前缀可以继续共享。

**与本节的联系。** 共享节省的是重复前缀的数据，不会把不同分支之后生成的内容混在一起。图中共享还发生在模型执行语义允许的范围内，代码或权重版本变化时要重新核验缓存有效性。
<!-- /paper-figure:paper-vllm-fig08 -->


<!-- chapter-figure:chapter-kv-copy-on-write -->
写时复制的目的，是让三条分支共享旧前缀，又不改到彼此的新内容。

[![写时复制的目的，是让三条分支共享旧前缀，又不改到彼此的新内容](../../_static/img/document-figures/chapter-kv-copy-on-write-screen.png)](../../_static/img/document-figures/chapter-kv-copy-on-write.png)

图采用每块4个token、前缀6个的独立教学布局。已满P0保持共享，未满P1在分支各自续写前复制；块号用于理解，不是实际地址。 [放大查看 SVG](../../_static/img/document-figures/chapter-kv-copy-on-write.svg)。
<!-- /chapter-figure:chapter-kv-copy-on-write -->


同一段 6-token 前缀要生成 3 条候选回答 A、B、C。最初它们都引用块 7、2，每块的引用计数为 3。引用计数表示有多少使用者仍需要这块数据。

块 7 已满，后续只读，可以继续共享。块 2 未满，A、B、C 接下来都要写槽位 2，内容可能不同，于是用 Copy-on-Write（写时复制）保持隔离：

| 步骤 | A 的末块 | B 的末块 | C 的末块 | 原块 2 引用数 |
|---|---:|---:|---:|---:|
| 初始共享 | 2 | 2 | 2 | 3 |
| A 先写，复制 2→8 | 8 | 2 | 2 | 2 |
| B 再写，复制 2→9 | 8 | 9 | 2 | 1 |
| C 写，已独占原块 | 8 | 9 | 2 | 1 |

先复制再改写，保证一个分支的新 token 不会改变其他分支的历史。在这一假设下，3 个共享者需要 2 次复制；一般 `k` 个都要改写的共享者，最后一个可保留原块，其余 `k-1` 个复制。

写满后继续生成时，各自申请全新下一块。请求结束则减少所引用块的计数，最后一个使用者离开后，块可以回收。

### 每层的数据怎样对应同一块号

Transformer 多层各自计算 K/V。同一个块号表达的是同一组 token 的逻辑分配，但每层保存自己的数值。写时复制需要复制所有相关层的数据，保证它们对应同一条历史。

以 Llama-3-8B 的参数计算，一个 16-token 块在一层的 Key 数据量为：

```text
16 token × 8 KV头 × 128维 × 2字节 = 32 KiB
32层 × 2（K和V）× 32 KiB = 2 MiB
```

因此一整块跨层 K/V 总量约 2 MiB。后端可以把多层复制合并为少数内核调用；数据需要复制多少，与启动多少次复制操作，是两个计量问题。

### 分页注意力怎样读完所有历史

对一个查询向量 `q`，注意力先计算它与历史各 `k_i` 的相似度 `s_i`，再按 softmax 权重加权 `v_i`：

```text
输出 = Σ exp(s_i)·v_i / Σ exp(s_i)
```

为了避免指数过大，实际可减去所有分数的最大值。分页读取时，无法一次拿齐所有历史，可以逐块更新三个量：当前最大值 `m`、归一化分母 `l`、加权和 `a`。

处理下一块，设该块分数为 `s`、Value 为 `v`：

```text
m_new = max(m, max(s))
l_new = exp(m-m_new)·l + Σ exp(s-m_new)
a_new = exp(m-m_new)·a + Σ exp(s-m_new)·v
最终输出 = a / l
```

当新块最大分数更大时，先把旧累计结果换到新的指数基准，再加上本块。这在实数数学上等价于一次性 softmax；浮点计算可能因求和顺序产生小差异。

例子：只有两个位置，分数都为 0，Value 为 2、4。第一块后 `m=0,l=1,a=2`；第二块后 `m=0,l=2,a=6`，输出为 3，正好是等权平均。块表只决定去哪里读，在线累计保证读到的数据组成完整注意力结果。

### 前缀哈希与缓存回收

跨请求复用需要识别“计算状态一致的完整前缀”。一种常见做法是链式哈希：

```text
本块摘要 = hash(前块摘要, 本块token, 影响模型结果的附加条件)
```

前块摘要把更早历史带进来，附加条件可包括适配器或多模态输入。新请求从开头逐块查找，复用连续命中的前缀，后续部分重新计算。缓存粒度、最后未满块如何处理及可支持的模型，以对应引擎版本为准。

请求结束后，缓存块可能继续保留，等待后续命中；资源紧张时再回收。共享活跃前缀可以减少重复物理存储，保留历史缓存则会占用池容量，因此需要共同衡量命中收益与可用空间。

相关机制与配置见 [vLLM 前缀缓存文档](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)。如果想继续追问，可以从“浮点舍入为何影响等价计算”“引用计数怎样避免提前回收”“循环状态模型如何保存前缀快照”三个问题进入；最后一个在[推理机制 §1.9](#19-linear-attention--ssm-层的-cache-管理什么时候需要多存一份-checkpoint)继续解释。
