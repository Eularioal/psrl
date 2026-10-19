# 推理与训练并行：公式速查及实验记录

> **核心结论**：集中查阅符号、公式、机制对应的实验脚本与历史测量条件；历史结果不代表当前环境的重新验证。

## 目录

- [七、符号表与常用公式速查](#七符号表与常用公式速查)
  - [符号](#符号)
  - [公式](#公式)
- [八、机制 → 实验验证设计](#八机制--实验验证设计)
  - [实验 1 实测结果(2026-09-11,H20,Qwen2.5-3B-Instruct)](#实验-1-实测结果2026-09-11h20qwen25-3b-instruct)
- [延伸问题](#延伸问题)
- [相关文档](#相关文档)
- [参考](#参考)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。原文整理日期为 2026-09-10，后续追加内容的日期以各节记载为准；保留章节编号以便对照旧引用。

[返回专题索引](../vllm-and-training-parallelism.md)

## 七、符号表与常用公式速查

### 符号

| 符号 | 含义 |
|---|---|
| `P` | 模型参数量 |
| `L` | transformer 层数 |
| `h` | hidden size |
| `b` | microbatch size |
| `s` | 序列长度 |
| `T` | 一个 batch 的 token 总数 = `b·s` |
| `t` | TP 并行度 |
| `p` | PP 并行度 |
| `d` | DP 并行度 |
| `m` | microbatch 数量（梯度累积步数） |
| `v` | virtual pipeline stage 数 |

### 公式

```
训练 FLOPs（fwd+bwd）      = 6 · P · T  (+ 12 · L · h · T²  attention 项)
推理 prefill FLOPs         = 2 · P · T  (+ 4 · L · h · T²)
推理 decode FLOPs/step     = 2 · P · B

KV cache/token             = 2 · L · n_kv · d_head · dtype_bytes
训练静态显存（bf16+Adam）   = 16 · P  bytes

All-Reduce 每卡流量         = 2(n-1)/n · D
RS / AG 每卡流量            = (n-1)/n · D

TP 通信/步/卡              = 4 · 2(t-1)/t · (b·s·h·2) · L
PP 通信/microbatch/边界     = b·s·h·2
DP 通信/步/卡              = 2(d-1)/d · P_bytes
FSDP(ZeRO-3) 通信/步/卡     = 3 · P_bytes（近似）

PP 气泡比                   = (p-1)/m        朴素/1F1B
                           = (p-1)/(m·v)    interleaved

Roofline ridge point       = Peak_FLOPS / HBM_BW
MFU                        = 6·P·T / (t_step · N_gpu · Peak_FLOPS)
```

---

## 八、机制 → 实验验证设计

前面的机制讲解都是理论推导。为了不停留在"应该是这样",配套写了 4 个可直接运行的实验脚本,放在
`../vllm_experiments/`（本地资料，未随仓库发布）。设计原则:**关键数字尽量实测,不查手册**
——比如显存带宽、matmul TFLOPS 都用一段简单的 torch 代码现场测,再拿理论公式去对照测出来的数字,
而不是直接抄 H100/H20 的标称参数(标称峰值和实际可达值经常差好几成)。

> **进度说明**:实验 1 已于 2026-09-11 在另一台空闲机器(8×H20,显存全空)上用本地
> Qwen2.5-3B-Instruct 跑通,结果见下方"实测结果"。实验 2/3/4 待补。

### 实验 1 实测结果(2026-09-11,H20,Qwen2.5-3B-Instruct)

实测 HBM 带宽 3246 GB/s,模型权重 6.17 GB,理论最小 `ms/step`(纯权重读取时间)= 1.90 ms。
固定输出长度 64 token,扫描 batch size:

| batch_size | tokens/s | ms/step |
|---|---|---|
| 1 | 45 | 22.05(冷启动/JIT 编译开销,不代表稳态) |
| 2 | 161 | 12.42 |
| 4 | 326 | 12.26 |
| 8 | 639 | 12.52 |
| 16 | 1002 | 15.97 |
| 32 | 2467 | 12.97 |
| 64 | 4841 | 13.22 |
| 128 | 8896 | 14.39 |
| 256 | 14548 | 17.60 |

**结论**:batch size 从 4 涨到 128(32 倍),吞吐同步涨了 27 倍(326 → 8896 tok/s),而每步耗时几乎
不变(12.26 → 14.39 ms)——这正是"decode 阶段 batching 近乎免费"的直接证据。实测 ms/step
(~12~15ms)比纯权重读取的理论下界(1.90ms)高出一个量级,差值来自调度开销、Python/CPU 侧开销、
attention kernel 等**与 batch size 无关的固定成本**,并不影响"该成本对所有 batch size 基本平坦"这一
定性结论。batch=256 时 ms/step 开始略微上升(17.60ms),是这轮扫描里开始看到边际效应的信号,但
还远没到饱和拐点——受限于这次只测到 256,没有继续往上扫到真正打满带宽/算力的量级。

| 机制 | 脚本 | 核心假设(预期现象) | 怎么解读输出 |
|---|---|---|---|
| decode 是**显存带宽瓶颈**([1.1](inference-mechanisms.md#一vllm-的核心机制)) | `01_bandwidth_and_decode_scaling.py`（本地资料，未随仓库发布） | 固定输出长度、扫描 batch size,`ms/step` 应接近"权重字节数 / 实测 HBM 带宽"这条理论线,直到某个 batch size 才明显偏离(转向算力瓶颈或 KV cache 顶满) | 看结果表里 `ms_per_step` 这一列是否随 `batch_size` 基本走平;和打印出的"理论最小 ms/step"比较,偏离方向和幅度就是 batching 还有多少"免费"空间 |
| prefill 是**算力瓶颈**,与 decode 对照([1.1](inference-mechanisms.md#一vllm-的核心机制)) | `02_prefill_vs_decode_roofline.py`（本地资料，未随仓库发布） | 扫描 prompt 长度(`output_len=1` 近似纯 prefill),`tokens/s` 应趋于一个由实测 TFLOPS 决定的上限,而不是像 decode 那样随 batch 线性伸缩 | 脚本会自动读 `01` 的结果 JSON,打印"decode ms/step vs prefill ms/token"对照表;两条线一条随 batch 走平、一条随长度走平,就是 compute-bound vs memory-bound 的直接证据 |
| **PagedAttention 的 prefix caching**([1.3](inference-mechanisms.md#13-pagedattention把-kv-cache-当虚拟内存管)) | `03_prefix_caching.py`（本地资料，未随仓库发布） | 关闭 `enable_prefix_caching` 时每条请求延迟接近常数(重复付共享前缀的 prefill 成本);打开后除第 1 条外显著变快 | 看 `speedup` 列从第 2 条请求起是否明显 > 1;命中率越高(前缀越长、后续请求越多),预期加速比越大 |
| **continuous batching**([1.4](inference-mechanisms.md#14-continuous-batchingiteration-level-scheduling)) | `04_continuous_batching.py`（本地资料，未随仓库发布） | 长短输出混合的请求,一次性提交(vLLM 真实的 continuous batching)应明显快于"按固定大小分块、每块等全部做完再发下一块"的静态批处理基线;加速比应接近"块数"(因为每块都混入一个长请求拖累整块) | 看 `speedup`(场景 B 耗时 / 场景 A 耗时);这正是文档 [6.2 节](rl-training-workflow.md#62-rollout-长尾)里 RL rollout 长尾问题在单机 GPU 调度层面的缩影 |

四个实验共用的设计取舍(细节见 `vllm_experiments/README.md`（本地资料，未随仓库发布）):

- 用 `min_tokens=max_tokens` + `ignore_eos=True` 让输出长度精确可控,避免 EOS 提前触发导致不同
  batch size / 长度之间不可比
- 用 tokenizer 精确构造定长 prompt(而不是按字符数估算),保证"prompt 长度"这个自变量是准确的
- 权重字节数优先从 `model.safetensors.index.json` 读实际值,只有模型还没下载到本地时才退化为用
  `AutoConfig` 的参数量公式估算

---

## 延伸问题

- FP8 / MXFP4 训练如何改变上面的通信量和 roofline 计算？
- MLA（DeepSeek）相比 GQA 在 KV cache 和 TP 切分上的具体差异？
- All-to-All 在多轨（multi-rail）IB 网络上的拓扑感知优化？
- 异步 RL 中 off-policy 程度（版本滞后）与收敛性的定量权衡？

## 相关文档

- [知识文档索引](../../README.md)
- [PSRL 项目概览](../../overview/index.md)
- vLLM 机制验证实验脚本（本地资料，未随仓库发布）

## 参考

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*（SOSP 2023，vLLM 原始论文）
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*（TP 方案）
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*（Sequence Parallelism + selective recompute）
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（3D 并行与 interleaved pipeline）
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*
- Agrawal et al., *Sarathi-Serve*（chunked prefill）
- Zhong et al., *DistServe*（prefill/decode 分离）
