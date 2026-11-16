# 推理与训练并行：公式速查及实验记录

2026-09-11 的 H20 测量中，batch 从 4 增至 128，吞吐约提高 27 倍，平均生成步耗时从 12.26 ms 增至 14.39 ms。它展示了这一配置下的合批收益；下面保留完整数据，并说明公式的使用条件。

2026-11-16 核对公式与文字口径，保留历史实测。原实验脚本和原始日志未随仓库发布，本次未重跑或从原始数据重新计算。

[返回专题索引](../vllm-and-training-parallelism.md)

## 目录

- [七、符号表与常用公式速查](#七符号表与常用公式速查)
  - [符号](#符号)
  - [公式](#公式)
- [八、机制 → 实验验证设计](#八机制--实验验证设计)
  - [实验 1 实测结果(2026-09-11,H20,Qwen2.5-3B-Instruct)](#实验-1-实测结果2026-09-11h20qwen25-3b-instruct)
- [延伸问题](#延伸问题)
- [相关文档](#相关文档)
- [参考](#参考)

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
训练 FLOPs（fwd+bwd）      = 6 · P · T  (+ 12 · L · h · b · s²  attention 项)
推理 prefill FLOPs         = 2 · P · T  (+ 4 · L · h · b · s²)
推理 decode FLOPs/step     = 2 · P · B

KV cache/token             = 2 · L · n_kv · d_head · dtype_bytes
训练静态显存（下述16字节假设）   = 16 · P  bytes

All-Reduce 每卡流量         = 2(n-1)/n · D
RS / AG 每卡流量            = (n-1)/n · D

TP 通信/步/卡              = 4 · 2(t-1)/t · (b·s·h·2) · L
PP 通信/microbatch/边界     = b·s·h·2
DP 通信/步/卡              = 2(d-1)/d · P_bytes
FSDP(ZeRO-3) 通信/步/卡     = 3 · P_bytes（近似）

PP 空闲/有效时间比          = (p-1)/m        均衡朴素/1F1B
                           ≈ (p-1)/(m·v)    interleaved

Roofline ridge point       = Peak_FLOPS / HBM_BW
MFU                        = 6·P·T / (t_step · N_gpu · Peak_FLOPS)
```

---

公式条件：注意力平方项应按每条序列长度计算，等长批量为 `b·s²`，而不是把总 token 数 `T=b·s` 整体平方。16 字节静态显存假设为 bf16 参数与梯度、fp32 主副本和两份 fp32 Adam 动量；FSDP 通信量采用简化同精度与重聚合假设。流水线空闲占总时间还需用空闲时间除以总时间，详见[训练并行](training-parallelism.md)。

## 八、机制 → 实验验证设计

前面的机制讲解都是理论推导。原记录设计了 4 个机制实验，脚本当时位于
`../vllm_experiments/`（本地资料，未随仓库发布）。设计原则：**记录实际有效带宽和算力，再与理论估计对照**
——比如显存带宽、matmul TFLOPS 都用一段简单的 torch 代码现场测,再拿理论公式去对照测出来的数字,
而不是直接抄 H100/H20 的标称参数(标称峰值和实际可达值经常差好几成)。

> **进度说明**:实验 1 已于 2026-09-11 在另一台空闲机器(8×H20,显存全空)上用本地
> Qwen2.5-3B-Instruct 跑通,结果见下方"实测结果"。实验 2/3/4 在本报告中保留为设计，未形成实测结果。

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

**怎样解读**：batch 从 4 到 128 增加 32 倍，吞吐从 326 到 8896 token/s，约增加 27.3 倍；平均步耗时增加约 17.4%。到 batch 256 时，步耗时为 17.60 ms，批量继续增加也提高了单步成本。

1.90 ms 仅估算理想权重读取，实测还包含注意力、调度、计算和通信等过程。现有数据没有性能剖析，无法把差值全部分配给固定 CPU 成本，也没有确定最佳延迟点或饱和批量。

| 机制 | 脚本 | 核心假设(预期现象) | 怎么解读输出 |
|---|---|---|---|
| decode 是**显存带宽瓶颈**([1.1](inference-mechanisms.md#一vllm-的核心机制)) | `01_bandwidth_and_decode_scaling.py`（本地资料，未随仓库发布） | 固定输出长度、扫描 batch size,`ms/step` 应接近"权重字节数 / 实测 HBM 带宽"这条理论线,直到某个 batch size 才明显偏离(转向算力瓶颈或 KV cache 顶满) | 看结果表里 `ms_per_step` 这一列是否随 `batch_size` 基本走平;和打印出的"理论最小 ms/step"比较,偏离需要结合算子与调度剖析，评估批量增加后的成本 |
| prefill 是**算力瓶颈**,与 decode 对照([1.1](inference-mechanisms.md#一vllm-的核心机制)) | `02_prefill_vs_decode_roofline.py`（本地资料，未随仓库发布） | 扫描 prompt 长度(`output_len=1` 近似纯 prefill),`tokens/s` 应趋于一个由实测 TFLOPS 决定的上限,而不是像 decode 那样随 batch 线性伸缩 | 脚本会自动读 `01` 的结果 JSON,打印"decode ms/step vs prefill ms/token"对照表;两条线一条随 batch 走平、一条随长度走平,就是 compute-bound vs memory-bound 的直接证据 |
| **PagedAttention 的 prefix caching**([1.3](inference-mechanisms.md#13-pagedattention把-kv-cache-当虚拟内存管)) | `03_prefix_caching.py`（本地资料，未随仓库发布） | 关闭 `enable_prefix_caching` 时每条请求延迟接近常数(重复付共享前缀的 prefill 成本);打开后除第 1 条外显著变快 | 看 `speedup` 列从第 2 条请求起是否明显 > 1;命中率越高(前缀越长、后续请求越多),预期加速比越大 |
| **continuous batching**([1.4](inference-mechanisms.md#14-continuous-batchingiteration-level-scheduling)) | `04_continuous_batching.py`（本地资料，未随仓库发布） | 长短输出混合的请求,一次性提交(vLLM 真实的 continuous batching)应明显快于"按固定大小分块、每块等全部做完再发下一块"的静态批处理基线;构造中的长短分布会决定加速范围(因为每块都混入一个长请求拖累整块) | 看 `speedup`(场景 B 耗时 / 场景 A 耗时);这正是文档 [6.2 节](rl-training-workflow.md#62-rollout-长尾)里 RL rollout 长尾问题在单机 GPU 调度层面的缩影 |

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
