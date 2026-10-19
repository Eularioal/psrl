# vLLM 推理与训练并行专题索引

> **核心结论**：原 1848 行长文已拆分为 5 篇，分别覆盖推理机制、训练并行、PSRL 训推流程、训练损失与实验速查。

## 目录

- [阅读路径](#阅读路径)
- [专题文档](#专题文档)
- [迁移说明](#迁移说明)

## 阅读路径

先读推理机制与训练并行，再进入 PSRL 训推流程和损失细节；查公式或历史实验时直接进入速查专题。

## 专题文档

- [vLLM 推理机制：缓存、调度与合批](vllm-and-training-parallelism/inference-mechanisms.md)：从 prefill/decode 的计算差异出发，理解 KV 缓存、PagedAttention、前缀复用、连续合批和选择性合批；原有算例与性能边界保留。
- [训练并行、通信量与带宽分析](vllm-and-training-parallelism/training-parallelism.md)：按切分对象理解训练并行，再用每卡通信量和互联带宽判断并行策略应如何排布。
- [PSRL 训推协作与 PPO、GRPO 基础](vllm-and-training-parallelism/rl-training-workflow.md)：串联轨迹生成、概率重算、优势估计、参数更新与权重同步，并保留 PPO/GRPO 的逐步推导和长尾分析。
- [PSRL 分组采样、DAPO 与训练损失](vllm-and-training-parallelism/rl-loss-and-sampling.md)：解释同一 prompt 的分组采样、DAPO 改动、熵与 KL 项，以及多轮 response_mask、奖励标签和 rollout correction。
- [推理与训练并行：公式速查及实验记录](vllm-and-training-parallelism/experiments-and-reference.md)：集中查阅符号、公式、机制对应的实验脚本与历史测量条件；历史结果不代表当前环境的重新验证。

## 迁移说明

2026-10-14 已按主题拆分为 5 篇子文档；本文件保留为索引。原有正文、例子、公式、代码块和历史实验记录已迁入对应专题，避免在索引中重复。已修复工作区内的章节引用；旧章节位置可通过下列对照表查找。

| 原章节 | 新位置 |
|---|---|
| 一、vLLM 的核心机制 | [阅读章节](vllm-and-training-parallelism/inference-mechanisms.md#一vllm-的核心机制) |
| 二、训练侧的并行维度 | [阅读章节](vllm-and-training-parallelism/training-parallelism.md#二训练侧的并行维度) |
| 三、Megatron 的具体切分方式 | [阅读章节](vllm-and-training-parallelism/training-parallelism.md#三megatron-的具体切分方式) |
| 四、通信量定量分析 | [阅读章节](vllm-and-training-parallelism/training-parallelism.md#四通信量定量分析) |
| 五、带宽的硬约束与策略排布 | [阅读章节](vllm-and-training-parallelism/training-parallelism.md#五带宽的硬约束与策略排布) |
| 六、RL 后训练场景（PSRL） | [阅读章节](vllm-and-training-parallelism/rl-training-workflow.md#六rl-后训练场景psrl) |
| 七、符号表与常用公式速查 | [阅读章节](vllm-and-training-parallelism/experiments-and-reference.md#七符号表与常用公式速查) |
| 八、机制 → 实验验证设计 | [阅读章节](vllm-and-training-parallelism/experiments-and-reference.md#八机制--实验验证设计) |
| Selective batching：投影按 token 合批，attention 按请求隔离 | [阅读章节](vllm-and-training-parallelism/inference-mechanisms.md#selective-batching投影按-token-合批attention-按请求隔离) |
| 延伸问题 | [阅读章节](vllm-and-training-parallelism/experiments-and-reference.md#延伸问题) |
| 相关文档 | [阅读章节](vllm-and-training-parallelism/experiments-and-reference.md#相关文档) |
| 参考 | [阅读章节](vllm-and-training-parallelism/experiments-and-reference.md#参考) |

本次为文档结构整理，不将历史实验视为当前验证，也未启动训练或集群实验。
