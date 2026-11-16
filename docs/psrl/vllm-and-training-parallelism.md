# vLLM 推理与训练并行专题索引

> **核心结论**：专题按 5 个阅读任务组织：推理机制、训练并行、PSRL 训推流程、训练损失与实验速查。缓存的解释和算例集中在推理机制一篇中。

## 目录

- [阅读路径](#阅读路径)
- [专题文档](#专题文档)
- [迁移说明](#迁移说明)

## 阅读路径

想理解 PSRL 的进程分工、数据流和版本协调，直接读训推流程；遇到显存、合批或通信问题，再查推理机制与训练并行。PPO、GRPO 与损失推导供需要核对算法细节时选读，历史测量集中在速查专题。

## 专题文档

- [vLLM 推理机制：缓存、调度与合批](vllm-and-training-parallelism/inference-mechanisms.md)：从 prefill/decode 的计算差异出发，解释 KV 缓存、PagedAttention、连续合批，并用三分支例子展开块表、写时复制与缓存回收。
- [训练并行、通信量与带宽分析](vllm-and-training-parallelism/training-parallelism.md)：按切分对象理解训练并行，再用每卡通信量和互联带宽判断并行策略应如何排布。
- [PSRL 训推协作与版本协调](vllm-and-training-parallelism/rl-training-workflow.md)：串联轨迹生成、概率重算、优势估计、参数更新与权重同步，并保留 PPO/GRPO 的逐步推导和长尾分析。
- [PSRL 分组采样、DAPO 与训练损失](vllm-and-training-parallelism/rl-loss-and-sampling.md)：解释同一 prompt 的分组采样、DAPO 改动、熵与 KL 项，以及多轮 response_mask、奖励标签和 rollout correction。
- [推理与训练并行：公式速查及实验记录](vllm-and-training-parallelism/experiments-and-reference.md)：集中查阅符号、公式、机制对应的实验脚本与历史测量条件；历史结果不代表当前环境的重新验证。

## 迁移说明

本页提供 5 个用途明确的专题入口，正文、例子、公式、代码块和历史实验各随所属主题维护。缓存管理与推理机制合并，避免理解同一问题时切换文件。旧章节位置可通过下列对照表查找。

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

推理正文按生成过程解释，公式与实现细节留在对应专题。2026-11-16 核对部分本地源码与官方说明，历史测量保留原条件，未启动训练或 GPU 实验。
