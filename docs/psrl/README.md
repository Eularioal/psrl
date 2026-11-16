# PSRL：从一次做题理解训练框架

这里按 PSRL 的运行链路组织文档，重点说明 Agent、环境、训练和推理进程怎样协作。阅读路径默认已有机器学习与 Agent 基础，系统机制在需要时解释；算法公式和完整推导可按问题进入专题。

## 目录

- [第一次读，从这里开始](#第一次读从这里开始)
- [运行、排查与源码](#运行排查与源码)
- [算法与系统专题](#算法与系统专题)
- [历史故障与深入推导](#历史故障与深入推导)

## 第一次读，从这里开始

1. [Agent Loop 做题例子](psrl-agent-loop-harbor-tool-calling.md#7-用一个修代码例子从零理解)：命令由哪个组件执行，观察怎样回传，TITO 怎样记录训练数据。
2. [一条回答怎样变成模型更新](vllm-and-training-parallelism/rl-training-workflow.md)：采样、训练与权重同步怎样重叠，模型版本差距怎样约束。
3. [Env 当前设计](../env-worker-deep-dive/knowledge/current-design.md)：任务容器怎样申请资源、执行和释放。

如果术语较多，随时查[全局术语表](../overview/glossary.md)。正文会解释理解本节必需的词，专题链接用于继续深入；也可以按其中一个具体步骤继续提问。

## 运行、排查与源码

| 想做什么 | 阅读入口 |
|---|---|
| 准备一次运行 | [基础运行](psrl-run-guide/getting-started.md) → [集群检查](psrl-run-guide/cluster-preflight.md) |
| 理解 Ray 怎样启动角色 | [Ray 与生成长尾](psrl-run-guide/ray-and-rollout.md) |
| 看指标、调批量、找进程 | [监控与批量配置](psrl-run-guide/monitoring-and-batches.md) |
| 从启动脚本读到具体代码 | [训练源码地图](psrl-training-script-source-map.md) |
| 找所有运行专题 | [运行指南索引](psrl-run-guide.md) |

## 算法与系统专题

| 想了解的问题 | 阅读入口 |
|---|---|
| 奖励从哪里来 | [RLVR：可验证奖励](rlvr-explained.md) |
| 一题要采几条回答 | [PPO、GRPO、DAPO 的样本组织](ppo-grpo-dapo-sample-requirements.md) |
| 优势估计怎样综合多步反馈 | [GAE 数值例子](gae-bias-variance-tradeoff.md) |
| 多条生成请求怎样一起运行 | [vLLM 机制](vllm-and-training-parallelism/inference-mechanisms.md) |
| 缓存怎样分页、共享与复制 | [缓存管理详解](vllm-and-training-parallelism/inference-mechanisms.md#kv-缓存管理详解块表写时复制与前缀复用) |
| 模型怎样分给多张卡 | [训练并行](vllm-and-training-parallelism/training-parallelism.md) |
| token 怎样参与训练损失 | [分组采样与损失](vllm-and-training-parallelism/rl-loss-and-sampling.md) |
| 输入、输出和总长度怎样限制 | [上下文长度与模型配置](hf-config-context-window.md) |
| 权重和缓存怎样传输 | [NIXL](nixl-explained.md) |
| 查公式与历史吞吐数据 | [实验与公式速查](vllm-and-training-parallelism/experiments-and-reference.md) |

完整阅读路线见[推理与训练并行索引](vllm-and-training-parallelism.md)。

## 历史故障与深入推导

| 记录 | 保留价值 |
|---|---|
| [启动故障](psrl-run-guide/startup-incident-reviews.md) | IP、数据类型、环境覆盖与上下文配置的真实失败 |
| [首步 backward OOM](../sciaccel-grpo-oom-analysis.md) | 长轨迹、训练分片和五步复测数据 |
| [DAPO 训练退化](dapo-training-collapse-diagnosis.md) | 联合读取质量指标，并链接指标与数值推导两篇深入资料 |
| [EnvWorker 早期设计](psrl-env-service-envworker-design.md) | 2026-09 提案的资源管理动机，现状看当前设计 |
| [Docker 挂起与恢复](psrl-run-guide/incidents/docker-storage-hang-20261020.md) | 线程与存储证据、独立恢复测试及清理结果 |

历史记录保留原版本、日期和证据条件，实际运行前重新检查节点与依赖。[返回文档总览](../README.md)。
