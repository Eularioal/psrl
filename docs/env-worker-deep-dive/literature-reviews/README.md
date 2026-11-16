# 文献调研与论文精读

> **核心结论**：这里按研究问题选择论文。先想清楚自己要减少哪种等待，再读对应机制；每篇区分作者的方法与证据、教学例子，以及对 PSRL 的分析建议。


## 目录

- [阅读入口](#阅读入口)
- [文档](#文档)

## 阅读入口

刚接触项目时先读[当前设计](../knowledge/current-design.md)。想理解已有工作与本地研究的关系，可从 RollArt / DynaRL 精读开始；想选真实实验任务，则先看 GPU 工作负载调研。研究进展统一查[提案索引](../research-proposals/README.md)。

## 文档

| 你想解决的问题 | 文档 | 阅读重点 |
|---|---|---|
| 先建立整个研究领域的认识 | [五篇系统论文](recent-top-conference-papers.md) | 环境长尾、动态分配、工作流、显存弹性和空闲窗口 |
| 模型与工具互相等待怎样解耦 | [RollArt 与 DynaRL 精读](rollart-dynarl-deep-reading.md) | 完整流程、数字例子、资源迁移与本地已有能力 |
| 共享 GPU 怎样可信测速 | [KREX 精读](krex-region-exclusivity.md) | 测量区域、阻止提交、设备排空和 CPU 保护 |
| 哪些任务适合检验 Env 调度 | [GPU 工作负载调研](gpu-workload-benchmark-survey.md) | 任务输入、GPU 阶段、评分方式和接入取舍 |
| 性能作为奖励时怎样设计实验 | [AI4Infra benchmark](ai4infra-performance-benchmarks.md) | 反馈是否失真、任务覆盖与最小验证 |
| 资源层与应用层应怎样分工 | [YARN 的调度启发](yarn-inspired-gpu-feedback-scheduling.md) | 分层协调、测量比较关系和状态保留 |
| 大模型能为调度提供什么新信息 | [模型调度大脑](llm-agent-scheduling-brain.md) | 在线决策、语义提取、离线策略生成和强基线 |

例如，只想知道“同一段 GPU 代码在有竞争时会不会被评错”，直接读 KREX 的算例；再按文章末尾的原文定位深入机制。论文发表信息沿用各文注明的核对版本，本轮文档整理没有重新进行完整文献检索。

[返回专题入口](../README.md) · [返回总索引](../../README.md)
