# EnvRouter 低负载配对实验与 Gate A/B 设计审视

> **最新修复状态（2026-10-27）**：资源遥测和清理交接见[第 13 节](router-paired-latency-and-design-review/remediation-and-handoff.md#13-物理资源遥测与清理交接确认)。关于“重派发尚未完成”，应准确理解为同请求跨 worker 转移协议未完成，框架早已有同 worker 重试和整组补采。新的[Worker 恢复专题](../knowledge/worker-retry-and-redispatch.md)核对了固定 baseline 中的实现，并记录本轮执行监督、熔断修复和真实 Ray/Docker 故障验证。这次 worker 改动尚未提交、推送或合并。第 1 至 13 节保留对应阶段的审计与实验条件。

> **核心结论**：当前“会话资源准入 + 命令阶段 GPU 预算 + 确认归还”的方向适合受控的多轮任务。Gate A 的预算拦截、共享准入和高压力下减少失败已有真实证据。旧 Gate B 是基于遥测的软节流，没有证明它能保证计算互斥或奖励归因。当前命令模式主动跳过旧 Gate B，不能将这条新路径的收益记到 Gate B 上。
>
> 本文对照用户指定的原始设计稿，区分原提案、当前实现、历史证据、配对实测和 CPU 教学输入。低负载数值来自此前同期 OFF → ON → ON → OFF 两对实验，见第 2 节。2026-10-21 新增第 9、10 节源码审计：统一后端、健康重派发与活动归因仍未闭环；原生 CUDA science 题尚需适配终端工具的真实 GPU 作业生命周期。本次没有运行模型训练，也没有改动被测生产代码。

## 专题导航

已拆分为 2 篇子文档。按完整章节迁移，未重新运行历史实验。

- [1. 原设计与当前执行链](router-paired-latency-and-design-review/paired-results-and-review.md)
- [11.1 已修复及不能继续使用的旧结论](router-paired-latency-and-design-review/remediation-and-handoff.md)
