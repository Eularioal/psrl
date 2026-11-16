# Env 实验脚本与证据目录

> **核心结论**：本目录保存“怎样复现”和“当时发生了什么”。理解实验结论先读[实验入口](../experiments/README.md)，了解代码现状读[当前设计](../knowledge/current-design.md)。脚本、教学演示与真实GPU结果在各报告中分别说明。

## 目录

- [按目的阅读](#按目的阅读)
- [目录与运行约束](#目录与运行约束)

## 按目的阅读

- [有效实验结论](../experiments/env-design-and-results-brief.md)：固定基线收益、功能闭环与研究反例分别列出。
- [当前设计](../knowledge/current-design.md)：预算持有、执行后端与生命周期的现有实现。
- [常驻模型的计算共享](../experiments/resident-compute-sharing.md)：`scripts/resident_sharing_study/` 提供真实 vLLM、command 租约与图像训练的有界三策略对照，分析脚本只读取证据。
- [主存受限的状态调度研究](../research-proposals/memory-bounded-state-scheduling.md)：`state_reuse_study/memory_*.py` 的教学入口，包含 24 项检查的推进方式、3/5 GiB 主存缓存、真实 GPU 条件及研究边界。
- [真实 benchmark 状态核查](../research-proposals/benchmark-state-reuse-audit.md)：`swe_state_audit.py` 扫描固定原题，`candidate_source_version_probe.py` 在真实 RE 入口比较旧字节码误读与修复，均不启动 GPU。
- [PostTrain 评分准备实验](../experiments/gpu-benchmarks/posttrain-verifier-reuse.md)：`scripts/posttrain_reuse_study/` 提供有界真实 GPU 对照、跨组汇总与独立 CPU 教学脚本，严格区分三者。
- [早期验证档案](../experiments/history/env-service-verification-20261013.md)：保留 2026-10-13 之前的失败、修复及原始条件，不代表后续训练、GPU 或原生后端仍未完成。

## 目录与运行约束

[scripts](scripts) 保存有界复现与只读分析脚本，[results](results) 保存报告引用的结果。原始 JSON、日志及固定源码快照不随文档清理改写。

需要重跑时先读[集群预检](../../psrl/psrl-run-guide/cluster-preflight.md)，按当前节点核对资源并设置预算。每个专题分别记录实际运行范围，索引日期不代表所有脚本都已重新验证。
