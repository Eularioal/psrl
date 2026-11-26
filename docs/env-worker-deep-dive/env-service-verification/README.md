# Env 实验脚本与证据目录

> **核心结论**：本目录保存“怎样复现”和“当时发生了什么”。理解实验结论先读[实验入口](../experiments/README.md)，了解代码现状读[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)。脚本、教学演示与真实GPU结果在各报告中分别说明。

<!-- figure:map-268b082887 -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![Env 实验脚本与证据目录：阅读地图（1/2）](../../_static/img/document-figures/map-268b082887-screen.png)](../../_static/img/document-figures/map-268b082887.png)

[![Env 实验脚本与证据目录：阅读地图（2/2）](../../_static/img/document-figures/map-268b082887-screen-02.png)](../../_static/img/document-figures/map-268b082887.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../_static/img/document-figures/map-268b082887.svg)。
<!-- /figure:map-268b082887 -->

## 目录

- [按目的阅读](#按目的阅读)
- [目录与运行约束](#目录与运行约束)

## 按目的阅读

- [近期Ray启动故障分诊](../../psrl/psrl-run-guide/03-启动排障_单机与多机故障复盘.md#近期env实验ray启动失败与导入阻塞)：现有日志的注册期限、预启动等待、导入链及确定依赖异常，补充摘录保存在[11月25日只读证据](results/ray-startup-review-20261125/evidence.json)，本次未启动新实例。

- [按九轮实验回顾资源分区](../experiments/09-资源分区_静态卡池与推理余量共享.md#实验总览每一轮验证了什么)：实际持卡模式、相同工作量结果、失败与预测边界。后续主基线统一为静态环境池＋episode 持卡＋满池排队。
- [每卡从两条提高到四条再比较](../experiments/09-资源分区_静态卡池与推理余量共享.md#实验四每卡从两条增至四条后再比较)：先测独立卡并发，再冻结共同上限做四步训练，Roofline 解码拟合放在该节补充说明。

- [有效实验结论](../experiments/01-实验总览_调度收益与证据范围.md)：固定基线收益、功能闭环与研究反例分别列出。
- [当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)：预算持有、执行后端与生命周期的现有实现。
- [常驻模型的计算共享](../experiments/06-训推共享_常驻vLLM与环境并行实验.md)与[长任务对照](../experiments/06-训推共享_常驻vLLM与环境并行实验.md#长生成与-4b-环境模型的-12-组对照)：`scripts/resident_sharing_study/` 提供真实 vLLM、command 租约、图像及 PostTrain 工具的有界三策略对照，分析脚本只读取证据。
- [主存受限的状态调度研究](../research-proposals/05-状态调度_主存受限时的准备状态保留.md)：`state_reuse_study/memory_*.py` 的教学入口，包含 24 项检查的推进方式、3/5 GiB 主存缓存、真实 GPU 条件及研究边界。
- [真实 benchmark 状态核查](../research-proposals/04-状态复用_真实任务中的收益空间核查.md)：`swe_state_audit.py` 扫描固定原题，`candidate_source_version_probe.py` 在真实 RE 入口比较旧字节码误读与修复，均不启动 GPU。
- [PostTrain 评分准备实验](../experiments/gpu-benchmarks/05-评分复用_PostTrain准备成本与整批耗时.md)：`scripts/posttrain_reuse_study/` 提供有界真实 GPU 对照、跨组汇总与独立 CPU 教学脚本，严格区分三者。
- [早期验证档案](../experiments/history/01-历史验证_2026年10月13日EnvService修复.md)：保留 2026-10-13 之前的失败、修复及原始条件，不代表后续训练、GPU 或原生后端仍未完成。

## 目录与运行约束

[scripts](scripts) 保存有界复现与只读分析脚本，[results](results) 保存报告引用的结果。原始 JSON、日志及固定源码快照不随文档清理改写。

需要重跑时先读[集群预检](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)，按当前节点核对资源并设置预算。每个专题分别记录实际运行范围，索引日期不代表所有脚本都已重新验证。

## 目录补充导航

- [辅助脚本](scripts/README.md)
- [原始运行结果](results)：各次执行的日志、遥测和产物，通过报告中的具体证据链接查阅，内部目录不批量改写。
