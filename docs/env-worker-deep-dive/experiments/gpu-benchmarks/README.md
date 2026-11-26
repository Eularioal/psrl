# GPU benchmark 实验导航

按研究问题选择报告。任务定义、评分、command 与 episode 的选择和运行步骤统一查[接入主文档](../../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)。

<!-- figure:map-66cda102ff -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![GPU benchmark 实验导航：阅读地图（1/2）](../../../_static/img/document-figures/map-66cda102ff-screen.png)](../../../_static/img/document-figures/map-66cda102ff.png)

[![GPU benchmark 实验导航：阅读地图（2/2）](../../../_static/img/document-figures/map-66cda102ff-screen-02.png)](../../../_static/img/document-figures/map-66cda102ff.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../../_static/img/document-figures/map-66cda102ff.svg)。
<!-- /figure:map-66cda102ff -->

## 按研究问题阅读

| 想研究的问题 | 先读哪份报告 |
|---|---|
| 显存不够时，提前排队是否比失败重试更快？ | [常驻模型压力](06-显存压力_提前排队与失败重试.md)：显存压力、失败重试与完整完成时间 |
| 模型所在GPU忙时，等待还是换卡加载？ | [热卡等待与冷卡加载](04-动态放置_热卡等待与冷卡加载对照.md)：实测选择过程、时间和显存代价；[代码指南](../../knowledge/02-资源与调度/09-动态放置_热卡等待与冷卡重建代码.md)讲实现 |
| 在两张卡之间怎样兼顾模型复用和排队？ | [联合放置探索](03-动态放置_真实模型驻留与跨卡探索.md)：同执行器内比较策略 |
| 评分是否能复用模型，省掉反复加载？ | [PostTrain评分复用](05-评分复用_PostTrain准备成本与整批耗时.md)：请求变快，但完整短批次还未胜过普通进程复用 |
| 同卡竞争会不会让算子速度测错？ | [性能反馈压力](08-性能反馈_高压力调度实验.md)：解释如何误选更慢的候选及保护手段 |
| 多个候选能否一起评估？ | [FlashInfer批量与分段](07-FlashInfer_进程复用与任务合批.md)：进程复用、筛选、分段分别测量 |

## 接入与证据入口

- [任务覆盖与路线图](../../milestones/02-Science现状_任务覆盖与接入路线.md)：逐题库状态与缺口。
- [接入与验收详情](../../milestones/03-Science验收_接入流程与环境隔离.md)：执行链、隔离和历史验收。
- [接入结果](01-任务验收_结果与证据导航.md)：实际工作量、失败和清理证据。
- [固定状态审计](onboarding-status-audit-20261116.json)：2026-11-16 的题目快照。

运行实验前按[集群预检](../../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)核对现场条件，历史结果不表示当前资源空闲。

## 目录补充导航

- [实验图表](figures/README.md)
- [FlashInfer 实验依赖](flashinfer-env/README.md)
- [辅助脚本](scripts/README.md)
- [状态复用：等待与重新加载的前期实验](02-状态复用_等待与重新加载的前期实验.md)
