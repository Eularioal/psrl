# 文献调研与论文精读

> **核心结论**：这里按研究问题选择论文。先想清楚自己要减少哪种等待，再读对应机制；每篇区分作者的方法与证据、教学例子，以及对 PSRL 的分析建议。


## 目录

- [阅读入口](#阅读入口)
- [文档](#文档)

## 阅读入口

刚接触项目时先读[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)。想理解已有工作与本地研究的关系，可从 RollArt / DynaRL 精读开始；想选真实实验任务，则先看 GPU 工作负载调研。研究进展统一查[提案索引](../research-proposals/README.md)。

八篇正文现在都在相关章节配有作者原图或官方架构图，图后解释英文标签、阅读方向、关键动作及与PSRL的联系。优先看[五篇综述中的系统原图](01-调度论文_五篇近期顶会工作综述.md#3-rollart环境长尾与轨迹级解耦)、[KREX的保护顺序](03-论文精读_KREX共享GPU测量保护.md#3-怎样建立真正可测量的独占窗口)和[Firepool的容器交接](08-基础设施_RSI与GPU沙盒及模型接管.md#firepool-场景与-psrl-三项能力对照)。[原图目录与来源](../../_static/img/literature-reviews/README.md#论文与官方原图)集中保存图号、版本、PDF页码和本地副本。正文中的教学算例继续明确标注，与作者原图分别阅读。

[![先辨认等待发生在哪里，再选择对应论文（1/3）](../../_static/img/document-figures/explained-reading-map-0eb10e0-screen.png)](../../_static/img/document-figures/explained-reading-map-0eb10e0.png)

[![先辨认等待发生在哪里，再选择对应论文（2/3）](../../_static/img/document-figures/explained-reading-map-0eb10e0-screen-02.png)](../../_static/img/document-figures/explained-reading-map-0eb10e0.png)

[![先辨认等待发生在哪里，再选择对应论文（3/3）](../../_static/img/document-figures/explained-reading-map-0eb10e0-screen-03.png)](../../_static/img/document-figures/explained-reading-map-0eb10e0.png)

<!-- explained-source:explained-reading-map-0eb10e0 -->
**读图结论：**先确定待解决的问题，再借用对应机制，不能从论文名字直接跳到本地收益。 阅读地图用于导航，完整方法和证据仍在各篇精读中连续解释。 [完整解释图（SVG）](../../_static/img/document-figures/explained-reading-map-0eb10e0.svg) · [原始机制图或实验图](../../_static/img/literature-reviews/reading-map.png)。
<!-- /explained-source:explained-reading-map-0eb10e0 -->

图中四列对应不同问题。例如，A 等待模型回复时不使用 GPU，B 正在等卡，这是借用空档的问题；A、B 同时运行后，计时评分发生变化，则还要处理反馈可信度。先把问题分清，才知道该借鉴哪一种机制。

建议沿一条主线阅读：先用[五篇论文的 A、B 时间线](01-调度论文_五篇近期顶会工作综述.md#8-一个贯穿五篇论文的例子)理解为什么存在调度空间，再读 [RollArt / DynaRL](02-论文精读_RollArt与DynaRL的Agent调度.md)区分轨迹推进与资源重分配，最后用[任务调研](05-基准调研_GPU任务类型与接入顺序.md)寻找真实触发场景。涉及性能评分时，再进入 KREX 与 AI4Infra 两篇；想研究决策信息或模型参与方式时，再读 YARN 与模型调度大脑。

正文还配有自绘的局部时间线、资源布局和状态图。例如RollArt / DynaRL用同一时间轴区分整批结束与流水线吞吐，RSI用CPU准备与GPU计算的对照解释卡上空等。作者原图用于对照实际方法，自绘教学图用于展开算例。两类图片均以屏幕版显示，点击打开完整图；[章节局部图集](../../_static/img/document-figures/README.md#长文章节局部图)和[论文原图与教学图集](../../_static/img/literature-reviews/README.md)保留来源与复现方法，教学数字不计作实测结果。

## 文档

| 你想解决的问题 | 文档 | 阅读重点 |
|---|---|---|
| 先建立整个研究领域的认识 | [五篇系统论文](01-调度论文_五篇近期顶会工作综述.md) | 环境长尾、动态分配、工作流、显存弹性和空闲窗口 |
| 模型与工具互相等待怎样解耦 | [RollArt 与 DynaRL 精读](02-论文精读_RollArt与DynaRL的Agent调度.md) | 完整流程、数字例子、资源迁移与本地已有能力 |
| 共享 GPU 怎样可信测速 | [KREX 精读](03-论文精读_KREX共享GPU测量保护.md) | 测量区域、阻止提交、设备排空和 CPU 保护 |
| 哪些任务适合检验 Env 调度 | [GPU 工作负载调研](05-基准调研_GPU任务类型与接入顺序.md) | [按场景选 bench](05-基准调研_GPU任务类型与接入顺序.md#按场景选择新增-benchmark)、Speedrunner 的完整训练与多卡接入、生成和仿真候选 |
| 性能作为奖励时怎样设计实验 | [AI4Infra benchmark](06-基准调研_AI4Infra性能反馈与验证.md) | 反馈是否失真、任务覆盖与最小验证 |
| 资源层与应用层应怎样分工 | [YARN 的调度启发](04-系统借鉴_YARN资源分配与反馈调度.md) | 分层协调、测量比较关系和状态保留 |
| 大模型能为调度提供什么新信息 | [模型调度大脑](07-智能调度_模型决策的已有工作与研究选择.md) | 在线决策、语义提取、离线策略生成和强基线 |
| infra Agent 训练中的 GPU 沙盒怎样调度 | [RSI 基础设施与 PSRL](08-基础设施_RSI与GPU沙盒及模型接管.md) | CUDA Agent、Firepool、[PSRL 三项能力与研究边界](08-基础设施_RSI与GPU沙盒及模型接管.md#firepool-场景与-psrl-三项能力对照)，以及模型接管扩展 |

例如，只想知道“同一段 GPU 代码在有竞争时会不会被评错”，直接读 KREX 的算例，再按文章末尾的原文定位深入机制。论文发表信息和产品现状以各文注明的核对日期为准，RSI 调研另含 2026-11-18 的 Firepool、CUDA Agent 与本地能力复核。2026-11-19 的图文整理保留这些历史条件，新增配图和解释不自动表示重跑实验或重新审核全部文献。具体范围见[图文可读性复核](../../tools/文档维护/01-文档组织_阅读结构与维护规范.md#图文可读性复核)。

[返回专题入口](../README.md) · [返回总索引](../../README.md)
