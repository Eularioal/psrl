# 全文档配图：彩图、来源与复现

本目录保存知识正文和阅读导航的解释图。图中区分教学示意、历史数据重绘与阅读地图，具体机制、实现状态及实验条件仍在对应正文维护。

<!-- figure:raster-vector-source -->
这张图沿同一个场景解释：PNG、SVG 与绘图脚本分别解决显示、编辑还是复现的问题？

[![配图资源：PNG负责直接看，SVG和脚本负责放大与维护（1/2）](raster-vector-source-screen.png)](raster-vector-source.png)

[![配图资源：PNG负责直接看，SVG和脚本负责放大与维护（2/2）](raster-vector-source-screen-02.png)](raster-vector-source.png)

同一幅解释图既要在 Markdown 里直接显示，也要能修改和追溯数字来源。 显示文件不是全部来源。保留源脚本与原始证据，才能避免下一轮只剩无法核对的截图。 SVG 内嵌原始栅格面板时，该面板不会自动变成矢量。本轮证据图明确保留这一来源。 [查看矢量图](raster-vector-source.svg)。
<!-- /figure:raster-vector-source -->

## 目录

- [所有解释图的屏幕显示](#所有解释图的屏幕显示)
- [Env现状的屏幕版](#env现状的屏幕版)
- [新版章节解释图](#新版章节解释图)
- [长文章节局部图](#长文章节局部图)
- [查看与复现](#查看与复现)
- [机制与操作图](#机制与操作图)
- [历史实验图](#历史实验图)
- [目录阅读地图](#目录阅读地图)
- [架构文献与实验的完整解释图](#架构文献与实验的完整解释图)
- [渲染复核](#渲染复核)
- [文件与子目录](#文件与子目录)

## 所有解释图的屏幕显示

本目录187张自绘或历史证据解释图都提供屏幕版，覆盖201篇文档的310处登记引用。长图拆成连续编号的小图，共498张，每张800像素宽、高度不超过600像素。文档按编号依次显示，点击任一分图都能打开对应完整图。

对应论文的作者原图另在[论文原图目录](../literature-reviews/README.md#论文与官方原图)维护，共22张、24张屏幕分图，配有中文读图说明。它们按来源清单独立登记，既有187张图及其复核记录保留原核对范围。

[打开可按主题筛选的屏幕版图集](screen-gallery.html)。下方按主题分类的目录先链接屏幕版首图，完整内容在对应正文与图集中连续阅读。Env四张常用回顾图另列在下一节。

机制图沿原有面板划分，目录图保留阅读入口；实验图保留曲线、坐标、数字和对照条件。只压缩完全空白的纵向间距，图例与说明单独保持可读。图例过小而绘图区较高时，先分离图例，再等比例缩放绘图区，避免整图一起缩小。

五张历史条形图与两张固定完成量实验图按已有数值重排，改善标签边距和子图分隔；Roofline的上界图与合批曲线分开显示，保留坐标和教学假设。这些是显示调整，没有新增实验。

完整PNG、SVG和66份原始证据不变。屏幕版SVG有两种：四张已有回顾图保留对应矢量布局，其他分图以PNG嵌入SVG保存显示布局，修改对象和数据时仍使用完整SVG及原始绘图脚本。[布局与来源记录](screen-layout.json)保留源图哈希、面板位置和分页顺序，[屏幕版复核](screen-review.json)记录本轮实际检查范围。498张屏幕分图是187张解释图的显示形式，不增加独立图或实验的数量。

在已安装Pillow、NumPy、CairoSVG、Matplotlib并具备中文字体的Python环境中，从仓库根目录运行：

```bash
python -I docs/_static/img/document-figures/screen.py
python -I docs/_static/img/document-figures/check.py
```

只重绘一张时给`screen.py`传入`--only 图ID`。`render.py --only 图ID`会先生成完整图，再同步屏幕版、分页清单与图集；面板数量改变时，还需同步正文中对应的全部分图引用。检查器同时核对屏幕大小、完整原图哈希及每篇正文是否嵌入全部分图。

## Env现状的屏幕版

日常回顾先打开下列小图，每张800像素宽、不高于600像素。模式与资源边界图重新组织为短图，另外两张按可读字号缩放。正文优先显示屏幕版，点击图片进入完整图；完整图的PNG、SVG、教学条件和细节仍保留。

| 回顾内容 | 屏幕版 | 尺寸 | 完整图 |
|---|---|---|---|
| 组件、主机资源与GPU卡池 | [PNG](host-and-gpu-ledgers-screen.png) / [SVG](host-and-gpu-ledgers-screen.svg) | 800 × 498 | [PNG](host-and-gpu-ledgers.png) |
| episode、command与auto | [PNG](command-episode-residency-screen.png) / [SVG](command-episode-residency-screen.svg) | 800 × 480 | [PNG](command-episode-residency.png) |
| 独立卡优先、显式共享卡后备 | [PNG](chapter-gpu-priority-screen.png) / [SVG](chapter-gpu-priority-screen.svg) | 800 × 578 | [PNG](chapter-gpu-priority.png) |
| 普通推理共运行与控制操作等待 | [PNG](chapter-sharing-control-screen.png) / [SVG](chapter-sharing-control-screen.svg) | 800 × 578 | [PNG](chapter-sharing-control.png) |

这四张适合日常快速回顾；完整图中的分面板细节仍从原图阅读。其他主题的屏幕分图、来源与生成方法统一在上一节维护。

## 新版章节解释图

当前图集共有187张独立解释图，在201篇维护中文档中引用310次。此前152张图完成全量排版；本轮进一步为24篇长文补充35张局部图，新增65个引用位置，并改写或删除与图重复的段落。以下八张是机制讲解的代表，完整图表在后文分类列出。每张围绕一个具体阅读任务组织，把起始条件、对象或资源布局、策略差异、状态变化和数值结论放在连续的阅读路径中。编号说明从哪里开始读，颜色和刻度在同一图内保持一致。信息量以能回答哪些问题衡量，避免靠增加名词和小字获得表面密度。

| 主图 | 单看图与图注应能回答什么 |
|---|---|
| [显存准入：从漏算风险到安全放行](memory-promises-screen.png)（[全部4张](screen-gallery.html#memory-promises)） | 实占与预约为什么不能直接取总量的max？<br>为什么也不能把实占与预约全部相加？<br>新公式怎样避免漏算和重复计算？<br>B何时等待，何时放行？ |
| [动态放置：等热卡，还是在冷卡重建？](wait-or-reload-screen.png)（[全部4张](screen-gallery.html#wait-or-reload)） | 两张卡分别保留什么状态？<br>等待与换卡的时间从哪里开始算？<br>为什么空闲卡也可能不值得使用？<br>A提前7秒为什么整批只提前4秒？ |
| [GPU借还：A多等了，整批为什么仍可能更快？](borrower-return-delay-screen.png)（[全部4张](screen-gallery.html#borrower-return-delay)） | A和B的角色及计时起点是什么？<br>为什么额外等待要用max取零下界？<br>A变慢与整批加速能否同时出现？<br>借用结束后为什么还要确认物理归还？ |
| [资源边界：会话留在主机，GPU按命令申请](host-and-gpu-ledgers-screen.png) | unit与会话资源是什么关系？<br>会话层与GPU命令层分别记录什么？<br>换卡时主存和容器是否一起移动？<br>热模型和跨节点迁移有哪些边界？ |
| [状态复用：保住昂贵准备，更新已变化的代码](state-version-match-screen.png)（[全部4张](screen-gallery.html#state-version-match)） | 写磁盘文件为何不自动更新进程？<br>哪些依赖对应哪些内存状态？<br>为什么只改评分代码时可以保留模型？<br>什么条件下拒绝复用或拒绝发布结果？ |
| [持卡模式：容器常驻，为什么模型仍要重载？](command-episode-residency-screen.png) | episode与command改变的是哪一段资源寿命？<br>容器保留为什么不等于Python模型保留？<br>B占用空档为何会让A再次等待？<br>auto的能力与时间预测分别怎样参与选择？ |
| [GPU并发：装得下，不等于一起跑就划算](gpu-capacity-and-compute-screen.png)（[全部4张](screen-gallery.html#gpu-capacity-and-compute)） | 显存容量和SM计算能力分别限制什么？<br>MPS能否保证任务互不减速？<br>哪些条件下10并发胜过5加5，哪些条件下相反？<br>应该测哪些指标才能决定是否限流？ |
| [Roofline：从硬件上界走到并发调度](roofline-capacity-decision-screen.png)（[全部4张](screen-gallery.html#roofline-capacity-decision)） | 计算和搬运为何同时给出时间下界？<br>Roofline的坐标、斜线和平台分别是什么？<br>合批如何提高权重搬运的复用程度？<br>硬件拐点为何还需要映射到真实服务曲线？ |

图解分为49张机制与操作主图、35张章节局部图、5张历史表格重绘、32张目录地图，以及66张保留原始机制或实验面板的完整解释图。后者在原始证据周围补充本图问题、条件、动作、结果与边界；原文件保留用于追溯。各图要回答的三至六个问题保存在[catalog.json](catalog.json)的`explanation_review`中，逐图替换对应关系见[全量清单](upgrade-manifest.json)。

以显存准入为例，自上而下按①至④读：先区分外部占用和已给A的承诺，再看旧算法如何错误放行B，接着看新公式怎样只补未兑现部分，最后判断B何时可以获准。图既呈现状态，也给出导致决定的算术。

<!-- figure:memory-promises -->
这张图沿同一个场景解释：为什么显存看起来有空余，调度器仍要让新任务等待？

[![显存准入：从漏算风险到安全放行（1/4）](memory-promises-screen.png)](memory-promises.png)

[![显存准入：从漏算风险到安全放行（2/4）](memory-promises-screen-02.png)](memory-promises.png)

[![显存准入：从漏算风险到安全放行（3/4）](memory-promises-screen-03.png)](memory-promises.png)

[![显存准入：从漏算风险到安全放行（4/4）](memory-promises-screen-04.png)](memory-promises.png)

沿①初始条件、②旧算法反例、③逐任务记账、④放行条件阅读。90 GiB容量、外部70 GiB、A和B各预约15 GiB是教学算例。A实占改变时剩余预约随之改变，总承诺仍为85 GiB；确认物理归还后才清除预约。 [查看矢量图](memory-promises.svg)。
<!-- /figure:memory-promises -->

## 长文章节局部图

总览回答整篇在解决什么，局部图回答这一段究竟发生了什么。以下35张图以900像素为设计宽度，使用同轴时间条、显存分区、队列变化、状态依赖、内存块和批次网格。它们就地插入24篇架构、资源调度、论文、训推和研究文档的65处，避免读者只在开头见到一张图，随后连续阅读大量机制文字。

例如，[固定30项任务的并发图](chapter-finite-concurrency-screen.png)同时给出并发数、每波耗时、波数和整批结束时间；[文件与进程版本图](chapter-code-versions-screen.png)直接画出磁盘已变成v2、进程仍持有v1的矛盾。正文补充原因与适用条件，移除重复复述同一过程的表格或段落。

| 要回答的问题 | 可直接查看与编辑 | 正文入口 |
|---|---|---|
| 先让谁用 GPU？要看它能解锁什么工作 | [PNG](chapter-critical-path-screen.png)（[全部2张](screen-gallery.html#chapter-critical-path)） / [SVG](chapter-critical-path.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 看见 16 GiB 空闲，为什么只能再放 4 GiB？ | [PNG](chapter-memory-headroom-screen.png) / [SVG](chapter-memory-headroom.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 借用空档：要装下整段交接，而不只是环境计算 | [PNG](chapter-idle-window-screen.png) / [SVG](chapter-idle-window.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 独立卡优先；显式共享卡只做后备 | [PNG](chapter-gpu-priority-screen.png) / [SVG](chapter-gpu-priority.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 普通生成可以共运行，模型控制操作要等环境归还 | [PNG](chapter-sharing-control-screen.png) / [SVG](chapter-sharing-control.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 卡归还了，积压请求为什么还没消失？ | [PNG](chapter-replica-backlog-screen.png) / [SVG](chapter-replica-backlog.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 并发更大、单条更慢：整批究竟快不快？ | [PNG](chapter-finite-concurrency-screen.png) / [SVG](chapter-finite-concurrency.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 只有 12 条待办，10 并发的吞吐最好也未必该选 | [PNG](chapter-tail-batch-screen.png) / [SVG](chapter-tail-batch.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 一个“并发数”不够：会话、准备、GPU 要分别看 | [PNG](chapter-three-limits-screen.png) / [SVG](chapter-three-limits.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/04-GPU作业_生命周期与任务遥测.md) |
| 一个模型实例接很多请求，不等于很多模型实例 | [PNG](chapter-instances-requests-screen.png) / [SVG](chapter-instances-requests.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 预测不该只报一个秒数，还要告诉调度器可能差多少 | [PNG](chapter-prediction-intervals-screen.png) / [SVG](chapter-prediction-intervals.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 一条轨迹结束以后，训练还在等什么？ | [PNG](chapter-rollout-pipeline-screen.png) / [SVG](chapter-rollout-pipeline.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 采样与训练重叠，省掉的是哪段等待？ | [PNG](chapter-pipeline-overlap-screen.png) / [SVG](chapter-pipeline-overlap.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md) |
| 八张卡怎样分：资源应该帮助较慢的那一段 | [PNG](chapter-resource-reallocation-screen.png) / [SVG](chapter-resource-reallocation.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md) |
| 文件已经改了，进程为什么还会跑旧代码？ | [PNG](chapter-code-versions-screen.png) / [SVG](chapter-code-versions.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/07-状态复用_基线状态寿命与版本正确性.md) |
| 改了哪里，就重建依赖它的状态 | [PNG](chapter-state-dependencies-screen.png) / [SVG](chapter-state-dependencies.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/07-状态复用_基线状态寿命与版本正确性.md) |
| CPU 先准备好，再占 GPU，能省掉哪段卡上空等？ | [PNG](chapter-prepare-gpu-screen.png) / [SVG](chapter-prepare-gpu.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md) |
| 重复测得很稳定，也可能稳定地选错代码 | [PNG](chapter-measurement-ranking-screen.png) / [SVG](chapter-measurement-ranking.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md) |
| 程序只算了 1 ms，为什么请求要等 24 ms？ | [PNG](chapter-local-service-time-screen.png) / [SVG](chapter-local-service-time.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md) |
| 谁分资源，谁懂任务：中央层和应用层怎样配合？ | [PNG](chapter-resource-layers-screen.png) / [SVG](chapter-resource-layers.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/04-系统借鉴_YARN资源分配与反馈调度.md) |
| “已经预热”到底保留了哪一层？ | [PNG](chapter-warm-layers-screen.png) / [SVG](chapter-warm-layers.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/06-训推借卡_Rollout睡眠与归还协议.md) |
| 父任务等子任务，为什么可能谁也走不下去？ | [PNG](chapter-nested-deadlock-screen.png) / [SVG](chapter-nested-deadlock.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md) |
| 工具里训出模型，不等于它已成为下一代 Agent | [PNG](chapter-model-promotion-screen.png) / [SVG](chapter-model-promotion.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md) |
| 另一台机器有空卡，为什么这组任务还在本机排队？ | [PNG](chapter-grouped-placement-screen.png) / [SVG](chapter-grouped-placement.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) |
| 结果已经打印出来，卡为什么还不能交给下一条命令？ | [PNG](chapter-gpu-return-states-screen.png) / [SVG](chapter-gpu-return-states.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/04-GPU作业_生命周期与任务遥测.md) |
| 投影可以合批，attention 仍要记住每条请求的边界 | [PNG](chapter-selective-batching-screen.png) / [SVG](chapter-selective-batching.svg) | [对应正文](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md) |
| 三条请求共享前缀，继续写时为什么要复制最后一块？ | [PNG](chapter-kv-copy-on-write-screen.png) / [SVG](chapter-kv-copy-on-write.svg) | [对应正文](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md) |
| 连续合批：一条完成，就让新请求补进下一轮 | [PNG](chapter-continuous-batching-screen.png) / [SVG](chapter-continuous-batching.svg) | [对应正文](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md) |
| 切开模型以后，省下显存也增加了搬运 | [PNG](chapter-parallel-communication-screen.png) / [SVG](chapter-parallel-communication.svg) | [对应正文](../../../psrl/推理与训练/02-训练并行_模型切分与通信带宽.md) |
| 整条回答只有最终得分，前面的 token 怎样得到训练信号？ | [PNG](chapter-ppo-credit-screen.png) / [SVG](chapter-ppo-credit.svg) | [对应正文](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md) |
| 下一条命令换卡，容器和主存需要一起搬吗？ | [PNG](chapter-gpu-pool-host-screen.png) / [SVG](chapter-gpu-pool-host.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) |
| MPS 提供共运行机会，不会把 GPU 计算能力翻倍 | [PNG](chapter-compute-sharing-screen.png) / [SVG](chapter-compute-sharing.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/04-GPU作业_生命周期与任务遥测.md) |
| 保留进程与模型，究竟省的是哪段时间？ | [PNG](chapter-process-reuse-screen.png) / [SVG](chapter-process-reuse.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) |
| 知道还剩多少小步，才比较容易估计结束时间 | [PNG](chapter-remaining-steps-screen.png) / [SVG](chapter-remaining-steps.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/08-命令预测_适用条件与时长估计.md) |
| 8 条尝试都在推进，不等于 8 个环境都已获准 | [PNG](chapter-episode-admission-screen.png) / [SVG](chapter-episode-admission.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) |

这些算例用于讲清机制，不是新增实测。各图的读者问题、文档改写范围、前后图片分布、哈希和实际查看记录保存在[局部图复核记录](chapter-readability-review.json)。

## 查看与复现

当前图集的187张图覆盖201篇维护中的文档、310个引用位置。相同概念在适合的段落复用；原始输出、冻结baseline、缓存与构建产物通过所属目录导航。完整对应关系见[catalog.json](catalog.json)，原图与新版图的哈希对应见[替换清单](upgrade-manifest.json)。

机制图用真实对象布局、状态变化与算例解释原因；实验图保留实际曲线，并明确对照、计时范围和可支持的结论；目录图列出各入口实际讲解的问题。五张表格重绘和其余实验面板均沿用历史条件，本次没有运行GPU实验。PNG用于直接显示，SVG可放大；其中复用的原始栅格面板保留原分辨率，不会自动成为矢量。

字号按900像素的文档显示宽度核对：新排版的主要标注以约16像素或更大、辅助文字不小于14像素为目标。四面板机制图改为自上而下阅读，正文卡片横向展开，标题、图例与脚注收拢字号差距。实验图逐面板显示，小图例单独放大展开，保留对应的颜色、线型、数值与原全图说明；同一曲线的双坐标轴仍放在一起。长图完整占据阅读宽度，避免高度上限将它再次缩小。文字放不下时换行或重排，不自动缩小字号。保留的历史原图内文字另外人工复核，不把新排版的字号检查当成原图全部小字的自动保证。

使用已安装Matplotlib、Pillow和CairoSVG的独立Python环境，从仓库根目录运行：

```bash
python -I docs/_static/img/document-figures/render.py
python -I docs/_static/img/document-figures/check.py
```

局部图由[chapters.py](chapters.py)生成，也通过统一的`render.py --only chapter-finite-concurrency`入口重绘。PNG先由CairoSVG绘制图形，再由Pillow使用明确的中文字体绘制文字，避免字体后端差异造成中文缺字；SVG保留可编辑文字，阅读端仍需安装对应字体。

覆盖检查只需要Python标准库和`rg`，可单独运行第二条命令。`render.py --only 图ID`可只重绘某张图；修改原始SVG面板后，再运行本目录绘图入口以更新完整解释图。生成需要一个可用中文字体，脚本默认查找Droid Sans Fallback等系统字体。绘图只用CPU，不启动训练、Docker或GPU任务。`check.py`验证维护中文档的解释图覆盖和图片引用，图意和版式还需人工查看。

原始证据面板默认保留不动。确需更正原图时，先核对对应正文与数据，再同步`evidence-explanations.json`和`catalog.json`中的来源、解释及SHA256，重绘后重新查看并更新复核记录。不能只刷新哈希来绕过来源完整性检查；`upgrade-manifest.json`保留本次替换时的版本对应关系。

历史实验有配套SVG时，从已有SVG提取坐标面板，并把图例的色标、线型和文字逐项展开。提取时保留各面板共用的字体定义，拒绝缺少引用定义的SVG，附加绘图源的哈希保存在`catalog.json`。只有PNG的横向多面板按固定区域拆开显示；少数小图例保留原面板，在下方复制原色标、重排文字，形成可读的大图例。区域、核对后的原标签与处理方法保留在`compose.py`，来源由哈希固定。已有16张文献教学图从原绘图函数重新排字，原PNG、SVG和实验数据不覆盖。派生面板缓存在本机临时目录，删除缓存后可由相同来源重新生成。

## 机制与操作图

按图要解释的问题查阅。图中的教学数字和实现范围由配套图注说明，具体机制在正文连续展开。

SSH 连接图的 SVG 是可编辑原图，解释两个端点如何对应四列、接口和端口分别在哪一层，以及同一监听端口怎样接收多条 TCP 连接。安装 GdkPixbuf 的 SVG 加载器及中文字体后，从仓库根目录重新生成单张屏幕图。

```bash
gdk-pixbuf-thumbnailer -s 800 docs/_static/img/document-figures/ssh-connection-endpoints.svg docs/_static/img/document-figures/ssh-connection-endpoints-screen.png
```


| 图解主题 | 图片 | 对应正文 |
|---|---|---|
| Rule、Memory与Skill：要求、背景和方法分别存放 | [PNG](agent-rule-memory-skill-screen.png)（[全部2张](screen-gallery.html#agent-rule-memory-skill)） / [SVG](agent-rule-memory-skill.svg) | [正文入口](../../../tools/codex/01-智能体配置_Rule与Memory及Skill.md) |
| GPU借还：A多等了，整批为什么仍可能更快？ | [PNG](borrower-return-delay-screen.png)（[全部4张](screen-gallery.html#borrower-return-delay)） / [SVG](borrower-return-delay.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/scripts/resident_sharing_study/README.md)，另用于3篇 |
| 主存缓存：保留状态可以省准备，容量不足就需要取舍 | [PNG](bounded-state-cache-screen.png)（[全部2张](screen-gallery.html#bounded-state-cache)） / [SVG](bounded-state-cache.svg) | [正文入口](../../../env-worker-deep-dive/research-proposals/05-状态调度_主存受限时的准备状态保留.md) |
| 网页呈现：内容、布局样式与交互共同组成页面 | [PNG](browser-page-layers-screen.png)（[全部2张](screen-gallery.html#browser-page-layers)） / [SVG](browser-page-layers.svg) | [正文入口](../../css/README.md)，另用于2篇 |
| 候选执行与可信判题：输入可以交给候选，隐藏答案留在边界内 | [PNG](candidate-and-trusted-verifier-screen.png)（[全部2张](screen-gallery.html#candidate-and-trusted-verifier)） / [SVG](candidate-and-trusted-verifier.svg) | [正文入口](../../../env-worker-deep-dive/airs-bench-audit/02-APPS判题_代码隔离与验证边界.md)，另用于1篇 |
| 持卡模式：容器常驻，为什么模型仍要重载？ | [PNG](command-episode-residency-screen.png) / [SVG](command-episode-residency.svg) | [正文入口](../../../env-worker-deep-dive/experiments/history/10-历史提案_阶段租约与回填调度.md)，另用于2篇 |
| 配置覆盖：有效值由默认配置与本次覆盖共同决定 | [PNG](configuration-layers-screen.png)（[全部2张](screen-gallery.html#configuration-layers)） / [SVG](configuration-layers.svg) | [正文入口](../../../tutorial/configuration.md) |
| 配置恢复：先比较备份，再恢复选中的设置 | [PNG](configuration-selective-restore-screen.png)（[全部2张](screen-gallery.html#configuration-selective-restore)） / [SVG](configuration-selective-restore.svg) | [正文入口](../../../tools/codex/02-Codex配置_备份与按需恢复.md) |
| 多轮上下文：历史工具输出也会占用下一轮输入空间 | [PNG](context-window-growth-screen.png)（[全部2张](screen-gallery.html#context-window-growth)） / [SVG](context-window-growth.svg) | [正文入口](../../../psrl/03-模型配置_上下文长度与生成限制.md) |
| 训练数据：结构化字段与样本内容要一起核对 | [PNG](dataset-fields-screen.png)（[全部2张](screen-gallery.html#dataset-fields)） / [SVG](dataset-fields.svg) | [正文入口](../../../examples/rlvr/common.md)，另用于1篇 |
| Docker挂起：线程等待写回，与磁盘还有多少空间是两条线索 | [PNG](docker-writeback-wait-screen.png)（[全部2张](screen-gallery.html#docker-writeback-wait)） / [SVG](docker-writeback-wait.svg) | [正文入口](../../../psrl/psrl-run-guide/incidents/01-Docker故障_2026年10月20日存储挂起.md)，另用于1篇 |
| 验证证据：不同实验分别回答不同问题 | [PNG](evidence-levels-screen.png)（[全部2张](screen-gallery.html#evidence-levels)） / [SVG](evidence-levels.svg) | [正文入口](../../../env-worker-deep-dive/airs-bench-audit/01-AIRS审计_修复状态与验证范围.md)，另用于15篇 |
| 实验预检：检查真正执行任务的节点与容器 | [PNG](experiment-preflight-node-screen.png)（[全部2张](screen-gallery.html#experiment-preflight-node)） / [SVG](experiment-preflight-node.svg) | [正文入口](../../../env-worker-deep-dive/knowledge/03-任务接入/02-任务接入_EnvWorker配置与CUDA验证.md)，另用于5篇 |
| 文件名、inode与打开引用：名字消失后谁还在用文件 | [PNG](file-references-screen.png)（[全部2张](screen-gallery.html#file-references)） / [SVG](file-references.svg) | [正文入口](../../../tools/linux/README.md)，另用于1篇 |
| GPU并发：装得下，不等于一起跑就划算 | [PNG](gpu-capacity-and-compute-screen.png)（[全部4张](screen-gallery.html#gpu-capacity-and-compute)） / [SVG](gpu-capacity-and-compute.svg) | [正文入口](../../../env-worker-deep-dive/experiments/history/08-历史过程_MPS训练并发对照.md)，另用于4篇 |
| GPU任务：代码、模型、输入与评分资产有不同寿命 | [PNG](gpu-task-asset-boundaries-screen.png)（[全部2张](screen-gallery.html#gpu-task-asset-boundaries)） / [SVG](gpu-task-asset-boundaries.svg) | [正文入口](../../../env-worker-deep-dive/knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)，另用于3篇 |
| GRPO：同题多次尝试，通过组内比较获得方向 | [PNG](group-relative-advantages-screen.png)（[全部2张](screen-gallery.html#group-relative-advantages)） / [SVG](group-relative-advantages.svg) | [正文入口](../../../examples/rlvr/grpo.md)，另用于2篇 |
| 资源边界：会话留在主机，GPU按命令申请 | [PNG](host-and-gpu-ledgers-screen.png) / [SVG](host-and-gpu-ledgers.svg) | [正文入口](../../../design/env_worker.md)，另用于6篇 |
| HTTP：主机、端口和路径分别解决不同问题 | [PNG](http-envelope-screen.png)（[全部2张](screen-gallery.html#http-envelope)） / [SVG](http-envelope.svg) | [正文入口](../../../tools/系统基础/04-网络通信_HTTP请求与GateB调用链.md) |
| SSH：连接两端怎样收发数据，服务怎样区分两条连接 | [PNG](ssh-connection-endpoints-screen.png) / [SVG](ssh-connection-endpoints.svg) | [正文入口](../../../tools/系统基础/06-远程登录_SSH连接与IP网卡端口.md) |
| KV缓存：共享已完成的前缀，分支继续时保留各自的新状态 | [PNG](kv-prefix-pages-screen.png)（[全部2张](screen-gallery.html#kv-prefix-pages)） / [SVG](kv-prefix-pages.svg) | [正文入口](../../../design/kv_cache.md) |
| 文件映射：虚拟地址背后仍需要完整的文件页面 | [PNG](mapped-file-pages-screen.png)（[全部2张](screen-gallery.html#mapped-file-pages)） / [SVG](mapped-file-pages.svg) | [正文入口](../../../tools/codex/session-management/03-Codex故障_会话数据库与SIGBUS排查.md)，另用于1篇 |
| Markdown与网页：同一份内容的两种阅读入口 | [PNG](markdown-and-site-screen.png)（[全部2张](screen-gallery.html#markdown-and-site)） / [SVG](markdown-and-site.svg) | [正文入口](../../../site-maintenance.md)，另用于1篇 |
| 性能反馈：同一段代码，在不同竞争条件下会得到不同计时 | [PNG](measurement-contention-screen.png)（[全部2张](screen-gallery.html#measurement-contention)） / [SVG](measurement-contention.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/scripts/feedback_pressure/README.md)，另用于2篇 |
| 显存准入：从漏算风险到安全放行 | [PNG](memory-promises-screen.png)（[全部4张](screen-gallery.html#memory-promises)） / [SVG](memory-promises.svg) | [正文入口](../../../env-worker-deep-dive/experiments/gpu-benchmarks/06-显存压力_提前排队与失败重试.md)，另用于4篇 |
| 训练并行：复制模型、切分张量与切分层是三种不同分工 | [PNG](parallel-model-shards-screen.png)（[全部2张](screen-gallery.html#parallel-model-shards)） / [SVG](parallel-model-shards.svg) | [正文入口](../../../psrl/推理与训练/02-训练并行_模型切分与通信带宽.md) |
| 进程归属：容器内编号与宿主编号可能不同 | [PNG](pid-namespace-identity-screen.png)（[全部2张](screen-gallery.html#pid-namespace-identity)） / [SVG](pid-namespace-identity.svg) | [正文入口](../../../env-worker-deep-dive/experiments/10-进程归属_Procfs挂载障碍与处理.md)，另用于1篇 |
| PPO裁剪：优势符号决定哪一侧的更新收益受到限制 | [PNG](ppo-clipped-objective-screen.png)（[全部3张](screen-gallery.html#ppo-clipped-objective)） / [SVG](ppo-clipped-objective.svg) | [正文入口](../../../examples/rlvr/ppo.md)，另用于1篇 |
| 统一执行：容器创建提供者与GPU执行协议分开 | [PNG](provider-execution-boundary-screen.png)（[全部2张](screen-gallery.html#provider-execution-boundary)） / [SVG](provider-execution-boundary.svg) | [正文入口](../../../env-worker-deep-dive/archive/03-Env历史_早期Router执行流程.md)，另用于2篇 |
| 配图资源：PNG负责直接看，SVG和脚本负责放大与维护 | [PNG](raster-vector-source-screen.png)（[全部2张](screen-gallery.html#raster-vector-source)） / [SVG](raster-vector-source.svg) | [正文入口](../../README.md)，另用于4篇 |
| 大块内存传输：先登记缓冲区，再确认搬运完成 | [PNG](registered-memory-transfer-screen.png)（[全部2张](screen-gallery.html#registered-memory-transfer)） / [SVG](registered-memory-transfer.svg) | [正文入口](../../../psrl/04-高速传输_NIXL内存搬运机制.md) |
| 资源回收：账本归零与进程显存释放要分别确认 | [PNG](release-logical-and-physical-screen.png)（[全部2张](screen-gallery.html#release-logical-and-physical)） / [SVG](release-logical-and-physical.svg) | [正文入口](../../../env-worker-deep-dive/experiments/05-容器后端_原生实现修复与Science验证.md)，另用于4篇 |
| 两道门：占卡互斥与回复节流控制不同的位置 | [PNG](reply-gate-vs-device-lock-screen.png)（[全部2张](screen-gallery.html#reply-gate-vs-device-lock)） / [SVG](reply-gate-vs-device-lock.svg) | [正文入口](../../../env-worker-deep-dive/archive/06-Env历史_72ca50c版本GateAB机制.md)，另用于1篇 |
| 异步采样与训练：一条轨迹完成后，先进入批次准备 | [PNG](rollout-batch-training-screen.png)（[全部2张](screen-gallery.html#rollout-batch-training)） / [SVG](rollout-batch-training.svg) | [正文入口](../../../env-worker-deep-dive/research-proposals/01-调度研究_Rollout完成时间与反馈保真.md)，另用于4篇 |
| TransferQueue：字段保存在样本下，阶段之间传递样本引用 | [PNG](sample-fields-and-handles-screen.png)（[全部2张](screen-gallery.html#sample-fields-and-handles)） / [SVG](sample-fields-and-handles.svg) | [正文入口](../../../design/router_tito.md)，另用于1篇 |
| 会话历史：共同前缀与分支后的新消息分开保存 | [PNG](session-history-branches-screen.png)（[全部2张](screen-gallery.html#session-history-branches)） / [SVG](session-history-branches.svg) | [正文入口](../../../tools/codex/session-archives/README.md)，另用于1篇 |
| 运行环境：子进程继承环境，Python导入状态各自保存 | [PNG](shell-and-python-screen.png)（[全部2张](screen-gallery.html#shell-and-python)） / [SVG](shell-and-python.svg) | [正文入口](../../../tools/系统基础/01-运行环境_Shell与Python及Conda.md)，另用于1篇 |
| SQLite：主库、WAL和共享索引分工不同 | [PNG](sqlite-three-files-screen.png)（[全部2张](screen-gallery.html#sqlite-three-files)） / [SVG](sqlite-three-files.svg) | [正文入口](../../../tools/系统基础/02-操作系统_进程与文件映射及SQLite.md) |
| 状态复用：保住昂贵准备，更新已变化的代码 | [PNG](state-version-match-screen.png)（[全部4张](screen-gallery.html#state-version-match)） / [SVG](state-version-match.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/scripts/state_reuse_study/README.md)，另用于3篇 |
| 静态映射风险：两个任务可能看见同一张物理GPU | [PNG](static-device-contention-screen.png)（[全部2张](screen-gallery.html#static-device-contention)） / [SVG](static-device-contention.svg) | [正文入口](../../../env-worker-deep-dive/archive/04-Env历史_Science整卡锁与设备账本.md) |
| 终端滚动：外层回滚、tmux历史和应用内部列表是三层 | [PNG](terminal-scroll-layers-screen.png)（[全部2张](screen-gallery.html#terminal-scroll-layers)） / [SVG](terminal-scroll-layers.svg) | [正文入口](../../../tools/claude-code/01-ClaudeCode_命令执行与权限设置.md)，另用于2篇 |
| 按轨迹平均与按token平均：长短回答的权重不同 | [PNG](token-loss-weighting-screen.png)（[全部2张](screen-gallery.html#token-loss-weighting)） / [SVG](token-loss-weighting.svg) | [正文入口](../../../examples/rlvr/dapo.md)，另用于1篇 |
| 训练健康：进程存活之外，还要看样本和更新信号 | [PNG](training-health-signals-screen.png)（[全部2张](screen-gallery.html#training-health-signals)） / [SVG](training-health-signals.svg) | [正文入口](../../../psrl/05-训练排障_DAPO退化诊断.md) |
| 训练显存峰值：长序列的中间状态会挤占余量 | [PNG](training-memory-peak-screen.png)（[全部2张](screen-gallery.html#training-memory-peak)） / [SVG](training-memory-peak.svg) | [正文入口](../../../design/resource_elasticity.md)，另用于1篇 |
| 命令动态放置：原卡排队时，检查另一个候选设备 | [PNG](two-gpu-command-queue-screen.png)（[全部2张](screen-gallery.html#two-gpu-command-queue)） / [SVG](two-gpu-command-queue.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/scripts/admission_study/README.md)，另用于4篇 |
| 两个模型、两层训练：工具里的模型与Agent策略分开更新 | [PNG](two-model-training-loops-screen.png)（[全部2张](screen-gallery.html#two-model-training-loops)） / [SVG](two-model-training-loops.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/scripts/posttrain_reuse_study/README.md)，另用于4篇 |
| 动态放置：等热卡，还是在冷卡重建？ | [PNG](wait-or-reload-screen.png)（[全部4张](screen-gallery.html#wait-or-reload)） / [SVG](wait-or-reload.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/scripts/warm_router_study/README.md)，另用于2篇 |
| 软件修复：只把提交补丁交给新的评分环境 | [PNG](software-patch-and-fresh-grader-screen.png)（[全部2张](screen-gallery.html#software-patch-and-fresh-grader)） / [SVG](software-patch-and-fresh-grader.svg) | [正文入口](../../../examples/agentic_rl/swe/index.md)，另用于1篇 |
| 模型反馈：评分者或教师为策略提供学习信号 | [PNG](model-derived-feedback-screen.png)（[全部2张](screen-gallery.html#model-derived-feedback)） / [SVG](model-derived-feedback.svg) | [正文入口](../../../examples/generative_reward_model/index.md)，另用于1篇 |
| Roofline：从硬件上界走到并发调度 | [PNG](roofline-capacity-decision-screen.png)（[全部4张](screen-gallery.html#roofline-capacity-decision)） / [SVG](roofline-capacity-decision.svg) | [正文入口](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md) |

## 历史实验图

横轴均标秒，分别展示局部阶段与整组完成时间。正文保留原版本、工作量、计时边界与失败证据。

| 图解主题 | 图片 | 对应正文 |
|---|---|---|
| FlashInfer准备复用：合并命令与复用进程收益不同 | [PNG](measured-flashinfer-preparation-screen.png)（[全部2张](screen-gallery.html#measured-flashinfer-preparation)） / [SVG](measured-flashinfer-preparation.svg) | [正文入口](../../../env-worker-deep-dive/experiments/gpu-benchmarks/07-FlashInfer_进程复用与任务合批.md) |
| 低负载对照：工作阶段省时，总时间未同步缩短 | [PNG](measured-low-load-screen.png)（[全部3张](screen-gallery.html#measured-low-load)） / [SVG](measured-low-load.svg) | [正文入口](../../../env-worker-deep-dive/experiments/02-基线对照_低负载EnvRouter修复验证.md) |
| MPS对照：GPU训练更快，完整运行仍有准备代价 | [PNG](measured-mps-boundaries-screen.png)（[全部3张](screen-gallery.html#measured-mps-boundaries)） / [SVG](measured-mps-boundaries.svg) | [正文入口](../../../env-worker-deep-dive/experiments/04-MPS训练_并行执行与控制修复对照.md) |
| PostTrain复用：单个请求变快，整组仍需计入准备 | [PNG](measured-posttrain-reuse-screen.png)（[全部3张](screen-gallery.html#measured-posttrain-reuse)） / [SVG](measured-posttrain-reuse.svg) | [正文入口](../../../env-worker-deep-dive/experiments/gpu-benchmarks/05-评分复用_PostTrain准备成本与整批耗时.md) |
| Science三步对照：训练时间与完整墙钟分别看 | [PNG](measured-science-three-steps-screen.png)（[全部3张](screen-gallery.html#measured-science-three-steps)） / [SVG](measured-science-three-steps.svg) | [正文入口](../../../env-worker-deep-dive/experiments/03-Science训练_三步对照与显存压力.md) |

## 目录阅读地图

每张地图使用对应目录的真实主题入口，适合先选阅读方向，再从README中的链接进入正文。

| 图解主题 | 图片 | 对应正文 |
|---|---|---|
| 集群预检与运行故障：阅读地图 | [PNG](map-019e476519-screen.png)（[全部2张](screen-gallery.html#map-019e476519)） / [SVG](map-019e476519.svg) | [正文入口](../../../psrl/psrl-run-guide/README.md) |
| PSRL 文档总览：阅读地图 | [PNG](map-0b5ca119d2-screen.png)（[全部2张](screen-gallery.html#map-0b5ca119d2)） / [SVG](map-0b5ca119d2.svg) | [正文入口](../../../README.md) |
| Designs & Features：阅读地图 | [PNG](map-0df1b907f1-screen.png)（[全部2张](screen-gallery.html#map-0df1b907f1)） / [SVG](map-0df1b907f1.svg) | [正文入口](../../../design/index.md) |
| Env任务接入：公共协议与各题库配方：阅读地图 | [PNG](map-0e0b082c0a-screen.png)（[全部2张](screen-gallery.html#map-0e0b082c0a)） / [SVG](map-0e0b082c0a.svg) | [正文入口](../../../env-worker-deep-dive/knowledge/03-任务接入/README.md) |
| Codex 使用、规则与故障：阅读地图 | [PNG](map-1c4901e702-screen.png)（[全部2张](screen-gallery.html#map-1c4901e702)） / [SVG](map-1c4901e702.svg) | [正文入口](../../../tools/codex/README.md) |
| Env 实验脚本与证据目录：阅读地图 | [PNG](map-268b082887-screen.png)（[全部2张](screen-gallery.html#map-268b082887)） / [SVG](map-268b082887.svg) | [正文入口](../../../env-worker-deep-dive/env-service-verification/README.md) |
| 开发工具知识索引：阅读地图 | [PNG](map-2f31c326af-screen.png)（[全部2张](screen-gallery.html#map-2f31c326af)） / [SVG](map-2f31c326af.svg) | [正文入口](../../../tools/README.md) |
| Env 环境执行：机制与接入指南：阅读地图 | [PNG](map-31ab314077-screen.png)（[全部2张](screen-gallery.html#map-31ab314077)） / [SVG](map-31ab314077.svg) | [正文入口](../../../env-worker-deep-dive/knowledge/README.md) |
| 环境执行与回收专题：阅读地图 | [PNG](map-31eb3fdaa1-screen.png)（[全部2张](screen-gallery.html#map-31eb3fdaa1)） / [SVG](map-31eb3fdaa1.svg) | [正文入口](../../../env-worker-deep-dive/README.md) |
| Env架构与执行：从任务申请到容器回收：阅读地图 | [PNG](map-37db5ffc0f-screen.png)（[全部2张](screen-gallery.html#map-37db5ffc0f)） / [SVG](map-37db5ffc0f.svg) | [正文入口](../../../env-worker-deep-dive/knowledge/01-架构与执行/README.md) |
| 文档维护：组织规范与历史巡检：阅读地图 | [PNG](map-4595f5dfc6-screen.png)（[全部2张](screen-gallery.html#map-4595f5dfc6)） / [SVG](map-4595f5dfc6.svg) | [正文入口](../../../tools/文档维护/README.md) |
| Examples：阅读地图 | [PNG](map-5e297a681a-screen.png)（[全部2张](screen-gallery.html#map-5e297a681a)） / [SVG](map-5e297a681a.svg) | [正文入口](../../../examples/index.md) |
| Tutorial：阅读地图 | [PNG](map-63e98fd058-screen.png)（[全部2张](screen-gallery.html#map-63e98fd058)） / [SVG](map-63e98fd058.svg) | [正文入口](../../../tutorial/index.md) |
| Env资源与调度：容量、借还、状态和动态放置：阅读地图 | [PNG](map-644bc9e974-screen.png)（[全部2张](screen-gallery.html#map-644bc9e974)） / [SVG](map-644bc9e974.svg) | [正文入口](../../../env-worker-deep-dive/knowledge/02-资源与调度/README.md) |
| GPU benchmark 实验导航：阅读地图 | [PNG](map-66cda102ff-screen.png)（[全部2张](screen-gallery.html#map-66cda102ff)） / [SVG](map-66cda102ff.svg) | [正文入口](../../../env-worker-deep-dive/experiments/gpu-benchmarks/README.md) |
| 推理与训练专题：阅读顺序与章节索引：阅读地图 | [PNG](map-6f366e2480-screen.png)（[全部2张](screen-gallery.html#map-6f366e2480)） / [SVG](map-6f366e2480.svg) | [正文入口](../../../psrl/推理与训练/README.md) |
| 计算机系统基础：环境、进程、文件、网络与GPU：阅读地图 | [PNG](map-7a7363b66b-screen.png)（[全部2张](screen-gallery.html#map-7a7363b66b)） / [SVG](map-7a7363b66b.svg) | [正文入口](../../../tools/系统基础/README.md) |
| AIRS 接入审查与隔离证据：阅读地图 | [PNG](map-907da87505-screen.png)（[全部2张](screen-gallery.html#map-907da87505)） / [SVG](map-907da87505.svg) | [正文入口](../../../env-worker-deep-dive/airs-bench-audit/README.md) |
| Codex 终端与历史恢复：阅读地图 | [PNG](map-9c01012189-screen.png)（[全部2张](screen-gallery.html#map-9c01012189)） / [SVG](map-9c01012189.svg) | [正文入口](../../../tools/codex/session-management/README.md) |
| 安装、运行与配置：阅读地图 | [PNG](map-9d52918d2e-screen.png)（[全部2张](screen-gallery.html#map-9d52918d2e)） / [SVG](map-9d52918d2e.svg) | [正文入口](../../../tutorial/README.md) |
| PSRL：从一次做题理解训练框架：阅读地图 | [PNG](map-9e82b92589-screen.png)（[全部2张](screen-gallery.html#map-9e82b92589)） / [SVG](map-9e82b92589.svg) | [正文入口](../../../psrl/README.md) |
| 历史实验与修复过程：阅读地图 | [PNG](map-a0cf851f52-screen.png)（[全部2张](screen-gallery.html#map-a0cf851f52)） / [SVG](map-a0cf851f52.svg) | [正文入口](../../../env-worker-deep-dive/experiments/history/README.md) |
| 可验证奖励训练配方：阅读地图 | [PNG](map-ac6079296a-screen.png)（[全部2张](screen-gallery.html#map-ac6079296a)） / [SVG](map-ac6079296a.svg) | [正文入口](../../../examples/rlvr/README.md) |
| PSRL Documentation：阅读地图 | [PNG](map-b4d68dc855-screen.png)（[全部2张](screen-gallery.html#map-b4d68dc855)） / [SVG](map-b4d68dc855.svg) | [正文入口](../../../index.md) |
| 算法与任务示例：阅读地图 | [PNG](map-bb83ca628a-screen.png)（[全部2张](screen-gallery.html#map-bb83ca628a)） / [SVG](map-bb83ca628a.svg) | [正文入口](../../../examples/README.md) |
| 历史实现与阶段分析：阅读地图 | [PNG](map-bc0d96902a-screen.png)（[全部2张](screen-gallery.html#map-bc0d96902a)） / [SVG](map-bc0d96902a.svg) | [正文入口](../../../env-worker-deep-dive/archive/README.md) |
| 架构与组件设计：阅读地图 | [PNG](map-c435a658e9-screen.png)（[全部2张](screen-gallery.html#map-c435a658e9)） / [SVG](map-c435a658e9.svg) | [正文入口](../../../design/README.md) |
| 项目概览与阅读起点：阅读地图 | [PNG](map-d345b2b349-screen.png)（[全部2张](screen-gallery.html#map-d345b2b349)） / [SVG](map-d345b2b349.svg) | [正文入口](../../../overview/README.md) |
| Env 实验与验证入口：阅读地图 | [PNG](map-df457d24c5-screen.png)（[全部2张](screen-gallery.html#map-df457d24c5)） / [SVG](map-df457d24c5.svg) | [正文入口](../../../env-worker-deep-dive/experiments/README.md) |
| Env 阶段成果与 benchmark 导览：阅读地图 | [PNG](map-e3e99949ff-screen.png)（[全部2张](screen-gallery.html#map-e3e99949ff)） / [SVG](map-e3e99949ff.svg) | [正文入口](../../../env-worker-deep-dive/milestones/README.md) |
| GPU 环境调度：研究问题与验证进展：阅读地图 | [PNG](map-ee40956879-screen.png)（[全部2张](screen-gallery.html#map-ee40956879)） / [SVG](map-ee40956879.svg) | [正文入口](../../../env-worker-deep-dive/research-proposals/README.md) |
| 工具与软件任务：阅读地图 | [PNG](map-f580c3ced5-screen.png)（[全部2张](screen-gallery.html#map-f580c3ced5)） / [SVG](map-f580c3ced5.svg) | [正文入口](../../../examples/agentic_rl/README.md) |

## 架构文献与实验的完整解释图

这66张图各自保留对应的原始面板，新增的解释与本张图的实际内容一致。实验数据不因换图而改变；教学算例、论文机制与本地实现的边界写在图内及正文中。

| 解释任务 | 完整图 | 正文与原始面板 |
|---|---|---|
| 系统总览：样本流与参数流 | [PNG](explained-overview-c46ca98-screen.png)（[全部3张](screen-gallery.html#explained-overview-c46ca98)） / [SVG](explained-overview-c46ca98.svg) | [对应正文](../../../design/architecture.md) · [原始面板](../overview.svg) |
| Rollout协调：路由、保留与冗余各解决什么 | [PNG](explained-rollout_techniques-cf838b0-screen.png)（[全部3张](screen-gallery.html#explained-rollout_techniques-cf838b0)） / [SVG](explained-rollout_techniques-cf838b0.svg) | [对应正文](../../../design/flexible_rollout.md) · [原始面板](../rollout_techniques.svg) |
| 参数服务：发布、传输、启用是三个动作 | [PNG](explained-ps-d3858c3-screen.png)（[全部3张](screen-gallery.html#explained-ps-d3858c3)） / [SVG](explained-ps-d3858c3.svg) | [对应正文](../../../design/parameter_server.md) · [原始面板](../ps.svg) |
| 样本滞后：版本年龄与可消费区间 | [PNG](explained-staleness_compatibility-ab32e0e-screen.png)（[全部2张](screen-gallery.html#explained-staleness_compatibility-ab32e0e)） / [SVG](explained-staleness_compatibility-ab32e0e.svg) | [对应正文](../../../design/staleness_control.md) · [原始面板](../staleness_compatibility.svg) |
| 短任务共享：暂停为什么没有更快 | [PNG](explained-resident-compute-sharing-f533859-screen.png)（[全部4张](screen-gallery.html#explained-resident-compute-sharing-f533859)） / [SVG](explained-resident-compute-sharing-f533859.svg) | [对应正文](../../../env-worker-deep-dive/experiments/06-训推共享_常驻vLLM与环境并行实验.md) · [原始面板](../../../env-worker-deep-dive/experiments/resident-compute-sharing.png) |
| 长任务共享：保留有益重叠 | [PNG](explained-resident-compute-sharing-long-be21a7c-screen.png)（[全部5张](screen-gallery.html#explained-resident-compute-sharing-long-be21a7c)） / [SVG](explained-resident-compute-sharing-long-be21a7c.svg) | [对应正文](../../../env-worker-deep-dive/experiments/06-训推共享_常驻vLLM与环境并行实验.md) · [原始面板](../../../env-worker-deep-dive/experiments/resident-compute-sharing-long.svg) |
| 单卡并发：吞吐增长与单请求变慢 | [PNG](explained-single-gpu-inference-curve-evidence-7df94d3-screen.png)（[全部5张](screen-gallery.html#explained-single-gpu-inference-curve-evidence-7df94d3)） / [SVG](explained-single-gpu-inference-curve-evidence-7df94d3.svg) | [对应正文](../../../env-worker-deep-dive/experiments/07-vLLM并发_吞吐与请求延迟曲线.md) · [原始面板](../../../env-worker-deep-dive/experiments/single-gpu-inference-curve-evidence.svg) |
| 自动持有：短空档与长空档分开看 | [PNG](explained-automatic-reservation-70f6d6f-screen.png)（[全部4张](screen-gallery.html#explained-automatic-reservation-70f6d6f)） / [SVG](explained-automatic-reservation-70f6d6f.svg) | [对应正文](../../../env-worker-deep-dive/experiments/08-模式选择_Command与Episode及等待预测.md) · [原始面板](../../../env-worker-deep-dive/experiments/automatic-reservation.svg) |
| 解码建模：拟合点与留出点怎样读 | [PNG](explained-capacity-model-c7da45b-screen.png)（[全部4张](screen-gallery.html#explained-capacity-model-c7da45b)） / [SVG](explained-capacity-model-c7da45b.svg) | [对应正文](../../../env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md) · [原始面板](../../../env-worker-deep-dive/experiments/capacity-model.svg) |
| 环境容量：并发越大是否越快 | [PNG](explained-environment-capacity-dd7a546-screen.png)（[全部4张](screen-gallery.html#explained-environment-capacity-dd7a546)） / [SVG](explained-environment-capacity-dd7a546.svg) | [对应正文](../../../env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md) · [原始面板](../../../env-worker-deep-dive/experiments/environment-capacity.svg) |
| 真实训练：四步累计怎样体现共享收益 | [PNG](explained-capacity-training-2330452-screen.png)（[全部4张](screen-gallery.html#explained-capacity-training-2330452)） / [SVG](explained-capacity-training-2330452.svg) | [对应正文](../../../env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md) · [原始面板](../../../env-worker-deep-dive/experiments/capacity-training.svg) |
| 命令时间线：共享加速与同步等待并存 | [PNG](explained-capacity-command-timeline-ea05e12-screen.png)（[全部3张](screen-gallery.html#explained-capacity-command-timeline-ea05e12)） / [SVG](explained-capacity-command-timeline-ea05e12.svg) | [对应正文](../../../env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md) · [原始面板](../../../env-worker-deep-dive/experiments/capacity-command-timeline.svg) |
| 六步对照：整批完成与异步边界 | [PNG](explained-integrated-partition-3748353-screen.png)（[全部4张](screen-gallery.html#explained-integrated-partition-3748353)） / [SVG](explained-integrated-partition-3748353.svg) | [对应正文](../../../env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md) · [原始面板](../../../env-worker-deep-dive/experiments/integrated-partition.svg) |
| 配对延迟：低负载的差异有多稳定 | [PNG](explained-paired-latency-d4ff470-screen.png)（[全部6张](screen-gallery.html#explained-paired-latency-d4ff470)） / [SVG](explained-paired-latency-d4ff470.svg) | [对应正文](../../../env-worker-deep-dive/experiments/11-历史审阅_低负载配对与GateAB设计.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-paired-review-20261020/paired-latency.png) |
| 热卡等待与冷卡加载：何时换卡 | [PNG](explained-warm-placement-core-0d59e83-screen.png)（[全部3张](screen-gallery.html#explained-warm-placement-core-0d59e83)） / [SVG](explained-warm-placement-core-0d59e83.svg) | [对应正文](../../../env-worker-deep-dive/experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md) · [原始面板](../../../env-worker-deep-dive/experiments/gpu-benchmarks/figures/warm-placement-core.svg) |
| 换卡成本分解：快在哪里又付出了什么 | [PNG](explained-warm-placement-decomposition-860b7c4-screen.png)（[全部3张](screen-gallery.html#explained-warm-placement-decomposition-860b7c4)） / [SVG](explained-warm-placement-decomposition-860b7c4.svg) | [对应正文](../../../env-worker-deep-dive/experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md) · [原始面板](../../../env-worker-deep-dive/experiments/gpu-benchmarks/figures/warm-placement-decomposition.svg) |
| 预测失败：过期、高估与容量不足 | [PNG](explained-warm-placement-faults-6ec458f-screen.png)（[全部4张](screen-gallery.html#explained-warm-placement-faults-6ec458f)） / [SVG](explained-warm-placement-faults-6ec458f.svg) | [对应正文](../../../env-worker-deep-dive/experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md) · [原始面板](../../../env-worker-deep-dive/experiments/gpu-benchmarks/figures/warm-placement-faults.svg) |
| CUDA准入：三条执行，一条等待 | [PNG](explained-comparison-b8c2052-screen.png)（[全部5张](screen-gallery.html#explained-comparison-b8c2052)） / [SVG](explained-comparison-b8c2052.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/02-历史对照_Science资源账本与Unit余量.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-cuda-ab-20261016-fixed/comparison.png) |
| CUDA压力预检：失败与监控故障分开 | [PNG](explained-cuda-preflight-9c45888-screen.png)（[全部4张](screen-gallery.html#explained-cuda-preflight-9c45888)） / [SVG](explained-cuda-preflight-9c45888.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-pressure-ab-20261018/preflight/cuda-preflight.png) |
| 等步数不等工作：前两步压力对照 | [PNG](explained-pressure-matched-c114981-screen.png)（[全部8张](screen-gallery.html#explained-pressure-matched-c114981)） / [SVG](explained-pressure-matched-c114981.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-pressure-ab-20261018/pressure-matched.png) |
| 训练曲线：没有完成的数据留空 | [PNG](explained-pressure-training-curves-ed8f503-screen.png)（[全部8张](screen-gallery.html#explained-pressure-training-curves-ed8f503)） / [SVG](explained-pressure-training-curves-ed8f503.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-pressure-ab-20261018/pressure-training-curves.png) |
| 历史耗时分解：等待采样不全是排队 | [PNG](explained-historical-timing-7286182-screen.png)（[全部3张](screen-gallery.html#explained-historical-timing-7286182)） / [SVG](explained-historical-timing-7286182.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-pressure-ab-20261018/baseline-0eeead8/historical-timing.png) |
| 固定完成量：成功数对齐后再看耗时 | [PNG](explained-fixed-work-outcomes-c85fca2-screen.png)（[全部8张](screen-gallery.html#explained-fixed-work-outcomes-c85fca2)） / [SVG](explained-fixed-work-outcomes-c85fca2.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-fixed-work-20261019/fixed-work-outcomes.png) |
| 固定完成量下的训练信号 | [PNG](explained-fixed-work-training-4f78708-screen.png)（[全部5张](screen-gallery.html#explained-fixed-work-training-4f78708)） / [SVG](explained-fixed-work-training-4f78708.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-fixed-work-20261019/fixed-work-training.png) |
| 物理显存空闲，为什么等待者进不来 | [PNG](explained-fixed-work-environment-memory-53e9b38-screen.png)（[全部3张](screen-gallery.html#explained-fixed-work-environment-memory-53e9b38)） / [SVG](explained-fixed-work-environment-memory-53e9b38.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-fixed-work-20261019/fixed-work-environment-memory.png) |
| 故障证据：外部占用导致训练 OOM | [PNG](explained-failure-evidence-dd5c2ec-screen.png)（[全部4张](screen-gallery.html#explained-failure-evidence-dd5c2ec)） / [SVG](explained-failure-evidence-dd5c2ec.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/04-历史对照_LAPS训练与EnvRouter开关.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-retry/failure-evidence.png) |
| LAPS两步训练：链路通过与加速分开 | [PNG](explained-training-curves-22c06d0-screen.png)（[全部8张](screen-gallery.html#explained-training-curves-22c06d0)） / [SVG](explained-training-curves-22c06d0.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/04-历史对照_LAPS训练与EnvRouter开关.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/training-curves.png) |
| LAPS资源：设备曲线能说明什么 | [PNG](explained-resource-comparison-7069978-screen.png)（[全部8张](screen-gallery.html#explained-resource-comparison-7069978)） / [SVG](explained-resource-comparison-7069978.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/04-历史对照_LAPS训练与EnvRouter开关.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/resource-comparison.png) |
| 并发容量：失败更快不等于吞吐更高 | [PNG](explained-capacity-comparison-7f2ba30-screen.png)（[全部4张](screen-gallery.html#explained-capacity-comparison-7f2ba30)） / [SVG](explained-capacity-comparison-7f2ba30.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/05-历史对照_同任务多协程GPU并发.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-gpu-capacity-20261022/capacity-comparison.svg) |
| 单卡压力：预算停止与成功完成不同 | [PNG](explained-comparison-38ffe21-screen.png)（[全部6张](screen-gallery.html#explained-comparison-38ffe21)） / [SVG](explained-comparison-38ffe21.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/06-历史过程_Science显存压力与原生任务.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-single-gpu-training-20261020/comparison.png) |
| 共同前缀：五步对照的收益与干扰 | [PNG](explained-matched-prefix-7cfb46d-screen.png)（[全部5张](screen-gallery.html#explained-matched-prefix-7cfb46d)） / [SVG](explained-matched-prefix-7cfb46d.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/06-历史过程_Science显存压力与原生任务.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-ten-step-training-20261020/matched-prefix.png) |
| 逐步曲线：完成数、token与OOM一起看 | [PNG](explained-per-step-fb816d0-screen.png)（[全部6张](screen-gallery.html#explained-per-step-fb816d0)） / [SVG](explained-per-step-fb816d0.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/06-历史过程_Science显存压力与原生任务.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-ten-step-training-20261020/per-step.png) |
| 原生任务：不注入压力时的边界 | [PNG](explained-comparison-f2dd9a5-screen.png)（[全部5张](screen-gallery.html#explained-comparison-f2dd9a5)） / [SVG](explained-comparison-f2dd9a5.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/06-历史过程_Science显存压力与原生任务.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-native-training-20261021/comparison.png) |
| 常驻显存压力：低压与高压的收益不同 | [PNG](explained-comparison-c991d3e-screen.png)（[全部6张](screen-gallery.html#explained-comparison-c991d3e)） / [SVG](explained-comparison-c991d3e.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/06-历史过程_Science显存压力与原生任务.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-resident-matrix-20261026/comparison.png) |
| 单卡固定基线：避免失败带来什么 | [PNG](explained-completion-a9a0126-screen.png)（[全部6张](screen-gallery.html#explained-completion-a9a0126)） / [SVG](explained-completion-a9a0126.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-baseline-control-20261020/completion.png) |
| 统一资源时间轴：预算与实占是否对齐 | [PNG](explained-resources-3247f1a-screen.png)（[全部4张](screen-gallery.html#explained-resources-3247f1a)） / [SVG](explained-resources-3247f1a.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-baseline-control-20261020/resources.png) |
| 低开销验证：启动与命令路径分开看 | [PNG](explained-low-overhead-771a5d1-screen.png)（[全部4张](screen-gallery.html#explained-low-overhead-771a5d1)） / [SVG](explained-low-overhead-771a5d1.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-low-overhead-20261020/low-overhead.png) |
| 预热对照：冷准备是否移出任务关键路径 | [PNG](explained-prewarm-comparison-b1a015e-screen.png)（[全部6张](screen-gallery.html#explained-prewarm-comparison-b1a015e)） / [SVG](explained-prewarm-comparison-b1a015e.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020/prewarm-comparison.png) |
| 预热资源时间线：提前准备是否越权占用 | [PNG](explained-resource-timeline-6931260-screen.png)（[全部3张](screen-gallery.html#explained-resource-timeline-6931260)） / [SVG](explained-resource-timeline-6931260.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020/resource-timeline.png) |
| 低负载长尾：获准后为什么还没开始脚本 | [PNG](explained-low-load-breakdown-37a32b1-screen.png)（[全部3张](screen-gallery.html#explained-low-load-breakdown-37a32b1)） / [SVG](explained-low-load-breakdown-37a32b1.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020/low-load-breakdown.png) |
| 同步查询移出事件循环：收益到哪里为止 | [PNG](explained-startup-offload-6f38130-screen.png)（[全部5张](screen-gallery.html#explained-startup-offload-6f38130)） / [SVG](explained-startup-offload-6f38130.svg) | [对应正文](../../../env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/router-sync-offload-20261020/startup-offload.png) |
| 作业重叠：结果可用与资源归还不同步 | [PNG](explained-latency-9f26f8b-screen.png)（[全部3张](screen-gallery.html#explained-latency-9f26f8b)） / [SVG](explained-latency-9f26f8b.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/01-架构与执行/04-GPU作业_生命周期与任务遥测.md) · [原始面板](../../../env-worker-deep-dive/gpu-job-validation/latency.svg) |
| 逐命令归还：怎样证明空档被复用 | [PNG](explained-completion-comparison-f9ce4be-screen.png)（[全部8张](screen-gallery.html#explained-completion-comparison-f9ce4be)） / [SVG](explained-completion-comparison-f9ce4be.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-command-work-20261019/completion-comparison.png) |
| Command资源轨迹：容器常驻而GPU间歇使用 | [PNG](explained-resource-timeline-a3c02cf-screen.png)（[全部3张](screen-gallery.html#explained-resource-timeline-a3c02cf)） / [SVG](explained-resource-timeline-a3c02cf.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-command-work-20261019/resource-timeline.png) |
| Command训练指标：执行成功与有效更新 | [PNG](explained-training-metrics-3137056-screen.png)（[全部5张](screen-gallery.html#explained-training-metrics-3137056)） / [SVG](explained-training-metrics-3137056.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/science-command-work-20261019/training-metrics.png) |
| 异步归还：安全重叠不保证总时间减少 | [PNG](explained-completion-c7d03c2-screen.png)（[全部3张](screen-gallery.html#explained-completion-c7d03c2)） / [SVG](explained-completion-c7d03c2.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/async-return-gpu-20261019/controlled-fixed/completion.png) |
| 异步时间线：谁先返回、谁后获准 | [PNG](explained-overlap-470dd69-screen.png)（[全部3张](screen-gallery.html#explained-overlap-470dd69)） / [SVG](explained-overlap-470dd69.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/async-return-gpu-20261019/controlled-fixed/overlap.png) |
| 控制路径修复：同步与异步都可能受益 | [PNG](explained-control-comparison-65d1b87-screen.png)（[全部5张](screen-gallery.html#explained-control-comparison-65d1b87)） / [SVG](explained-control-comparison-65d1b87.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md) · [原始面板](../../../env-worker-deep-dive/env-service-verification/results/async-admission-diagnosis-20261020/control-comparison.png) |
| 权重同步：等待、并行还是取消重做 | [PNG](explained-weight-sync-choices-6bf07fc-screen.png)（[全部3张](screen-gallery.html#explained-weight-sync-choices-6bf07fc)） / [SVG](explained-weight-sync-choices-6bf07fc.svg) | [对应正文](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md) · [原始面板](../../../env-worker-deep-dive/experiments/weight-sync-choices.png) |
| 借卡时间线：空档如何被利用 | [PNG](explained-borrowing-timeline-2487ead-screen.png)（[全部3张](screen-gallery.html#explained-borrowing-timeline-2487ead)） / [SVG](explained-borrowing-timeline-2487ead.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md) · [原始面板](../literature-reviews/borrowing-timeline.png) |
| 独立推进：快轨迹提前是否等于整批提前 | [PNG](explained-rollout-independence-3e2aa22-screen.png)（[全部3张](screen-gallery.html#explained-rollout-independence-3e2aa22)） / [SVG](explained-rollout-independence-3e2aa22.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md) · [原始面板](../literature-reviews/rollout-independence.png) |
| 流水线瓶颈：资源应转给谁 | [PNG](explained-pipeline-bottleneck-77cbdcf-screen.png)（[全部3张](screen-gallery.html#explained-pipeline-bottleneck-77cbdcf)） / [SVG](explained-pipeline-bottleneck-77cbdcf.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md) · [原始面板](../literature-reviews/pipeline-bottleneck.png) |
| KREX控制：准备复用与测量保护 | [PNG](explained-krex-control-flow-0e8d632-screen.png)（[全部3张](screen-gallery.html#explained-krex-control-flow-0e8d632)） / [SVG](explained-krex-control-flow-0e8d632.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md) · [原始面板](../literature-reviews/krex-control-flow.png) |
| 保护窗口：共享准备、独占测量 | [PNG](explained-protected-regions-0a75ded-screen.png)（[全部2张](screen-gallery.html#explained-protected-regions-0a75ded)） / [SVG](explained-protected-regions-0a75ded.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md) · [原始面板](../literature-reviews/protected-regions.png) |
| 关门与排空：为什么必须做两步 | [PNG](explained-gate-and-drain-1739525-screen.png)（[全部3张](screen-gallery.html#explained-gate-and-drain-1739525)） / [SVG](explained-gate-and-drain-1739525.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md) · [原始面板](../literature-reviews/gate-and-drain.png) |
| YARN借鉴：任务语义与节点执行分工 | [PNG](explained-yarn-responsibilities-180f1c7-screen.png)（[全部3张](screen-gallery.html#explained-yarn-responsibilities-180f1c7)） / [SVG](explained-yarn-responsibilities-180f1c7.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/04-系统借鉴_YARN资源分配与反馈调度.md) · [原始面板](../literature-reviews/yarn-responsibilities.png) |
| GPU任务生命周期：会话与命令分两层 | [PNG](explained-gpu-task-lifecycle-336b197-screen.png)（[全部3张](screen-gallery.html#explained-gpu-task-lifecycle-336b197)） / [SVG](explained-gpu-task-lifecycle-336b197.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/05-基准调研_GPU任务类型与接入顺序.md) · [原始面板](../literature-reviews/gpu-task-lifecycle.png) |
| 基准选择：准确率是否一定与竞争无关 | [PNG](explained-benchmark-choice-c6aaac2-screen.png)（[全部3张](screen-gallery.html#explained-benchmark-choice-c6aaac2)） / [SVG](explained-benchmark-choice-c6aaac2.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/05-基准调研_GPU任务类型与接入顺序.md) · [原始面板](../literature-reviews/benchmark-choice.png) |
| 奖励偏差：重复测量也可能稳定地选错 | [PNG](explained-reward-distortion-5865243-screen.png)（[全部3张](screen-gallery.html#explained-reward-distortion-5865243)） / [SVG](explained-reward-distortion-5865243.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/06-基准调研_AI4Infra性能反馈与验证.md) · [原始面板](../literature-reviews/reward-distortion.png) |
| 调度两条路径：低频分析与快速派发 | [PNG](explained-scheduling-two-lanes-e460719-screen.png)（[全部3张](screen-gallery.html#explained-scheduling-two-lanes-e460719)） / [SVG](explained-scheduling-two-lanes-e460719.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/07-智能调度_模型决策的已有工作与研究选择.md) · [原始面板](../literature-reviews/scheduling-two-lanes.png) |
| 决策成本：选对动作仍可能更慢 | [PNG](explained-model-decision-cost-05c9647-screen.png)（[全部3张](screen-gallery.html#explained-model-decision-cost-05c9647)） / [SVG](explained-model-decision-cost-05c9647.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/07-智能调度_模型决策的已有工作与研究选择.md) · [原始面板](../literature-reviews/model-decision-cost.png) |
| 两层训练与模型接管：参数流别混淆 | [PNG](explained-rsi-two-loops-cb6e04f-screen.png)（[全部3张](screen-gallery.html#explained-rsi-two-loops-cb6e04f)） / [SVG](explained-rsi-two-loops-cb6e04f.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md) · [原始面板](../literature-reviews/rsi-two-loops.png) |
| 调研阅读地图：按问题选择论文机制 | [PNG](explained-reading-map-0eb10e0-screen.png)（[全部3张](screen-gallery.html#explained-reading-map-0eb10e0)） / [SVG](explained-reading-map-0eb10e0.svg) | [对应正文](../../../env-worker-deep-dive/literature-reviews/README.md) · [原始面板](../literature-reviews/reading-map.png) |
| PSRL完整循环：工具反馈怎样进入下一次更新 | [PNG](explained-psrl-training-loop-b6943fc-screen.png)（[全部3张](screen-gallery.html#explained-psrl-training-loop-b6943fc)） / [SVG](explained-psrl-training-loop-b6943fc.svg) | [对应正文](../../../overview/01-PSRL入门_从一道题到模型更新.md) · [原始面板](../literature-reviews/psrl-training-loop.png) |
| PSRL入口：生成与训练怎样解耦 | [PNG](explained-PSRL_arch-4d2b637-screen.png)（[全部3张](screen-gallery.html#explained-PSRL_arch-4d2b637)） / [SVG](explained-PSRL_arch-4d2b637.svg) | [对应正文](../../../overview/index.md) · [原始面板](../PSRL_arch.svg) |
| Roofline与合批：拐点从哪里来 | [PNG](explained-roofline-and-batching-1ddd7ab-screen.png)（[全部4张](screen-gallery.html#explained-roofline-and-batching-1ddd7ab)） / [SVG](explained-roofline-and-batching-1ddd7ab.svg) | [对应正文](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md) · [原始面板](../literature-reviews/roofline-and-batching.png) |

## 渲染复核

此前152张图均实际生成PNG与SVG，并通过19页总览逐张查看布局，再对密集机制图、坐标曲线、图例和历史栅格等代表图按900像素阅读宽度复核，具体清单保存在下方记录。另生成76页双图复核页，每页只放两张完整图，长图保留实际宽度，不再因高度限制而进一步缩窄。目录地图核对真实专题；机制图核对同一场景下的对象、数字和变化；实验图核对原始面板与新增解释是否一致。字号检查覆盖新生成SVG中的矢量文字，嵌入的原始栅格和派生面板仍需查看实际渲染。

复核中修正了配置恢复字段、HTTP端口、KV前缀块编号、异步批次图的文字间距，以及一张常驻显存压力图与模型复用解释的主题错配。复核结果见[review.json](review.json)和[quality-review.json](quality-review.json)，覆盖与来源哈希检查见[coverage.json](coverage.json)。这些记录不代替完整历史实验重跑或全量源码审计。

| 拼图 | 包含的图 |
|---|---|
| [review-01.png](review-01.png) | agent-rule-memory-skill、borrower-return-delay |
| [review-02.png](review-02.png) | bounded-state-cache、browser-page-layers |
| [review-03.png](review-03.png) | candidate-and-trusted-verifier、command-episode-residency |
| [review-04.png](review-04.png) | configuration-layers、configuration-selective-restore |
| [review-05.png](review-05.png) | context-window-growth、dataset-fields |
| [review-06.png](review-06.png) | docker-writeback-wait、evidence-levels |
| [review-07.png](review-07.png) | experiment-preflight-node、file-references |
| [review-08.png](review-08.png) | gpu-capacity-and-compute、gpu-task-asset-boundaries |
| [review-09.png](review-09.png) | group-relative-advantages、host-and-gpu-ledgers |
| [review-10.png](review-10.png) | http-envelope、kv-prefix-pages |
| [review-11.png](review-11.png) | map-019e476519、map-0b5ca119d2 |
| [review-12.png](review-12.png) | map-0df1b907f1、map-0e0b082c0a |
| [review-13.png](review-13.png) | map-1c4901e702、map-268b082887 |
| [review-14.png](review-14.png) | map-2f31c326af、map-31ab314077 |
| [review-15.png](review-15.png) | map-31eb3fdaa1、map-37db5ffc0f |
| [review-16.png](review-16.png) | map-4595f5dfc6、map-5e297a681a |
| [review-17.png](review-17.png) | map-63e98fd058、map-644bc9e974 |
| [review-18.png](review-18.png) | map-66cda102ff、map-6f366e2480 |
| [review-19.png](review-19.png) | map-7a7363b66b、map-907da87505 |
| [review-20.png](review-20.png) | map-9c01012189、map-9d52918d2e |
| [review-21.png](review-21.png) | map-9e82b92589、map-a0cf851f52 |
| [review-22.png](review-22.png) | map-ac6079296a、map-b4d68dc855 |
| [review-23.png](review-23.png) | map-bb83ca628a、map-bc0d96902a |
| [review-24.png](review-24.png) | map-c435a658e9、map-d345b2b349 |
| [review-25.png](review-25.png) | map-df457d24c5、map-e3e99949ff |
| [review-26.png](review-26.png) | map-ee40956879、map-f580c3ced5 |
| [review-27.png](review-27.png) | mapped-file-pages、markdown-and-site |
| [review-28.png](review-28.png) | measured-flashinfer-preparation、measured-low-load |
| [review-29.png](review-29.png) | measured-mps-boundaries、measured-posttrain-reuse |
| [review-30.png](review-30.png) | measured-science-three-steps、measurement-contention |
| [review-31.png](review-31.png) | memory-promises、parallel-model-shards |
| [review-32.png](review-32.png) | pid-namespace-identity、ppo-clipped-objective |
| [review-33.png](review-33.png) | provider-execution-boundary、raster-vector-source |
| [review-34.png](review-34.png) | registered-memory-transfer、release-logical-and-physical |
| [review-35.png](review-35.png) | reply-gate-vs-device-lock、rollout-batch-training |
| [review-36.png](review-36.png) | sample-fields-and-handles、session-history-branches |
| [review-37.png](review-37.png) | shell-and-python、sqlite-three-files |
| [review-38.png](review-38.png) | state-version-match、static-device-contention |
| [review-39.png](review-39.png) | terminal-scroll-layers、token-loss-weighting |
| [review-40.png](review-40.png) | training-health-signals、training-memory-peak |
| [review-41.png](review-41.png) | two-gpu-command-queue、two-model-training-loops |
| [review-42.png](review-42.png) | wait-or-reload、software-patch-and-fresh-grader |
| [review-43.png](review-43.png) | model-derived-feedback、roofline-capacity-decision |
| [review-44.png](review-44.png) | explained-overview-c46ca98、explained-rollout_techniques-cf838b0 |
| [review-45.png](review-45.png) | explained-ps-d3858c3、explained-staleness_compatibility-ab32e0e |
| [review-46.png](review-46.png) | explained-resident-compute-sharing-f533859、explained-resident-compute-sharing-long-be21a7c |
| [review-47.png](review-47.png) | explained-single-gpu-inference-curve-evidence-7df94d3、explained-automatic-reservation-70f6d6f |
| [review-48.png](review-48.png) | explained-capacity-model-c7da45b、explained-environment-capacity-dd7a546 |
| [review-49.png](review-49.png) | explained-capacity-training-2330452、explained-capacity-command-timeline-ea05e12 |
| [review-50.png](review-50.png) | explained-integrated-partition-3748353、explained-paired-latency-d4ff470 |
| [review-51.png](review-51.png) | explained-warm-placement-core-0d59e83、explained-warm-placement-decomposition-860b7c4 |
| [review-52.png](review-52.png) | explained-warm-placement-faults-6ec458f、explained-comparison-b8c2052 |
| [review-53.png](review-53.png) | explained-cuda-preflight-9c45888、explained-pressure-matched-c114981 |
| [review-54.png](review-54.png) | explained-pressure-training-curves-ed8f503、explained-historical-timing-7286182 |
| [review-55.png](review-55.png) | explained-fixed-work-outcomes-c85fca2、explained-fixed-work-training-4f78708 |
| [review-56.png](review-56.png) | explained-fixed-work-environment-memory-53e9b38、explained-failure-evidence-dd5c2ec |
| [review-57.png](review-57.png) | explained-training-curves-22c06d0、explained-resource-comparison-7069978 |
| [review-58.png](review-58.png) | explained-capacity-comparison-7f2ba30、explained-comparison-38ffe21 |
| [review-59.png](review-59.png) | explained-matched-prefix-7cfb46d、explained-per-step-fb816d0 |
| [review-60.png](review-60.png) | explained-comparison-f2dd9a5、explained-comparison-c991d3e |
| [review-61.png](review-61.png) | explained-completion-a9a0126、explained-resources-3247f1a |
| [review-62.png](review-62.png) | explained-low-overhead-771a5d1、explained-prewarm-comparison-b1a015e |
| [review-63.png](review-63.png) | explained-resource-timeline-6931260、explained-low-load-breakdown-37a32b1 |
| [review-64.png](review-64.png) | explained-startup-offload-6f38130、explained-latency-9f26f8b |
| [review-65.png](review-65.png) | explained-completion-comparison-f9ce4be、explained-resource-timeline-a3c02cf |
| [review-66.png](review-66.png) | explained-training-metrics-3137056、explained-completion-c7d03c2 |
| [review-67.png](review-67.png) | explained-overlap-470dd69、explained-control-comparison-65d1b87 |
| [review-68.png](review-68.png) | explained-weight-sync-choices-6bf07fc、explained-borrowing-timeline-2487ead |
| [review-69.png](review-69.png) | explained-rollout-independence-3e2aa22、explained-pipeline-bottleneck-77cbdcf |
| [review-70.png](review-70.png) | explained-krex-control-flow-0e8d632、explained-protected-regions-0a75ded |
| [review-71.png](review-71.png) | explained-gate-and-drain-1739525、explained-yarn-responsibilities-180f1c7 |
| [review-72.png](review-72.png) | explained-gpu-task-lifecycle-336b197、explained-benchmark-choice-c6aaac2 |
| [review-73.png](review-73.png) | explained-reward-distortion-5865243、explained-scheduling-two-lanes-e460719 |
| [review-74.png](review-74.png) | explained-model-decision-cost-05c9647、explained-rsi-two-loops-cb6e04f |
| [review-75.png](review-75.png) | explained-reading-map-0eb10e0、explained-psrl-training-loop-b6943fc |
| [review-76.png](review-76.png) | explained-PSRL_arch-4d2b637、explained-roofline-and-batching-1ddd7ab |

本轮新增35张局部图另有18页完整900像素宽复核页，逐张检查中文、字号、布局与算例。发现并修正KV块标签重叠、训练准备框文字越界、资源归还图缺乏占用变化，以及命令预测正文与旧配图的A/B角色错配。修正后的图片再次查看。以下复核页只涵盖新增图，既有152张图保留前一轮记录。

| 局部图复核页 | 包含的图 |
|---|---|
| [chapter-review-01.png](chapter-review-01.png) | chapter-critical-path、chapter-memory-headroom |
| [chapter-review-02.png](chapter-review-02.png) | chapter-idle-window、chapter-gpu-priority |
| [chapter-review-03.png](chapter-review-03.png) | chapter-sharing-control、chapter-replica-backlog |
| [chapter-review-04.png](chapter-review-04.png) | chapter-finite-concurrency、chapter-tail-batch |
| [chapter-review-05.png](chapter-review-05.png) | chapter-three-limits、chapter-instances-requests |
| [chapter-review-06.png](chapter-review-06.png) | chapter-prediction-intervals、chapter-rollout-pipeline |
| [chapter-review-07.png](chapter-review-07.png) | chapter-pipeline-overlap、chapter-resource-reallocation |
| [chapter-review-08.png](chapter-review-08.png) | chapter-code-versions、chapter-state-dependencies |
| [chapter-review-09.png](chapter-review-09.png) | chapter-prepare-gpu、chapter-measurement-ranking |
| [chapter-review-10.png](chapter-review-10.png) | chapter-local-service-time、chapter-resource-layers |
| [chapter-review-11.png](chapter-review-11.png) | chapter-warm-layers、chapter-nested-deadlock |
| [chapter-review-12.png](chapter-review-12.png) | chapter-model-promotion、chapter-grouped-placement |
| [chapter-review-13.png](chapter-review-13.png) | chapter-gpu-return-states、chapter-selective-batching |
| [chapter-review-14.png](chapter-review-14.png) | chapter-kv-copy-on-write、chapter-continuous-batching |
| [chapter-review-15.png](chapter-review-15.png) | chapter-parallel-communication、chapter-ppo-credit |
| [chapter-review-16.png](chapter-review-16.png) | chapter-gpu-pool-host、chapter-compute-sharing |
| [chapter-review-17.png](chapter-review-17.png) | chapter-process-reuse、chapter-remaining-steps |
| [chapter-review-18.png](chapter-review-18.png) | chapter-episode-admission |

## 文件与子目录

- [绘图入口](render.py)、[对象与证据组合绘图](compose.py)、[多面板机制图](detailed.py)与[章节局部图](chapters.py)
- [机制场景输入](explanations.json)与[原始面板的专项解释](evidence-explanations.json)
- [覆盖与链接检查](check.py)
- [绘图和配文输入](catalog.json)
- [覆盖检查结果](coverage.json)
- [本轮验证与证据范围](validation.json)
- [实际渲染查看记录](review.json)与[全量章节解释复核](quality-review.json)
- [局部图与正文可读性复核](chapter-readability-review.json)
- [复核拼图与图片对应](review-sheets.json)与[逐图替换清单](upgrade-manifest.json)
- [屏幕版生成器](screen.py)、[屏幕阅读复核](screen-review.json)、[分页与来源记录](screen-layout.json)及[可筛选图集](screen-gallery.html)

本目录没有子目录。[返回图片导航](../README.md) · [返回总导航](../../../README.md)
