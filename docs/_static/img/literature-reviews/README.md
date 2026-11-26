# 论文原图与调度教学图集

这些图配合[论文调研正文](../../../env-worker-deep-dive/literature-reviews/README.md)和 [PSRL 阅读指南](../../../overview/01-PSRL入门_从一道题到模型更新.md)使用。PNG 可直接预览，SVG 便于放大。图中的假设数字只用于解释机制，论文证据与本地实现状态在对应正文维护。

<!-- figure:raster-vector-source -->
这张图沿同一个场景解释：PNG、SVG 与绘图脚本分别解决显示、编辑还是复现的问题？

[![配图资源：PNG负责直接看，SVG和脚本负责放大与维护（1/2）](../document-figures/raster-vector-source-screen.png)](../document-figures/raster-vector-source.png)

[![配图资源：PNG负责直接看，SVG和脚本负责放大与维护（2/2）](../document-figures/raster-vector-source-screen-02.png)](../document-figures/raster-vector-source.png)

同一幅解释图既要在 Markdown 里直接显示，也要能修改和追溯数字来源。 显示文件不是全部来源。保留源脚本与原始证据，才能避免下一轮只剩无法核对的截图。 SVG 内嵌原始栅格面板时，该面板不会自动变成矢量。本轮证据图明确保留这一来源。 [查看矢量图](../document-figures/raster-vector-source.svg)。
<!-- /figure:raster-vector-source -->

## 论文与官方原图

这里保存从15篇论文和Apache Hadoop官方文档节选的22张原图，正文引用25处，以24张屏幕分图显示。图中对象、箭头、坐标和英文标签保留原貌，中文讲解放在正文。与下方自绘教学图分开登记，原图中的实验条件和结论不自动成为PSRL实验结果。

| 来源与原图 | 图回答什么 | 配套中文讲解 |
|---|---|---|
| [rollart 图7](paper-rollart-fig07-screen.png) · [完整原图](paper-rollart-fig07.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/gao) | 资源、执行与控制三层 | [调度论文：五篇近期顶会工作综述](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md#3-rollart环境长尾与轨迹级解耦) |
| [rollart 图8](paper-rollart-fig08-screen.png) · [完整原图](paper-rollart-fig08.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/gao) | 每条轨迹独立调用环境与推理 | [论文精读：RollArt与DynaRL的Agent调度](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md#3-rollart-的贡献与执行过程) |
| [rollart 图9](paper-rollart-fig09-screen.png) · [完整原图](paper-rollart-fig09.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/gao) | 样本收集、权重切换与训练重叠 | [论文精读：RollArt与DynaRL的Agent调度](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md#3-rollart-的贡献与执行过程) |
| [dynarl 图6](paper-dynarl-fig06-screen.png) · [完整原图](paper-dynarl-fig06.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) | 全局决策与本地迁移反馈环 | [调度论文：五篇近期顶会工作综述](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md#4-dynarl随瓶颈变化重新分配资源) |
| [dynarl 图9](paper-dynarl-fig09-screen.png) · [完整原图](paper-dynarl-fig09.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) | 三种迁移动作及其状态开销 | [论文精读：RollArt与DynaRL的Agent调度](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md#6-dynarl-的贡献与执行过程) |
| [murakkab 图5](paper-murakkab-fig05-screen.png) · [完整原图](paper-murakkab-fig05.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/chaudhry) | 工作流声明、优化部署与运行执行 | [调度论文：五篇近期顶会工作综述](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md#5-murakkab按完整工作流优化) |
| [prism 图4](paper-prism-fig04-screen.png) · [完整原图](paper-prism-fig04.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/yu-shan) | 虚拟地址与物理显存页的解耦 | [调度论文：五篇近期顶会工作综述](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md#6-prism让模型显存真正具有弹性) |
| [weave 图1](paper-weave-fig01-screen.png) · [完整原图](paper-weave-fig01.png) · [原文](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan) | 两个RL作业交错利用依赖空泡 | [调度论文：五篇近期顶会工作综述](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md#7-weave利用依赖造成的空闲窗口) / [训推共享：vLLM余量与容量调度](../../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#借出的卡什么时候能收回来) |
| [krex 图1](paper-krex-fig01-screen.png) · [完整原图](paper-krex-fig01.png) · [原文](https://arxiv.org/html/2609.30057v1) | 整命令与测量区间的保护范围 | [论文精读：KREX共享GPU测量保护](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md#1-从完整流程理解论文) / [基准调研：AI4Infra性能反馈与验证](../../../env-worker-deep-dive/literature-reviews/06-基准调研_AI4Infra性能反馈与验证.md#为什么可靠反馈会改变调度目标) |
| [krex 图4](paper-krex-fig04-screen.png) · [完整原图](paper-krex-fig04.png) · [原文](https://arxiv.org/html/2609.30057v1) | 关门、排空、冻结与解冻的顺序 | [论文精读：KREX共享GPU测量保护](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md#3-怎样建立真正可测量的独占窗口) |
| [kernelbench 图1](paper-kernelbench-fig01-screen.png) · [完整原图](paper-kernelbench-fig01.png) · [原文](https://arxiv.org/html/2502.10517v1) | 生成代码、正确性检查与性能测量 | [基准调研：GPU任务类型与接入顺序](../../../env-worker-deep-dive/literature-reviews/05-基准调研_GPU任务类型与接入顺序.md#比较维度) |
| [schedmate 图5](paper-schedmate-fig05-screen.png) · [完整原图](paper-schedmate-fig05.png) · [原文](https://arxiv.org/html/2510.03334v1) | 代码语义提取与历史近邻估计 | [智能调度：模型决策的已有工作与研究选择](../../../env-worker-deep-dive/literature-reviews/07-智能调度_模型决策的已有工作与研究选择.md#语义补信息schedmate) |
| [alphaevolve 图1](paper-alphaevolve-fig01-screen.png) · [完整原图](paper-alphaevolve-fig01.png) · [原文](https://arxiv.org/abs/2506.13131v1) | 生成、评价与程序库反馈环 | [智能调度：模型决策的已有工作与研究选择](../../../env-worker-deep-dive/literature-reviews/07-智能调度_模型决策的已有工作与研究选择.md#离线生成算法alphaevolve-与-adrs) |
| [firepool 图2](paper-firepool-fig02-screen.png) · [完整原图](paper-firepool-fig02.png) · [原文](https://arxiv.org/html/2611.10832v1) | 作业请求与预热容器池的分工 | [基础设施：RSI与GPU沙盒及模型接管](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md#firepool最直接的工业-gpu-沙盒调度系统) |
| [firepool 图3](paper-firepool-fig03-screen.png) · [完整原图](paper-firepool-fig03.png) · [原文](https://arxiv.org/html/2611.10832v1) | 零显存备用容器怎样缩短交接 | [基础设施：RSI与GPU沙盒及模型接管](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md#firepool-场景与-psrl-三项能力对照) / [运行准备：重复准备成本与GPU就绪调度](../../../env-worker-deep-dive/research-proposals/08-运行准备_重复准备成本与GPU就绪调度.md#已有研究与创新边界) |
| [cuda-agent 图2](paper-cuda-agent-fig02-screen.png) · [完整原图](paper-cuda-agent-fig02.png) · [原文](https://cuda-agent.github.io/) | GPU工具反馈回到内核生成Agent | [基础设施：RSI与GPU沙盒及模型接管](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md#cuda-agent最贴近的工业训练场景) |
| [seal 图1](paper-seal-fig01-screen.png) · [完整原图](paper-seal-fig01.png) · [原文](https://arxiv.org/html/2506.10943v2) | 临时适应模型与外层策略更新 | [基础设施：RSI与GPU沙盒及模型接管](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md#seal学习怎样生成有效的自训练指令) |
| [vllm 图6](paper-vllm-fig06-screen.png) · [完整原图](paper-vllm-fig06.png) · [原文](https://arxiv.org/abs/2309.06180v1) | 逻辑KV块通过块表映射到物理显存 | [vLLM推理：计算瓶颈与KV缓存](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md#13-pagedattention把-kv-cache-当虚拟内存管) |
| [vllm 图8](paper-vllm-fig08-screen.png) · [完整原图](paper-vllm-fig08.png) · [原文](https://arxiv.org/abs/2309.06180v1) | 共享前缀的引用计数与写时复制 | [vLLM推理：计算瓶颈与KV缓存](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md#三条分支怎样共享并各自续写) |
| [distserve 图6](paper-distserve-fig06-screen.png) · [完整原图](paper-distserve-fig06.png) · [原文](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | P池、D池与跨池KV传输 | [vLLM推理：计算瓶颈与KV缓存](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md#16-prefilldecode-分离disaggregation) |
| [serverlessllm 图3](paper-serverlessllm-fig03-screen.png) · [完整原图](paper-serverlessllm-fig03.png) · [原文](https://www.usenix.org/conference/osdi24/presentation/fu) | 空卡、状态局部性与在线迁移的四种选择 | [动态放置：热卡等待与冷卡重建代码](../../../env-worker-deep-dive/knowledge/02-资源与调度/09-动态放置_热卡等待与冷卡重建代码.md#5-类型化成本与三种放置策略) |
| [yarn 官方架构图](paper-yarn-architecture-screen.png) · [完整原图](paper-yarn-architecture.png) · [原文](https://hadoop.apache.org/docs/r3.4.2/hadoop-yarn/hadoop-yarn-site/YARN.html) | 资源管理器、应用管理器与节点容器的关系 | [系统借鉴：YARN资源分配与反馈调度](../../../env-worker-deep-dive/literature-reviews/04-系统借鉴_YARN资源分配与反馈调度.md#先理解-yarn-提供的分工) |

各图的作者、原文链接、版本、PDF页码、裁切坐标、许可记录、源文件哈希及使用章节见[原图清单](paper-figures.json)。CC许可图保留相应许可链接；其他图保留作者版权，作为对应机制的研究评述节选，不将arXiv托管许可标为开放转载许可。只保留与讨论有关的图形，未复制整篇论文。

复现需要Python环境中已有PyMuPDF、Pillow；下载来源另需Requests。源PDF与官方GIF默认保存在临时缓存，目录内保留完整图PNG、PDF截取的矢量SVG与屏幕PNG。论文版本按清单哈希核对，不覆盖旧实验或教学图。

```bash
python -I docs/_static/img/literature-reviews/paper_figures.py --download
python -I docs/_static/img/literature-reviews/paper_figures.py --check
python -I docs/_static/img/document-figures/check.py
```

用`--cache 路径`指定已有来源，用`--only 图ID`仅重提取指定图。PDF页码从1计数，包含会议PDF封面；SVG只含所引用的图。Murakkab图5按原有(a)、(b)、(c)三个面板分别显示。原图没有在本轮重新测量，数值示意与本地实验仍分别查看对应正文。

## 自绘教学图与正文

[原始机制面板缩略预览](preview.png)。正文现用的16张完整解释图见下表，已加入各图特有的条件、算例读法、结果与限制；全量查看记录和新版PNG入口在[总图集](../document-figures/README.md#架构文献与实验的完整解释图)。

| 图 | 配套正文 |
|---|---|
| [阅读地图](../document-figures/explained-reading-map-0eb10e0.svg) | [论文导航](../../../env-worker-deep-dive/literature-reviews/README.md) |
| [借卡时间线](../document-figures/explained-borrowing-timeline-2487ead.svg) | [五篇论文综述](../../../env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md#8-一个贯穿五篇论文的例子) |
| [轨迹独立推进](../document-figures/explained-rollout-independence-3e2aa22.svg)、[流水线瓶颈](../document-figures/explained-pipeline-bottleneck-77cbdcf.svg) | [RollArt / DynaRL 精读](../../../env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md) |
| [区域保护流程](../document-figures/explained-krex-control-flow-0e8d632.svg)、[关门与排空](../document-figures/explained-gate-and-drain-1739525.svg)、[准备并行与测量串行](../document-figures/explained-protected-regions-0a75ded.svg) | [KREX 精读](../../../env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md) |
| [评分失真](../document-figures/explained-reward-distortion-5865243.svg) | [AI4Infra](../../../env-worker-deep-dive/literature-reviews/06-基准调研_AI4Infra性能反馈与验证.md#为什么可靠反馈会改变调度目标) |
| [任务执行关系](../document-figures/explained-gpu-task-lifecycle-336b197.svg)、[benchmark 选择](../document-figures/explained-benchmark-choice-c6aaac2.svg) | [GPU 任务调研](../../../env-worker-deep-dive/literature-reviews/05-基准调研_GPU任务类型与接入顺序.md) |
| [语义分析与派发](../document-figures/explained-scheduling-two-lanes-e460719.svg)、[模型决策成本](../document-figures/explained-model-decision-cost-05c9647.svg) | [模型调度大脑](../../../env-worker-deep-dive/literature-reviews/07-智能调度_模型决策的已有工作与研究选择.md) |
| [应用与资源分工](../document-figures/explained-yarn-responsibilities-180f1c7.svg) | [YARN 启发](../../../env-worker-deep-dive/literature-reviews/04-系统借鉴_YARN资源分配与反馈调度.md) |
| [两层训练与接管](../document-figures/explained-rsi-two-loops-cb6e04f.svg) | [RSI 调研](../../../env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md#现有两层训练与原生权重同步) |
| [PSRL 完整主线](../document-figures/explained-psrl-training-loop-b6943fc.svg) | [从一道题理解 PSRL](../../../overview/01-PSRL入门_从一道题到模型更新.md#先把完整流程连起来) |
| [Roofline 与合批曲线](../document-figures/explained-roofline-and-batching-1ddd7ab.svg) | [推理机制的 Roofline 与吞吐算例](../../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md#11-prefill-与-decode-的根本差异) |

## 重新生成

使用已有安装 Matplotlib、Pillow、CairoSVG 的独立 Python 环境，还需一个中文字体。脚本依次查找 Noto Sans CJK SC、Source Han Sans SC、Microsoft YaHei、Droid Sans Fallback、SimHei，缺少字体时明确退出。无需 CUDA、Docker、网络或训练数据，不修改共享环境。

从仓库根目录执行：

```bash
python -I docs/_static/img/literature-reviews/render.py
python -I docs/_static/img/document-figures/render.py
python -I docs/_static/img/document-figures/check.py
```

`-I` 让 Python 忽略额外的模块搜索路径与用户级包配置，绘图无需导入 PSRL。它不会修改已有环境。[render.py](render.py)保存全部布局、颜色、文字及教学数字，输出本目录的原始机制面板和缩略预览。第二条命令把这些面板与[专项解释输入](../document-figures/evidence-explanations.json)组合，生成正文正在使用的完整图；只运行第一条不会更新新版解释图。修改数字或布局时同时重新生成两种格式，正文中的条件和解释也要同步。图片采用文字、位置与颜色共同区分角色，角色标签和计时单位写入图中。

[原图提取与检查脚本](paper_figures.py)及[全目录补图审阅记录](paper-figure-review.json)保留维护入口。

本目录没有子目录。[返回图片导航](../README.md)
