# Env任务接入：公共协议与各题库配方

先看公共接入主线，再按题库选择配方。配方、容器CUDA验证、任务评分和完整训练的证据层次各有条件，使用前核对对应章节。

| 顺序 | 文档 | 主要回答 |
|---|---|---|
| 01 | [任务接入：公共GPU任务与PSRL配方](01-任务接入_公共GPU任务与PSRL配方.md) | 数据、任务容器、GPU命令和PSRL怎样接起来 |
| 02 | [任务接入：EnvWorker配置与CUDA验证](02-任务接入_EnvWorker配置与CUDA验证.md) | 自定义环境怎样调用EnvWorker并验证真实GPU执行 |
| 03 | [AIRSBench：接入流程与有界验收](03-AIRSBench_接入流程与有界验收.md) | 怎样准备AIRS任务并选择最小验收 |
| 04 | [MLGym与SWE及Serve：任务协议与评分训练](04-MLGym与SWE及Serve_任务协议与评分训练.md) | 提交什么，评分器读取什么，策略模型训练什么 |
| 05 | [REBench：聊天模型训练接入](05-REBench_聊天模型训练接入.md) | 怎样接入聊天模型训练任务与资产 |
| 06 | [InferenceBench：评分机制与开关验证](06-InferenceBench_评分机制与开关验证.md) | 性能评分怎样产生，Env开关对照测到了什么 |

题库覆盖与阶段介绍见[Science任务导航](../../milestones/README.md)。选择研究场景见[基准调研](../../literature-reviews/05-基准调研_GPU任务类型与接入顺序.md)。

本目录无子目录。[返回机制导航](../README.md) · [返回总导航](../../../README.md)

<!-- figure:map-0e0b082c0a -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![Env任务接入：公共协议与各题库配方：阅读地图（1/2）](../../../_static/img/document-figures/map-0e0b082c0a-screen.png)](../../../_static/img/document-figures/map-0e0b082c0a.png)

[![Env任务接入：公共协议与各题库配方：阅读地图（2/2）](../../../_static/img/document-figures/map-0e0b082c0a-screen-02.png)](../../../_static/img/document-figures/map-0e0b082c0a.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../../_static/img/document-figures/map-0e0b082c0a.svg)。
<!-- /figure:map-0e0b082c0a -->

