# Env架构与执行：从任务申请到容器回收

先读当前架构，再看执行层怎样实现它。编号给出推荐顺序，每篇仍可按问题独立查阅。

| 顺序 | 文档 | 主要回答 |
|---|---|---|
| 01 | [Env架构：现状与完整执行流程](01-Env架构_现状与完整执行流程.md) | 谁申请资源，谁创建容器，命令结束后归还什么 |
| 02 | [统一执行：EnvRouter与容器及预算策略](02-统一执行_EnvRouter与容器及预算策略.md) | 调度、容器提供者和执行策略怎样分工 |
| 03 | [容器后端：EnvWorker与Harbor统一实现](03-容器后端_EnvWorker与Harbor统一实现.md) | 两种容器创建路径怎样共用执行与生命周期接口 |
| 04 | [GPU作业：生命周期与任务遥测](04-GPU作业_生命周期与任务遥测.md) | 怎样提交、等待、取消与观察一个GPU作业 |
| 05 | [EnvRouter：配置恢复与Worker重试](05-EnvRouter_配置恢复与Worker重试.md) | 怎样配置资源，失败后在哪一层重试 |
| 06 | [工程审查：待修复问题与优先级](06-工程审查_待修复问题与优先级.md) | 哪些工程问题仍影响正确性与恢复 |

接着理解容量和命令借还，进入[资源与调度](../02-资源与调度/README.md)。要接新题库，进入[任务接入](../03-任务接入/README.md)。现有实现、研究原型与提案以各文注明的证据范围为准。

本目录无子目录。[返回机制导航](../README.md) · [返回总导航](../../../README.md)

<!-- figure:map-37db5ffc0f -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![Env架构与执行：从任务申请到容器回收：阅读地图（1/2）](../../../_static/img/document-figures/map-37db5ffc0f-screen.png)](../../../_static/img/document-figures/map-37db5ffc0f.png)

[![Env架构与执行：从任务申请到容器回收：阅读地图（2/2）](../../../_static/img/document-figures/map-37db5ffc0f-screen-02.png)](../../../_static/img/document-figures/map-37db5ffc0f.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../../_static/img/document-figures/map-37db5ffc0f.svg)。
<!-- /figure:map-37db5ffc0f -->

