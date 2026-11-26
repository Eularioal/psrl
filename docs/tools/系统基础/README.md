# 计算机系统基础：环境、进程、文件、网络与GPU

这里集中理解PSRL和Env调度时常遇到的系统知识。默认已有机器学习背景，每篇用一个具体场景解释实际机制。编号给出阅读顺序，遇到问题时也可直接选读。

| 顺序 | 文档 | 从什么问题开始 |
|---|---|---|
| 01 | [运行环境：Shell与Python及Conda](01-运行环境_Shell与Python及Conda.md) | 为什么source能改变当前环境，而子进程的修改回不来 |
| 02 | [操作系统：进程与文件映射及SQLite](02-操作系统_进程与文件映射及SQLite.md) | 为什么程序读文件会遇到SIGBUS，文件怎样映射进进程内存 |
| 03 | [文件系统：inode与链接及Docker写回](03-文件系统_inode与链接及Docker写回.md) | 文件改名和删除后，正在运行的进程究竟持有什么 |
| 04 | [网络通信：HTTP请求与GateB调用链](04-网络通信_HTTP请求与GateB调用链.md) | 请求怎样经IP、端口和路径到达服务函数 |
| 05 | [GPU并行：CUDA上下文与计算竞争](05-GPU并行_CUDA上下文与计算竞争.md) | 显存放得下时，为什么多个任务仍会互相拖慢 |
| 06 | [远程登录：SSH 连接与 IP、网卡、端口](06-远程登录_SSH连接与IP网卡端口.md) | `SSH_CONNECTION` 四列怎么读，机器有多个 IP 时怎样核对这条连接 |

每篇正文保存完整解释、算例及对应代码入口。Linux的CPU演示脚本与原始结果继续由[脚本与证据目录](../linux/README.md)维护，GPU基础课引用的示例按正文说明运行。这里只整理知识入口，本次没有重跑历史演示或GPU实验。

继续读[PSRL入门](../../overview/01-PSRL入门_从一道题到模型更新.md)、[推理与训练专题](../../psrl/推理与训练/README.md)或[Env机制](../../env-worker-deep-dive/knowledge/README.md)。

本目录无子目录。[返回工具导航](../README.md) · [返回总导航](../../README.md)

<!-- figure:map-7a7363b66b -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![计算机系统基础：环境、进程、文件、网络与GPU：阅读地图（1/2）](../../_static/img/document-figures/map-7a7363b66b-screen.png)](../../_static/img/document-figures/map-7a7363b66b.png)

[![计算机系统基础：环境、进程、文件、网络与GPU：阅读地图（2/2）](../../_static/img/document-figures/map-7a7363b66b-screen-02.png)](../../_static/img/document-figures/map-7a7363b66b.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../_static/img/document-figures/map-7a7363b66b.svg)。
<!-- /figure:map-7a7363b66b -->

