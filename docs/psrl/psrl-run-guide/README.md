# 集群预检与运行故障

[上级导航](../README.md) · [总导航](../../README.md)

从下面的主题文档开始阅读，子目录入口列在文末。

近期遇到actor没起来，先读[Ray启动失败与导入阻塞](03-启动排障_单机与多机故障复盘.md#近期env实验ray启动失败与导入阻塞)：区分进程创建、worker注册和actor业务初始化，包含11月19日与23日的真实失败、24日栈线索和已做修复的边界。

训练前导入预检超过 180 秒时，查看副本对照与修复后的启动流程。 [依赖导入排查](03-启动排障_单机与多机故障复盘.md#180-秒导入预检的排查与修复)

<!-- figure:map-019e476519 -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![集群预检与运行故障：阅读地图（1/2）](../../_static/img/document-figures/map-019e476519-screen.png)](../../_static/img/document-figures/map-019e476519.png)

[![集群预检与运行故障：阅读地图（2/2）](../../_static/img/document-figures/map-019e476519-screen-02.png)](../../_static/img/document-figures/map-019e476519.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../_static/img/document-figures/map-019e476519.svg)。
<!-- /figure:map-019e476519 -->

## 文件与子目录

- [PSRL运行：准备启动与监控](01-PSRL运行_准备启动与监控.md)，其中的 [Ray 章节](01-PSRL运行_准备启动与监控.md#ray-角色与-rollout-流程)沿一条 Science 轨迹讲清 actor、资源放置和 EnvRouter 的分工。
- [集群预检：依赖地址与GPU资源核验](02-集群预检_依赖地址与GPU资源核验.md)
- [启动排障：单机与多机故障复盘](03-启动排障_单机与多机故障复盘.md)
- [独立故障记录](incidents/README.md)
