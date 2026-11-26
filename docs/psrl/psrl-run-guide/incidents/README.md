# 独立故障记录

[上级导航](../README.md) · [总导航](../../../README.md)

从下面的主题文档开始阅读，子目录入口列在文末。

<!-- figure:docker-writeback-wait -->
这张图沿同一个场景解释：显存已经下降，Docker 为什么还迟迟不能完成清理？

[![Docker挂起：线程等待写回，与磁盘还有多少空间是两条线索（1/2）](../../../_static/img/document-figures/docker-writeback-wait-screen.png)](../../../_static/img/document-figures/docker-writeback-wait.png)

[![Docker挂起：线程等待写回，与磁盘还有多少空间是两条线索（2/2）](../../../_static/img/document-figures/docker-writeback-wait-screen-02.png)](../../../_static/img/document-figures/docker-writeback-wait.png)

排障场景：GPU 工作已结束，任务仍卡在容器退出或文件系统清理阶段。 GPU、容器服务和存储是不同资源。释放一类资源，不会自动结束另一类等待。 机制排障图，未对当前机器下故障诊断。历史恢复动作不是可直接重放的处方。 [查看矢量图](../../../_static/img/document-figures/docker-writeback-wait.svg)。
<!-- /figure:docker-writeback-wait -->

## 文件与子目录

- [Docker故障：2026年10月20日存储挂起](01-Docker故障_2026年10月20日存储挂起.md)
