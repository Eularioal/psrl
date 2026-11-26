# 辅助脚本

[上级导航](../README.md) · [总导航](../../../README.md)

[会话管理](../session-management/01-Codex会话_恢复命名与历史回看.md)介绍终端标题演示，[历史恢复](../session-management/03-Codex故障_会话数据库与SIGBUS排查.md)说明修复脚本的适用条件。修改会话数据库前先按恢复文档备份。

<!-- figure:evidence-levels -->
这张图沿同一个场景解释：一次检查通过，究竟能支持多大的结论？

[![验证证据：不同实验分别回答不同问题（1/2）](../../../_static/img/document-figures/evidence-levels-screen.png)](../../../_static/img/document-figures/evidence-levels.png)

[![验证证据：不同实验分别回答不同问题（2/2）](../../../_static/img/document-figures/evidence-levels-screen-02.png)](../../../_static/img/document-figures/evidence-levels.png)

同一套调度代码可能通过 CPU 回归，但容器 CUDA、真实评分和训练仍各有条件。 验证是结论与证据的对应关系，不是每次必须爬完的一整套阶梯。 证据类别示意，不把进程存活、历史成功或测试替身写成本次完整训练。 [查看矢量图](../../../_static/img/document-figures/evidence-levels.svg)。
<!-- /figure:evidence-levels -->

## 文件与子目录

- [repair-session-history.py](repair-session-history.py)
- [terminal-title-demo.py](terminal-title-demo.py)
