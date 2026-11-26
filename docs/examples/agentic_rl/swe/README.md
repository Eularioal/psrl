# SWE 软件修复

[上级导航](../README.md) · [总导航](../../../README.md)

从下面的主题文档开始阅读，子目录入口列在文末。

<!-- figure:software-patch-and-fresh-grader -->
这张图沿同一个场景解释：工作容器里测试通过，为什么还要在新容器里重新评分？

[![软件修复：只把提交补丁交给新的评分环境（1/2）](../../../_static/img/document-figures/software-patch-and-fresh-grader-screen.png)](../../../_static/img/document-figures/software-patch-and-fresh-grader.png)

[![软件修复：只把提交补丁交给新的评分环境（2/2）](../../../_static/img/document-figures/software-patch-and-fresh-grader-screen-02.png)](../../../_static/img/document-figures/software-patch-and-fresh-grader.png)

Agent 在原始仓库上反复编辑与测试，提交物是代码补丁，评分从干净基准重新开始。 新评分容器隔开探索状态与最终成果，避免依赖一次偶然留下的文件或进程。 SWE 类任务机制示意。测试协议与允许提交内容按具体 benchmark 确认。 [查看矢量图](../../../_static/img/document-figures/software-patch-and-fresh-grader.svg)。
<!-- /figure:software-patch-and-fresh-grader -->

## 文件与子目录

- [SWE Agentic RL](index.md)
