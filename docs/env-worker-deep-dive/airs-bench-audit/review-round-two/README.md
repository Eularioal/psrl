# AIRS 第二轮审查

[上级导航](../README.md) · [总导航](../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [AIRS 两轮审查与结果](../01-AIRS审计_修复状态与验证范围.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:evidence-levels -->
这张图沿同一个场景解释：一次检查通过，究竟能支持多大的结论？

[![验证证据：不同实验分别回答不同问题（1/2）](../../../_static/img/document-figures/evidence-levels-screen.png)](../../../_static/img/document-figures/evidence-levels.png)

[![验证证据：不同实验分别回答不同问题（2/2）](../../../_static/img/document-figures/evidence-levels-screen-02.png)](../../../_static/img/document-figures/evidence-levels.png)

同一套调度代码可能通过 CPU 回归，但容器 CUDA、真实评分和训练仍各有条件。 验证是结论与证据的对应关系，不是每次必须爬完的一整套阶梯。 证据类别示意，不把进程存活、历史成功或测试替身写成本次完整训练。 [查看矢量图](../../../_static/img/document-figures/evidence-levels.svg)。
<!-- /figure:evidence-levels -->

## 文件与子目录

- [scoped-checks.json](scoped-checks.json)
- [summary.json](summary.json)
