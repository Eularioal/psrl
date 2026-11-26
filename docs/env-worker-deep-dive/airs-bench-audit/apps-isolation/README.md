# APPS 隔离复现

[上级导航](../README.md) · [总导航](../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [APPS 隔离机制与证据](../02-APPS判题_代码隔离与验证边界.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:candidate-and-trusted-verifier -->
这张图沿同一个场景解释：候选程序怎样得到输入，同时不能读到隐藏答案或篡改最终分数？

[![候选执行与可信判题：输入可以交给候选，隐藏答案留在边界内（1/2）](../../../_static/img/document-figures/candidate-and-trusted-verifier-screen.png)](../../../_static/img/document-figures/candidate-and-trusted-verifier.png)

[![候选执行与可信判题：输入可以交给候选，隐藏答案留在边界内（2/2）](../../../_static/img/document-figures/candidate-and-trusted-verifier-screen-02.png)](../../../_static/img/document-figures/candidate-and-trusted-verifier.png)

同一次 APPS 检查：可信判题器拥有测试输入与答案，候选程序只负责计算输出。 隔离的是可见数据与写入权限。把候选进程放进容器，本身不足以隐藏挂载的答案。 APPS 隔离目标。已验证用例、兼容性与具体权限仍以正文为准。 [查看矢量图](../../../_static/img/document-figures/candidate-and-trusted-verifier.svg)。
<!-- /figure:candidate-and-trusted-verifier -->

## 文件与子目录

- [Dockerfile](Dockerfile)
- [cleanup.json](cleanup.json)
- [first-remote-results.json](first-remote-results.json)
- [real-reference-results.json](real-reference-results.json)
- [remote-results.json](remote-results.json)
- [review-error-classification.py](review-error-classification.py)
- [smoke.py](smoke.py)
- [summary.json](summary.json)
