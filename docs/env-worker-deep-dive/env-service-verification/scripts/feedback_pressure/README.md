# 性能测量干扰实验脚本

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [性能反馈压力实验与复现](../../../experiments/gpu-benchmarks/08-性能反馈_高压力调度实验.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:measurement-contention -->
这张图沿同一个场景解释：候选代码没有变，为什么它的计时分数却变差？

[![性能反馈：同一段代码，在不同竞争条件下会得到不同计时（1/2）](../../../../_static/img/document-figures/measurement-contention-screen.png)](../../../../_static/img/document-figures/measurement-contention.png)

[![性能反馈：同一段代码，在不同竞争条件下会得到不同计时（2/2）](../../../../_static/img/document-figures/measurement-contention-screen-02.png)](../../../../_static/img/document-figures/measurement-contention.png)

教学条件：同一份候选与输入，单独执行需 2 个时间单位，有竞争时需 4 个单位。 调度可以改善整批时间，同时改变单次测速结果。把两种指标混用会误判代码质量。 数字为教学假设。竞争敏感性因候选而异，重复测量未必消除系统偏差。 [查看矢量图](../../../../_static/img/document-figures/measurement-contention.svg)。
<!-- /figure:measurement-contention -->

## 文件与子目录

- [analyze.py](analyze.py)
- [driver.py](driver.py)
- [prepare.py](prepare.py)
- [worker.py](worker.py)
