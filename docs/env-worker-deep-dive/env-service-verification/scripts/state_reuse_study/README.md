# 状态复用实验脚本

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [等待热状态还是重建](../../../experiments/gpu-benchmarks/02-状态复用_等待与重新加载的前期实验.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:state-version-match -->
这张图沿同一个场景解释：文件变了以后，怎样避免常驻进程读旧版本，又保留可复用状态？

[![状态复用：保住昂贵准备，更新已变化的代码（1/4）](../../../../_static/img/document-figures/state-version-match-screen.png)](../../../../_static/img/document-figures/state-version-match.png)

[![状态复用：保住昂贵准备，更新已变化的代码（2/4）](../../../../_static/img/document-figures/state-version-match-screen-02.png)](../../../../_static/img/document-figures/state-version-match.png)

[![状态复用：保住昂贵准备，更新已变化的代码（3/4）](../../../../_static/img/document-figures/state-version-match-screen-03.png)](../../../../_static/img/document-figures/state-version-match.png)

[![状态复用：保住昂贵准备，更新已变化的代码（4/4）](../../../../_static/img/document-figures/state-version-match-screen-04.png)](../../../../_static/img/document-figures/state-version-match.png)

版本标签W1、D1、C1、v1/v2是教学记号。沿①磁盘与进程不一致、②依赖分类、③核对与局部重建、④正确性和成本对照阅读。正文的固定基线实验在CPU模型上验证，受控适配器使用文件快照、新评分进程及发布前核验，不是任意程序通用热更新。 [查看矢量图](../../../../_static/img/document-figures/state-version-match.svg)。
<!-- /figure:state-version-match -->

## 文件与子目录

- [baseline_compare.py](baseline_compare.py)
- [contract_checks.py](contract_checks.py)
- [memory_demo.py](memory_demo.py)
- [memory_experiment.py](memory_experiment.py)
- [memory_policy.py](memory_policy.py)
- [memory_replay.py](memory_replay.py)
- [memory_runtime.py](memory_runtime.py)
- [memory_study.py](memory_study.py)
- [memory_summarize.py](memory_summarize.py)
- [memory_walkthrough.py](memory_walkthrough.py)
- [model_process.py](model_process.py)
- [summarize.py](summarize.py)
- [workspace.py](workspace.py)
