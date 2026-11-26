# PostTrain 裁判复用实验脚本

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [PostTrain 裁判模型复用](../../../experiments/gpu-benchmarks/05-评分复用_PostTrain准备成本与整批耗时.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:two-model-training-loops -->
这张图沿同一个场景解释：Agent 让环境训练了一个模型，PSRL 更新的到底是哪一个模型？

[![两个模型、两层训练：工具里的模型与Agent策略分开更新（1/2）](../../../../_static/img/document-figures/two-model-training-loops-screen.png)](../../../../_static/img/document-figures/two-model-training-loops.png)

[![两个模型、两层训练：工具里的模型与Agent策略分开更新（2/2）](../../../../_static/img/document-figures/two-model-training-loops-screen-02.png)](../../../../_static/img/document-figures/two-model-training-loops.png)

策略模型 θ 提出训练方案，环境任务模型 φ 执行更新并保存产物，评分器返回质量。 φ 的内部训练次数与 θ 的策略 step 属于两层循环。任务产物不会自动成为下一轮 Agent。 θ、φ 是教学符号。任务模型接管 Agent 是额外设计，需要能力评估与发布协议。 [查看矢量图](../../../../_static/img/document-figures/two-model-training-loops.svg)。
<!-- /figure:two-model-training-loops -->

## 文件与子目录

- [bounded_worker.py](bounded_worker.py)
- [explain_reuse.py](explain_reuse.py)
- [run_study.py](run_study.py)
- [summarize.py](summarize.py)
- [test_bounded_worker.py](test_bounded_worker.py)
