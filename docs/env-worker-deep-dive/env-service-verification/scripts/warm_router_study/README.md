# 热状态放置实验脚本

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [热状态放置矩阵](../../../experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:wait-or-reload -->
这张图沿同一个场景解释：同一个请求换卡后更早完成，为什么整批收益又不同？

[![动态放置：等热卡，还是在冷卡重建？（1/4）](../../../../_static/img/document-figures/wait-or-reload-screen.png)](../../../../_static/img/document-figures/wait-or-reload.png)

[![动态放置：等热卡，还是在冷卡重建？（2/4）](../../../../_static/img/document-figures/wait-or-reload-screen-02.png)](../../../../_static/img/document-figures/wait-or-reload.png)

[![动态放置：等热卡，还是在冷卡重建？（3/4）](../../../../_static/img/document-figures/wait-or-reload-screen-03.png)](../../../../_static/img/document-figures/wait-or-reload.png)

[![动态放置：等热卡，还是在冷卡重建？（4/4）](../../../../_static/img/document-figures/wait-or-reload-screen-04.png)](../../../../_static/img/document-figures/wait-or-reload.png)

统一教学条件：A是新请求，执行4秒；B正在热卡运行，还需12秒；冷卡加载A需5秒。A留热卡16秒完成，转冷卡9秒完成，A、B整批分别16秒和12秒。另一个分支只把B剩余改成1秒，说明保留热卡更好。文件和主存留在同机容器，GPU模型在冷卡重新准备。 [查看矢量图](../../../../_static/img/document-figures/wait-or-reload.svg)。
<!-- /figure:wait-or-reload -->

## 文件与子目录

- [analyze.py](analyze.py)
- [analyze_long.py](analyze_long.py)
- [analyze_matrix.py](analyze_matrix.py)
- [asyncio_warm_demo.py](asyncio_warm_demo.py)
- [baseline.py](baseline.py)
- [client.py](client.py)
- [demo.py](demo.py)
- [experiment.py](experiment.py)
- [gpu_sampler.py](gpu_sampler.py)
- [guarded.py](guarded.py)
- [long_study.py](long_study.py)
- [main.py](main.py)
- [matrix.py](matrix.py)
- [matrix_suite.py](matrix_suite.py)
- [metadata_cache.py](metadata_cache.py)
- [plot_matrix.py](plot_matrix.py)
- [prediction_provider.py](prediction_provider.py)
- [runtime_guard.py](runtime_guard.py)
- [server.py](server.py)
- [suite.py](suite.py)
- [telemetry.py](telemetry.py)
