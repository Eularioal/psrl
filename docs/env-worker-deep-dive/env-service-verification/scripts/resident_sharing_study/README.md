# 常驻模型计算共享实验脚本

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [短任务与长生成共享实验](../../../experiments/06-训推共享_常驻vLLM与环境并行实验.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:borrower-return-delay -->
这张图沿同一个场景解释：把局部额外等待、填空收益和安全归还放在同一个场景里。

[![GPU借还：A多等了，整批为什么仍可能更快？（1/4）](../../../../_static/img/document-figures/borrower-return-delay-screen.png)](../../../../_static/img/document-figures/borrower-return-delay.png)

[![GPU借还：A多等了，整批为什么仍可能更快？（2/4）](../../../../_static/img/document-figures/borrower-return-delay-screen-02.png)](../../../../_static/img/document-figures/borrower-return-delay.png)

[![GPU借还：A多等了，整批为什么仍可能更快？（3/4）](../../../../_static/img/document-figures/borrower-return-delay-screen-03.png)](../../../../_static/img/document-figures/borrower-return-delay.png)

[![GPU借还：A多等了，整批为什么仍可能更快？（4/4）](../../../../_static/img/document-figures/borrower-return-delay-screen-04.png)](../../../../_static/img/document-figures/borrower-return-delay.png)

教学简化协议：A空档6秒，之后计算4秒；B计算8秒；保存归还和恢复各1秒，并按串行计算。保留GPU时A在10秒、B在18秒结束；借给B时B在9秒、A在14秒结束。A多等4秒，而整批节省4秒。协议是否启用、预测条件与误差以正文为准。 [查看矢量图](../../../../_static/img/document-figures/borrower-return-delay.svg)。
<!-- /figure:borrower-return-delay -->

## 文件与子目录

- [analyze.py](analyze.py)
- [analyze_long.py](analyze_long.py)
- [experiment.py](experiment.py)
- [inference.py](inference.py)
- [long_experiment.py](long_experiment.py)
- [posttrain_worker.py](posttrain_worker.py)
- [protocol.py](protocol.py)
- [worker.py](worker.py)
