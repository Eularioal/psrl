# FlashInfer 实验依赖

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [FlashInfer 实验条件与依赖](../07-FlashInfer_进程复用与任务合批.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:shell-and-python -->
这张图沿同一个场景解释：source 环境脚本后，哪些内容能被子进程继承，哪些状态仍各自独立？

[![运行环境：子进程继承环境，Python导入状态各自保存（1/2）](../../../../_static/img/document-figures/shell-and-python-screen.png)](../../../../_static/img/document-figures/shell-and-python.png)

[![运行环境：子进程继承环境，Python导入状态各自保存（2/2）](../../../../_static/img/document-figures/shell-and-python-screen-02.png)](../../../../_static/img/document-figures/shell-and-python.png)

同一 Shell source 环境脚本，然后分别启动 Python P1 和 P2。 环境继承、文件缓存与进程内模块复用处在不同层。保留容器不等于保留 Python 模型。 Python 内存与环境语义教学，未测 import 时间。实际包来源要在 driver 和 worker 分别核对。 [查看矢量图](../../../../_static/img/document-figures/shell-and-python.svg)。
<!-- /figure:shell-and-python -->

## 文件与子目录

- [evidence.json](evidence.json)
