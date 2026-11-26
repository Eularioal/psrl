# 联合 GPU 放置实验脚本

[上级导航](../README.md) · [总导航](../../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [联合 GPU 放置实验](../../../experiments/gpu-benchmarks/03-动态放置_真实模型驻留与跨卡探索.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:two-gpu-command-queue -->
这张图沿同一个场景解释：老会话下条命令被堵住时，为什么可以考虑同机另一张空卡？

[![命令动态放置：原卡排队时，检查另一个候选设备（1/2）](../../../../_static/img/document-figures/two-gpu-command-queue-screen.png)](../../../../_static/img/document-figures/two-gpu-command-queue.png)

[![命令动态放置：原卡排队时，检查另一个候选设备（2/2）](../../../../_static/img/document-figures/two-gpu-command-queue-screen-02.png)](../../../../_static/img/document-figures/two-gpu-command-queue.png)

会话 S 的容器和文件留在节点 N；GPU0 排队，配置候选池中的 GPU1 有空余。 可移动的对象是下一条命令的设备选择。热状态不自动迁移，空卡也不保证更快。 仅解释配置池内同机调度，不表示跨节点迁移或默认借用全部推理 GPU。 [查看矢量图](../../../../_static/img/document-figures/two-gpu-command-queue.svg)。
<!-- /figure:two-gpu-command-queue -->

## 文件与子目录

- [analyze.py](analyze.py)
- [experiment.py](experiment.py)
- [planner.py](planner.py)
- [warm_control.py](warm_control.py)
- [worker.py](worker.py)
