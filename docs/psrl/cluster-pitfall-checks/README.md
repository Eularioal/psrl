# 集群问题检查脚本与证据

[上级导航](../README.md) · [总导航](../../README.md)

本目录保存集群问题的最小检查脚本和历史 JSON。[集群预检](../psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)统一解释检查顺序、现象与修复。历史探测结果仅反映当时节点，执行前仍需现场检查。

<!-- figure:evidence-levels -->
这张图沿同一个场景解释：一次检查通过，究竟能支持多大的结论？

[![验证证据：不同实验分别回答不同问题（1/2）](../../_static/img/document-figures/evidence-levels-screen.png)](../../_static/img/document-figures/evidence-levels.png)

[![验证证据：不同实验分别回答不同问题（2/2）](../../_static/img/document-figures/evidence-levels-screen-02.png)](../../_static/img/document-figures/evidence-levels.png)

同一套调度代码可能通过 CPU 回归，但容器 CUDA、真实评分和训练仍各有条件。 验证是结论与证据的对应关系，不是每次必须爬完的一整套阶梯。 证据类别示意，不把进程存活、历史成功或测试替身写成本次完整训练。 [查看矢量图](../../_static/img/document-figures/evidence-levels.svg)。
<!-- /figure:evidence-levels -->

## 文件与子目录

- [docker-forensics-2026-10-20.json](docker-forensics-2026-10-20.json)
- [docker-recovery-2026-10-20.json](docker-recovery-2026-10-20.json)
- [docker_forensics.py](docker_forensics.py)
- [probe.py](probe.py)
- [results-2026-10-13.json](results-2026-10-13.json)
- [source-checks-2026-10-13.json](source-checks-2026-10-13.json)
