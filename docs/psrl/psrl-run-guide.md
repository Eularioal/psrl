# PSRL 运行指南索引

> **核心结论**：原 840 行运行指南已拆分为 5 篇。先核对集群环境约束，再查启动步骤、Ray 调度、监控配置及历史故障复盘。

## 目录

- [阅读路径](#阅读路径)
- [专题文档](#专题文档)
- [迁移说明](#迁移说明)

## 阅读路径

实际运行前先读[共享环境与集群检查](psrl-run-guide/cluster-preflight.md)，再按需要查基础运行、调度、监控和故障复盘。

## 专题文档

- [PSRL 基础运行与示例脚本](psrl-run-guide/getting-started.md)：以 DAPO 示例说明安装、模型与数据准备、启动参数和单机 Demo；执行前先核对集群检查专题中的环境约束。
- [PSRL 的 Ray 调度与生成长尾](psrl-run-guide/ray-and-rollout.md)：分别说明 Ray 的角色、资源调度流程和 rollout 长尾，区分集群调度职责与模型计算职责。
- [PSRL 日志监控、批量配置与进程排查](psrl-run-guide/monitoring-and-batches.md)：从训练日志提取指标，理解 train_prompt_bsz 的配置映射，并确认训练进程所在节点及运行状态。
- [PSRL 单机与多机启动故障复盘](psrl-run-guide/startup-incident-reviews.md)：保留 IP、数据格式、环境变量覆盖及上下文长度问题的历史故障链、修复过程与验证边界。
- [PSRL 共享环境与集群检查](psrl-run-guide/cluster-preflight.md)：运行前核对源码与依赖来源、IP/Ray、临时路径、Docker、GPU 身份、数据和验证预算；2026-10-13 的证据只代表当时的条件。

## 迁移说明

2026-10-14 已按主题拆分为 5 篇子文档；本文件保留为索引。原有正文、例子、公式、代码块和历史实验记录已迁入对应专题，避免在索引中重复。已修复工作区内的章节引用；旧章节位置可通过下列对照表查找。

| 原章节 | 新位置 |
|---|---|
| 项目结构速览 | [阅读章节](psrl-run-guide/getting-started.md#项目结构速览) |
| 一、安装 | [阅读章节](psrl-run-guide/getting-started.md#安装) |
| 二、运行前准备 | [阅读章节](psrl-run-guide/getting-started.md#运行前准备) |
| 三、逐段拆解示例脚本 `examples/dapo_trainer/qwen2.5_3b_fsdp.sh` | [阅读章节](psrl-run-guide/getting-started.md#逐段拆解示例脚本-examplesdapo_trainerqwen25_3b_fsdpsh) |
| 四、监控指标 | [阅读章节](psrl-run-guide/getting-started.md#监控指标) |
| 五、实操建议 | [阅读章节](psrl-run-guide/getting-started.md#实操建议) |
| 延伸阅读 | [阅读章节](psrl-run-guide/getting-started.md#延伸阅读) |
| 六、为什么 RL rollout 会有“生成耗时不均”（long-tail latency） | [阅读章节](psrl-run-guide/ray-and-rollout.md#为什么-rl-rollout-会有生成耗时不均long-tail-latency) |
| 七、单机 Demo 脚本与常用 bash 语法：`qwen2.5_3b_fsdp_singlenode_demo.sh` | [阅读章节](psrl-run-guide/getting-started.md#单机-demo-脚本与常用-bash-语法qwen25_3b_fsdp_singlenode_demosh) |
| 八、单机 Demo 踩坑记录：两处根因与修复 | [阅读章节](psrl-run-guide/startup-incident-reviews.md#单机-demo-踩坑记录两处根因与修复) |
| 十一、Ray 在 psrl 里是做什么的 | [阅读章节](psrl-run-guide/ray-and-rollout.md#ray-的基础职责与-actor-协作) |
| 九、如何阅读训练日志、抓取每步指标 | [阅读章节](psrl-run-guide/monitoring-and-batches.md#如何阅读训练日志抓取每步指标) |
| 十、`train_prompt_bsz` 对应哪个配置、是什么意思 | [阅读章节](psrl-run-guide/monitoring-and-batches.md#train_prompt_bsz-对应哪个配置是什么意思) |
| 十一、Ray 在 PSRL 里究竟做什么：不只是"拉起集群" | [阅读章节](psrl-run-guide/ray-and-rollout.md#ray-的资源调度与应用启动流程) |
| 十二、让 `qwen2.5_3b_fsdp.sh` 跑通 2 节点实验：脚本改了什么 | [阅读章节](psrl-run-guide/startup-incident-reviews.md#让-qwen25_3b_fsdpsh-跑通-2-节点实验脚本改了什么) |
| 十三、laps/repair_easy 多机训练踩坑复盘：从 `max_model_len` 崩溃到重启，给下一次训练任务的检查清单 | [阅读章节](psrl-run-guide/startup-incident-reviews.md#lapsrepair_easy-多机训练踩坑复盘从-max_model_len-崩溃到重启给下一次训练任务的检查清单) |
| 十四、查一个训练 pid 的运行情况：去哪儿看、看什么 | [阅读章节](psrl-run-guide/monitoring-and-batches.md#查一个训练-pid-的运行情况去哪儿看看什么) |
| 十五、共享 conda env 里 `pip install -e` 是全局的：为什么日志/checkpoint 会悄悄跑到别人的目录下 | [阅读章节](psrl-run-guide/cluster-preflight.md#共享-conda-env-里-pip-install--e-是全局的为什么日志checkpoint-会悄悄跑到别人的目录下) |
| 十六、集群踩坑总表与最小复现（2026-10-13） | [阅读章节](psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13) |

本次为文档结构整理，不将历史实验视为当前验证，也未启动训练或集群实验。
