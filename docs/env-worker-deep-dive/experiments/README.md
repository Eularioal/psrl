# EnvWorker 实验与验证

> **核心结论**：按验证对象选择报告，先核对版本、工作负载、完成边界和计时口径。CPU 模拟、GPU 组件验证与真实训练的结论不能互相替代。报告中的历史「当前」只指其记录时的状态，理解实现请看[知识索引](../knowledge/README.md)。

向他人介绍项目时，可直接转发[Env 实现、实测收益与后续规划](env-status-and-roadmap.md)，其图示、条件和结果数据均包含在同一文件中。需要查完整实验记录时，再通过下表选择报告。

## 目录

- [真实训练对照](#真实训练对照)
- [调度与组件对照](#调度与组件对照)
- [故障与接入验证](#故障与接入验证)
- [脚本与原始结果](#脚本与原始结果)

## 真实训练对照

| 报告 | 验证对象与边界 |
|---|---|
| [Science 单卡真实训练对照](science-single-gpu-training-control.md) | 固定 OFF 与 ON 的显存压力、原生任务及多步训练。各节分别记录完成范围，三步修复后结果集中在第 12 节。 |
| [Science GPU 压力与 Router 训练对照](science-gpu-pressure-ab.md) | 跨任务池竞争、显存不足补采与等完成量比较。固定历史基线和更早的整卡锁对照分别标注。 |
| [真实 LAPS 训练的 Router 开关对照](science-training-router-ab.md) | 训练曲线、资源账本和回收。该轮任务容器使用 CPU，不能证明 GPU 容器装箱或同卡借还。 |

## 调度与组件对照

| 报告 | 验证对象与边界 |
|---|---|
| [修复后 Router 与固定 OFF 的单卡对照](router-baseline-control-validation.md) | 合成 CUDA 任务的等完成量、低负载开销、预热与长尾定位，不是完整模型训练。 |
| [同期配对实验与 Gate A/B 设计审视](router-paired-latency-and-design-review.md) | 低负载配对、剩余提案审计，以及资源限额、物理遥测、负载派发和清理确认的修复与验收边界。 |
| [Science 资源账本对照](science-env-router-comparison.md) | 旧整卡锁与预算共享的 GPU 容器实验，以及未完成训练对照的原因。 |
| [同任务多协程组件对照](science-natural-gpu-concurrency.md) | 单容器多命令的高低压力实验。文中已纠正需求范围，不能视为同题多 rollout 验证完成。 |

## 故障与接入验证

| 报告 | 验证对象与边界 |
|---|---|
| [Worker 重试与执行监督](../knowledge/worker-retry-and-redispatch.md#7-可运行验证与-results) | 真实 Ray worker 死亡、训练等待方错误传播和 Harbor 容器回收，不是完整 science 训练。 |
| [procfs 挂载障碍与实验处理](host-proc-blocker.md) | 现场 GPU 进程归属障碍、观察模式配置及未验证风险。 |
| [Env Service 设计与修复验证](../env-service-verification/README.md) | 早期进程、Docker、跨节点与训练功能验证，按报告中注明的版本和范围阅读。 |
| [AIRS-Bench 修复与小实验](../airs-bench-audit/fixes-and-results.md) | 评分隔离、shell 超时、资源回收，以及 MLGym 回放和 Docker 验证。 |
| [APPS 判题隔离实现与验证](../airs-bench-audit/apps-isolation.md) | judge 隔离、评分错误分类、小样本对照与吞吐验收边界。 |

## 脚本与原始结果

| 目录 | 用途 |
|---|---|
| [env-service-verification/scripts/](../env-service-verification/scripts/) | 共用复现与分析脚本，也包含知识文章的 CPU 教学示例。 |
| [env-service-verification/results/](../env-service-verification/results/) | 报告引用的原始结果与图表，以各报告链接为准。 |
| [gpu-job-validation/](../gpu-job-validation/) | GPU 作业接口与任务遥测的配套验证。 |
| [airs-bench-audit/](../airs-bench-audit/) | AIRS-Bench 的复现脚本、基线快照和审查证据。 |

需要重跑时，先读[集群运行约束](../../psrl/psrl-run-guide/cluster-preflight.md)，再使用具体报告的复现步骤。不能把目录里的历史输出当作本次验证结果。

[返回专题入口](../README.md) · [查看知识与实现](../knowledge/README.md)
