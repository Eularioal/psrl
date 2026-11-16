# Env 实验与验证入口

> **核心结论**：想了解结论，先读[实验结论简报](env-design-and-results-brief.md)；想选GPU研究题，读[任务与实验导览](gpu-benchmarks/README.md)。每份报告注明版本、工作量和计时范围，历史故障与原始证据在专门目录保留。

## 目录

- [固定基线性能对照](#固定基线性能对照)
- [功能验收与当前机制](#功能验收与当前机制)
- [研究探索](#研究探索)
- [历史与审计](#历史与审计)

## 固定基线性能对照

| 报告 | 有效结论与边界 |
|---|---|
| [Science 修复后三步训练](science-single-gpu-training-control.md) | 人工高压力下减少显存不足重试，完成相同三步目标，完整时间少27.59% |
| [常驻模型整卡压力](gpu-benchmarks/resident-model-pressure.md) | 真实 Mistral 服务、相同 7 项任务 14 请求，完整时间少 10.6%，没有 PSRL 策略训练 |
| [低负载同期配对](router-baseline-control-validation.md) | 修复后两对总时间相差约 1.49%，未证明稳定提速，替代最早高开销参考 |

## 功能验收与当前机制

| 报告 | 验收范围 |
|---|---|
| [GPU任务验收证据](gpu-benchmarks/FINDINGS.md) | 按问题定位固定运行；[任务路线图](../milestones/science-gpu-bench-status-and-roadmap.md)维护当前完整数量与缺口 |
| [MPS 修复后配对](mps-training-comparison.md) | 真实计算并发与控制修复，两组都开 Router，不是固定 OFF/ON |
| [原生后端 lifecycle 修复](native-backend-fixes-and-science.md) | 当时 CPU LAPS oracle 接线的有限对照，不代表原生 GPU 仍未完成 |
| [Worker 重试与监督](../knowledge/worker-retry-and-redispatch.md#7-可运行验证与-results) | Ray worker 死亡、错误传播和所属容器回收 |
| [进程归属部署条件](host-proc-blocker.md) | procfs 条件与可选 eBPF 归属，不将历史节点故障推广到所有部署 |
| [AIRS GPU 接入](../knowledge/airs-gpu-recipe.md)、[接入审计](../airs-bench-audit/fixes-and-results.md)与[APPS 隔离](../airs-bench-audit/apps-isolation.md) | 19 个非 Kaggle 任务的有界 GPU 与代表闭环证据、当前 CPU 回归及剩余缺口，历史回放保留原条件 |

## 研究探索

- [常驻 vLLM 与环境共享计算](resident-compute-sharing.md)：三策略真实对照已完成，合作式暂停可正确继续，小图像任务上未胜过现有共享。
- [热卡等待与冷卡加载的对照实验](gpu-benchmarks/warm-placement-matrix.md)：比较直接等待、优先复制与预测选择，用实测结果解释等待和加载的取舍。
- [等待或重建的代码阅读指南](../knowledge/warm-placement-code-guide.md)：研究执行链、协程并发、类型化成本、准入与清理，附 CPU 教学实验。
- [热卡等待还是冷卡重建](gpu-benchmarks/warm-wait-or-rebuild.md)：前期真实 PSRL 固定步数长命令的历史探索，保留其完成数量与计时边界。
- [PostTrain 候选评分准备复用](gpu-benchmarks/posttrain-verifier-reuse.md)：九次真实 GPU 评分，已加载模型的请求变快，但完整短批次未胜过普通进程复用。
- [GPU 性能反馈压力实验](gpu-benchmarks/feedback-pressure.md)：144 组矩阵与 15 组持续竞争，简单保护避免严重选优错误，自适应未胜过固定两轮。

- [同机 command 池](../research-proposals/dynamic-command-placement.md#14-results真实-ray-与相同工作量对照)：同版本机制验证与安全边界。
- [真实服务准备前移](../research-proposals/non-timing-gpu-scheduling-scenarios.md#results真实-score-服务题与准备前移对照)：减少预算持有，未测得整组加速。
- [FlashInfer 批量与分段](gpu-benchmarks/flashinfer-batching.md)：进程复用、候选筛选与分段可行性，尚无新联合调度收益。

## 历史与审计

- [历史实验索引](history/README.md)：旧整任务保留、旧整卡锁、容量不对称配置、修复前失败与完整时间线。
- [全量文档审查](../../tools/docs-freshness-audit.md#7-2026-11-11-实验文档全量审查与重整)：逐文件清单、替代关系与核验边界。
- [脚本与原始证据](../env-service-verification/README.md)：重跑前必须重新预检，不能沿用历史空闲状态。
- [当前设计](../knowledge/current-design.md)：查现在的代码能力。阶段报告和实验快照不能覆盖它。
