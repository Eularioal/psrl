# GPU 环境调度：研究问题与验证进展

> **核心结论**：研究主线是让一组 Agent 尽快获得可信反馈，同时保护资源和代码版本。先看真实任务是否存在问题，再比较简单策略能做到什么，最后决定是否需要新算法。下面把已实现基础、仍在检验的想法和收窄后的原型分开列出。


## 目录

- [阅读入口](#阅读入口)
- [文档](#文档)
- [怎样解读证据](#怎样解读证据)

## 阅读入口

了解实际能力先读[当前设计](../knowledge/current-design.md)。研究选题建议按“[真实任务核查](benchmark-state-reuse-audit.md) → 对应提案 → [实验报告](../experiments/README.md)”阅读。这样可以先理解场景，再判断方案是否带来额外价值。

## 文档

| 研究问题 | 文档 | 当前定位 |
|---|---|---|
| 什么状态真的值得复用 | [benchmark 状态核查](benchmark-state-reuse-audit.md) | 真实参考补丁、状态寿命和计时；定位 RE 源码读取错误，收窄人工缓存竞争假设 |
| 如何让整组更早获得可信反馈 | [command 调度研究](command-scheduling-research.md) | 任务范围、反馈精度、时长预测与依赖；各候选分别验收 |
| 下一条命令能否换到空闲 GPU | [动态命令放置](dynamic-command-placement.md) | 同机 direct 命令池已实现；跨节点和活状态迁移仍需设计 |
| 等待热卡还是在冷卡重建模型 | [系统对照矩阵](../experiments/gpu-benchmarks/warm-placement-matrix.md)、[代码指南](../knowledge/warm-placement-code-guide.md) | 核心配对完整；长阻塞有复制空间，边界场景仍需与简单亲和比较 |
| 只按正确性评分时调度什么 | [非计时场景](non-timing-gpu-scheduling-scenarios.md) | 准备前移与服务题实验；当前优先核查真实 checkpoint 和评估版本 |
| 缓存装不下全部准备状态怎么办 | [主存受限原型](memory-bounded-state-scheduling.md) | 受控四会话模型，保留机制证据；真实任务代表性与稳定增量不足 |
| vLLM 有新请求时怎样及时收回计算 | [可撤回执行与反馈依赖](reclaim-aware-env-sharing.md)、[真实共存对照](../experiments/resident-compute-sharing.md) | 合作式适配器可正确继续，当前小图像任务未胜过现有共享；继续研究须先找到真实干扰与依赖 |

前期探索保存在[独立命令放置实验](../experiments/gpu-benchmarks/joint-gpu-placement.md)和[热卡等待/冷卡重建](../experiments/gpu-benchmarks/warm-wait-or-rebuild.md)。它们用于解释后续系统矩阵的设计与条件。

## 怎样解读证据

“有界原型”指固定资源、任务数与时间预算下的受控验证。例如两个模型缓存共 3.91 GiB，3 GiB 主存池装不下两份、5 GiB 装得下，这能研究淘汰与重建，却还要检查真实 Agent 是否反复使用这些不变模型。局部请求变快、整个任务批次变快、策略训练效果提高，是三个分别测量的结果；每篇以实际完成范围陈述结论。

[文献调研](../literature-reviews/README.md) · [返回专题入口](../README.md) · [返回总索引](../../README.md)
