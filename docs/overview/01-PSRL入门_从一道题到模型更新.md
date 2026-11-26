# PSRL入门：从一道题到模型更新

> **核心结论**：PSRL 把“让模型做题”和“根据做题结果更新模型”分开执行，让两边能重叠工作。生成结果通过样本存储交给训练端，新模型参数通过参数服务器交回生成端。调度器还要限制样本有多旧，避免训练一直使用落后太多的结果。

## 目录

- [先把完整流程连起来](#先把完整流程连起来)
- [跟着一道题走一遍](#跟着一道题走一遍)
- [为什么需要这些组件](#为什么需要这些组件)
- [跟着图读源码](#跟着图读源码)
- [运行一个不需要 GPU 的例子](#运行一个不需要-gpu-的例子)
- [按自己的问题继续阅读](#按自己的问题继续阅读)
- [核对范围](#核对范围)

## 先把完整流程连起来

本文默认已有机器学习与 agent 基础，重点看 PSRL 怎样衔接生成、样本存储、训练和参数分发。下面是主要数据流，图里省略了重试、验证和可选优化。

[![PSRL 做题、环境执行、样本收集、训练与参数发布之间的数据流（1/3）](../_static/img/document-figures/explained-psrl-training-loop-b6943fc-screen.png)](../_static/img/document-figures/explained-psrl-training-loop-b6943fc.png)

[![PSRL 做题、环境执行、样本收集、训练与参数发布之间的数据流（2/3）](../_static/img/document-figures/explained-psrl-training-loop-b6943fc-screen-02.png)](../_static/img/document-figures/explained-psrl-training-loop-b6943fc.png)

[![PSRL 做题、环境执行、样本收集、训练与参数发布之间的数据流（3/3）](../_static/img/document-figures/explained-psrl-training-loop-b6943fc-screen-03.png)](../_static/img/document-figures/explained-psrl-training-loop-b6943fc.png)

<!-- explained-source:explained-psrl-training-loop-b6943fc -->
**读图结论：**生成路由与 EnvRouter 各管不同队列，工具更快还要通过批次依赖才影响训练开始。 图省略部分版本控制箭头，调用关系与源码定位在正文补齐。 [完整解释图（SVG）](../_static/img/document-figures/explained-psrl-training-loop-b6943fc.svg) · [原始机制图或实验图](../_static/img/literature-reviews/psrl-training-loop.png)。
<!-- /explained-source:explained-psrl-training-loop-b6943fc -->

[查看可放大的 SVG](../_static/img/document-figures/explained-psrl-training-loop-b6943fc.svg)。先看上排的一次交互：Worker 请求模型生成，按任务需要调用环境工具，再将输出交回原任务继续。任务产生轨迹后，沿下排进入评分、样本存储和训练，新参数再回到生成实例。参数服务还参与请求版本和样本位置检查，图中为保持主线清楚省略这条控制关系，下表单独列出。

图中有两个容易混淆的调度对象：生成 Router 分发模型请求，EnvRouter 管环境的 CPU、主存和 GPU 预算。模型推理和工具执行可以各自排队。工具快一点，可能让下一轮生成更早就绪；但训练端什么时候开始更新，还要看规定的一批样本是否齐备，因此一条 episode 完成不会自动触发一次训练更新。

图例：图里每个部分对应的源码位置。

| 编号 | 在做什么 | 源码入口 |
|---|---|---|
| 1 | 把待处理题目交给 Agent Worker | [Manager 的训练派发循环](../../psrl/workers/agent_loop/manager.py#L432) |
| 2 | 创建任务，执行具体 Agent 流程 | [generate_trajectory](../../psrl/workers/agent_loop/worker.py#L262)、[_run_agent_loop_inner](../../psrl/workers/agent_loop/worker.py#L313) |
| 3 | 启动生成网关，接入外部 SMG 路由实现 | [launch_router](../../psrl/workers/gen/rollout_gateway.py#L123) |
| 4 | 运行模型生成服务 | [vllm_async_server.py](../../psrl/workers/gen/vllm_async_server.py) |
| 5 | 写入模型输出的训练字段，组织可训练批次 | [postprocess_output](../../psrl/workers/agent_loop/worker.py#L461)、[wait_for_training_batch](../../psrl/workers/agent_loop/manager.py#L1918) |
| 6 | 读取一批数据，计算奖励和优势，执行参数更新 | [FullBatchStepStrategy.run_step](../../psrl/trainer/ppo/strategies/full_batch.py#L26) |
| 7 | 发布参数版本，通知生成协调器 | [push_model_state_dict_nixl](../../psrl/workers/ps/ps_manager.py#L1194) |
| 3 ↔ 7 | 判断请求能否获得预留位置 | [can_reserve_request](../../psrl/workers/ps/ps_manager.py#L356) |

SMG 是单独的路由组件，本仓库包含它的启动和集成代码。图例中的本地入口可用于追踪集成关系，完整路由选择算法需结合 SMG 源码阅读。

## 跟着一道题走一遍

以下是**教学假设**，数字用来解释流程，不是实测结果。假设题目是“2 + 3 等于多少”，模型版本是 10，每道题生成两种回答，一个训练批次需要两道题。

1. **派发题目。** Manager 把题目和采样要求交给 Worker。Worker 负责串联过程，真正执行模型计算的是生成实例。
2. **生成两次。** Router 选择能服务版本 10 且有准入空间的实例，得到两条轨迹。本例分别回答“5”和“6”。
3. **给结果打分。** 两条轨迹得到奖励 `[1, 0]`，与生成时使用的版本一起进入后续处理。
4. **存数据、等一批。** 回答的 token、分数和版本号写入 TransferQueue。另一道题也完成后，两个题目组组成一批。训练端先收到样本键等轻量信息，再按需读取大数组，避免所有数组都由中心协调进程反复转发。
5. **更新参数。** 训练端读取这批轨迹，按所选算法计算更新，得到下一版参数。
6. **发布版本 11。** 训练端把新参数交给参数服务器。生成实例按协调策略获取新版本，之后的回答逐渐使用版本 11。

两边可以重叠：训练端处理这批回答时，生成端可能正在做下一批题。代价是部分新到的样本仍由旧模型生成，因此需要限制**陈旧度**，即训练使用样本时，参数版本比生成时领先多少步。例如训练端处于版本 13、样本始于版本 10，版本差就是 3。

如果任务要求运行 Python 代码，步骤 2 还会插入“Agent 发命令 → 环境容器执行 → 返回输出 → Agent 决定下一步”的循环。容器的 CPU、内存和 GPU 需求由环境系统管理，见[环境当前设计](../env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)。一次环境训练任务与 PSRL 更新主模型的训练步骤，是两种不同工作。

## 为什么需要这些组件

| 组件 | 如果没有它，会遇到什么问题 | 在上例中如何使用 |
|---|---|---|
| Agent Worker | 多轮生成、工具和奖励没人串起来 | 组织同一道题的两次回答及结果处理 |
| Router（路由器） | 多个生成实例之间缺少统一选择和版本检查 | 为每次生成挑选可服务的实例 |
| TransferQueue（样本存储） | 组件反复通过中心进程搬运整批大数组 | 按样本键存取回答和分数 |
| 参数服务器 | 训练端的新参数缺少统一发布和分发位置 | 保存版本 11，供生成实例拉取 |
| 陈旧度控制 | 训练可能不断收到落后很多版本的样本 | 约束版本 10 的回答最晚能用于哪一步 |

这些机制解决的是工作衔接和资源利用。实际速度还取决于模型、题目长度、工具耗时、网络和资源分配；需要相同工作量的对照实验来判断收益。

## 跟着图读源码

先看流程，再按下表进入实现。无需先读完所有底层文件。

| 步骤 | 图中编号 | 文件与函数 | 输入、输出或状态变化 |
|---|---|---|---|
| 启动流水线 | 1–7 | [ray_trainer.py：fit](../../psrl/trainer/ppo/ray_trainer.py#L3361) | 连接 Manager、奖励与生成协调器，启动后台工作 |
| 派发 | 1 | [manager.py：_train_dispatch_data](../../psrl/workers/agent_loop/manager.py#L432) | 从待办队列取题，送往 Worker |
| 执行轨迹 | 2–5 | [worker.py：_run_agent_loop_inner](../../psrl/workers/agent_loop/worker.py#L313) | 根据配置运行 Agent；内部可有生成、工具和重试分支 |
| 等待批次 | 5 → 6 | [manager.py：wait_for_training_batch](../../psrl/workers/agent_loop/manager.py#L1918) | 等待目标缓冲区就绪，返回样本元数据 |
| 更新模型 | 6 → 7 | [full_batch.py：run_step](../../psrl/trainer/ppo/strategies/full_batch.py#L26) | 读取数据并执行所选算法的训练阶段 |
| 更新版本 | 7 → 4 | [ps_manager.py：push_model_state_dict_nixl](../../psrl/workers/ps/ps_manager.py#L1194) | 记录新版本并通知协调器；参数大数组由存储 Worker 承载 |
| 停止 Agent 后台工作 | 1–2 | [manager.py：stop_busy_loop](../../psrl/workers/agent_loop/manager.py#L356) | 取消派发与收集任务，停止 Worker 程序，关闭后处理池 |

表中是阅读顺序。真实运行中，题目派发、结果收集、生成协调与训练由不同任务并发推进。`fine_grain_overlap` 还可选择更细的训练执行策略；上表用默认整批策略讲解。停止 Agent 工作只是系统清理的一部分，环境容器还需要各自的回收流程。

这些代码或其中某个术语读不顺时，可以直接围绕某一步继续追问，例如“Worker 怎样知道一组回答齐了”或“新参数怎样传到另一张 GPU”。

## 运行一个不需要 GPU 的例子

更深入的算法与推理机制可从[PSRL 知识索引](../psrl/README.md)选择。本页先用一个可手算的版本检查例子，在 Python 中运行：

```python
# 教学模拟：只有样本字段和版本判断，不加载模型、不计算梯度。
sample = {"question": "2 + 3", "answers": ["5", "6"], "version": 10}
sample["rewards"] = [int(answer == "5") for answer in sample["answers"]]
train_version, allowed_gap = 13, 3
gap = train_version - sample["version"]
print(sample["rewards"], gap, gap <= allowed_gap)
```

预期输出是 `[1, 0] 3 True`：奖励来自答案是否正确；版本差是 `13 - 10 = 3`，处于本例允许范围。把 `train_version` 改为 14，最后一个值就变为 `False`。真实系统在派发、预留和样本入批时共同约束版本，并处理等待和中止，本例只演示最终不等式。

## 按自己的问题继续阅读

| 你想了解什么 | 可选入口 |
|---|---|
| 某个专业词是什么意思 | [术语速查](02-PSRL入门_系统与实验术语速查.md)：中文解释、具体例子和深入入口 |
| 整体代码如何分工 | [系统架构](../design/architecture.md) |
| 参数版本和批次如何配合 | [陈旧度控制](../design/staleness_control.md) |
| 多轮 Agent 如何保存工具运行状态 | [环境当前设计](../env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) |
| GPU 任务目前接入到什么程度 | [对外简报](../env-worker-deep-dive/milestones/01-Science介绍_已接入任务与研究价值.md)、[覆盖与路线图](../env-worker-deep-dive/milestones/02-Science现状_任务覆盖与接入路线.md) |
| 如何准备一次真实运行 | [安装与运行](../tutorial/index.md)、[集群运行前检查](../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md) |

## 核对范围

2026-11-16 对照本地提交 `6aa328d` 及主目录已有未提交代码，核对了表内入口和默认整批训练路径。本页 CPU 例子已实际运行，得到 `[1, 0] 3 True`，与上述解释一致。本次核对覆盖流程和教学例子，未运行训练、GPU 实验或远端部署验证。
