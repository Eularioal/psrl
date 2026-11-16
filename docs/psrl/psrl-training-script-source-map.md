# 从 Science 启动脚本读懂 PSRL 训练代码

**核心结论**：一次训练沿四条职责展开：脚本组合配置，PSRL 安排采样和训练任务，Agent 与环境完成做题，veRL 等依赖执行底层模型计算。读代码时先沿一条样本走完流程，再按问题进入资源调度、权重同步或损失细节。

## 目录

- [先跟着一道题走一遍](#先跟着一道题走一遍)
- [1. 启动脚本与配置](#1-启动脚本与配置)
- [2. 训练主流程](#2-训练主流程)
- [3. 模型请求与工具执行](#3-模型请求与工具执行)
- [4. 权重怎样送到生成端](#4-权重怎样送到生成端)
- [5. 奖励与训练样本](#5-奖励与训练样本)
- [6. 训练计算与外部依赖](#6-训练计算与外部依赖)
- [按问题选择下一篇](#按问题选择下一篇)

## 先跟着一道题走一遍

假设训练题目是修复一个函数。这是教学流程，具体调度分支由配置决定：

```text
启动脚本 → 合并配置 → TaskRunner 创建资源和 Trainer
                              ↓
数据中的题目 → Agent Loop → 请求模型 → 得到操作命令
                   ↑                    ↓
                   └── 工具结果 ← 做题环境执行命令
                              ↓ 完成任务
验证器评分 → 奖励适配 → 训练样本与优势值 → 模型参数更新
                                                   ↓
                                    参数服务器 → 生成端加载新版本
```

图中的箭头表示数据依赖，不同题目与批次可以并发。Trainer 组织训练阶段，各类 worker 进程执行模型计算、轨迹采样或参数管理。下表沿调用链给出源码入口，重点看跨进程的数据交接与配置分支。

本文于 2026-11-16 静态核对所列本地入口，没有启动训练。旧版标题中的 `fsdp_qwen2.5_3b_laps.sh` 已不在当前仓库，现以 [fsdp_qwen35_4b.sh](../../examples/sciaccel_rl/fsdp_qwen35_4b.sh)为入口。旧运行的参数与故障继续保存在 [OOM 复盘](../sciaccel-grpo-oom-analysis.md)和[上下文长度说明](hf-config-context-window.md)。

## 1. 启动脚本与配置

[Science 脚本](../../examples/sciaccel_rl/fsdp_qwen35_4b.sh)先准备模型、数据、批量和长度，再把它们传给 `python -m psrl.trainer.main_ppo`。**Hydra** 是组合配置的工具，命令行上的 `data.train_batch_size=16` 等键会覆盖相应默认值。

| 先读哪里 | 它回答什么问题 |
|---|---|
| [长度与训练批量](../../examples/sciaccel_rl/fsdp_qwen35_4b.sh#L68) | 一轮用多少题，一条轨迹允许多长 |
| [根配置](../../psrl/trainer/config/ppo_trainer.yaml) | 配置由哪些主题组合而来 |
| [部署配置](../../psrl/trainer/config/psrl/deployment.yaml) | 训练、生成和验证如何分配节点与 GPU |
| [Agent 配置](../../examples/sciaccel_rl/config/sciaccel_agent_config.yaml) | `sciaccel` 这个名称对应哪个 Agent Loop |

例如发现长任务 OOM，先沿长度变量查其传递位置，随后进入训练预算与并行配置。单看某个 YAML 默认值，可能漏掉启动脚本的覆盖。

## 2. 训练主流程

| 代码入口 | 输入、执行与结果 |
|---|---|
| [main](../../psrl/trainer/main_ppo.py#L41) | 接收合并配置，校验后调用 `run_ppo` |
| [run_ppo](../../psrl/trainer/main_ppo.py#L126) | 准备 Ray 运行环境并调用 TaskRunner。Ray 是跨进程、跨节点派发任务的框架 |
| [TaskRunner.run](../../psrl/trainer/main_ppo.py#L374) | 选择各类 worker，创建资源池，调用 Trainer 的 `init_workers()` 与 `fit()` |
| [init_workers](../../psrl/trainer/ppo/ray_trainer.py#L1559) | 初始化训练、生成等执行组件 |
| [fit](../../psrl/trainer/ppo/ray_trainer.py#L3361) | 恢复进度、启动数据管道，选择步骤策略并迭代更新 |

当前 `fit()` 通过 `_step_strategy.run_step()` 组织每步工作。阅读批量等待、概率重算与训练重叠时，还要进入 [strategies](../../psrl/trainer/ppo/strategies)，按启用的策略继续追踪。原文将所有工作都写成 `fit` 中一条固定同步循环，现按实际接线区分调度策略与计算依赖。

## 3. 模型请求与工具执行

模型请求和终端执行是做题过程中交替出现的两条路径。

| 路径 | 代码入口 | 实际工作 |
|---|---|---|
| 接到题目 | [Agent worker](../../psrl/workers/agent_loop/worker.py#L262) | 为轨迹创建异步 task，同一个 worker 可跟踪多个任务 |
| 多轮模型交互 | [SciAccelAgentLoop](../../examples/sciaccel_rl/agent_loop.py) | 创建会话、运行 episode、取回训练数据 |
| 请求代理 | [RolloutGateway](../../psrl/workers/gen/rollout_gateway.py#L123) | 启动 SMG 路由和 SessionRouter 会话代理 |
| 执行模型命令 | [agent](../../examples/sciaccel_rl/agent.py#L94) | 普通命令交给 Harbor 终端，受管 GPU 命令交给专门接口 |
| 受管 GPU 操作 | [GPU 工具分发](../../examples/sciaccel_rl/gpu_tools.py#L32) | 将提交、等待、取消等操作转换为环境调用 |

Harbor 是执行做题流程的第三方框架。**SMG** 负责模型请求路由，**SessionRouter** 为多轮任务提供会话接口，做题环境则运行命令并返回观察。详细例子见 [Agent Loop 与工具调用](psrl-agent-loop-harbor-tool-calling.md)。

## 4. 权重怎样送到生成端

训练端更新了模型，生成端需要加载相应版本。PS（Parameter Server，参数服务器）管理权重存储与版本，生成端按协调策略同步。

- [PSManager](../../psrl/workers/ps/ps_manager.py)管理版本与同步控制。
- [PSStorageWorker](../../psrl/workers/ps/ps_storage_worker.py)保存分片权重，`nixl_cpu` 路径使用可传输的主机内存。
- [RolloutCoordinator](../../psrl/workers/gen/rollout_coordination/coordinator.py)协调生成实例的版本与状态。
- [NIXL client](../../psrl/utils/nixl/client.py#L301)注册内存并支持大块数据传输。

**分片**是将模型张量分成多块存储或计算。控制消息决定“哪一版、发给谁”，数据传输负责真正搬运权重字节，两者共同组成同步流程。原理和数字例子见 [NIXL 入门](nixl-explained.md)，配置细节见 [参数服务器设计](../design/parameter_server.md)。

## 5. 奖励与训练样本

Science 的验证器先对任务成果评分。[reward.compute_score](../../examples/sciaccel_rl/reward.py#L23)按 `reward_key` 从 `harbor_rewards` 中选择实际训练分数，同时返回诊断字段。查看分数异常时，应核对数据选择的奖励键和验证器实际输出。

**TITO** 是记录真实输入输出 token、重建训练轨迹的机制。[训练数据构建](../../psrl/utils/tito/training_data.py)生成模型输入、输出及哪些位置计算损失的掩码。Trainer 的[优势计算](../../psrl/trainer/ppo/ray_trainer.py#L3011)随后读取奖励、概率和分组标识，形成参数更新需要的学习信号。

排查训练分数异常时，沿 `reward_key → compute_score → rm_scores → advantage` 核对字段和样本分组，避免把 Harbor 的通用奖励当作实际训练分数。

## 6. 训练计算与外部依赖

本仓库的 [EngineTrainWorker](../../psrl/workers/train/engine_train_worker.py)适配训练工作与 PSRL 的权重流转、显存管理。底层模型训练大量复用外部 **veRL** 库，包括模型并行、优化器更新及策略损失。具体符号与行为取决于实际安装版本。

| 要查的问题 | 常见依赖入口或专题 |
|---|---|
| FSDP 怎样分摊模型状态 | veRL 的 FSDP engine，[训练并行](vllm-and-training-parallelism/training-parallelism.md) |
| Ulysses 怎样分摊长序列 | veRL 的序列并行实现，同上专题 |
| GRPO 优势、PPO 裁剪损失怎样计算 | veRL 的 PPO core algorithms，[损失专题](vllm-and-training-parallelism/rl-loss-and-sampling.md) |
| 采样策略与训练策略概率不同怎样修正 | rollout correction，[损失专题](vllm-and-training-parallelism/rl-loss-and-sampling.md) |

`patch/verl/` 保存本项目需要的补丁，补丁文件存在与当前进程实际加载补丁依赖是两个需要分别核对的条件。执行前按[集群指南](psrl-run-guide/cluster-preflight.md)检查 driver 与 worker 的实际导入来源。本次仅核对本仓库的调用入口，没有重新审查外部 veRL 或 Harbor 的全部内部代码。

## 按问题选择下一篇

- 想从运行开始：[基础运行指南](psrl-run-guide/getting-started.md)。
- 想了解做题样本如何变成训练输入：[Agent Loop 与工具调用](psrl-agent-loop-harbor-tool-calling.md)。
- 想理解公式与并行：[训推专题索引](vllm-and-training-parallelism.md)。

可以继续指定任意表格的一行追问，例如“请沿 `compute_score` 到 advantage 解释一个分数怎样影响参数更新”。
