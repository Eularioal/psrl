# PSRL 基础运行：先理解示例，再准备一次有界任务

一个 PSRL 训练任务需要模型、题目数据、生成服务、训练进程和评分逻辑共同工作。先确认这些角色及路径，再按目标选择最小运行规模。本页以仓库 DAPO 脚本解释参数；Science 的当前源码链路另见[训练源码地图](../psrl-training-script-source-map.md)。

## 目录

- [项目结构速览](#项目结构速览)
- [安装](#安装)
- [运行前准备](#运行前准备)
- [逐段拆解示例脚本 `examples/dapo_trainer/qwen2.5_3b_fsdp.sh`](#逐段拆解示例脚本-examplesdapo_trainerqwen25_3b_fsdpsh)
  - [1. 基本变量与路径](#1-基本变量与路径)
  - [2. 集群拓扑（几节点、几卡，谁做生成谁做训练）](#2-集群拓扑几节点几卡谁做生成谁做训练)
  - [3. 算法超参（DAPO/GRPO 相关）](#3-算法超参dapogrpo-相关)
  - [4. 核心 PSRL 命令（真正的启动命令）](#4-核心-psrl-命令真正的启动命令)
  - [5. 运行时会发生什么](#5-运行时会发生什么)
- [监控指标](#监控指标)
- [实操建议](#实操建议)
- [延伸阅读](#延伸阅读)
- [单机 Demo 脚本与常用 bash 语法：`qwen2.5_3b_fsdp_singlenode_demo.sh`](#单机-demo-脚本与常用-bash-语法qwen25_3b_fsdp_singlenode_demosh)

## 项目结构速览

| 位置 | 用途 |
|---|---|
| `examples/` | 各类任务的启动脚本与配置 |
| `psrl/trainer/main_ppo.py` | 解析配置、连接 Ray、创建训练任务 |
| `psrl/trainer/ppo/` | 调度采样与训练阶段 |
| `psrl/workers/` | 生成、训练、评分、参数服务器等执行角色 |
| `docs/` | 使用说明、机制解释与历史证据 |

Ray 是跨进程、跨机器的任务管理框架；Hydra 是把默认配置与命令行覆盖项合并的工具。先知道它们分别管执行位置和运行参数即可，后面再按问题深入。

## 安装

新环境的安装与版本要求统一读[安装说明](../../tutorial/installation.md)。本工作区使用共享环境时，先按[集群检查](cluster-preflight.md)激活指定环境、确认实际导入路径，依赖修复放到本次隔离副本。

`pip install -e` 会改变所用 Python 环境的源码指向，适合个人新环境；共享环境应沿用已经约定的导入与隔离方法。原文中的重复安装命令移除，避免和安装入口维护两份版本要求。

## 运行前准备

先确认下面五件事：

1. 验收目标：只查配置、验证容器，还是确实需要训练更新；写清时间、GPU、步骤与重试上限。
2. 文件：目标模型目录、训练/验证 Parquet、输出目录都由本次任务明确指定。
3. 环境：source 后检查 `PSRL_WORKSPACE`，再确认 driver 与 worker 的 PSRL、veRL 等导入来源。
4. 节点：核对路由地址、Ray 节点和目标 GPU UUID，确认实际占用与 Docker 能力。
5. 收尾：明确本次进程、容器、租约和日志归属，结束后核实物理释放。

Parquet 是列式数据格式，题目和元信息还需要正确的嵌套类型；上下文长度和训练显存也分别计算。具体最小检查见[运行前五项](cluster-preflight.md#开跑前先完成这五项)。

## 逐段拆解示例脚本 `examples/dapo_trainer/qwen2.5_3b_fsdp.sh`

以下依据 2026-11-16 实际阅读的[仓库脚本](../../../examples/dapo_trainer/qwen2.5_3b_fsdp.sh)。它保留了早期环境假设：source 后使用工作区路径，通过 `import psrl` 定位代码，地址依赖外部 `LOCAL_IP`，并会重设 shell 的家目录变量。原文所说“该脚本已自动保护全部路径并自算 IP”与当前源码不符，现已修正。

这份脚本适合用来理解参数。准备真实任务时，先按上述检查固定环境和路径，并核对选用脚本的实际行为。

### 1. 基本变量与路径

脚本读取 `PSRL_WORKSPACE` 下的环境、模型和数据。例如工作区为教学路径 `/my-work` 时，模型预期位于 `/my-work/models/Qwen2.5-3B-Instruct`，数据位于 `/my-work/data/gsm8k/`。这些路径表示目录关系。

`staleness=${1:-3}` 表示第一个脚本参数作为版本滞后设置，缺省为 3。滞后允许生成和训练在有限版本差距内推进，详细机制见[训推协作](../vllm-and-training-parallelism/rl-training-workflow.md)。

共享环境脚本可能覆盖工作区变量，应在实际启动前检查 source 后的值。环境脚本作用见[activate 说明](../../tools/shell-python/python-environments.md#lhyactivate-脚本说明与新-tunnel-启动配置)。

### 2. 集群拓扑（几节点、几卡，谁做生成谁做训练）

脚本设置 2 节点、每节点 8 GPU：一个节点生成，一个节点训练。生成 `TP=1,PP=1`，所以 8 张生成卡对应 8 个单卡实例。

```text
16 GPU = 8 GPU 训练池 + 8 GPU 生成池
生成实例数 = 生成 GPU 总数 / 每实例使用 GPU 数 = 8 / 1 = 8
```

训练侧另配置验证实例，训练与验证的切换按相应资源管理配置处理。Ray 负责把这些角色安排到资源池；改变节点数时，要一起重算角色数量与资源声明，详见[Ray 调度](ray-and-rollout.md)。

### 3. 算法超参（DAPO/GRPO 相关）

脚本设置 GRPO 优势估计，64 道题每题采 8 条回答，目标工作量为 512 条轨迹。动态采样组处理开关为 `False`；冗余采样、部分 rollout 则另有自己的配置。

非对称裁剪范围参数为 0.2/0.28，按 token 平均损失；回答上限 4096，超长惩罚缓冲为 2048。每项的计算例子见[分组采样与损失](../vllm-and-training-parallelism/rl-loss-and-sampling.md)。

### 4. 核心 PSRL 命令（真正的启动命令）

实际入口是 `python -m psrl.trainer.main_ppo`，后面一长串 `字段=值` 是 Hydra 配置覆盖。下面按前缀读即可：

| 前缀 | 管什么 | 例子 |
|---|---|---|
| `psrl.*` | 调度、部署、权重版本 | 生成实例数、PS 模式 |
| `gen_actor_rollout_ref.*` | 生成引擎 | TP、批量 token 预算 |
| `train_actor_rollout_ref.*` | 训练与验证模型 | 学习率、分片、验证采样 |
| `reward.*` | 评分和奖励处理 | 超长惩罚 |
| `data.*` | 文件与数据预算 | 训练文件、每批题目数 |
| `algorithm.*` | 优势与修正算法 | `adv_estimator=grpo` |
| `trainer.*` | 总步骤、日志和保存 | 该脚本 200 步、每 200 步保存 |

这里只解释真实脚本，不拼接带省略号的伪可运行启动命令。选定任务后应从实际脚本和最终解析配置核对具体值。

### 5. 运行时会发生什么

加载配置、连接 Ray 后，训练器创建生成、训练、参数服务器等角色。生成请求经网关到模型执行，回答获得奖励后进入样本队列；训练侧计算概率、优势和损失，更新参数，再发布新权重。

TransferQueue 是保存与传递样本字段的队列；参数服务器则传递模型权重，两者内容不同。沿一条回答阅读这条链路，见[完整训练路径](../vllm-and-training-parallelism/rl-training-workflow.md#一条回答的完整路径)。

## 监控指标

先看训练 step 是否推进、退出码和异常，再看奖励、验证准确率、回答长度、熵和梯度是否协调。熵下降和奖励上升都需要结合任务阶段解释，单个数字无法保证训练质量。方法见[日志与批量配置](monitoring-and-batches.md)。

## 实操建议

先用能回答目标的最小检查。如果目标要求真实训练，缩小规模时同时核对模型容量、数据批量、角色资源池与运行预算。使用共享集群时沿用已分配资源，集群级停止或重建需明确归属。

`examples/ray/ray_start.sh` 包含面向目标节点的清理与启动操作，使用前应先阅读脚本；运行指南不再把重建共享 Ray 当成普通启动步骤。

## 延伸阅读

- 按代码定位：[训练源码地图](../psrl-training-script-source-map.md)。
- 准备 Science 环境：[Env 当前设计](../../env-worker-deep-dive/knowledge/current-design.md)。
- 处理启动报错：[历史故障复盘](startup-incident-reviews.md)。

## 单机 Demo 脚本与常用 bash 语法：`qwen2.5_3b_fsdp_singlenode_demo.sh`

这个名称来自早期本地四卡 Demo，当前仓库未提供该文件。历史记录为 2 卡生成、2 卡训练、3 个训练 step，故障证据保存在[单机复盘](startup-incident-reviews.md#单机-demo-踩坑记录两处根因与修复)。

阅读类似脚本时，几个语法就足够入门：`${VAR:-默认值}` 为未设置变量提供默认值，`export` 将变量传给子进程，`$((...))` 做整数计算，行尾反斜杠表示续行，`2>&1 | tee 文件` 将输出显示并写日志。`source` 对当前终端环境的作用见[进程示例](../../tools/shell-python/python-environments.md#source让环境设置留在当前终端)。
