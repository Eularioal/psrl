# EnvRouter：配置恢复与Worker重试

> **核心结论**：先核对共享 Router 的配置契约，再申请与归还资源；节点故障需要健康检查与物理清理，rollout 失败则由 Worker 和 Manager 的重试、整组补采及有界退出链处理。本篇把启动、执行、故障可见性与恢复放在一起，各轮验证仍保留原日期和版本。

## 目录

- [1. 先把整个过程连起来](#1-先把整个过程连起来)
  - [1.1 启动方、题目、rollout 与环境申请](#11-启动方题目rollout-与环境申请)
  - [1.2 跟着训练流程图读源码](#12-跟着训练流程图读源码)
  - [1.3 运行四条 rollout 的教学模拟](#13-运行四条-rollout-的教学模拟)
- [2. 进程对象和远程调用是什么](#2-进程对象和远程调用是什么)
- [3. 为什么同名服务可能使用旧配置](#3-为什么同名服务可能使用旧配置)
- [4. 配置快照指纹和协议版本](#4-配置快照指纹和协议版本)
- [5. 为什么服务活着还不能恢复使用](#5-为什么服务活着还不能恢复使用)
- [6. 有限重试与受控恢复](#6-有限重试与受控恢复)
- [7. 跟着图读源码](#7-跟着图读源码)
- [8. 运行教学实验与回归](#8-运行教学实验与回归)
- [9. 本次完成与剩余边界](#9-本次完成与剩余边界)
- [10. 一条 rollout 中的命令与最终评分](#10-一条-rollout-中的命令与最终评分)
- [11. 当前方案是否必然快于固定基线](#11-当前方案是否必然快于固定基线)
- [Worker 失败、重试与整组补采](#worker-失败重试与整组补采)
  - [1. 先把角色和执行链连起来](#1-先把角色和执行链连起来)
  - [2. 原框架支持到哪一步](#2-原框架支持到哪一步)
  - [3. 用四条轨迹解释重试和补采](#3-用四条轨迹解释重试和补采)
  - [4. 为什么不能直接把原任务扔给另一台机器](#4-为什么不能直接把原任务扔给另一台机器)
  - [5. 本轮修复了哪些确定问题](#5-本轮修复了哪些确定问题)
  - [6. 跟着图读源码](#6-跟着图读源码)
  - [7. 可运行验证与 Results](#7-可运行验证与-results)
  - [8. 尚未实现的范围与后续顺序](#8-尚未实现的范围与后续顺序)

> **核心结论**：复用 Router 前先核对配置，恢复节点前先核实资源。配置契约把本次预期的预算、设备和后端与现有服务比较；诊断与人工恢复入口则重新探测节点，并保留尚未确认清理的资源记录。

例如第一次启动允许同时运行 4 个容器，第二次配置改成 8 个。如果第二次只是找到同名服务，它可能仍按 4 个运行。启动核对会把这个差异直接报出来，避免任务按错误预算继续执行。第 3、4 节展开完整过程。

本文从训练怎样发起环境任务讲起，再解释服务对象、远程调用、配置比较和故障恢复。R1/R2 实现与 CPU 回归为 2026-11-10 记录；11-11 对 `ab74120` 加工作区内容核对训练链，并补隔离真实 Ray/Docker 探测验证，范围集中在第 8、9 节。本文提供机制解释；总体能力见[当前设计](01-Env架构_现状与完整执行流程.md)，工程优先级见[工程审查](06-工程审查_待修复问题与优先级.md)。


## 1. 先把整个过程连起来

训练会不断产生环境任务。环境任务需要创建容器、运行命令、使用 GPU，结束后释放资源。如果多个任务各自决定能否使用 GPU，它们可能同时许诺超过总容量的显存。因此需要一个共同决定“谁可以进入”的资源服务，这里叫 EnvRouter。

<!-- figure:release-logical-and-physical -->
这张图沿同一个场景解释：租约数已经是零，为什么下一组实验仍可能撞上残留显存？

[![资源回收：账本归零与进程显存释放要分别确认（1/2）](../../../_static/img/document-figures/release-logical-and-physical-screen.png)](../../../_static/img/document-figures/release-logical-and-physical.png)

[![资源回收：账本归零与进程显存释放要分别确认（2/2）](../../../_static/img/document-figures/release-logical-and-physical-screen-02.png)](../../../_static/img/document-figures/release-logical-and-physical.png)

任务结束后，逻辑租约、GPU 进程与 Docker 容器通过不同路径退出。 账本描述使用权，设备遥测描述实际消耗。安全交接要把两者重新对齐。 不执行全局 Ray stop、Docker 重启或共享缓存 prune。图不代表本次启动了实验。 [查看矢量图](../../../_static/img/document-figures/release-logical-and-physical.svg)。
<!-- /figure:release-logical-and-physical -->

以下是实际实现的调用关系。右边的恢复分支由管理员显式调用，不是每次任务都自动恢复节点。

```text
[1] 训练启动方：请求配置和 Router 名称
                 ↓ 创建或找到同名服务
[2] 核对服务启动契约
       ├─ 不一致 → 报错，不修改旧服务
       └─ 一致 → 允许复用及预热
                 ↓
[3] 节点探测：Docker、GPU 容量、遗留清理
       ├─ 通过 → [4] 准入、发放 lease、执行任务
       └─ 失败 → 重试预算耗尽后拒绝新准入
                                  ↑
[5] 管理员读取 node_diagnostics    │
       ↓ 明确发起 recover_node    │
[6] 合并在途探测或执行一次有界检查 ┘
       ├─ 仍失败或待清理 → 保留阻断和原资源记录
       └─ 检查通过 → 健康条件恢复，其他准入限制仍有效
```

图例：图里每个部分对应的源码位置。

| 节点或箭头 | 函数和职责 | 源码 |
|---|---|---|
| 1 | `create_env_router` 创建或复用命名服务 | [router.py](../../../../psrl/workers/env_service/router.py#L752) |
| 2 | `router_contract` 生成配置快照，`startup_contract` 返回服务实际启动契约，`verify_router_contract` 比较 | [生成](../../../../psrl/workers/env_service/router_contract.py#L10)、[返回](../../../../psrl/workers/env_service/router.py#L437)、[比较](../../../../psrl/workers/env_service/router_contract.py#L29) |
| 3 | `_check_node` 检查真实节点，`_health_task` 限制自动重试次数 | [检查](../../../../psrl/workers/env_service/router.py#L524)、[次数控制](../../../../psrl/workers/env_service/router.py#L566) |
| 4 | `try_admit` 同时检查健康状态、待清理记录与预算 | [router.py](../../../../psrl/workers/env_service/router.py#L706) |
| 5 | `node_diagnostics` 只读列出错误和资源归属 | [router.py](../../../../psrl/workers/env_service/router.py#L581) |
| 6 | `recover_node` 复用在途探测或启动一次检查，不清空账本 | [router.py](../../../../psrl/workers/env_service/router.py#L605) |
| 3 的物理清理 | `EnvNodeService.cleanup` 清理所属容器，再向 Router 确认 | [node.py](../../../../psrl/workers/env_service/node.py#L182) |

这些代码如果有理解上的问题，可以随时追问。异步任务会交错推进，因此该图不能理解为所有请求排成一个单线程流水线。

### 1.1 启动方、题目、rollout 与环境申请

先看启用 Env Service 后的完整调用关系。这里的“训练任务”指一次完整训练运行，而“题目”指数据集中要求模型完成的一项工作。两者都常被简称为任务，所以容易混淆。rollout 指模型针对一道题实际尝试求解一次的采样过程，可能包含多轮模型回复和多次工具执行。

```text
一次完整训练运行，可以涉及多个进程和多台机器
│
├─ [F1] Trainer 初始化代码，即本文的 Router 启动方
│        │ create_env_router(服务名, 服务配置)
│        ▼
│  [F2] 命名 EnvRouter actor，内部维护共享资源账本
│        ↑ 创建发生在初始化阶段，下面多个环境申请复用它
│        │
└─ [F3] 题目按采样次数展开 → AgentLoopManager 分派题目组
         ├─ worker A：题目 Q1 的 rollout 1、rollout 2
         └─ worker B：题目 Q2 的 rollout 1、rollout 2
                    │ 每条 rollout 创建一个主协程
                    ▼
          [F4] AgentLoopWorker 内的 rollout 协程
                    │ Science agent loop 组织一次环境任务
                    ▼
          [F5] runner → EnvServiceClient.acquire(session_id, demand)
                    │ try_admit：向同一个 F2 申请，而非创建 Router
                    ├─ 等待 → 协程让出执行机会，稍后再问
                    ├─ 拒绝 → 返回环境申请错误
                    └─ 获准 → 返回 lease，登记预约并维持心跳
                                  ▼
          [F6] Harbor 和容器提供者创建环境、运行求解和评分
                    │ 多轮：模型回复 → 环境执行 → 结果供模型继续推理
                    ▼
          [F7] 清理所属容器 → 归还 lease → 收集采样数据供训练使用
```

图例：图里每个部分对应的源码位置。Q1、Q2 和 worker A/B 是教学名称，实际 worker 数量和分派结果由配置及节点资源决定。

| 节点或箭头 | 职责、函数 | 源码 |
|---|---|---|
| F1 → F2 | Trainer `_init_env_service` 读取配置，调用 `create_env_router` 创建或复用服务并核对契约 | [ray_trainer.py:549](../../../../psrl/trainer/ppo/ray_trainer.py#L549)、[router.py:752](../../../../psrl/workers/env_service/router.py#L752) |
| F2 | `EnvRouterCore.__init__` 建立共享账本 | [router.py:49](../../../../psrl/workers/env_service/router.py#L49) |
| F3 题目展开 | 数据处理器按 `rollout_n` 展开同题的请求，并保留父题目身份 | [data_processor.py:567](../../../../psrl/utils/dataset/data_processor.py#L567) |
| F3 分派 | `get_dispatch_plan` 按题目组选择 worker，`_supervise_worker_batch` 调用 worker | [manager.py:991](../../../../psrl/workers/agent_loop/manager.py#L991)、[manager.py:931](../../../../psrl/workers/agent_loop/manager.py#L931) |
| F4 | `add_agent_program` 逐请求调用 `generate_trajectory`，后者用 `asyncio.create_task` 创建主协程任务 | [worker.py:220](../../../../psrl/workers/agent_loop/worker.py#L220)、[worker.py:262](../../../../psrl/workers/agent_loop/worker.py#L262) |
| F4 → F5 | Science agent loop 启动 Harbor 求解，使用进程内缓存的 Env client | [agent_loop.py:182](../../../../examples/sciaccel_rl/agent_loop.py#L182)、[agent_loop.py:95](../../../../examples/sciaccel_rl/agent_loop.py#L95)、[agent_loop.py:391](../../../../examples/sciaccel_rl/agent_loop.py#L391) |
| F5 查找服务 | `EnvServiceClient.from_config` 用 `ray.get_actor` 取得已有服务句柄 | [client.py:124](../../../../psrl/workers/env_service/client.py#L124) |
| F5 申请与等待 | runner 创建容器前 `acquire`，client 调用 `try_admit`，账本返回获准、等待或拒绝 | [runner.py:366](../../../../examples/sciaccel_rl/runner.py#L366)、[client.py:186](../../../../psrl/workers/env_service/client.py#L186)、[ledger.py:113](../../../../psrl/workers/env_service/ledger.py#L113) |
| F6 | runner 注入租约与执行配置后调用 Harbor `job.run`，容器提供者按配置创建环境 | [runner.py:412](../../../../examples/sciaccel_rl/runner.py#L412)、[runner.py:551](../../../../examples/sciaccel_rl/runner.py#L551)、[原生容器后端](../../../../psrl/workers/env_worker/harbor.py#L19) |
| F7 | runner 的 `finally` 清理容器后归还租约，agent loop 收集本次会话采样数据 | [runner.py:610](../../../../examples/sciaccel_rl/runner.py#L610)、[client.py:69](../../../../psrl/workers/env_service/client.py#L69)、[agent_loop.py:228](../../../../examples/sciaccel_rl/agent_loop.py#L228) |

这些代码如果有理解上的问题，可以随时追问。

**启动方具体是谁。** 它不是新增的一种专门进程，而是“发起创建或查找 Router 的调用方”的描述。在当前正常训练路径里，具体就是 Trainer 初始化时执行 `_init_env_service` 的代码。从使用者角度可以说“这次训练启动了 Router”，但一次训练包含 Trainer、worker、模型推理服务等多个角色，不能把它们当成一个进程。

**一个题目为什么会有多个协程。** 假设题目 Q1 是“修改程序使测试通过”，配置每题采样 2 次，就生成 Q1/1、Q1/2 两次独立尝试。每次尝试可以得到不同回复、执行不同命令、获得不同评分。数据处理器先把题目展开为两个请求，worker 再为每个请求创建一个主协程。因此更准确的单位是“每条 rollout 一个主协程”，不是“每个不同题目只有一个协程”。心跳和清理还可能创建辅助协程，实际协程总数不止主协程数量。

协程是在程序中可以暂停、稍后继续的一段执行流程。例如 Q1/1 发出模型请求后，要等模型回复，`await` 会让出当前线程上的执行机会，Q1/2 就有机会继续。**并发**表示这些流程在一段时间内交错推进，不表示每个流程都拥有独立 CPU 核或 GPU。worker 是执行这些流程的进程角色，一个 worker 可以运行许多协程。当前 Science 实现还把 Harbor 工作提交到同一 worker 的专用事件循环线程，事件循环负责安排哪些等待结束的协程继续运行。它不是为每条 rollout 新建一个操作系统进程。

**环境、资源需求和租约分别是什么。** 环境是一次求解所用的执行空间，包括代码、文件、依赖和容器等。`demand` 是资源需求描述，例如需要几个 CPU、多少内存或 GPU 显存。`lease` 中文称租约，是 Router 批准后登记的资源预约，包含租约身份、所属节点和预算。租约不是容器，也不是实际显存分配，实际创建和执行由容器后端完成。`session_id` 关联本次求解，`lease_id` 标识获准的预约，两者与 Router 服务名是不同层次的身份。

严格说，不是尚未创建的容器自己跑来申请资源。当前路径是 worker 中的 runner 代表本次环境任务先申请，拿到租约后才启动相关容器。worker 的 Env client 相当于发送申请的调用入口，同一进程会缓存并共享它。多个 client 仍可以指向同一个 Router，共享同一本账。

**两种调用传的东西不同。** 创建调用传入的是整项服务的规则，例如服务名、节点、资源单元和容量。环境申请传入的是本次求解的身份、需求和所在节点。一个资源单元 `unit` 是配置中受管理的一组节点资源，不等于一个题目或一条 rollout。训练初始化一般创建或查找一个配置指定的 Router，后面可以有成千上万次环境申请。

用第 1.3 节脚本的教学设置理解：Q1、Q2 各采样两次，共有 4 个主协程，环境容量为 2。先获准的两个会话创建环境，另两个保留等待流程；一个会话清理并归还后，等待 client 再次申请即可接续。会话准入通过带随机退避的轮询实现，顺序不保证严格先到先服务，上游分派也可能提前延缓整组工作。

资源范围、两张卡的 CPU/主存与显存预算、同题 8 条 rollout 的完整接续表，以及 episode 与 command 的共同流程，统一见[当前设计第 2.1 至 2.3 节](01-Env架构_现状与完整执行流程.md#21-先声明环境资源范围再分派和申请)。[第 2.4 节](01-Env架构_现状与完整执行流程.md#24-为什么会在本节点排队其他节点仍有空闲)进一步用 A、B 各有 4 个名额的例子解释同题成组派发，以及等待请求固定在 worker 节点的原因。episode 本身就支持初次选卡、资源不足等待和释放后接纳新会话；`command_gpu_pool` 另外控制已存在会话下一条命令能否重新选卡。

**同一条 rollout 内的多轮交互。** Q1/1 可以先让模型读题，再执行测试，再把错误输出给模型，模型修改代码后再次测试，最后评分。这些是同一次求解内部的多轮，不是每执行一条 shell 命令就新建 Router 或生成一条独立 rollout。模型生成由模型推理服务完成，EnvRouter 管理环境资源准入，不负责模型文本请求的路由。项目里用于模型会话的 SessionRouter 是另一个组件，即使二者都叫 Router，也不能当成同一个服务。

如果使用 `episode` GPU 持有方式，本次求解期间保留 GPU 预约。如果使用 `command` 方式，环境会话仍存在，但 GPU 预算在实际命令执行阶段进一步申请和归还。unit 配置 `auto` 时，在准入时根据可信任务协议解析成上述一种方式，具体见[自动模式及其配置](../02-资源与调度/05-训推共享_vLLM余量与容量调度.md#自动模式第一版怎样工作)。两种情况都共享前面的初始化和环境会话流程，不能把“一条 rollout”“一次环境会话”和“一次 GPU 命令”当作同一层。容器集合还可能包含独立评分器或网络辅助组件，不能从一条租约推导恰好一个容器。

**再回到 R1 的两个启动方。** 前文的启动方 A/B 表示两次 `create_env_router` 调用，并不表示 Q1/1 与 Q2/1，也不表示每条 rollout 都创建 Router。这两次调用可能来自同一控制进程的重复初始化，也可能来自能找到已有 actor 的另一个控制进程。只有在同一 Ray 集群、同一命名空间且旧服务仍存在时，才会复用。同名而规则不同才是 R1 要阻止的情况。普通四条 rollout 向同一 Router 申请不同租约，是正常的资源共享，不会因此触发配置冲突。当前代码没有设置独立存活的 detached 生命周期，不能假定任意训练重启后旧服务都会留下。

### 1.2 跟着训练流程图读源码

| 步骤 | 图中编号 | 文件、行号、函数 | 输入与输出或状态变化 |
|---|---|---|---|
| 训练初始化 | F1 → F2 | [ray_trainer.py:549 `_init_env_service`](../../../../psrl/trainer/ppo/ray_trainer.py#L549)、[router.py:752 `create_env_router`](../../../../psrl/workers/env_service/router.py#L752) | 服务配置与节点规则传入，得到已经核对启动契约的远程句柄 |
| 题目展开 | F3 | [data_processor.py:567](../../../../psrl/utils/dataset/data_processor.py#L567) | Q1、Q2 按采样次数展开，每条请求有身份，并保留父题目身份 |
| 派发题目组 | F3 → F4 | [manager.py:991 `get_dispatch_plan`](../../../../psrl/workers/agent_loop/manager.py#L991)、[manager.py:931 `_supervise_worker_batch`](../../../../psrl/workers/agent_loop/manager.py#L931) | 同题的采样组选择 worker，随后远程调用 worker，管理器异步监督其完成 |
| 并发求解 | F4 | [worker.py:220 `add_agent_program`](../../../../psrl/workers/agent_loop/worker.py#L220)、[worker.py:262 `generate_trajectory`](../../../../psrl/workers/agent_loop/worker.py#L262) | 每个请求创建主协程，多个主协程交错运行，不是单线程逐题跑完才继续 |
| 提交 Harbor 工作 | F4 → F5 | [agent_loop.py:391 `_gated`](../../../../examples/sciaccel_rl/agent_loop.py#L391)、[agent_loop.py:417](../../../../examples/sciaccel_rl/agent_loop.py#L417) | 通过进程缓存取得 client，将工作提交到专用 Harbor 事件循环线程并等待结果 |
| 正式申请资源 | F5 → F2 | [runner.py:372](../../../../examples/sciaccel_rl/runner.py#L372)、[client.py:186 `acquire`](../../../../psrl/workers/env_service/client.py#L186) | 身份与需求交给 Router，获准则启动心跳，等待则暂停后轮询，拒绝则报错 |
| 执行与取结果 | F6 | [runner.py:551](../../../../examples/sciaccel_rl/runner.py#L551) | Harbor 求解、工具交互与评分，内部可能多轮和并发，不展开为一个同步函数 |
| 收尾与数据收集 | F7 | [runner.py:610](../../../../examples/sciaccel_rl/runner.py#L610)、[agent_loop.py:228](../../../../examples/sciaccel_rl/agent_loop.py#L228) | runner 的 finally 完成所属资源清理后归还，agent loop 收集会话采样数据。训练消费由下游组织，不是每次释放租约就立刻执行一次参数更新 |

### 1.3 运行四条 rollout 的教学模拟

从仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_rollout_env_flow.py
```

[脚本](../../env-service-verification/scripts/model_rollout_env_flow.py)使用 Python 标准库 `asyncio`。`ModelRouter(capacity=2)` 模拟 F1/F2 的一个共享账本。两个题目各创建两个协程，模拟 F3/F4。`acquire` 模拟 F5，`asyncio.sleep` 只模拟 F6 的等待，`finally` 先打印 `ENV_CLEANED` 再释放，模拟 F7 的清理顺序。

`asyncio.Semaphore(2)` 是用于教学的信号量，可以理解成初始有两张许可证。申请拿走一张，归还放回一张，没有许可证时协程等待。这样能直接观察四个求解流程为何最多只有两个环境一起持有预约。真实 EnvRouter 除数量外还核对节点健康、CPU、内存、GPU 与归属，不能用这个信号量替换真实账本。

预期首行包含 `routers=1 rollout_coroutines=4 capacity=2`，结尾为 `DONE admitted=4 peak=2 active=0`。中间打印四次申请与获准，活动租约不超过 2，每次 `RELEASE` 前先出现本次 `ENV_CLEANED`。本次实际在 CPU 运行，观察到这些结果。

这是教学模拟，worker A/B 只是标签，全部协程都在一个本地进程和一个事件循环里运行。它不启动真实 Ray actor、模型、Docker、GPU，不模拟实际心跳、故障回收、命令级 GPU 预约，也不证明生产性能。真实 Science 的 Harbor 事件循环线程与 worker 主事件循环是分开的。

## 2. 进程对象和远程调用是什么

**进程**是操作系统中正在运行的程序实例。启动两次 Python，一般会得到两个各有内存的进程。某个进程里的 Python 字典，并不会因为另一进程创建同名变量就被修改。

**对象**是程序中的一份数据和操作这些数据的方法。例如 Router 对象持有资源账本，还有申请资源的方法。普通对象通常只能被所在进程直接调用。

**Ray**是这里使用的分布式执行框架。它能把一个对象放进常驻进程，由其他进程发送消息调用其方法。这样的常驻对象叫 **actor**，中文可理解为“接收调用并保存状态的服务对象”。本项目的 Router 就采用这种形式。

客户端获得的是 actor 的 **handle，即远程句柄**。它表示如何找到服务，不是把服务的全部内存复制回来。`handle.method.remote()` 发送一次远程方法调用。`ray.get(...)` 等待结果返回。服务内部的 `await` 表示等待某个异步操作完成，同时允许其他任务继续运行。

这个问题不要求训练进程先重启。同一启动方重复初始化，或其他启动方能找到尚存的同名 actor，都可能触发复用。复用要求处于同一个 Ray 集群、同一个命名空间，并且旧 actor 实际仍然存在。命名空间是服务名称的查找范围，不同范围里的同名对象不是同一个对象。

当前创建入口没有设置 `lifetime="detached"`。detached 表示与创建者独立存活，本地安装的 Ray 说明默认 actor 与创建者共享生命周期。因此不能把本问题解释成“训练主进程每次重启，旧 Router 都必然留下”。旧服务已销毁时会新建，不存在这次讨论的旧配置复用。

## 3. 为什么同名服务可能使用旧配置

Router 名称是查找服务的标识。配置是服务应当怎样工作的内容，两者作用不同。

教学例子：第一次启动时，名字是 X，GPU 预算是 80 GiB。Router 在启动时创建账本，之后允许任务申请资源。第二次启动仍指定名字 X，但把预算改成 40 GiB。

原入口使用 `get_if_exists=True`，含义是“已经有同名服务就复用”。如果找到旧服务，Ray 不会为这个旧对象重新执行构造函数。新调用传入 40 GiB，不代表旧账本从 80 变成了 40。

**2026-11-11 补充：到底混淆了什么。** Router 不负责给每个训练进程发一本独立账本。每个 Router 内维护一份资源账本，为环境任务发放租约。复用同一个 Router 时，各调用方共享这个已有账本。

这里有三个不同的身份，不要混为一谈：

| 身份 | 标识什么 | 是否属于 R1 |
|---|---|---|
| Router 的 actor 名称 | 查找哪个资源管理服务，受 Ray 集群和命名空间约束 | 是，本次防止同名服务与新配置不一致 |
| `session_id` | 一次环境任务，用于找到或创建租约 | 不是本次配置复用问题。同一 ID 重试会返回已有租约，避免重复计费 |
| `lease_id` | 一次已经获准的资源预约 | 标识资源归属，不是训练进程名 |

创建入口见 [create_env_router](../../../../psrl/workers/env_service/router.py#L752)，共享账本在 [EnvRouterCore.__init__](../../../../psrl/workers/env_service/router.py#L49) 建立，任务身份与已有租约复用见 [EnvUnitLedger.try_admit](../../../../psrl/workers/env_service/ledger.py#L113)。worker 按 Router 名称取得句柄，见 [EnvServiceClient.from_config](../../../../psrl/workers/env_service/client.py#L124)。

最直接的错误是“启动方以为新配置已经生效，实际请求交给按旧配置运行的服务”。不要求服务把两个进程认成同一个进程，也不意味着账本自己把加减算错。

进一步的容量例子，以下全部为教学假设：物理 GPU 容量 80 GiB，旧 unit 配置允许环境使用 80 GiB，新启动方希望只使用 40 GiB，给其他工作留余量。任务 A、B 各申请 30 GiB，假设其余条件允许，且实测物理使用尚未触及限制。

| 时刻 | 真正采用新配置时 | 错误复用旧配置时 |
|---|---|---|
| A 申请 30 GiB | 获准，已预约 30 | 获准，已预约 30 |
| B 再申请 30 GiB | 30 + 30 超过 40，B 等待 | 30 + 30 未超过 80，B 可能获准 |
| 最终 | 保持新设定的 40 上限 | 可能承诺 60，超出新意图 20 |

这个例子说明新资源策略被违反，不证明必然显存不足。旧 Router 的实测显存检查、物理容量和原有归属保护仍在，不能把 R1 夸大为这些检查全部失效。若与其他工作叠加竞争，可能进一步造成资源争抢、等待或显存不足，但需要真实负载才能确定。

反方向也有危害：旧上限 40、新意图 80，服务仍按 40 管理，明明希望增加容量，任务却继续排队。旧节点选择范围和旧执行后端也可能与新意图不一致，例如以为已切 MPS 并发，服务实际仍按 direct 运行。哪条路径最终失败或继续运行取决于其他校验，本次未将这些推演当作实机事故。

仅调用 `ping` 也不够。ping 只能回答“服务能否回应”，不能回答“服务是不是按我的配置启动的”。这就是 R1 的背景。

修复后，启动方先取得旧服务实际保存的启动契约，再比较。80 与 40 不一致，启动方直接报错，预热也不会继续。旧服务的活动任务和预算保持原样。

不能偷偷把旧账本改成 40。假如旧服务已经批准 60 GiB，修改上限会立即产生“已承诺 60，但总容量只有 40”的矛盾。安全修改需要明确结束活动租约、确认清理、停止旧服务，再用新配置创建。

这里 GiB 是二进制容量单位，1 GiB 等于 1024 MiB。项目配置中的 `gpu_memory_mb` 延续现有 MB 字段约定，例子的整数单位用于教学比较，不是实测显存。

## 4. 配置快照指纹和协议版本

**快照**就是保存某个时刻的一份独立数据。本次在 Router 初始化时保存配置快照，避免调用方之后修改嵌套字典，连带改变“服务启动时到底是什么配置”的记录。

**规范化表示**用于消除无意义的写法差异。例如字典 `{"a":1,"b":2}` 和 `{"b":2,"a":1}` 内容相同，按键排序后都会生成同一份 JSON 文本。JSON 是一种用文本表示字典、列表、数字等数据的格式。本次还把列表和元组转成列表，对训练及推理节点身份集合排序并去重。

**指纹**是这份文本的短摘要。本次使用 SHA-256，即一种将任意长度数据计算为固定长度摘要的算法。相同输入产生相同摘要，内容改变通常会改变摘要。它用于配置身份，不是访问密码，不会证明服务有权限管理某张 GPU。

实现同时比较指纹和完整配置结构，没有只依赖摘要相等。配置原文不会被放进报错日志，避免产生不必要的信息泄露。

**协议版本**是通信双方对接口契约的版本号。配置相同但双方理解不同，也不能安全复用，因此版本也进入比较。目前版本为 1，未来改变契约含义时需要维护版本。它不是整个 PSRL 的 Git 提交号，也不自动检测全部代码差异。

本次采用严格的“声明配置”核对。字典键顺序不影响比较，但任意列表顺序并不自动忽略。“省略一个默认项”和“显式写出相同默认值”也可能被判成不同配置。调用方应传入同一份解析后的配置，不能假设系统会推理任意两份配置的语义等价。

## 5. 为什么服务活着还不能恢复使用

R2 涉及两种不同条件。

- **活性检查**：服务能否回应，Docker 和 GPU 查询能否完成。
- **安全条件**：旧任务是否仍拥有资源，旧容器是否清理，显存是否确认归还。

“查询失败”只表示不知道实际情况，不能把未知当成没有占用。**隔离**在这里指暂时不允许继续使用不确定资源，不是删除旧任务，也不是已经回收显存。

**lease，即租约**，记录某个任务获准使用哪些资源和它的身份。**预算**是账本对资源的预约，可能早于进程实际分配显存。**物理清理确认**则来自真实执行节点，证明所属容器等资源已经处理完成。三者不能互相替代。

教学例子：一张 80 GiB GPU 上，任务 A 已获准 60 GiB。Docker 查询突然失败。若此时清空 A 的记录，再批准 B 的 60 GiB，真实机器上的 A 可能根本没退出，总需求便可能达到 120 GiB。

正确做法是暂时阻止不确定资源复用。Docker 恢复后，先检查旧任务归属，执行受控清理并获得确认，之后才释放相应预算。单纯收到一次心跳或把错误字符串清空，都不能证明安全。

本实现中 `pending_reaped` 保存已经退休但尚未确认清理的租约。即使健康检查返回成功，`try_admit` 仍会检查这个集合。`blocked_units` 和 GPU 命令的 `quarantined` 状态也不会因为人工探测而自动删除。

## 6. 有限重试与受控恢复

**重试预算**是允许自动尝试的次数。默认连续失败达到 3 次，自动健康探测停止，任务准入明确失败。这避免节点长期故障时无限请求服务、无限等待，也让故障暴露出来。

问题是：预算耗尽以后，节点可能已经被修好，但单纯再次提交任务不会让计数自动归零。因此需要一个明确的人工作业入口。

新增 `node_diagnostics(node_ip)` 只读返回失败次数、最后错误、待清理状态、活动租约、退休待清理租约、阻断 unit、未完成 GPU 命令和旧借卡阶段。读取不会改变计数或释放资源。`healthy` 只是健康维度，不是保证所有类型的新任务都能获准。

新增 `recover_node(node_ip)` 的步骤是：

1. 检查节点属于当前配置，并且服务不在关闭过程中。
2. 如果已有探测在运行，等待同一个探测，不再额外创建一套检查。
3. 否则重新开放自动重试计数，先把健康状态标成“等待物理检查”，然后执行一次带超时的真实探测。
4. 探测通过且无节点清理待办时，健康条件恢复。若失败，保留错误。若仍有 GPU 隔离或账本待办，继续遵守这些限制。

这里的**超时**是等待的时间上限。它避免一次恢复调用无限挂起，不代表超时后后台物理资源已自动清理。重新开放计数后，常规路径仍使用原来有限的自动重试预算。

**并发合并**表示两个管理员同时点恢复时，尽量共享正在运行的那次检查。服务在同一个事件循环里先登记任务，再等待它，第二次调用就能看到并复用这个任务。取消一个等待者也不会取消被共享的检查。

实际调用形态如下，仅适用于已经连接正确 Ray 集群并取得当前 Router 句柄的管理程序：

```python
before = ray.get(router.node_diagnostics.remote(node_ip))
after = ray.get(router.recover_node.remote(node_ip))
```

`before` 用于判断阻塞原因，`after` 用于确认新的探测结果。这里没有提供自动连接任意集群的命令，防止管理错运行中的服务。该接口依赖现有可信 Ray 管理域，不新增用户身份鉴权，也不能处理所有需要人工接管的孤儿 MPS 服务。

## 7. 跟着图读源码

| 步骤 | 图中编号 | 文件、函数和输入输出 |
|---|---|---|
| 创建请求 | 1 | `create_env_router` 接收配置和训推节点身份，生成预期契约，再创建或获取同名 actor |
| 服务端快照 | 2 | `EnvRouterCore.__init__` 保存快照，`startup_contract` 返回独立副本，不返回内部可修改引用 |
| 复用核对 | 2 | `verify_router_contract` 比较预期与实际，不一致报错，一致才进入预热 |
| 正常健康路径 | 3 → 4 | `_health_task` 创建异步检查，`_check_node` 探测节点、容量和清理，`try_admit` 最后核对账本并返回许可 |
| 只读排查 | 5 | `node_diagnostics` 返回已知错误和资源记录，不重试、不删除 |
| 人工恢复分支 | 6 → 3 | `recover_node` 创建或合并一次探测，等待其结束，再返回诊断 |
| 清理确认 | 3 → 4 | 节点 `cleanup` 向 Router 发送确认，未确认的租约继续挡住准入 |

各函数的可跳转链接见第一节图例。节点服务异步执行，清理确认会回调 Router，不能把本节当成始终单向的同步函数调用栈。节点 actor 已死且仍有活动归属时，原逻辑继续保留旧句柄，恢复入口不会擅自接管。无活动归属的死服务可以按原检查逻辑被移除，下一次探测才创建新服务。

## 8. 运行教学实验与回归

在仓库根目录、项目 Python 环境执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_router_recovery.py
PYTHONPATH=. python -m pytest tests/env_service/test_router_contract.py tests/env_service/test_node_prewarm.py tests/env_service/test_design_remediation.py -q
```

[教学脚本](../../env-service-verification/scripts/model_router_recovery.py)使用标准库，直接加载本次生产契约比较函数。恢复部分是教学模拟，用布尔值表示 Docker 可用和清理确认，不启动 Ray、Docker、GPU，也不真正释放显存。

预期先输出相同配置 accepted、变化配置 rejected。随后显示连续失败 3 次时保留 60000 的预约，服务恢复但未确认清理时仍不健康且预约不变，确认清理后健康且预约归零。数字只演示状态变化，真实系统仍要实际检查节点与资源。

生产接口回归使用节点故障替身，覆盖耗尽后恢复、继续失败、超时、并发合并、活动归属保留、退休租约未确认不准入、无归属死服务重新探测。它们与真正的多进程 Ray 或 GPU 故障注入有明确区别。

本次实际运行上述教学脚本，得到相同配置通过、变化配置拒绝、未清理保留预约、确认后释放预约的预期输出。扩展回归同时覆盖 Router client、GPU command 和 MPS 作业分支，共 82 项 CPU 测试通过，耗时 19.86 秒。修改涉及的六个 Python 文件 Ruff、格式与 prose 检查通过。全仓 pre-commit 在临时隔离副本运行，因原有文件的长行、格式和注释规则问题失败，未把自动修改带回主目录。

2026-11-11 增量实机验证使用 [router_live_recovery_check.py](../../env-service-verification/scripts/router_live_recovery_check.py)。在实际执行节点激活项目环境后，从仓库根目录执行，`NODE_IP` 为经路由核对的该节点地址，`RESULT_FILE` 为本次独立结果文件：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=/tmp
unset RAY_ADDRESS
timeout --signal=INT --kill-after=15 120 python docs/env-worker-deep-dive/env-service-verification/scripts/router_live_recovery_check.py --node-ip "$NODE_IP" --output "$RESULT_FILE"
```

脚本创建隔离 Ray 集群和独立命名空间，只分配 CPU 逻辑资源。R1 调用生产创建入口，确认同配置得到同一 actor，配置改变返回拒绝。R2 继承生产 Router 与节点服务，只有本次测试节点提供故障开关，实际 Docker 探测和生产恢复逻辑保留。失败注入、超时、未确认清理都保持不健康，清除注入后恢复健康，实机运行全部通过。

这次验证不包含真实 Docker 停服、活动租约遗留、双 driver 生命周期或 MPS 服务失联。未创建环境容器或 GPU 作业。结束后已确认本次 Ray 控制进程退出，节点八张 GPU 显存仍均为 0 MiB。原始结果保留在执行节点的本次结果文件，包含节点身份与故障堆栈，公开文档只记录验证边界。SSH 使用用户现有配置即可，先前额外开启严格主机密钥校验造成的阻断不是节点不可达。

## 9. 本次完成与剩余边界

R1 已实现启动契约核对，R2 已实现节点只读诊断及受控重新探测。没有热改活动配置，不清空租约，不解除 GPU 隔离，也不新增原请求迁移。

旧 actor 没有契约接口时明确拒绝复用。应先通过旧版本管理流程结束任务并确认清理，再停止旧 actor、用新代码创建。不能仅为了升级而全局停止共享 Ray 或 Docker。

仍待完成的 R3 是 Router 重启后恢复完整状态、跨 Router 设备唯一所有权，以及旧控制器失权后的授权阻断。它要求更多持久化和所有权协议，本次人工重新探测不等于已经解决这些问题。


## 10. 一条 rollout 中的命令与最终评分

本节于 2026-11-11 核对主目录源码，完成核对时 HEAD 为 `6aa328d`，包含本次未提交文档和实验脚本。范围是 FlashInfer 的普通算子子集、单任务单最终评分模式，不涵盖 sampling 特殊评分或 Harbor 多阶段任务。此前章节的验证日期和范围保持独立。

```text
[V1] 一道题 → 创建一次 rollout 的环境
                  ↓
[V2] 模型生成本轮动作（使用模型推理服务）
       ├─ CPU command：读文件、改代码 ───────────┐
       └─ [V3] psrl-gpu：申请 GPU → 自测 → 归还 ─┤
                  ↑                             │
                  └──── 工具输出反馈给下一轮 ────┘
                  │ Agent 宣告完成或到达结束条件
                  ↓
[V4] 收集提交文件，结束 Agent 环境
                  ↓
[V5] 独立 verifier 环境，读取提交与受保护的测试
     一个评分过程：准备 → 必做测试单元 1 → 单元 2 … → 汇总
                                  ↑
                     提案的观测点在这里，尚未完成评分
                  ↓
[V6] 最终 reward → 清理资源 → 已评分轨迹交付训练
```

图例：图里每个部分对应的源码位置。

| 节点或箭头 | 职责与函数 | 源码 |
|---|---|---|
| V1 | `run_harbor_episode` 先申请环境，再创建 Harbor Job | [runner.py:372](../../../../examples/sciaccel_rl/runner.py#L372)、[runner.py:550](../../../../examples/sciaccel_rl/runner.py#L550) |
| V2 | `_query_llm` 调用父类模型交互，`_execute_commands` 区分普通命令与 GPU 工具 | [agent.py:125](../../../../examples/sciaccel_rl/agent.py#L125)、[agent.py:104](../../../../examples/sciaccel_rl/agent.py#L104) |
| V3 | `dispatch_gpu_tool` 将 start、wait 等操作交给环境，`exec_gpu` 进入统一执行层 | [gpu_tools.py:32](../../../../examples/sciaccel_rl/gpu_tools.py#L32)、[managed_execution.py:53](../../../../psrl/workers/env_service/managed_execution.py#L53) |
| V4 → V5 | `prepare` 指定 separate 模式与提交位置，`job.run` 调用 Harbor 编排 | [prepare.py:37](../../../../examples/flashinfer_bench/prepare.py#L37)、[prepare.py:72](../../../../examples/flashinfer_bench/prepare.py#L72)、[runner.py:551](../../../../examples/sciaccel_rl/runner.py#L551) |
| V5 GPU 授权 | 选用 `KernelCommandVerifier` 时，将评分 shell 调用包装为 `exec_gpu` | [verifier.py:34](../../../../examples/kernel_bench/verifier.py#L34)、[verifier.py:43](../../../../examples/kernel_bench/verifier.py#L43) |
| V5 测试和 V6 奖励 | `main` 调用官方 `Benchmark.run_all`，完整通过才写奖励 1 | [verify.py:70](../../../../examples/flashinfer_bench/verify.py#L70)、[verify.py:79](../../../../examples/flashinfer_bench/verify.py#L79) |
| V6 清理与训练数据 | runner 先清容器再释放租约，agent loop 将评分附到采样输出 | [runner.py:625](../../../../examples/sciaccel_rl/runner.py#L625)、[runner.py:635](../../../../examples/sciaccel_rl/runner.py#L635)、[agent_loop.py:262](../../../../examples/sciaccel_rl/agent_loop.py#L262) |

V4 到 V5 的具体编排在已安装的外部 Harbor 依赖中。本次实际读取其 `harbor/trial/single_step.py`，`SingleStepTrial._run` 第 41 行调用 `_run_agent`，第 52 行调用 `_run_verifier`。separate 模式先收集产物并停止 Agent 环境，再评分。依赖源码不在本仓库，因此表中给出本地调用入口和配置，不伪造仓库内实现。Agent 超时或异常后的评分与恢复另有分支，不能将图中的正常完成路径当作所有异常路径。

**先区分层次。** rollout 是针对一道题的一次完整尝试，里面可以有许多 turn。turn 是一次模型生成动作并获得工具反馈的交互轮次。command 是动作里的具体命令，一轮可以发多个命令。一个 GPU command 又可以运行许多 GPU kernel，即真正交给 GPU 并行执行的计算函数。它们没有一一对应关系。这里把最终评分也计入交付训练的一条完整 rollout，若只说“生成结束”，则可能尚未评分。

**用一条实际任务类型举例。** 题目要求优化 RMSNorm，即用一行数字的均方根调整数值尺度的运算。输入每行 2048 个数，受保护测试有 1 行和 16254 行两种大小。下面的 Agent 修改和成败顺序是教学假设，这次实验没有调用模型生成代码。

1. 第一轮，模型决定查看函数定义，用普通 command 读文件。CPU 执行读取，环境 GPU 不需要参与。模型产生这条回复通常仍使用推理服务的 GPU，不能把“环境没用 GPU”理解为整个系统没有 GPU 工作。
2. 第二轮，模型生成候选实现，用普通命令写入文件。候选就是本次准备提交、可能对也可能错的程序版本。
3. 第三轮，模型通过 `psrl-gpu` 的 start 操作启动自测。start 返回作业编号，任务可能仍在排队或运行。wait 操作取得状态和输出，等待超时本身不取消计算。假设小输入通过、大输入失败，错误成为下一轮模型输入。自测没有生成训练使用的最终分数。
4. 第四轮，模型根据失败修改实现，再发 GPU 自测。假设两种输入通过，完成后确认 GPU 归还。在 command 模式，后续模型思考和 CPU 修改期间可让其他任务申请环境 GPU。episode 模式则可能保留整次会话的预约，不能混为一谈。
5. Agent 提交最终文件并结束生成。Harbor 收集文件，单独启动 verifier，即可信的最终评分程序。它读提交代码和受保护的题目测试，不信任 Agent 自称“全部通过”的文本。
6. verifier 调用官方 evaluator，即实际生成测试输入、执行候选、比较参考结果和测量耗时的评估库。必须完成整个子集，且所有结果通过，当前普通算子 recipe 才写 reward=1，否则为 0。reward 是训练使用的数值奖励，本 recipe 的奖励是正确性，记录了速度不等于速度已经进入奖励。
7. 等结果和资源清理完成，带奖励的轨迹才供训练使用。得到一条奖励不是立即完成一次模型参数更新，训练还需要收集相应样本并按训练配置处理。

**verify 究竟在哪里。** Agent 可以主动运行一个自测脚本，文件名即使叫 verify.py，也不代表框架在每条 command 后自动执行最终 verifier。此处的官方 verifier 在 Agent 阶段结束后运行一次，但它内部会跑多项测试。别的多阶段任务可以每阶段评分，本文不将单阶段规则推广到所有任务。

**研究建议的准确位置。** 候选代码提交后，未来还会改什么代码这一不确定性消失了，已知测试列表也已经固定。让最终 verifier 完成第一个必做单元后，可用观察更新其余单元的成本估计，再决定如何安排剩余工作。这里是“一次最终评分内部的中间点”，不是“最终评分结束以后”，更不是“每个 Agent command 之后”。第一项测试包括哪些准备、是否代表后续输入，都必须核验。暂停 Python 循环也不等于已经释放 GPU，进程可能仍持有显存和 CUDA 状态，实际让其他任务使用还需要资源协议。

跟着图读源码：V1 的 runner 创建 Job → V2 的 Agent 父类交互循环反复生成命令 → `_execute_commands` 普通分支或 V3 的 `dispatch_gpu_tool` → `start_gpu_job` 异步执行，wait 观察状态 → Agent 结束后由 Harbor 依赖执行 V4、V5 → `verify.main` 的 `run_all` 和汇总 → V6 的 runner 清理、lease 释放及 agent loop 评分输出。V3 的 GPU 作业与独立 CPU 操作可并发，多个 rollout 也可交错。这个阅读顺序是控制依赖，不代表全系统单线程。上述代码如果有不清楚的地方，可以随时追问。

运行 [CPU 教学脚本](../../env-service-verification/scripts/rollout_command_verifier_demo.py)：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/rollout_command_verifier_demo.py
```

脚本模拟两轮“改代码加自测”，最终提交后再运行两项评分测试，打印 V1 至 V6，最终 reward=1。它用字典真假值替代真实候选，用打印替代模型、容器和 GPU，目的是分清顺序，不是验证真实吞吐。本次已实际运行并通过断言。真实 GPU 的分段可行性结果见[实验报告](../../experiments/gpu-benchmarks/07-FlashInfer_进程复用与任务合批.md#保留评估器分段的-results)。

## 11. 当前方案是否必然快于固定基线

当前已经有修复后的正收益证据。固定 Env OFF 基线为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，不能用当前代码关闭开关替代。2026-11-11 对照保存的 JSON 复核后，应采用以下结果，而不是早期整任务占卡与修复前慢查询反例。

| 场景 | 固定 OFF | 修复后 ON | 解释 |
|---|---:|---:|---|
| Science 人工高压力，3 步、12 条接受轨迹，完整驱动时间 | 1954.11 秒，18 次 OOM | 1415.01 秒，0 次 OOM | 少 27.59%，主要减少失败补采 |
| 真实常驻模型整卡压力，7 项任务、14 请求 | 158.48 秒，1 次 OOM | 141.73 秒，0 次 OOM | 少 10.6%，无人工降低 ON 容量 |
| 修复后低负载同期配对，两对总时间 | 213.96 秒 | 217.15 秒 | 接近，尚未证明稳定提速 |

OOM 是显存不足导致分配失败。前两项说明 Env 在明确压力条件下已经超过固定旧版。它们来自各自冻结版本，不是今天 HEAD 重跑，也不支持所有负载都必然更快。具体配置、计时口径和机器证据统一见[有效实验结论](../../experiments/01-实验总览_调度收益与证据范围.md)。

此前本节列出的早期整任务保留、ON 慢 64.3% 的结果已经移除。该执行模式的缺口后来由 command 借还与相关修复处理，不能继续当作当前方案的性能反例。同样，最早低负载开销已有同步准备与连接路径优化和后续配对，不能跨版本挑选旧慢值代表现状。

教学例子：同样成功计算需要 100 秒，旧版失败重试另花 40 秒，Env 管理花 10 秒，则 140 秒变成 110 秒。若本来不失败，管理成本仍需与实际省下的等待比较。这个算例解释收益机制，不替代上表的实测。

新增预测策略还要与当前已有的批量执行、进程复用、command 借还比较。旧版对照证明系统已有价值，新策略对照证明额外增量，两者应分别报告。本次分段探索属于后者的前置可行性，不能倒过来说当前 Env 尚无任何性能收益。


## Worker 失败、重试与整组补采

> **核心结论**：PSRL 已有同 worker 重试、失败整组补采、重新选择 worker 和连续失败熔断。worker 是实际执行采样任务的工作进程；rollout 是 Agent 从接题到提交的一条完整尝试。恢复时，框架可以重试或补出新的轨迹，原请求跨 worker 转移则还需要完整协议。

例如一题需要 4 条轨迹，原工作进程只成功返回 3 条，框架可以按既有规则重新采样；若要把失败的那一个请求交给另一进程，必须先确认旧执行停止，避免两份结果同时返回。第 3 节用四条轨迹说明两种情况。

历史参照提交 `0eeead861ffafed6bf31c962d259fd847775dcf9` 已包含基本恢复链，不能把框架原有恢复能力全归为新增 Env 调度收益。本文记录执行监督与重试链路的正确性修复，并解释原请求迁移仍需补齐的身份、清理和次数约束。


### 1. 先把角色和执行链连起来

下面是本轮修复后的实际调用关系。虚线含义用文字标明，不把后台并发画成单线程顺序。

```text
[1] DataProcessor 提供一组请求
            ↓
[2] AgentLoopManager 选择 worker，启动并监督远程调用
            ↓ 远程调用直到整批执行结束才返回
[3] AgentLoopWorker 运行每条 trajectory，并按配置在本 worker 重试
            ↓
[4] science agent → Harbor runner → EnvRouter 准入 → Docker 容器
            │                                            │
            │ 正常结束，清理环境                          │ 心跳消失
            ↓                                            ↓
[5] 成功结果进入训练数据队列                     [8] Node 按租约清理孤儿容器
            │
            └─ 可恢复失败 → [6] notify_group_failed
                              ├─ 训练：丢弃失败组，补一组新请求 → 回到 [1][2]
                              └─ 验证：缩减需要收集的组数

[3] 进程被杀或执行异常未处理
            ↓ 远程调用失败，由 [2] 持续监督到
[7] 锁存失败原因 → 唤醒当前训练/验证等待者 → 后来的等待者也立即失败

[8] 是独立清理分支。收到 [7] 的错误不等于容器已经清空。
```

图例：图里每个部分对应的源码位置。

| 编号 | 角色、函数及职责 | 源码 |
|---|---|---|
| 1 | `_retry_data` 向 DataProcessor 请求新训练样本 | [manager.py](../../../../psrl/workers/agent_loop/manager.py#L498) |
| 2 | `_inner_dispatch_data` 选择执行者，`_supervise_worker_batch` 观察远程执行结果 | [派发](../../../../psrl/workers/agent_loop/manager.py#L891)、[监督](../../../../psrl/workers/agent_loop/manager.py#L931) |
| 3 | `add_agent_program` 执行并等待整批任务，`_run_agent_loop_inner` 做本地重试 | [worker.py](../../../../psrl/workers/agent_loop/worker.py#L220)、[重试](../../../../psrl/workers/agent_loop/worker.py#L313) |
| 4 | SciAccel `run` 每次进入时创建会话并执行 Harbor 任务 | [agent_loop.py](../../../../examples/sciaccel_rl/agent_loop.py#L144)、[会话](../../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L64) |
| 5 | `postprocess_output` 处理成功轨迹并向队列和 Manager 提交结果 | [worker.py](../../../../psrl/workers/agent_loop/worker.py#L461) |
| 6 | `notify_group_failed` 幂等处理组失败，`_refill_failed_group` 受熔断状态约束 | [组失败](../../../../psrl/workers/agent_loop/manager.py#L764)、[补采](../../../../psrl/workers/agent_loop/manager.py#L630) |
| 7 | `_trip_refill_breaker` 保存错误并唤醒等待者，`wait_for_training_batch` 检查失败状态 | [失败锁存](../../../../psrl/workers/agent_loop/manager.py#L581)、[训练等待](../../../../psrl/workers/agent_loop/manager.py#L1918) |
| 8 | `cleanup` 删除物理容器、查询确认后回执 | [node.py](../../../../psrl/workers/env_service/node.py#L83) |

这些代码如果有不清楚的地方，可以按编号继续追问。

这里的 **AgentLoopWorker** 是运行 agent 整段任务的 Ray 进程。Ray actor 是可被远程调用的常驻进程对象。**Manager** 管派发和训练数据组，**EnvRouter** 管环境资源准入，二者职责不同。另一个组件 **EnvWorker** 面向环境和沙箱操作，不能仅凭名字相似就认为它负责整段 agent 任务的恢复。

RPC 是远程方法调用。调用返回“已放进本地队列”，只证明接收成功，不能证明任务完成。本轮把 AgentLoopWorker 的批次 RPC 保持到执行结束，Manager 仍然异步发出多批请求，不会串行等待上一批跑完才发下一批。差别在于它保留并检查执行回执。

### 2. 原框架支持到哪一步

| 能力 | 本轮之前已经支持什么 | 不能据此推出什么 |
|---|---|---|
| 单条任务本地重试 | `retry_limit` 控制同 worker 调用 agent 的总次数 | 不会因为重试自动更换 worker |
| 失败分类 | `needs_worker_retry` 筛选错误、轨迹超时等需要重跑的情况 | 模型答错、奖励为零不自动触发基础设施重试 |
| 训练组失败恢复 | `notify_group_failed` 取消同组其他子请求、清除部分结果，再补采一组新请求 | 不保证重跑同一题，不恢复中断的容器或 shell |
| 补采后的放置 | 新请求重新进入 Manager 的派发逻辑，可被分配给其他 worker | 没有“必须避开这次失败的 worker”这一同请求转移协议 |
| 补采上限 | 连续失败达到 `refill_failure_threshold`，默认 32，会锁存错误 | 原实现的锁存不等于所有调用路径都已经停止继续发补采 |
| 验证集失败 | 丢失一组后缩减本轮需要收集的组数，全失败可返回空批次 | 验证失败不会自动换一道训练题补上 |
| Env Service 派发前重试 | 暂时没有可用节点时有限重读容量、再做选择 | 不等于任务执行中崩溃后换 worker 重跑 |
| 运行中 worker 进程死亡 | 环境租约和独立清理器可回收容器 | Manager 原来没有通过执行回执可靠获知所有进程死亡 |

默认 [retry_limit](../../../../psrl/trainer/config/rollout/psrl_rollout.yaml#L154) 为 1，含第一次执行，所以默认没有第二次本地尝试。设为 2 才表示第一次加一次重试。没有在本轮直接调大默认值，因为确定性的内存不足、损坏任务或错误配置会被重复执行。

[TerminateReason](../../../../psrl/workers/agent_loop/loops/utils.py#L138) 是任务结束原因分类。可保留训练数据的预算截断与“没有可用数据的执行错误”不同。`ABORTED` 表示框架已经接管请求取消，worker 不再额外补采，否则会重复补偿同一个训练位置。

固定 baseline 中的证据可以在本地只读核对：

```bash
git show 0eeead861ffafed6bf31c962d259fd847775dcf9:psrl/workers/agent_loop/worker.py \
  | rg -n 'retry_limit|notify_group_failed|def add_agent_program'
git show 0eeead861ffafed6bf31c962d259fd847775dcf9:psrl/workers/agent_loop/manager.py \
  | rg -n 'def _retry_data|def notify_group_failed|refill_failure_threshold|add_agent_program.remote'
```

预期能找到两层恢复函数和默认阈值。这个检查不切换工作目录，也不修改用于性能比较的固定 baseline。

### 3. 用四条轨迹解释重试和补采

下面是教学假设，不是本轮训练测量：训练需要 2 组数据，每组对同一个题目采 4 条 trajectory。trajectory 是 agent 从题目输入到结束的一整段交互。组内相对奖励算法需要同组轨迹一起处理，所以不能随便拿其他题目的单条轨迹补缺口。

题目 P 的组号为 10，四条子请求编号是 40、41、42、43，被放到 worker A。另一组题目 Q 被放到 worker B。

1. **同 worker 重试**：40 遇到临时环境错误。如果 `retry_limit=2`，A 会再调用一次 agent，逻辑请求还是 40。Science 会重新进入任务执行和创建会话，不代表从容器内第 7 条命令接着跑。如果第二次成功，结果归到原来的 P 组。
2. **训练整组补采**：假设 40 最终失败，Manager 取消 41、42、43 并清掉 P 组的部分结果。DataProcessor 新采一组 R，组号和子请求身份重新分配。R 可能被派到 B，也可能再次落到 A，取决于派发策略和当时资源状态。训练仍需要 2 组成功数据，这不是“把 P 转移到了 B”。
3. **验证集处理**：同样是 2 组验证任务，P 组失败后目标可缩减为 1 组，以已有成功的 Q 组结束本轮。它不会把另一道 R 当成 P 的验证结果。
4. **进程直接被杀**：A 没有机会执行 `notify_group_failed`。本轮新增监督后，Manager 接收到 Ray 的 actor 死亡错误，通知等待训练数据的调用方失败。容器清理由独立节点服务继续完成。这一步没有自动把 P 放到 B。

再看本轮修复的一个具体错误：第一次尝试抛异常，变量记录了异常。第二次成功，原先该变量没有清空，于是提交成功结果之后仍可能重新抛出第一次的异常。现在每次尝试开始都重置它，第二次成功就按成功结束，避免“结果已经提交，执行却又被报告失败”。

### 4. 为什么不能直接把原任务扔给另一台机器

真正的同请求跨 worker 重派发至少要满足这些条件：

1. **确认上一轮不会继续运行**。网络暂时不通不等于旧 worker 已死。若同时在 A 和 B 跑，可能重复写文件、重复提交结果。对有 GPU 句柄的任务，还要确认设备授权和旧进程已收回。
2. **区分每一次尝试**。同一个逻辑请求需要不同的 attempt 身份。Manager 只接受当前有效尝试的结果，忽略旧尝试迟到的回包。只有一个 uid 不足以区分这两者。
3. **定义重跑还是接着跑**。从头执行需要清晰的输入、输出和幂等约束。幂等指同一个操作重复执行不会造成额外副作用，例如不能重复写同一份最终结果。接着跑则需要保存 shell、文件系统、agent 状态甚至进程状态，属于更大的状态恢复问题。
4. **保留组语义和次数上限**。一条轨迹重跑后不能破坏整组奖励、模型版本和训练数据的对应关系。必须明确同请求最多尝试几次、哪些故障值得换节点、清理超时如何退出。

现有“新采一组”已经能补训练缺口。在这些条件未闭环前，先把 worker 死亡明确暴露给训练方，比静默无期限等待更可靠，也不会偷偷引入重复执行。自动同请求转移仍是待实现能力。

### 5. 本轮修复了哪些确定问题

| 修复 | 原先可能发生什么 | 当前行为 |
|---|---|---|
| 每次尝试清空异常状态 | 第二次成功后重新抛第一次异常 | 只传播最终失败尝试的错误 |
| 执行回执持续到任务结束 | 接收入队即返回，后续异常只写日志，Manager 可能永久等数据 | Manager 保留批次监督任务，执行异常和 actor 死亡锁存到等待方 |
| 熔断后禁止补采和派发 | 失败计数达到阈值后，当前路径仍可能继续采新题 | 达到阈值不再调用补采，后续派发也检查失败状态 |
| 所有等待入口共享失败状态 | 验证等待者或晚注册的训练等待者可能漏掉错误 | 整批训练、分块训练、验证等待者都收到失败，远程创建 buffer 后再次检查 |
| 关闭时取消并等待执行清理 | 管理进程结束了轮询，worker 后台任务可能仍在跑 | 停止接收新任务，取消并等待所属任务，保留有界的 worker 收尾等待 |

补采熔断有一个教学小例子：把阈值设为 3，且每一组都失败。第一次失败补第 2 组，第二次失败补第 3 组，第三次失败停止，因此只应调用新样本采集 **2 次**，不能再补第 4 组。默认阈值仍是 32，这个较小数字只用于回归验证。

`raise_on_error=True` 下最终未处理异常会由执行监督传到训练等待方，形成明确失败。需要允许普通任务失败后补采的配方，应使用其已有的错误分类和 `raise_on_error=False` 行为。这个配置不会把进程死亡变成可安全重放的普通任务错误。

### 6. 跟着图读源码

| 执行步骤 | 图中编号 | 函数及输入输出 |
|---|---|---|
| 选择并提交一批请求 | 2 | `_inner_dispatch_data` 接收包含 uid、parent_id、version_tag 的批次，更新运行状态，创建监督任务 |
| 运行任务 | 3 | `add_agent_program` 为每条请求建立执行任务，远程方法等待整批结束 |
| 单条任务处理错误 | 3、4 | `_run_agent_loop_inner` 根据结束原因决定是否再尝试，Science 进入 Harbor 创建和清理环境 |
| 发送成功数据或处理组失败 | 5、6 | 成功数据由 `postprocess_output` 提交。可恢复组失败调用 `notify_group_failed`，训练路径调用 `_refill_failed_group` |
| 后台监督异常 | 2、7 | `_supervise_worker_batch` 等待 Ray 回执。进程死亡或未处理异常触发 `_trip_refill_breaker` |
| 唤醒训练调用方 | 7 | `wait_for_training_batch` 在远程 buffer 创建前后检查错误，并在登记等待前再次检查 |
| 清理退出 | 8 | worker 正常退出等待环境清理，异常死亡由 Node 的租约清理路径接手，Manager 关闭时取消自身监督任务 |

逐项源码链接见第 1 节图例。Manager 监督任务与 worker 执行并发运行，训练等待方也是独立远程调用。成功数据依然走原训练数据通道，不通过批次 RPC 返回大张量。

### 7. 可运行验证与 Results

在仓库根目录、项目 Python 环境中执行 CPU 回归：

```bash
python -m pytest tests/agent_loop/test_worker_supervision.py \
  tests/agent_loop/test_manager_refill_breaker.py \
  tests/agent_loop/test_empty_training_group.py \
  tests/env_service/test_manager_dispatch.py tests/trainer/test_chunk_manager.py -q
```

这些测试使用真实 worker/manager 方法和故障替身，覆盖第二次尝试成功、执行期间回执保持未完成、取消后等待清理、训练与验证等待者收到异常，以及熔断上限不再补采。没有启动 GPU 或外部服务，不能仅据这些测试宣称真实 actor 死亡链路已通过。

真实实验脚本：[real_worker_supervision.py](../../env-service-verification/scripts/real_worker_supervision.py)。预检通过后，在有 Docker、已缓存 Python 镜像的目标节点运行：

```bash
read -r -p '目标节点 IP: ' WORKER_TEST_IP
read -r -p '已有 Python 镜像: ' WORKER_TEST_IMAGE
run_dir=$(mktemp -d /tmp/worker-supervision.XXXXXX)
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" nohup timeout 300 python \
  docs/env-worker-deep-dive/env-service-verification/scripts/real_worker_supervision.py \
  --node-ip "$WORKER_TEST_IP" --image "$WORKER_TEST_IMAGE" --output "$run_dir" \
  > "$run_dir/driver.log" 2>&1 < /dev/null &
```

预期依次验证正常结束、丢弃回执后杀进程、监督回执后杀进程。每阶段最多一个 0.5 核、128 MiB 的真实 Harbor 容器，容器运行 Python 计算 `6 × 7`。采用实际 Ray actor 和生产 `add_agent_program`、监督、熔断、训练等待、EnvRouter 租约清理方法。任务体是小型 CPU 夹具，PSManager 的 buffer 创建接口由最小 actor 替身提供，没有模型生成、训练数据提交和优化器更新。

丢弃回执对照是故障机制对照，不是固定历史提交的 science 性能 baseline。它有意不检查远程失败，预期训练等待超时。监督组预期收到明确 actor 死亡错误，后来的等待也立即失败。脚本在观测结束后按本次 unit 标签和 Compose project 清理资源。外层超时仍需按 `ownership.json` 核实没有遗留资源。

**Results：** 本轮在用户指定的私有节点后台执行，单次总预算 300 秒，脚本从开始到清理完成约 180.78 秒，退出码 0。阶段结果如下，时间是单次故障观测，不作为吞吐或延迟统计结论。

| 阶段 | 真实观察 | 说明 |
|---|---|---|
| 正常执行 | 容器输出 42，批次监督正常完成，没有失败锁存 | 正常结束不会误触发熔断 |
| 故意不监督回执，再杀 worker | 训练等待到 2 秒观测期限仍未结束 | 复现丢弃执行回执造成的故障不可见，不证明一定永久等待 |
| 监督回执，再杀 worker | 约 0.175 秒收到包含 `ActorDiedError` 的错误 | 实际经过生产 Manager 的监督、熔断与训练等待方法 |
| 后来的等待者 | 同样立即收到已锁存的死亡错误 | 错误不是一次性通知 |
| 所有阶段资源回收 | 每阶段租约和实际容器都回到 0 | 由生产 EnvRouter / Node 清理路径执行，再单独核验物理资源 |

[完整结果](../../env-service-verification/results/worker-supervision-20261027/evidence.json)、[实际执行脚本快照](../../env-service-verification/results/worker-supervision-20261027/executed-script.py)、[原始日志](../../env-service-verification/results/worker-supervision-20261027/driver.log)。日志中的未领取异步异常来自故意不监督的对照组结束时终止 Manager，验收本身通过。复现脚本后续只补了该对照等待任务的显式取消和循环变量绑定，生产代码不变，未为这项脚本收尾修改重复容器实验。

本轮最终 **42 项针对性回归通过**，覆盖重试、补采、空训练组、派发、分块等待与关闭清理。更宽的初始回归有 6 项多模态测试因直接实例化抽象 `AgentLoopBase` 失败，另有 2 项旧测试夹具缺少验证等待者字段，后者已补齐并回归通过。不能把这一轮写成“全仓测试全通过”。本轮代码在主工作区，尚未提交、推送或合并，其他会话提交的 Env Service 工作保持独立记录。

收尾另行核对本次所属容器、网络、隔离 Ray 进程均已释放，见[物理释放记录](../../env-service-verification/results/worker-supervision-20261027/physical-release.json)。

本轮涉及的 6 个 Python 文件定向 Ruff 和格式检查通过，339 个本地链接、章节锚点和源码行号范围检查通过。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因历史脚本和其他已有风格问题失败，见[完整检查日志](../../env-service-verification/results/worker-supervision-20261027/precommit.log)，未将自动修改复制回工作区。

### 8. 尚未实现的范围与后续顺序

本轮没有实现同请求跨 worker 自动重放、容器状态迁移、失联但未确认死亡进程的强制接管，也没有证明训练速度或模型效果提升。可信进程归属仍由其他工作单独维护。它修复的是现有重试/补采链路和故障可见性。

仍存活但永久阻塞的 worker 不会产生 actor 死亡错误，不能靠本次监督自动识别。任务时间边界由 `trajectory_timeout` 等配置决定，本轮不擅自缩短长任务超时。

后续若推进跨 worker 重派发，建议先限定为“已确认旧执行结束的、可从头重跑的任务”，复用现有整组处理与有界失败策略，再增加 attempt 身份、旧结果隔离、坏 worker 排除和物理清理确认。不要先全局调高 `retry_limit`，也不要把任意 OOM 或低奖励都当成换节点后必定成功。

相关总审计与物理清理机制见[设计审视第 13 节](../../experiments/11-历史审阅_低负载配对与GateAB设计.md#13-物理资源遥测与清理交接确认)。
