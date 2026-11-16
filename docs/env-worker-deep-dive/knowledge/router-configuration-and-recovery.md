# 从零理解 Router 配置复用与节点恢复

> **核心结论**：复用 Router 前先核对配置，恢复节点前先核实资源。配置契约把本次预期的预算、设备和后端与现有服务比较；诊断与人工恢复入口则重新探测节点，并保留尚未确认清理的资源记录。

例如第一次启动允许同时运行 4 个容器，第二次配置改成 8 个。如果第二次只是找到同名服务，它可能仍按 4 个运行。启动核对会把这个差异直接报出来，避免任务按错误预算继续执行。第 3、4 节展开完整过程。

本文从训练怎样发起环境任务讲起，再解释服务对象、远程调用、配置比较和故障恢复。R1/R2 实现与 CPU 回归为 2026-11-10 记录；11-11 对 `ab74120` 加工作区内容核对训练链，并补隔离真实 Ray/Docker 探测验证，范围集中在第 8、9 节。本文提供机制解释；总体能力见[当前设计](current-design.md)，工程优先级见[工程审查](engineering-review.md)。


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

## 1. 先把整个过程连起来

训练会不断产生环境任务。环境任务需要创建容器、运行命令、使用 GPU，结束后释放资源。如果多个任务各自决定能否使用 GPU，它们可能同时许诺超过总容量的显存。因此需要一个共同决定“谁可以进入”的资源服务，这里叫 EnvRouter。

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
| 1 | `create_env_router` 创建或复用命名服务 | [router.py](../../../psrl/workers/env_service/router.py#L752) |
| 2 | `router_contract` 生成配置快照，`startup_contract` 返回服务实际启动契约，`verify_router_contract` 比较 | [生成](../../../psrl/workers/env_service/router_contract.py#L10)、[返回](../../../psrl/workers/env_service/router.py#L437)、[比较](../../../psrl/workers/env_service/router_contract.py#L29) |
| 3 | `_check_node` 检查真实节点，`_health_task` 限制自动重试次数 | [检查](../../../psrl/workers/env_service/router.py#L524)、[次数控制](../../../psrl/workers/env_service/router.py#L566) |
| 4 | `try_admit` 同时检查健康状态、待清理记录与预算 | [router.py](../../../psrl/workers/env_service/router.py#L706) |
| 5 | `node_diagnostics` 只读列出错误和资源归属 | [router.py](../../../psrl/workers/env_service/router.py#L581) |
| 6 | `recover_node` 复用在途探测或启动一次检查，不清空账本 | [router.py](../../../psrl/workers/env_service/router.py#L605) |
| 3 的物理清理 | `EnvNodeService.cleanup` 清理所属容器，再向 Router 确认 | [node.py](../../../psrl/workers/env_service/node.py#L182) |

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
| F1 → F2 | Trainer `_init_env_service` 读取配置，调用 `create_env_router` 创建或复用服务并核对契约 | [ray_trainer.py:549](../../../psrl/trainer/ppo/ray_trainer.py#L549)、[router.py:752](../../../psrl/workers/env_service/router.py#L752) |
| F2 | `EnvRouterCore.__init__` 建立共享账本 | [router.py:49](../../../psrl/workers/env_service/router.py#L49) |
| F3 题目展开 | 数据处理器按 `rollout_n` 展开同题的请求，并保留父题目身份 | [data_processor.py:567](../../../psrl/utils/dataset/data_processor.py#L567) |
| F3 分派 | `get_dispatch_plan` 按题目组选择 worker，`_supervise_worker_batch` 调用 worker | [manager.py:991](../../../psrl/workers/agent_loop/manager.py#L991)、[manager.py:931](../../../psrl/workers/agent_loop/manager.py#L931) |
| F4 | `add_agent_program` 逐请求调用 `generate_trajectory`，后者用 `asyncio.create_task` 创建主协程任务 | [worker.py:220](../../../psrl/workers/agent_loop/worker.py#L220)、[worker.py:262](../../../psrl/workers/agent_loop/worker.py#L262) |
| F4 → F5 | Science agent loop 启动 Harbor 求解，使用进程内缓存的 Env client | [agent_loop.py:182](../../../examples/sciaccel_rl/agent_loop.py#L182)、[agent_loop.py:95](../../../examples/sciaccel_rl/agent_loop.py#L95)、[agent_loop.py:391](../../../examples/sciaccel_rl/agent_loop.py#L391) |
| F5 查找服务 | `EnvServiceClient.from_config` 用 `ray.get_actor` 取得已有服务句柄 | [client.py:124](../../../psrl/workers/env_service/client.py#L124) |
| F5 申请与等待 | runner 创建容器前 `acquire`，client 调用 `try_admit`，账本返回获准、等待或拒绝 | [runner.py:366](../../../examples/sciaccel_rl/runner.py#L366)、[client.py:186](../../../psrl/workers/env_service/client.py#L186)、[ledger.py:113](../../../psrl/workers/env_service/ledger.py#L113) |
| F6 | runner 注入租约与执行配置后调用 Harbor `job.run`，容器提供者按配置创建环境 | [runner.py:412](../../../examples/sciaccel_rl/runner.py#L412)、[runner.py:551](../../../examples/sciaccel_rl/runner.py#L551)、[原生容器后端](../../../psrl/workers/env_worker/harbor.py#L19) |
| F7 | runner 的 `finally` 清理容器后归还租约，agent loop 收集本次会话采样数据 | [runner.py:610](../../../examples/sciaccel_rl/runner.py#L610)、[client.py:69](../../../psrl/workers/env_service/client.py#L69)、[agent_loop.py:228](../../../examples/sciaccel_rl/agent_loop.py#L228) |

这些代码如果有理解上的问题，可以随时追问。

**启动方具体是谁。** 它不是新增的一种专门进程，而是“发起创建或查找 Router 的调用方”的描述。在当前正常训练路径里，具体就是 Trainer 初始化时执行 `_init_env_service` 的代码。从使用者角度可以说“这次训练启动了 Router”，但一次训练包含 Trainer、worker、模型推理服务等多个角色，不能把它们当成一个进程。

**一个题目为什么会有多个协程。** 假设题目 Q1 是“修改程序使测试通过”，配置每题采样 2 次，就生成 Q1/1、Q1/2 两次独立尝试。每次尝试可以得到不同回复、执行不同命令、获得不同评分。数据处理器先把题目展开为两个请求，worker 再为每个请求创建一个主协程。因此更准确的单位是“每条 rollout 一个主协程”，不是“每个不同题目只有一个协程”。心跳和清理还可能创建辅助协程，实际协程总数不止主协程数量。

协程是在程序中可以暂停、稍后继续的一段执行流程。例如 Q1/1 发出模型请求后，要等模型回复，`await` 会让出当前线程上的执行机会，Q1/2 就有机会继续。**并发**表示这些流程在一段时间内交错推进，不表示每个流程都拥有独立 CPU 核或 GPU。worker 是执行这些流程的进程角色，一个 worker 可以运行许多协程。当前 Science 实现还把 Harbor 工作提交到同一 worker 的专用事件循环线程，事件循环负责安排哪些等待结束的协程继续运行。它不是为每条 rollout 新建一个操作系统进程。

**环境、资源需求和租约分别是什么。** 环境是一次求解所用的执行空间，包括代码、文件、依赖和容器等。`demand` 是资源需求描述，例如需要几个 CPU、多少内存或 GPU 显存。`lease` 中文称租约，是 Router 批准后登记的资源预约，包含租约身份、所属节点和预算。租约不是容器，也不是实际显存分配，实际创建和执行由容器后端完成。`session_id` 关联本次求解，`lease_id` 标识获准的预约，两者与 Router 服务名是不同层次的身份。

严格说，不是尚未创建的容器自己跑来申请资源。当前路径是 worker 中的 runner 代表本次环境任务先申请，拿到租约后才启动相关容器。worker 的 Env client 相当于发送申请的调用入口，同一进程会缓存并共享它。多个 client 仍可以指向同一个 Router，共享同一本账。

**两种调用传的东西不同。** 创建调用传入的是整项服务的规则，例如服务名、节点、资源单元和容量。环境申请传入的是本次求解的身份、需求和所在节点。一个资源单元 `unit` 是配置中受管理的一组节点资源，不等于一个题目或一条 rollout。训练初始化一般创建或查找一个配置指定的 Router，后面可以有成千上万次环境申请。

完整手算例子，以下为教学假设，不是本次训练实测：一次训练，这一批有 Q1、Q2 两题，每题采样 2 次，分派给两个 worker。资源配置只允许同时存在两个环境，其他条件都满足。

| 时刻 | 主协程和申请 | Router 的共享账本 | 还在等待的请求 |
|---|---|---|---|
| 初始化 | Trainer 创建名为 X 的 Router | 0 个活动租约，容量 2 | 无 |
| 展开与分派 | A 持有 Q1/1、Q1/2，B 持有 Q2/1、Q2/2，共 4 个主协程 | 仍为 0，创建协程不等于获准资源 | 尚未申请 |
| 前两个申请获准 | 假设 Q1/1、Q1/2 先申请并创建环境 | 租约 L1、L2，共 2/2 | Q2/1、Q2/2 |
| 一个环境结束 | Q1/1 的容器清理完成，归还 L1 | 只剩 L2，共 1/2 | 两个请求竞争余量，顺序不保证 |
| 后续申请获准 | 假设 Q2/1 下次询问时获得 L3 | L2、L3，共 2/2 | Q2/2 |
| 全部结束 | 4 条 rollout 各自完成清理和归还 | 回到 0/2 | 无 |

这里同时有 4 个主协程，却最多只有 2 个持有环境租约。等待资源的协程仍存在，但尚未为这次尝试创建受管理的环境容器。容量释放之后，等待 client 会稍后再次询问 Router。当前申请是轮询并带随机退避，不应从例子推断严格先到先服务。上游分派也可能提前延缓整组工作，本例特意只演示请求已经抵达 worker 后的准入。

**同一条 rollout 内的多轮交互。** Q1/1 可以先让模型读题，再执行测试，再把错误输出给模型，模型修改代码后再次测试，最后评分。这些是同一次求解内部的多轮，不是每执行一条 shell 命令就新建 Router 或生成一条独立 rollout。模型生成由模型推理服务完成，EnvRouter 管理环境资源准入，不负责模型文本请求的路由。项目里用于模型会话的 SessionRouter 是另一个组件，即使二者都叫 Router，也不能当成同一个服务。

如果使用 `episode` GPU 持有方式，本次求解期间保留 GPU 预约。如果使用 `command` 方式，环境会话仍存在，但 GPU 预算在实际命令执行阶段进一步申请和归还。两种情况都共享前面的初始化和环境会话流程，不能把“一条 rollout”“一次环境会话”和“一次 GPU 命令”当作同一层。容器集合还可能包含独立评分器或网络辅助组件，不能从一条租约推导恰好一个容器。

**再回到 R1 的两个启动方。** 前文的启动方 A/B 表示两次 `create_env_router` 调用，并不表示 Q1/1 与 Q2/1，也不表示每条 rollout 都创建 Router。这两次调用可能来自同一控制进程的重复初始化，也可能来自能找到已有 actor 的另一个控制进程。只有在同一 Ray 集群、同一命名空间且旧服务仍存在时，才会复用。同名而规则不同才是 R1 要阻止的情况。普通四条 rollout 向同一 Router 申请不同租约，是正常的资源共享，不会因此触发配置冲突。当前代码没有设置独立存活的 detached 生命周期，不能假定任意训练重启后旧服务都会留下。

### 1.2 跟着训练流程图读源码

| 步骤 | 图中编号 | 文件、行号、函数 | 输入与输出或状态变化 |
|---|---|---|---|
| 训练初始化 | F1 → F2 | [ray_trainer.py:549 `_init_env_service`](../../../psrl/trainer/ppo/ray_trainer.py#L549)、[router.py:752 `create_env_router`](../../../psrl/workers/env_service/router.py#L752) | 服务配置与节点规则传入，得到已经核对启动契约的远程句柄 |
| 题目展开 | F3 | [data_processor.py:567](../../../psrl/utils/dataset/data_processor.py#L567) | Q1、Q2 按采样次数展开，每条请求有身份，并保留父题目身份 |
| 派发题目组 | F3 → F4 | [manager.py:991 `get_dispatch_plan`](../../../psrl/workers/agent_loop/manager.py#L991)、[manager.py:931 `_supervise_worker_batch`](../../../psrl/workers/agent_loop/manager.py#L931) | 同题的采样组选择 worker，随后远程调用 worker，管理器异步监督其完成 |
| 并发求解 | F4 | [worker.py:220 `add_agent_program`](../../../psrl/workers/agent_loop/worker.py#L220)、[worker.py:262 `generate_trajectory`](../../../psrl/workers/agent_loop/worker.py#L262) | 每个请求创建主协程，多个主协程交错运行，不是单线程逐题跑完才继续 |
| 提交 Harbor 工作 | F4 → F5 | [agent_loop.py:391 `_gated`](../../../examples/sciaccel_rl/agent_loop.py#L391)、[agent_loop.py:417](../../../examples/sciaccel_rl/agent_loop.py#L417) | 通过进程缓存取得 client，将工作提交到专用 Harbor 事件循环线程并等待结果 |
| 正式申请资源 | F5 → F2 | [runner.py:372](../../../examples/sciaccel_rl/runner.py#L372)、[client.py:186 `acquire`](../../../psrl/workers/env_service/client.py#L186) | 身份与需求交给 Router，获准则启动心跳，等待则暂停后轮询，拒绝则报错 |
| 执行与取结果 | F6 | [runner.py:551](../../../examples/sciaccel_rl/runner.py#L551) | Harbor 求解、工具交互与评分，内部可能多轮和并发，不展开为一个同步函数 |
| 收尾与数据收集 | F7 | [runner.py:610](../../../examples/sciaccel_rl/runner.py#L610)、[agent_loop.py:228](../../../examples/sciaccel_rl/agent_loop.py#L228) | runner 的 finally 完成所属资源清理后归还，agent loop 收集会话采样数据。训练消费由下游组织，不是每次释放租约就立刻执行一次参数更新 |

### 1.3 运行四条 rollout 的教学模拟

从仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_rollout_env_flow.py
```

[脚本](../env-service-verification/scripts/model_rollout_env_flow.py)使用 Python 标准库 `asyncio`。`ModelRouter(capacity=2)` 模拟 F1/F2 的一个共享账本。两个题目各创建两个协程，模拟 F3/F4。`acquire` 模拟 F5，`asyncio.sleep` 只模拟 F6 的等待，`finally` 先打印 `ENV_CLEANED` 再释放，模拟 F7 的清理顺序。

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

创建入口见 [create_env_router](../../../psrl/workers/env_service/router.py#L752)，共享账本在 [EnvRouterCore.__init__](../../../psrl/workers/env_service/router.py#L49) 建立，任务身份与已有租约复用见 [EnvUnitLedger.try_admit](../../../psrl/workers/env_service/ledger.py#L113)。worker 按 Router 名称取得句柄，见 [EnvServiceClient.from_config](../../../psrl/workers/env_service/client.py#L124)。

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

[教学脚本](../env-service-verification/scripts/model_router_recovery.py)使用标准库，直接加载本次生产契约比较函数。恢复部分是教学模拟，用布尔值表示 Docker 可用和清理确认，不启动 Ray、Docker、GPU，也不真正释放显存。

预期先输出相同配置 accepted、变化配置 rejected。随后显示连续失败 3 次时保留 60000 的预约，服务恢复但未确认清理时仍不健康且预约不变，确认清理后健康且预约归零。数字只演示状态变化，真实系统仍要实际检查节点与资源。

生产接口回归使用节点故障替身，覆盖耗尽后恢复、继续失败、超时、并发合并、活动归属保留、退休租约未确认不准入、无归属死服务重新探测。它们与真正的多进程 Ray 或 GPU 故障注入有明确区别。

本次实际运行上述教学脚本，得到相同配置通过、变化配置拒绝、未清理保留预约、确认后释放预约的预期输出。扩展回归同时覆盖 Router client、GPU command 和 MPS 作业分支，共 82 项 CPU 测试通过，耗时 19.86 秒。修改涉及的六个 Python 文件 Ruff、格式与 prose 检查通过。全仓 pre-commit 在临时隔离副本运行，因原有文件的长行、格式和注释规则问题失败，未把自动修改带回主目录。

2026-11-11 增量实机验证使用 [router_live_recovery_check.py](../env-service-verification/scripts/router_live_recovery_check.py)。在实际执行节点激活项目环境后，从仓库根目录执行，`NODE_IP` 为经路由核对的该节点地址，`RESULT_FILE` 为本次独立结果文件：

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
| V1 | `run_harbor_episode` 先申请环境，再创建 Harbor Job | [runner.py:372](../../../examples/sciaccel_rl/runner.py#L372)、[runner.py:550](../../../examples/sciaccel_rl/runner.py#L550) |
| V2 | `_query_llm` 调用父类模型交互，`_execute_commands` 区分普通命令与 GPU 工具 | [agent.py:125](../../../examples/sciaccel_rl/agent.py#L125)、[agent.py:104](../../../examples/sciaccel_rl/agent.py#L104) |
| V3 | `dispatch_gpu_tool` 将 start、wait 等操作交给环境，`exec_gpu` 进入统一执行层 | [gpu_tools.py:32](../../../examples/sciaccel_rl/gpu_tools.py#L32)、[managed_execution.py:53](../../../psrl/workers/env_service/managed_execution.py#L53) |
| V4 → V5 | `prepare` 指定 separate 模式与提交位置，`job.run` 调用 Harbor 编排 | [prepare.py:37](../../../examples/flashinfer_bench/prepare.py#L37)、[prepare.py:72](../../../examples/flashinfer_bench/prepare.py#L72)、[runner.py:551](../../../examples/sciaccel_rl/runner.py#L551) |
| V5 GPU 授权 | 选用 `KernelCommandVerifier` 时，将评分 shell 调用包装为 `exec_gpu` | [verifier.py:34](../../../examples/kernel_bench/verifier.py#L34)、[verifier.py:43](../../../examples/kernel_bench/verifier.py#L43) |
| V5 测试和 V6 奖励 | `main` 调用官方 `Benchmark.run_all`，完整通过才写奖励 1 | [verify.py:70](../../../examples/flashinfer_bench/verify.py#L70)、[verify.py:79](../../../examples/flashinfer_bench/verify.py#L79) |
| V6 清理与训练数据 | runner 先清容器再释放租约，agent loop 将评分附到采样输出 | [runner.py:625](../../../examples/sciaccel_rl/runner.py#L625)、[runner.py:635](../../../examples/sciaccel_rl/runner.py#L635)、[agent_loop.py:262](../../../examples/sciaccel_rl/agent_loop.py#L262) |

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

运行 [CPU 教学脚本](../env-service-verification/scripts/rollout_command_verifier_demo.py)：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/rollout_command_verifier_demo.py
```

脚本模拟两轮“改代码加自测”，最终提交后再运行两项评分测试，打印 V1 至 V6，最终 reward=1。它用字典真假值替代真实候选，用打印替代模型、容器和 GPU，目的是分清顺序，不是验证真实吞吐。本次已实际运行并通过断言。真实 GPU 的分段可行性结果见[实验报告](../experiments/gpu-benchmarks/flashinfer-batching.md#保留评估器分段的-results)。

## 11. 当前方案是否必然快于固定基线

当前已经有修复后的正收益证据。固定 Env OFF 基线为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，不能用当前代码关闭开关替代。2026-11-11 对照保存的 JSON 复核后，应采用以下结果，而不是早期整任务占卡与修复前慢查询反例。

| 场景 | 固定 OFF | 修复后 ON | 解释 |
|---|---:|---:|---|
| Science 人工高压力，3 步、12 条接受轨迹，完整驱动时间 | 1954.11 秒，18 次 OOM | 1415.01 秒，0 次 OOM | 少 27.59%，主要减少失败补采 |
| 真实常驻模型整卡压力，7 项任务、14 请求 | 158.48 秒，1 次 OOM | 141.73 秒，0 次 OOM | 少 10.6%，无人工降低 ON 容量 |
| 修复后低负载同期配对，两对总时间 | 213.96 秒 | 217.15 秒 | 接近，尚未证明稳定提速 |

OOM 是显存不足导致分配失败。前两项说明 Env 在明确压力条件下已经超过固定旧版。它们来自各自冻结版本，不是今天 HEAD 重跑，也不支持所有负载都必然更快。具体配置、计时口径和机器证据统一见[有效实验结论](../experiments/env-design-and-results-brief.md)。

此前本节列出的早期整任务保留、ON 慢 64.3% 的结果已经移除。该执行模式的缺口后来由 command 借还与相关修复处理，不能继续当作当前方案的性能反例。同样，最早低负载开销已有同步准备与连接路径优化和后续配对，不能跨版本挑选旧慢值代表现状。

教学例子：同样成功计算需要 100 秒，旧版失败重试另花 40 秒，Env 管理花 10 秒，则 140 秒变成 110 秒。若本来不失败，管理成本仍需与实际省下的等待比较。这个算例解释收益机制，不替代上表的实测。

新增预测策略还要与当前已有的批量执行、进程复用、command 借还比较。旧版对照证明系统已有价值，新策略对照证明额外增量，两者应分别报告。本次分段探索属于后者的前置可行性，不能倒过来说当前 Env 尚无任何性能收益。
