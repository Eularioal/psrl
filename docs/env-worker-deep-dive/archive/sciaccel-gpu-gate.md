# Science GPU Gate 的早期设计与验证：静态映射、整卡锁和设备账本

**早期 science 任务会给多个容器重复挂载同一张 GPU，整卡锁让它们按设备排队。** 假设 4 个任务都使用固定的 `/dev/nvidia0`，它们可能同时分配显存；改为 3 张卡的互斥池后，前三个各拿一张，第 4 个等释放。Gate A 负责这种设备分配，Gate B 在模型回复出口根据 GPU 读数进行有上限的等待。

这份记录保存 2026-09 至 10 月的设计讨论、提交 `72ca50c` 的实现，以及 CPU 诊断输出。它解释“为什么先用整卡锁，后来又需要共享显存账本”。完整任务尝试称为 **episode**；当时的锁覆盖整个 episode，等待模型期间也持续持有设备名额。

从第 16 节看静态映射怎样改为互斥，第 17 节看训练与环境的设备账本，第 18 节看资源单元提案，第 19 节用完整算例解释跨进程文件锁。章节沿用原文的 16—19 编号，原跳转保留。

了解现行能力请读[当前设计](../knowledge/current-design.md)和[GPU 预约机制](../knowledge/command-gpu-reservations.md)。本篇的“已实现”“待实现”均限定到早期阶段；历史代码行号和输出保留，本地源码链接用于查找同名模块。固定版本可用 `git show 72ca50c:psrl/utils/concurrency/gpu_device_gate.py` 恢复。

## 目录

- [16. Science 任务如何取得 GPU](#16-science-任务如何取得-gpu)
  - [16.1 先看一个具体场景：mitgcm-biogeo 的一道 GPU 题，GPU 到底是怎么给的](#161-先看一个具体场景mitgcm-biogeo-的一道-gpu-题gpu-到底是怎么给的)
  - [16.1 补充：为什么需要跨进程锁](#161-补充为什么需要跨进程锁)
  - [16.2 Harbor 与 PSRL 各负责什么](#162-harbor-与-psrl-各负责什么)
  - [16.3 评估：science 任务要不要接入 EnvWorker 管理 GPU，怎么接](#163-评估science-任务要不要接入-envworker-管理-gpu怎么接)
  - [16.4 实测脚本：验证 GPU 调度确实会出现抢占](#164-实测脚本验证-gpu-调度确实会出现抢占)
  - [16.5 整卡锁与回复节流的实现记录](#165-整卡锁与回复节流的实现记录)
- [17. 训练、推理与环境的设备账本](#17-训练推理与环境的设备账本)
  - [17.1 账本①内部：训练设备和 vLLM 推理设备是怎么分的](#171-账本①内部训练设备和-vllm-推理设备是怎么分的)
  - [17.2 账本②：SciAccel episode 容器的设备是怎么分的](#172-账本②sciaccel-episode-容器的设备是怎么分的)
  - [17.3 两套设备账本的安全条件](#173-两套设备账本的安全条件)
- [18. 资源单元方案怎样解决共享与设备边界](#18-资源单元方案怎样解决共享与设备边界)
  - [18.1 从整卡独占改为容量准入](#181-从整卡独占改为容量准入)
  - [18.2 配置边界与训推设备保护](#182-配置边界与训推设备保护)
- [19. 文件锁如何保证一张卡只发给一个 episode](#19-文件锁如何保证一张卡只发给一个-episode)
  - [19.1 先看一个具体场景：两个 episode 进程抢两张卡，发生了什么](#191-先看一个具体场景两个-episode-进程抢两张卡发生了什么)
  - [19.2 背景知识：什么是文件锁，advisory 锁和 mandatory 锁有什么区别](#192-背景知识什么是文件锁advisory-锁和-mandatory-锁有什么区别)
  - [19.3 逐行解读抢锁循环的源码](#193-逐行解读抢锁循环的源码)
  - [19.4 分配粒度：整卡与资源单元](#194-分配粒度整卡与资源单元)
- [延伸阅读](#延伸阅读)

---

<a id="16-追加问题2026-09-27sciencesciaccel任务接入-envworker-管理-gpu-是否可行harbor-自己管理-docker能不能手动管理gpu-抢占怎么实测"></a>

## 16. Science 任务如何取得 GPU

### 16.1 先看一个具体场景：`mitgcm-biogeo` 的一道 GPU 题，GPU 到底是怎么给的

假设数据集里一行样本是 `mitgcm-biogeo/repair_easy` 的某道题，它的 `extra_info` 里带着 `{"gpus": 1, "task_path": "...", ...}`。这行样本被 `SciAccelAgentLoop.run()` 取出后：

```python
needs_gpu = int(extra_info.get("gpus", 0)) > 0   # 这里算出 True
```
—— [`agent_loop.py:131`](../../../examples/sciaccel_rl/agent_loop.py)

`needs_gpu=True` 只做了一件事：如果这次运行的 `config.harbor.gpu_compose_override`（一个配置里的文件路径字符串，默认是空串 `""`，需要显式配置才会生效，见 [`sciaccel_agent_config.yaml:21`](../../../examples/sciaccel_rl/config/sciaccel_agent_config.yaml)）非空，就把这个文件作为一层 Compose overlay 叠加进去：

```python
if needs_gpu and config.harbor.gpu_compose_override:
    extra_compose_paths.append(Path(config.harbor.gpu_compose_override))
```
—— [`runner.py:190-191`](../../../examples/sciaccel_rl/runner.py)

而这个 overlay 文件本身，内容是写死的：

```yaml
services:
  main:
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
      - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools
```
—— [`gpu-compose-override.yaml:1-7`](../../../examples/sciaccel_rl/config/gpu-compose-override.yaml)

**关键点**：这一行 `/dev/nvidia0:/dev/nvidia0` 是同一个文件、同一份字符串，不管当前是第 1 个 episode 还是第 20 个 episode 在跑，`extra_compose_paths.append(Path(config.harbor.gpu_compose_override))` 追加的都是**指向磁盘上同一份 yaml 的同一个路径**。所以只要这个 override 一旦启用（`gpu_compose_override` 配了非空路径），**每一个 `needs_gpu=True` 的 episode，不看当前有几个这样的 episode 同时在跑，也不看这台机器实际有几张卡，都会被分配到完全相同的 `/dev/nvidia0`**。

这跟 [`implementation-and-gaps.md` §15.3(1)](implementation-and-gaps.md) 已经记录的问题不是同一件事，需要区分清楚：

| | §15.3(1) 已记录的问题 | 本节新发现的问题 |
|---|---|---|
| 层面 | **数量**层面：`max_concurrent_episodes`/Harbor 自己的 `n_concurrent_trials` 只数"有几个 episode 在跑"，不看 cpu/mem/gpu 类型 | **设备身份**层面：就算数量控制得当（比如只放行 1 个 GPU episode），只要**同时**有 ≥2 个 GPU episode 被放行，它们拿到的会是**同一个物理设备字符串**，而不是像 Path A 那样各分到不同的卡 |
| 后果 | 8 卡机器同时放行 8 个 `gpus=1` 的任务，可能超过总显存 → OOM | 哪怕只放行 2 个 GPU 任务，也会**共享同一张卡**，而不是分散到 2 张不同的卡上 |
| 对照 Path A 的机制 | Path A 也只在 `WorkerSlot.has_gpu_capacity` 里做**计数**检查（[`coordinator.py:40-41`](../../../psrl/workers/env_worker/coordinator.py)，`used_gpu` 也只是个整数），本身不区分具体是哪张卡 | Path A 在计数检查通过之后，还有一层 Path B 完全没有的东西：`EnvWorker._allocate_gpu_indices` 维护一个真实的 `_busy_gpu_indices: set[int]`，从 `CUDA_VISIBLE_DEVICES` 解析出的 `available_gpu_indices` 里挑一个当前空闲的物理编号，选中的编号会被登记为占用，直到 `destroy_sandbox` 才释放（[`worker.py:169-187`](../../../psrl/workers/env_worker/worker.py)、释放见 [`worker.py:344-349`](../../../psrl/workers/env_worker/worker.py)） |

一句话总结这个场景：**Path A 是"先数个数、再分配具体哪张卡、卡与卡互斥"；Path B 现状是"不数具体哪张卡的个数（只数 episode 总数），也压根不存在'分配具体哪张卡'这一步——配置文件里的设备号是写死的常量"**。16.4 节的实测脚本会把这个场景变成一段可以亲手跑的代码。

<a id="161-补充2026-10-07这里说的多进程具体指什么从哪行代码能看出来"></a>

### 16.1 补充：为什么需要跨进程锁

16.1 的场景、以及 [`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md) 第 3 节开头"两个进程各开 8 个任务"的例子，都直接假设了"存在多个进程"，但没有交代这些进程具体从哪来、由谁创建、创建了几个——而这正是 16.5 节"Gate A 为什么必须换成 `fcntl` 文件锁、内存信号量管不住"这个结论成立的前提，补在这里。

运行 `SciAccelAgentLoop`（从而驱动 `run_harbor_episode`）的宿主类是 `PSRL_AgentLoopWorker`，类定义上带着一个 `@ray.remote` 装饰器：

```python
@ray.remote
class PSRL_AgentLoopWorker:
    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle: ray.actor.ActorHandle,
        rollout_gateway_url: str,
        session_router_url: str,
        worker_id: int = 0,
        worker_num: int = 1,
    ):
        ...
```
—— [`worker.py:45-67`](../../../psrl/workers/agent_loop/worker.py)

`@ray.remote` 是 Ray（PSRL 依赖的分布式计算框架）声明"Actor"的方式，含义是：这个类的每一个实例不是普通的本地 Python 对象，而是由 Ray 在后台另外拉起一个**独立的 OS 进程**（甚至是独立的 Python 解释器实例）来承载；调用它的方法时走的是跨进程 RPC，不是本地函数调用。构造参数里的 `worker_id`/`worker_num` 就是训练入口按配置创建**多个**这样的 Actor 实例时传进去的编号（`worker_id` 从 `0` 到 `worker_num - 1`）——这就是例子中“进程 A、进程 B”的来源。`worker_num` 决定这组 worker 的数量，实际节点由 Ray 资源与放置配置决定；部署在同一节点时，它们需要跨进程协调设备。

理解这一点后再回看 [`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md) 第 3 节和本文 16.5 节会更具体：第 3 节讲的那个 `asyncio.Semaphore`（也就是 `_episode_gate`）是在**每一个 `PSRL_AgentLoopWorker` Actor 自己的解释器里**单独创建的模块级全局变量；`worker_num` 个 Actor 就有 `worker_num` 份互不相通的计数器——这正是 16.5 节 Gate A 必须换成 `fcntl` 文件锁（锁状态挂在磁盘 inode 上，不属于任何一个进程的内存）而不能继续用内存信号量的直接原因，不是随便挑了个更复杂的方案。

<a id="162-harbor-自己管理-docker我们能不能手动管理-能而且现在已经在这么做"></a>

### 16.2 Harbor 与 PSRL 各负责什么

这个问题看起来是"能不能"，但更准确的问法应该是"现在到底是怎么管的"，答案里已经包含了"能"：

**Harbor 不是一个独立跑着的、把 Docker 藏起来的服务进程**。它是一个 Python 依赖包（`from harbor.job import Job`，见 [`runner.py:20`](../../../examples/sciaccel_rl/runner.py)），在 PSRL 自己的进程里被同步调用：`Job.create(config)` 再 `await job.run()`（调用处见 [`runner.py:214-263`](../../../examples/sciaccel_rl/runner.py) 附近的 `JobConfig`/`Job.create`）。这个调用背后，Harbor 把 `task.toml` 里声明的沙箱翻译成一份 `docker-compose` 配置，用 `docker compose --project-name <session_id 净化后的字符串>` 的方式起停容器——这一点不是猜测，是 [`docker_utils.py:153-172`](../../../psrl/utils/common/docker_utils.py) 里 `sanitize_compose_project_name` 函数注释明确写的：

> "Mirrors `harbor.environments.docker.docker._sanitize_docker_compose_project_name`. Harbor passes `--project-name <sanitized session_id>`, so Compose stamps that value onto every container of the episode as `com.docker.compose.project`."

也就是说，Harbor 起的每一个容器，在宿主机的 `docker ps` 里看，**和你自己手写 `docker compose up` 起的容器没有任何区别**——都会带标准的 `com.docker.compose.project` 标签，都能被 `docker inspect`/`docker exec`/`docker stop`/`docker rm` 直接操作。

**更进一步的证据是：PSRL 现在已经在手动管理它了，而且管理点还不止一处**，全部在 [`psrl/utils/common/docker_utils.py`](../../../psrl/utils/common/docker_utils.py)：

| 手动管理的动作 | 怎么定位到 Harbor 起的容器 | 源码位置 |
|---|---|---|
| 按 Compose 项目名强制删容器 | 用 `com.docker.compose.project` 标签（值 = 净化后的 `session_id`），绕开 Harbor 自己的 `job.cleanup()`，直接 `docker rm -f` | [`docker_utils.py:175-200`](../../../psrl/utils/common/docker_utils.py)（`force_remove_compose_project`），调用处见 [`implementation-and-gaps.md` §13.7](implementation-and-gaps.md) 讨论过的取消/超时兜底路径 |
| 删除 Compose 构建出的镜像 | Compose 给镜像命名的规则是 `<project>-<service>`，用这个前缀过滤 | [`docker_utils.py:203-252`](../../../psrl/utils/common/docker_utils.py)（`force_remove_compose_images`） |
| 按 actor 级自定义标签清理（不是 Harbor 自带的标签，是 PSRL 自己塞进去的） | `psrl.actor_id=<actor_id>`，这个标签是 PSRL 通过**动态生成一份新的 compose overlay 文件**、追加到 `extra_docker_compose` 里塞进去的，Harbor 本身根本不知道这个标签的存在 | 标签生成：[`runner.py:193-207`](../../../examples/sciaccel_rl/runner.py)（`main_override["labels"] = [f"psrl.actor_id={actor_id}"]`，写成临时 yaml 文件后追加到 `extra_compose_paths`）；用标签清理：[`docker_utils.py:105-150`](../../../psrl/utils/common/docker_utils.py)（`force_remove_containers_by_label`） |
| 进程退出后兜底回收（哪怕 Python 进程被杀，容器也要清） | 同样按 `psrl.actor_id` 标签，用一个独立于 Harbor/Python 的 `bash` 常驻进程轮询父进程是否存活 | [`docker_utils.py:332-427`](../../../psrl/utils/common/docker_utils.py)（`spawn_actor_reaper`） |

这一条"用动态生成的 compose overlay 塞自定义标签"（`runner.py:193-207`）是本节后面 16.3 小节要复用的关键机制：**它证明"往 Harbor 的 compose 配置里插入一份运行时才决定内容的 overlay 文件"这件事本来就是这份代码已经在做、已经跑通的模式，不是要新发明的能力**——现在只是拿它塞了 label，完全可以同样拿它塞"这次分配到的真实设备号"，取代 16.1 节里那份写死的静态文件。

**结论**：可以手动管理，而且不需要碰 Harbor 的任何内部代码/私有 API——所有手动管理动作都发生在 Harbor 进程之外，靠的是"Harbor 起的容器在宿主机上就是普通 Compose 容器"这个事实，加上 Docker 标签做筛选。唯一要小心的是 `_sanitize_docker_compose_project_name` 是 Harbor 的私有函数（`docker_utils.py` 的注释也承认"copied rather than imported because it is private to Harbor"），如果未来 Harbor 升级改了净化规则，这份拷贝需要跟着同步，这是手动管理相对于官方 API 的唯一代价。

### 16.3 评估：science 任务要不要接入 EnvWorker 管理 GPU，怎么接

结合 16.1（问题：设备身份没有互斥）和 16.2（能力：可以在 Harbor 外层任意插入动态 overlay），给出的评估是：**可行，且不需要动 Harbor 内部，改动可以完全限制在 `runner.py`/`agent_loop.py` 这两个文件内**。具体推荐路径：

```
现状（16.1 的场景）：
  needs_gpu=True
      │
      ▼
  extra_compose_paths.append(静态文件 gpu-compose-override.yaml)   ← 问题在这一行
      │
      ▼
  Job.create(...).run()  → 不管几个 episode 同时跑，都是 /dev/nvidia0

建议改法：
  needs_gpu=True
      │
      ▼
  【新增】向一个节点级 GPU 账本申请一个真实空闲的物理设备号
      （复用 worker.py 已有的 _allocate_gpu_indices 语义，
       或复用更轻量的 psrl/utils/concurrency/slot.py 的 SlotManager）
      │  没有空闲设备 → 排队等待，而不是硬塞进同一张卡
      ▼
  拿到 device_index 后，用 runner.py:199-207 同一套"动态生成 overlay yaml"机制，
  写一份内容为 `- /dev/nvidia{device_index}:/dev/nvidia0` 的临时文件，
  追加到 extra_compose_paths（取代静态的 gpu-compose-override.yaml）
      │
      ▼
  Job.create(...).run()  → 这次真正拿到的是当前空闲的那张卡
      │
      ▼
  finally: 释放这个 device_index（挂在 agent_loop.py:366-367 已有的
           `finally: gate.release()` 旁边，同一个生命周期钩子，
           不引入新的失败面）
```

两种"节点级账本"的选型对比：

| 方案 | 复用什么 | 优点 | 代价 |
|---|---|---|---|
| 接入 `EnvWorkerCoordinator`（Path A 的账本） | [`coordinator.py:123`](../../../psrl/workers/env_worker/coordinator.py) 全局路由 + [`worker.py:171-187`](../../../psrl/workers/env_worker/worker.py) 的 `_allocate_gpu_indices` | 一步到位统一了 Path A/Path B（正是 [`implementation-and-gaps.md` §15.3(5)](implementation-and-gaps.md) 里说的 Roadmap Phase 0 目标），CPU/GPU/跨节点路由都有了 | 改动面最大：SciAccel 的 Harbor 调用需要挪到一个 `EnvWorker` actor 背后才能拿到 `self.available_gpu_indices`/`_busy_gpu_indices` 这套状态；工作量对应原文 Phase 0 四步（§15.3(5) 已列出） |
| 只加一个轻量的节点级设备账本（复用 [`psrl/utils/concurrency/slot.py`](../../../psrl/utils/concurrency/slot.py) 的 `SlotManager`，`fcntl` 文件锁天然跨进程） | `SlotManager.acquire(max_slots=本机 GPU 数, output_dir=...)` 返回的 `slot_index` 直接当成物理设备号用 | 改动面最小：只在 `runner.py`/`agent_loop.py` 内加十几行，不用挪 Harbor 调用的宿主进程，`SlotManager` 本来就是跨进程安全的（`fcntl.flock`），天然解决"多个 worker 进程各自独立信号量、互相看不见"的问题（对应 §15.3(5) 提到的"节点级原语，只是 SciAccel 没用"） | 只解决了本节两个具体问题（设备互斥 + 跨进程可见），不统一 Path A/Path B 的整体架构，CPU 账本、跨节点路由仍然是两套 |

**给的建议**：如果目标只是"尽快让 science 任务的 GPU 分配不再互相踩踏"，选轻量方案（`SlotManager`），改动小、风险低，一两天内可以落地并用 16.4 的脚本验证；如果目标是"从架构上让 Path A/Path B 用同一套账本"（对应 §15.3(5) 的长期目标），需要按 Phase 0 四步走完整迁移，工作量和风险都更大，但轻量方案不会与之冲突——轻量方案本质是 Phase 0 里"把进程级 Semaphore 换成节点级原语"这一步的 GPU 版本，即使先落地轻量方案，之后往 `EnvWorkerCoordinator` 迁移时这部分逻辑基本可以直接搬过去，不是要推倒重来。

### 16.4 实测脚本：验证 GPU 调度确实会出现抢占

配套脚本：`docs/psrl/scripts/sciaccel_gpu_contention_demo.py`（本地资料，未随仓库发布）。运行方式：

```bash
cd /Users/liuwei/CodeBuddy/20260811161322
python3 docs/psrl/scripts/sciaccel_gpu_contention_demo.py --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1
```

脚本做两件事，对应 16.1 场景里的两层问题：

1. **静态配置检查**（不需要 docker/GPU，纯读文件）：用正则从真实的 `gpu-compose-override.yaml` 里抽出 `services.main.devices` 列表，直接打印出来，证明它就是写死的常量。
2. **动态并发检查**（本节的重点，也是用户要的"验证会出现抢占"）：把 `agent_loop.py:60-83` 的 `asyncio.Semaphore(limit)` 准入门逻辑原样复刻（不是导入，因为直接 `import` 那个模块需要 ray/vllm 等本沙箱没装的重依赖；复刻的是完全相同的三行门禁代码），提交若干个"episode"（一半带 `needs_gpu`，模拟真实数据集里 GPU/非 GPU 任务混合的情况），被信号量放行后，GPU episode 一律记录为占用同一个 `/dev/nvidia0`（真实反映 `runner.py:190-191` 的行为），用真实的 `time.monotonic()` 时间戳记录每个 episode 的起止区间，最后做一次扫描线重叠分析。

本机（这次会话所在的沙箱）实测输出（无 docker、无 GPU，走的是"进程内 asyncio 模拟"分支，时间戳和并发行为是真实的，只是没有真的起 Docker 容器）：

```
STEP 1 -- static config check (no docker/GPU needed, deterministic)
  services.main.devices (host side) = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']
STEP 2 -- dynamic concurrency check (in-process asyncio (no docker/GPU on this machine))
  max_concurrent_episodes=4, num_episodes=6, fixed_devices=False
Timeline (t = seconds since first BEGIN):
  t=  0.00  BEGIN  episode=0   device=/dev/nvidia0
  t=  0.00  BEGIN  episode=2   device=/dev/nvidia0
  t=  0.00  BEGIN  episode=4   device=/dev/nvidia0
  t=  1.00  END    episode=0   device=/dev/nvidia0
  t=  1.00  END    episode=2   device=/dev/nvidia0
  t=  1.00  END    episode=4   device=/dev/nvidia0
Overlap analysis per device:
  device='/dev/nvidia0': peak simultaneous holders = 3  -> CONTENTION -- multiple episodes held this device at once
RESULT: reproduced real preemption/oversubscription. Up to 3 episodes were simultaneously
        assigned the same physical device string at the same time.
```

3 个 episode（0、2、4，因为脚本按"偶数编号需要 GPU"交替生成）在同一时刻都拿到了 `/dev/nvidia0`——这就是用户问的"多个 worker 同时争抢一份资源"的真实复现：只要 `max_concurrent_episodes >= 2`（真实配置默认是 16，见 [`sciaccel_agent_config.yaml:19`](../../../examples/sciaccel_rl/config/sciaccel_agent_config.yaml)）且一批里有 ≥2 个 GPU 任务，就会触发。

加 `--fixed-devices --num-gpus 3` 参数会切到 16.3 节建议的"修复后"逻辑（真实按物理编号互斥分配，逻辑复刻自 `worker.py:171-187`），同样 6 个 episode，输出变成每张卡 `peak simultaneous holders = 1`，零重叠——这组对照证明"抢占"不是无解的固有特性，而是 Path B 现状缺一层设备互斥account 导致的，加上之后就没有了。

**当次验证范围：** 脚本运行的是进程内 asyncio 模型。并发调度、时间戳和门禁逻辑实际执行，设备用字符串表示。`--use-docker` 分支留给具备 Docker 与 GPU 的节点，当次没有执行。

<a id="165-修复已部署gate-a设备互斥-gate-b回复节流代码位置与验证脚本"></a>

### 16.5 整卡锁与回复节流的实现记录

16.3 节比较的两个方案里，选的是"轻量方案"（复用 `SlotManager`），理由和 16.3 节给的建议一致：改动面小、风险低、不影响 Path A 的架构。同时按 [`implementation-and-gaps.md` §15.3(2)](implementation-and-gaps.md) 的思路把 Gate B 也接了进去。下面是实际改了哪些文件、怎么验证。

> 这套改动对应 psrl 仓库的提交 `72ca50c`（2026-09-29）。想看**逐段代码讲解**、这个提交**没有做**的部分（它不含 `EnvRouter`，也没有改动 `EnvWorker`），以及从代码结构推断出的几处注意点（例如 Gate B 的 `nvidia-smi` 跑在 SessionRouter 所在节点），见 [提交 72ca50c 代码讲解](commit-72ca50c-gate-ab-walkthrough.md)。

**Gate A：跨进程物理设备互斥分配**

| 内容 | 位置 |
|---|---|
| 核心逻辑：`GpuDeviceGate`（`acquire`/`release`）、`discover_gpu_devices`、`build_gpu_override` | [`psrl/psrl/utils/concurrency/gpu_device_gate.py`](../../../psrl/utils/concurrency/gpu_device_gate.py) |
| 接入点：`run_harbor_episode` 里原来"永远 append 同一份静态 override 文件"的位置，改成先 `await gate.acquire()` 拿到物理设备号，再用 `build_gpu_override` 现场生成一份指向该设备的临时 override | [`runner.py`](../../../examples/sciaccel_rl/runner.py) 的 `run_harbor_episode` 函数（搜索 `GpuDeviceGate` 即可定位） |
| 释放：episode 的 compose 项目被 `force_remove_compose_project` 确认删除之后才 `release`，避免"设备号已经能被下一个 episode 拿到，但上一个容器还没退干净" | 同上，`finally` 块末尾 |
| 没有发现任何 `/dev/nvidiaN` 时的降级行为：回退成旧的静态映射（不是直接报错），并打印 WARNING——不能悄悄假装提供了互斥保证 | 同上 |

`SlotManager.acquire(max_slots, jobs_dir)` 返回的是设备列表中的位置。取得 `slot_index` 后，用 `devices[slot_index]` 找到真实设备路径；例如列表为 `[nvidia2, nvidia7]`，位置 1 对应 `nvidia7`。锁文件由同节点、使用同一锁目录的进程共同遵守，因此独立 Python 进程也能协调。

**Gate B：SessionRouter 出口的 GPU 空闲节流**

> 下表信息密度高，看不懂可先读 [Gate B 新手版讲解](../knowledge/gate-b-beginner-guide.md)：用"后厨传菜窗口"的故事 + 全局图把这五行逐行拆开，附可运行的判定逻辑演示脚本。

| 内容 | 位置 |
|---|---|
| `SessionState` 新增 `gpu_device` 字段，默认 `None`（未绑定的 session 完全不受影响） | [`session_router.py`](../../../psrl/workers/gen/session_router.py)，`SessionState` 类 |
| 新控制端点 `POST /control/gpu_bind`：Gate A 拿到/释放设备时各调用一次，绑定或清空某个 session 的 `gpu_device` | 同上，`control_gpu_bind` 方法 |
| 真实 GPU 空闲判断：`_read_gpu_utilization` 用 `nvidia-smi --query-gpu=index,utilization.gpu,memory.used` 读整卡计数器，带 1 秒缓存避免多个 session 同一瞬间各发一次子进程；`nvidia-smi` 不存在或超时时"失败开放"（当成空闲），不会把没有 GPU 的开发机卡死 | 同上，`_read_gpu_utilization` / `_wait_for_gpu_idle` |
| 出口 hook：`session_chat_completions` 返回响应前，若该 session 绑定了 `gpu_device`，先 `await _wait_for_gpu_idle(...)`，最长等待 `PSRL_GATE_B_MAX_WAIT_SEC`（默认 600 秒，与本节前面"初步实现路径"给的默认值一致） | 同上，`session_chat_completions` 末尾 |
| `runner.py` 侧的绑定/解绑调用（best-effort，失败只记日志，不影响 episode 本身） | [`runner.py`](../../../examples/sciaccel_rl/runner.py) 的 `_bind_gate_b` 函数 |

**整卡读数与任务归属的区别。** 早期实现用整卡 `nvidia-smi --query-gpu`，而提案考虑按进程查询。整卡上有其他任务时，读数会把它们一起算入，从而增加等待；它也无法证明活动来自本任务。再加上轮询间隔、超时放行和遥测失败放行，Gate B 只能提供软节流。严格评分需要可信进程归属，现行机制见[GPU 作业与任务遥测](../knowledge/gpu-jobs-and-task-telemetry.md)。

**对照验证脚本（真实导入部署好的模块，不是复刻）**

脚本放在 `psrl/` 代码仓库里（不是 `docs/`），这样 `git push` 之后服务器上 `git pull` 就能直接拿到，不需要额外手动同步文件。和 16.4 节那个纯模拟脚本不同，下面这对脚本里的"after"脚本是**直接 `import` 上面部署的 `GpuDeviceGate`**，不是照抄逻辑：

```bash
cd /Users/liuwei/CodeBuddy/20260811161322/psrl

# BEFORE：复现问题（没有 Gate A，读真实的静态 override 文件）
python3 examples/sciaccel_rl/diagnostics/gate_a_before.py \
    --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1

# AFTER：单进程内验证（真实 import GpuDeviceGate）
python3 examples/sciaccel_rl/diagnostics/gate_a_after.py \
    --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1 --num-gpus 3

# AFTER：跨进程验证（真实起多个 OS 子进程，证明 fcntl 锁跨进程也生效——
# 这才是对应生产环境"多个 AgentLoopWorker 独立进程"的真实场景）
python3 examples/sciaccel_rl/diagnostics/gate_a_after.py --cross-process \
    --num-workers 4 --num-episodes 3 --hold-seconds 0.3 --num-gpus 3

# 在真正有 GPU 的服务器上实验：去掉 --num-gpus（或不设 PSRL_SCIACCEL_GPU_DEVICES
# 环境变量），GpuDeviceGate 会自动 glob 真实的 /dev/nvidia[0-9]*；before 脚本
# 也可以加 --use-docker 改用真正的 docker run --device 容器。
```

本机（本次会话所在沙箱，无 docker、无真实 GPU）实测：

```
# before: peak simultaneous holders = 4  -> CONTENTION（4 个 episode 同时占用 /dev/nvidia0），exit code 1
# after（单进程）: 3 个设备各自 peak = 1 -> ok, exclusive，exit code 0
# after（跨进程，4 个真实 OS 子进程）: 3 个设备各自 peak = 1 -> ok, exclusive，exit code 0
```

这组 CPU 诊断实际执行了文件锁、进程和协程调度，使用 `PSRL_SCIACCEL_GPU_DEVICES` 提供设备字符串。三张“卡”各自的持有峰值为 1，说明锁遵守了互斥关系；实际显存、CUDA 和容器分配仍需单独的实机证据。

<a id="17-追加问题2026-10-08训练设备vllm-推理设备sciaccel-episode-容器设备分别是谁分的彼此有没有关系"></a>

## 17. 训练、推理与环境的设备账本

这三类设备分配，答案是**两套完全独立、互不通信的"账本"**：训练设备和 vLLM 推理设备同属一套（Ray 自己的资源账本），SciAccel episode 容器设备是另一套（本文 Gate A 的节点级账本）。先看全景图，再逐层拆开看代码证据。

```
┌───────────────────── 账本①：Ray 自己的 GPU 资源账本（全局唯一，跨节点生效）─────────────────────┐
│                                                                                              │
│  训练启动时刻（main_ppo.py: TaskRunner.init_resource_pool_mgr，只执行一次）：                      │
│  按 config 里的数字，一次性划出几块互不重叠的 Ray 资源池：                                          │
│                                                                                              │
│    train_pool          ← Actor/Critic/RefPolicy/DummyPolicy 共用（训练设备）                     │
│    rollout_pool_0..N   ← 每个 vLLM 实例一份（推理设备）                                           │
│    validate_pool_i     ← 开启"训练/验证同卡复用"时，和 train_pool 按 0.9/0.1 的资源比例           │
│                           共享同一批物理卡（仅有的一处"同一张卡两个角色都登记"的情况）              │
│    env_worker（可选）   ← 如果某个 Environment 配置了 gpu_slots_per_worker>0，也走这同一套         │
│                           Ray num_gpus 请求，一样被纳入这个账本                                   │
│                                                                                              │
│  Ray 的调度器保证：这几块资源池互不重叠——同一张物理卡的"已被预订"状态，                            │
│  对 Ray 调度的所有 Actor 全局可见，不会被发给两个池子                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                          ✕ 没有任何代码把这两套账本打通
┌───────────────────── 账本②：Gate A 的节点级账本（只在单台物理机内生效，Ray 完全不知情）────────────┐
│                                                                                              │
│  SciAccel 的 `PSRL_AgentLoopWorker`（跑 Harbor/Docker 的那个 Ray Actor）                         │
│  声明时是 `@ray.remote`，不带任何 `num_gpus` 参数 → Ray 认为它是纯 CPU Actor，                     │
│  调度时完全不会为它预留/隔离任何物理 GPU                                                          │
│                                                                                              │
│  每次 episode 需要 GPU：`GpuDeviceGate.acquire()` 直接 glob 这台宿主机上                           │
│  所有 `/dev/nvidia[0-9]*` 设备文件（不看 Ray 分给了谁），                                          │
│  用 `fcntl` 文件锁在这台机器的本地临时目录里登记"这个索引现在被哪个 episode 占着"                    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 17.1 账本①内部：训练设备和 vLLM 推理设备是怎么分的

两者是`ResourcePoolManager`管理下的两个不同资源池，划分逻辑都在 `TaskRunner.init_resource_pool_mgr` 这一个函数里，训练开始前执行一次：

```python
train_pool_id = "train_pool"
resource_pool_spec = {
    train_pool_id: [deployment_config.train_ngpus_per_node] * deployment_config.train_nnodes,
}
...
rollout_pool_id_list = [f"rollout_pool_{i}" for i in range(deployment_config.n_rollout_instances)]
for i in range(deployment_config.n_rollout_instances):
    resource_pool_spec[rollout_pool_id_list[i]] = [
        deployment_config.rollout_ngpus_per_node_per_instance
    ] * deployment_config.rollout_nnodes_per_instance
```
—— [`main_ppo.py:234-238,247-249,293-297`](../../../psrl/trainer/main_ppo.py)

`Actor`/`Critic`/`RefPolicy`/`DummyPolicy` 这几个训练角色全部登记进同一个 `"train_pool"`（[`main_ppo.py:210,218,228`](../../../psrl/trainer/main_ppo.py)），`Rollout`（vLLM）角色登记进各自独立的 `rollout_pool_i`（[`main_ppo.py:310`](../../../psrl/trainer/main_ppo.py)）。这些池子名字和 GPU 数量的字典传给 `ResourcePoolManager.create_resource_pool()`（[`utils.py:118-135`](../../../psrl/trainer/ppo/utils.py)），每个池子各自建一个 `RayResourcePool`（底层是 Ray 的 placement group），在 `ray_trainer.py` 的 `init_workers()` 里**只调用一次**（[`ray_trainer.py:1458`](../../../psrl/trainer/ppo/ray_trainer.py)），之后整个训练过程不再重新划分。

拿项目里**真实的默认配置**举例（不是我编的教学数字）：

```yaml
# psrl/psrl/trainer/config/psrl/deployment.yaml（默认值）
n_rollout_instances: 1
rollout_nnodes_per_instance: 1
rollout_ngpus_per_node_per_instance: 1
train_nnodes: 1
train_ngpus_per_node: 1
```
—— [`deployment.yaml:66,72,75,84,87`](../../../psrl/trainer/config/psrl/deployment.yaml)

代入上面的公式：`train_pool` = 1 节点 × 1 卡 = 1 张卡；`rollout_pool_0` = 1 节点 × 1 卡 = 1 张卡。这就是**单机最小可跑配置至少要 2 张物理 GPU**的原因——`train_pool` 和 `rollout_pool_0` 是 Ray 账本里两个不重叠的池子，哪怕都在同一台机器上，Ray 也会把它们当成两份独立预订的资源，即使只有 1 张卡可用，`ResourcePoolManager._check_resource_available`（[`utils.py:147-168`](../../../psrl/trainer/ppo/utils.py)）也会在训练一开始就报错拒绝启动，而不是让两个池子挤在同一张卡上"悄悄共享"。

唯一的例外、也是**账本①内部唯一一处"同一张卡登记给两个角色"**的情况：`colocate_validate_and_train=True` 时，`validate_pool_i` 和 `train_pool` 会用 Ray 资源包的**小数资源份额**（`resource_num_per_bundle`）分别登记 `0.9` 和 `0.1`（[`main_ppo.py:235,241,307`](../../../psrl/trainer/main_ppo.py)），让 Ray 同意把同一张物理卡同时"发"给训练和验证两个角色——但这不是真的同时用：谁在用由 `is_rollout_mode_in_actor` 这个布尔开关和 `switch_to_rollout_mode()`/`switch_to_trainer_mode()`（[`ray_trainer.py:2155-2312`](../../../psrl/trainer/ppo/ray_trainer.py)）手动协调，训练在用的时候验证侧处于"meta-sleep"（只保留元数据描述符，不占真实显存），反过来也一样，是**显式握手的轮流使用**，不是并发抢占。

### 17.2 账本②：SciAccel episode 容器的设备是怎么分的

这套完全不经过 `ResourcePoolManager`，走的是本文 §16.5 已经讲过的 `GpuDeviceGate`：

```python
devices = discover_gpu_devices()              # glob("/dev/nvidia[0-9]*")，不看 CUDA_VISIBLE_DEVICES
slot = await SlotManager.acquire(len(devices), self.jobs_dir, prefix=self.slot_prefix)  # fcntl 文件锁
device = devices[slot_index]
```
—— [`gpu_device_gate.py:69-90,117-147`](../../../psrl/utils/concurrency/gpu_device_gate.py)，接入点 [`runner.py:240-245,403-404`](../../../examples/sciaccel_rl/runner.py)

关键一行区别：`EnvWorker`（账本①的一部分，见下 17.3）是从 `os.getenv("CUDA_VISIBLE_DEVICES")` 派生可用设备——这个环境变量是 **Ray 在调度这个 Actor 时根据账本①写入的**，天然被限制在 Ray 分给它的那部分卡里；而 `GpuDeviceGate.discover_gpu_devices()` 是直接 `glob` 宿主机文件系统上的 `/dev/nvidia[0-9]*`，**不读 `CUDA_VISIBLE_DEVICES`，也不知道 Ray 账本的存在**。它解决的问题范围很明确：保证"这台机器上同时跑的多个 SciAccel episode 不会抢到同一个物理设备号"（本文 §16.4/§16.5 的实测脚本验证的就是这一点），但没有、也没打算去检查"这张卡是不是 Ray 已经分给了同一台机器上的某个 `train_pool`/`rollout_pool` Actor"。

<a id="173-对比envworker-路径为什么天然安全sciaccelgate-a路径为什么需要额外小心"></a>

### 17.3 两套设备账本的安全条件

早期两条路径的区别在于：一条从 Ray 分给 actor 的设备中选择，另一条自行扫描宿主机设备。

| 环节 | 旧 EnvWorker | science 的 `GpuDeviceGate` |
|---|---|---|
| 取得设备范围 | actor 声明 GPU 资源需求，Ray 设置 `CUDA_VISIBLE_DEVICES` | 扫描 `/dev/nvidia[0-9]*`，或使用显式覆盖列表 |
| 同路径内部协调 | Ray 的资源账本，加 actor 内 `_busy_gpu_indices` 集合 | 同节点、同锁目录中的 `fcntl` 文件锁 |
| 与训练、推理协调的条件 | 使用同一套 Ray 资源管理，正确声明设备需求，并遵守设备可见性约定 | 显式配置安全设备池，避免把已分给训练、推理的设备再次纳入 |

教学例子：Ray 已把 GPU 0 分给推理 actor。EnvWorker 若按同一套整卡资源声明取得 GPU 1，随后只把 GPU 1 挂给容器，两条路径可以分开使用。旧 Gate 扫描到 `[nvidia0, nvidia1]` 时，却可能把 GPU 0 也当成候选；它的文件锁只能协调其他遵守同一锁协议的 Gate 使用者。

Ray 的资源声明属于调度约定，设备可见性也需要正确传递到容器。能否安全共用节点，还取决于实际设备集合、分配粒度和权限。原文把 EnvWorker 描述为“天然安全、结构上不可能碰到训练卡”，保证范围过大；准确的结论是：统一账本减少了重复分配来源，正确的设备映射与部署约束仍然必要。后续的具体保护见[当前节点策略](../knowledge/router-configuration-and-recovery.md)。

<a id="18-追加问题2026-10-08env-service-design-这份方案能修复-173-这两个问题吗"></a>

## 18. 资源单元方案怎样解决共享与设备边界

这一节分析 2026-09-21 的资源服务提案。提案用资源单元和显存预算支持一卡多容器，同时要求配置安全的设备边界。下面先解释容量准入，再解释它怎样与训练、推理的设备分配配合；实施后的早期记录见[2026-10-12 实现](env-service-implementation.md)。

<a id="181-问题1能不能让多个低显存的-episode-共享一张物理卡能这正是这份设计的核心改动"></a>

### 18.1 从整卡独占改为容量准入

现状（本文 §16.5 已部署的 `GpuDeviceGate`）把"一张物理卡"当成不可再分的互斥单位：`fcntl` 文件锁一次只把一个设备索引登记给一个 episode，哪怕这个 episode 只用 2 GB 显存，这张卡剩下的几十 GB 在它跑完之前也不能分给别的 episode——这是本文 §17.3 表格里"只认设备索引、不认显存用量"的直接后果。

`env-service-design.md` 先做了一个实测来确认"共享"这件事本身是否可行：两个容器各自在同一张卡（`/dev/nvidia0`）上分配 8 GB 显存、跑一个持续占用 GPU 的 kernel，结果两个都正常跑完、退出码都是 0，`nvidia-smi` 读到的是 `utilization.gpu=100`、`memory.used=17030 MiB`（即 8508+8508 两个进程显存的简单相加）：

```
utilization.gpu = 100      memory.used = 17030 MiB      (8508 + 8508)
compute-apps:  pid 3762699   8508 MiB
               pid 3763387   8508 MiB
```
—— `2026-09-21-env-service-design.md:361-369`（本地资料，未随仓库发布）

这个实测揭示了一个关键的不对称性，也是新设计敢把"分配单位"从"一整张卡"改成"一个可以塞多个容器的 unit"的依据：

| 资源 | 共享时的行为 | 能不能靠调度化解 |
|---|---|---|
| **SM / 算力** | SM 是执行 GPU 指令的计算单元；普通独立 CUDA 上下文通常通过时间片轮换共享计算机会 | 共享会带来竞争，需要测量各任务时延和吞吐 |
| **显存** | 多进程分配通常分别占用设备容量，放在同卡时累加 | 在准入时为实测占用和未兑现预约留足容量 |

—— `2026-09-21-env-service-design.md:377-380`（本地资料，未随仓库发布）

基于这个结论，新设计把"可分配对象"从设备改成了 **unit**——一个可以同时装下多个容器的资源包：

```python
@dataclass(frozen=True)
class EnvUnit:
    unit_id: str                      # stable, unique cluster wide
    node_ip: str
    cpu_quota: float                  # shared by this unit's containers
    memory_mb: int                    # host memory, shared
    gpu_indices: tuple[int, ...]      # whole devices owned outright, () means CPU only
    max_containers: int               # configured hyperparameter, not derived
    gpu_memory_mb: int                # device memory budget across gpu_indices
```
—— `2026-09-21-env-service-design.md:429-439`（本地资料，未随仓库发布）

配置示例里，`n1-gpu01` 这个 unit 占 2 张卡（`gpu_indices: [0, 1]`），显存预算 `gpu_memory_mb: 195742`，`max_containers: 3`——意味着最多允许 3 个 episode 的容器同时挂在这 2 张卡上，只要它们声明的显存总和不超过预算，不要求每个容器独占一整张卡：

```yaml
units:
  - unit_id: n1-gpu01
    node_ip: 192.168.1.14
    gpu_indices: [0, 1]
    max_containers: 3
    gpu_memory_mb: 195742
```
—— `2026-09-21-env-service-design.md:1043-1049`（本地资料，未随仓库发布）

对应的准入逻辑（新设计里的 Gate A）校验的也不再是"这个设备索引有没有被占"，而是"把这个新容器加进去之后，CPU/host memory/device memory/container 数这几项加总会不会超预算"：

| | Gate A：episode 准入 |
|---|---|
| 位置 | agent loop 里，`Job.create` **之前** |
| 这一刻容器存在吗 | 还不存在 |
| 校验维度 | **Device memory、host memory、CPU、container count**（加总判断，不是"按设备互斥排队"） |

—— `2026-09-21-env-service-design.md:487-491`（本地资料，未随仓库发布）

**这组结果说明了什么。** 两个容器可以共存在同一张卡，显存占用随进程累加。程序是否真正重叠执行、每个任务得到多少计算时间，需要进一步测量；普通 CUDA 上下文与 [MPS](../knowledge/mps-parallel-execution.md) 的共享机制也不同。Gate B 只按整卡读数等待，任务评分还需要独立的归属证据。

提案把 `max_containers` 和显存预算作为配置参数。上面的双容器测量可以说明可行性，实际任务应按峰值、设备容量和干扰要求确定并发数。对这种计费方式的进一步改进见[现行显存准入](../knowledge/gpu-memory-admission.md)。

<a id="182-问题2会不会和-train_poolrollout_pool-抢同一张卡必然触发的-bug-被修了但没有自动化的结构性保证"></a>

### 18.2 配置边界与训推设备保护

§17.3 指出的根因是：`PSRL_AgentLoopWorker` 声明时是不带 `num_gpus` 的纯 CPU Actor，Ray 完全不知道它后面会摸真实 GPU，`GpuDeviceGate.discover_gpu_devices()` 又是直接 `glob` 宿主机上的全部 `/dev/nvidia*`，不读 `CUDA_VISIBLE_DEVICES`——如果运维把跑 SciAccel 的物理机和跑 `train_pool`/`rollout_pool` 的物理机设成同一批，这台机器上所有卡（包括 Ray 已经预订给训练/推理的那几张）都会被当成"可分配池"去抢。

**比这更糟的是，现状里这个风险不是"可能"，而是"必然"**——`env-service-design.md` 把它列进了 Known bugs（**B3**）：

> `gpu-compose-override.yaml` hardcodes `/dev/nvidia0` ... Every GPU episode on a node
> lands on device 0, which during training is almost certainly a rollout engine's device.

—— `2026-09-21-env-service-design.md:884`（本地资料，未随仓库发布）

因为当前这份 override 配置文件把设备路径写死成常量字符串 `/dev/nvidia0`，**不管同时有几个 episode、也不看这台机器上 0 号卡实际分给了谁**，每个需要 GPU 的 episode 读到的都是同一行配置——如果这台机器同时也在跑训练/推理，0 号卡几乎肯定已经被 Ray 分给了某个 vLLM 实例或训练 rank，SciAccel 的容器会直接撞上去。新设计明确承诺在 Phase 2 修掉这个问题（"Fix B1, B2, B3"，`2026-09-21-env-service-design.md:851`（本地资料，未随仓库发布）），具体做法是用"每个 unit 生成专属 compose overlay"替换这份写死的静态文件（`2026-09-21-env-service-design.md:848-849`（本地资料，未随仓库发布）），并且在 unit 的两条不变量里，第一条就明确写着"unit 之间不会在同一张设备上重叠，这份 unit 地图本身就是让 env 容器不去碰 rollout 引擎的 GPU 的台账"：

> **Units never overlap on a device.** The unit map is therefore itself the ledger that
> keeps env containers off a rollout engine's GPU. Ray cannot provide this, because env
> containers are invisible to it (§12.1).

—— `2026-09-21-env-service-design.md:443-445`（本地资料，未随仓库发布）

配置层面也加了一条校验规则，从"复制粘贴同一个常量"升级成了"每个 unit 显式声明自己的 `gpu_indices`，且不允许两个 unit 填同一个设备号"：

> Units are declared explicitly. Each owns whole devices, and no two units may name the
> same device index.

—— `2026-09-21-env-service-design.md:1040-1041`（本地资料，未随仓库发布）

**但这条校验规则的作用范围，只是"SciAccel 自己这些 unit 互相之间不撞"，并不是"SciAccel 的 unit 和 Ray 的 `train_pool`/`rollout_pool` 之间不撞"**——配置文件的 schema 里（§12.4 的 `units` 列表）压根没有任何字段会去读取 Ray 那边 `train_pool`/`rollout_pool` 实际占用了哪些设备索引，`gpu_indices: [0, 1]` 这两个数字填对还是填错，完全靠运维自己知道"这台机器上 0、1 号卡 Ray 有没有在用"——这和 §17.3 指出的缺口是**同一类性质的问题，只是严重程度从"必然撞"（硬编码同一个常量）降级为"人工配置失误仍可能撞"（没有自动交叉校验）**，不是被彻底自动化地堵死了。

**文档里确实设计了一套能和活跃的训练/推理池真正交互的机制**——unit 的状态机专门为"从 Ray 那边借一张卡给 env 用"设计了 `LEASE_PENDING`/`RETURNING` 两个中间状态，必须验证设备真的空了才能转换：

```
        ┌────────┐   borrow requested    ┌───────────────┐
        │  FREE  │ ────────────────────► │ LEASE_PENDING │
        └───▲────┘                       └───────┬───────┘
            │                    devices verified empty
            │                                    │
      verified empty                             ▼
            │                              ┌──────────┐
     ┌──────┴──────┐   release requested   │   ENV    │
     │  RETURNING  │ ◄──────────────────── │          │
     └─────────────┘                       └──────────┘

        ┌──────────┐
        │ ROLLOUT  │   devices held by a vLLM engine, not selectable for env
        └──────────┘
```
—— `2026-09-21-env-service-design.md:452-465`（本地资料，未随仓库发布）

配合 `nvidia-smi --query-compute-apps` 的后置校验（"借"之前确认卡上真的没有残留进程、"还"回去之前也要验证清空），这是 Phase 2 的第 5 项任务（`2026-09-21-env-service-design.md:853`（本地资料，未随仓库发布））。这套协议如果真正投入使用，是能在代码层面安全地和 Ray 的资源池做一次性握手的——但它要解决的前提是"要从活跃的训练/推理池里借卡"，而文档自己在 Phase 2 末尾直接浇了一盆冷水：

> Current topology is 3 nodes times 8 GPUs equals 24, allocated as 8 generation plus 16
> training, with **zero headroom**. Borrowing comes out of generation or training. **This
> is a topology decision, not a code decision.**

—— `2026-09-21-env-service-design.md:861-864`（本地资料，未随仓库发布）

也就是说，当前集群根本没有空闲的卡可借——真要触发这套借还协议，就必然要从正在跑的训练或推理里**暂停、挤出**一张卡给 env 用，这不是这份设计本身能用代码解决的问题，而是被明确标记为"拓扑决策"甩给了 Open Question 1。Open Question 1 给出的推荐方案，恰恰是**不要触发借还协议**：

> Options: cut generation by 2 (costs rollout throughput), cut training by 2 (costs step
> time and risks the 85 of 95 GB memory peak), or add a dedicated env node (costs nothing
> in the existing pools). **Recommendation: the dedicated node**, precisely because §4.3
> shows env GPU holds cannot be reclaimed quickly.

—— `2026-09-21-env-service-design.md:919-924`（本地资料，未随仓库发布）

换句话说，作者自己也认为"从活跃训练/推理池里借卡"风险和代价太高（episode 一旦持有 GPU 可能长达 4 小时不释放，训练/推理等不起），**真正推荐的落地方式是加一台专属的 dedicated 节点跑 SciAccel，从物理拓扑上让它和 `train_pool`/`rollout_pool` 所在节点完全不重叠**——这和本文 §17.3 建议的规避方式（"让跑 SciAccel 的节点和跑训练/推理的节点物理隔离，不复用同一批机器"）是同一个思路，也和 §10 里提到的现状治理手段（`agent.node_ips` 手工维护的节点排除名单，`2026-09-21-env-service-design.md:905-907`（本地资料，未随仓库发布））本质上是同一类"人工在部署层面保证不重叠"的治理方式，只是换了一种更规范的呈现形式（声明式的 unit 节点列表），**不是写进代码里的自动化保证**。

**小结**：

| | 问题1：显存能否共享 | 问题2：会不会和 train/rollout 抢卡 |
|---|---|---|
| 这份设计是否解决 | **是**，unit 模型 + Gate A 的加总准入就是为此设计的 | **部分**：修掉了"必然撞"的 B3 bug，新增了可选的借还验证协议，但没有自动交叉校验 unit 和 Ray 资源池的设备号 |
| 依据是否已验证 | 实测过（§5.3 两容器共存跑通），但"每个 unit 装几个合适"未实测，需 Phase 2 调优 | B3 修复方案未实现（仍在 Roadmap），借还协议本身也未实现且不被推荐启用 |
| 最终落地靠什么兜底 | 代码层面的加总判断（Gate A） | 作者推荐的是**部署层面的物理隔离**（专属 dedicated 节点），不是代码层面的自动互斥 |

<a id="19-追加问题2026-10-08fcntl-文件锁背景知识与-gate-a-的分配粒度"></a>

## 19. 文件锁如何保证一张卡只发给一个 episode

本节回答两个问题：(1) `SlotManager.acquire` 用的 `fcntl`/`flock` 文件锁，背景原理是什么、具体是怎么轮询抢锁的；(2) 现在部署的 Gate A（`GpuDeviceGate`）到底是"每个 episode 分配一整张卡"，还是按 §18.1 介绍的 `EnvUnit` 资源包模型分配。先说结论：**现状是前者**——`GpuDeviceGate` 里压根没有"unit"这个概念，`EnvUnit`/一张卡塞多个容器只存在于 §18 讨论的 `2026-09-21-env-service-design.md` 设计提案里，尚未落地到这份已部署的代码。

> **2026-10-12 更新：** 上面这段是 §19 写作时（2026-10-08）的状态，对 `GpuDeviceGate` 本身依然正确，且它仍是 `psrl.env_service.enable=False`（默认）时走的路径。但 `EnvUnit` 和一卡多容器此后已在 `psrl/workers/env_service/` 里实现为一条**独立的、默认关闭的新路径**，详见 [Env Service 实现讲解](env-service-implementation.md)。

### 19.1 先看一个具体场景：两个 episode 进程抢两张卡，发生了什么

延续 16.1 的场景：这台机器只有 2 张物理卡，`discover_gpu_devices()` 会返回 `['/dev/nvidia0', '/dev/nvidia1']`（[`gpu_device_gate.py:69-90`](../../../psrl/utils/concurrency/gpu_device_gate.py)）。假设两个不同的 `AgentLoopWorker` **进程**（回忆 16.1 补充：`worker_num` 个 Ray Actor，每个都是独立 OS 进程）P1（pid=1001）和 P2（pid=1002）几乎同时各自跑一个 `needs_gpu=True` 的 episode，都会调用到 `GpuDeviceGate.acquire()` → `SlotManager.acquire(max_slots=2, jobs_dir, prefix="sciaccel_gpu_gate")`（[`gpu_device_gate.py:137`](../../../psrl/utils/concurrency/gpu_device_gate.py)）。

手工画一条时间线（`poll_interval` 默认 0.2 秒，见 [`slot.py:35`](../../../psrl/utils/concurrency/slot.py)）：

```
t=0.000  P1 进入 for slot_idx in range(2) 循环，先试 slot_idx=0
t=0.000  P1 打开 /tmp/psrl_slots_<hash>/slot_0.lock，flock(EX|NB) 成功
         → P1 立刻拿到 (fd_P1, 0)，对应 /dev/nvidia0，不需要等待
t=0.001  P2 也进入循环，先试 slot_idx=0
t=0.001  P2 打开同一个 slot_0.lock（另开一个 fd，但指向磁盘上同一个 inode）
         flock(EX|NB) 失败：P1 已经持有独占锁 → 抛 BlockingIOError
         P2 关闭这次打开的 fd，继续试 slot_idx=1
t=0.001  P2 打开 slot_1.lock，flock(EX|NB) 成功
         → P2 拿到 (fd_P2, 1)，对应 /dev/nvidia1
```

两个槽位、两个竞争者，所以一轮循环内都能拿到槽位，不需要睡眠。如果这时来了第三个 episode（P3），它会先后对 `slot_0.lock`、`slot_1.lock` 都尝试加锁并都失败，`for` 循环跑完一轮仍然两手空空，于是执行 [`slot.py:73`](../../../psrl/utils/concurrency/slot.py) 的 `await asyncio.sleep(0.2)`，睡 0.2 秒后回到 `while True` 的开头重新扫一遍两个槽位——直到 P1 或 P2 中有一个 episode 结束、调用 `SlotManager.release` 解锁，P3 才能在下一次扫描时抢到那个刚释放的槽位。这个"扫一遍所有槽位 → 都占用就睡一下 → 再扫一遍"的循环，就是用户问的"具体怎么轮询抢锁"。

### 19.2 背景知识：什么是文件锁，advisory 锁和 mandatory 锁有什么区别

**先明确这里用的是哪一种锁**。Python 的 `fcntl` 标准库模块名字叫 `fcntl`，但它同时封装了两套不同的 POSIX 文件锁 API：一套是 `fcntl()` 系统调用本身的 `F_SETLK`/`F_SETLKW`（可以锁文件里的任意字节区间，称为"记录锁"/record lock）；另一套是 `flock()` 系统调用（只能锁整个文件，不能锁字节区间，称为"BSD 锁"）。`slot.py` 里用的是 `fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)`（[`slot.py:66`](../../../psrl/utils/concurrency/slot.py)）——函数名带 `flock`，调用的就是后一种、锁整个文件的语义。因为这里每个 `slot_N.lock` 文件本身就是专门为"一把锁"创建的空文件，不需要在同一个文件内部再划分字节区间，所以选最简单的整文件锁已经够用，不需要记录锁的额外复杂度。

**advisory（劝告式）锁 vs mandatory（强制式）锁**：这是文件锁领域一个必须先搞清楚的区分，否则会误以为"加了锁，这个文件就被保护起来了"。`flock`/`fcntl` 在 Linux/POSIX 系统上默认都是 **advisory（劝告式）** 的，意思是：操作系统内核并不会阻止任何进程直接 `open()`/`read()`/`write()`/`unlink()` 这个文件——加锁本身只在"调用者主动调用 `flock()` 并检查返回值"这个协作协议内生效。一个不遵守这个协议、直接用 `open()` 读写文件的进程，完全可以在别人持有锁的情况下照样把文件改了，操作系统不会报错拦截。

打个类比：几个人共用一间会议室，约定"进门前看看门口挂牌，挂着'使用中'就不进"。这块牌子（对应这里的 `slot_N.lock` 文件）本身不会把门锁上，谁不看牌硬闯都能推门进去；这套机制之所以有效，是因为**所有相关方都约定好了同一套"先查牌子再决定进不进"的协议**，而不是靠物理上锁死房门。`SlotManager.acquire`/`release` 就是在维护这块"牌子"：所有需要这把锁的代码（这里只有 `GpuDeviceGate`）都遵守"先 `flock` 再用对应的 GPU"这个协议，这把锁就等价于互斥；但如果有另一段完全不知情的代码直接往这台机器摸 `/dev/nvidia0`（比如手工 `docker run --gpus device=0` 而不经过这套 Gate A），这把 `fcntl` 锁完全拦不住它——这也是 §17.3 讨论"Gate A 拦不住 Ray 的 train_pool/rollout_pool 去抢同一张卡"的根本原因：那两边压根不是"遵守同一套协议的协作方"。

（与之相对的 mandatory lock 是由内核在系统调用层面强制拒绝不符合锁状态的 I/O 操作，Linux historically 支持过一种依附于文件系统挂载选项 `-o mand` 的强制锁，但已经被标记废弃、现代 Linux 内核默认不再支持，这里只是帮助理解"advisory"这个词是相对谁而言的，不是这份代码会用到的机制。）

**`LOCK_EX`、`LOCK_NB` 这两个标志位分别是什么**（[`slot.py:66`](../../../psrl/utils/concurrency/slot.py) 的 `fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)`）：

| 标志 | 含义 | 在这里的作用 |
|---|---|---|
| `LOCK_EX` | 独占锁（exclusive）：同一时刻最多一个持有者。另有 `LOCK_SH` 共享锁，多个持有者可以同时读，这里没用到 | 保证一个 `slot_N.lock` 同一时刻只会被一个 episode 的进程持有，对应"一张卡只能分给一个 episode" |
| `LOCK_NB` | 非阻塞（non-blocking）：如果锁已被别人占着，`flock()` 立刻返回错误（Python 里体现为抛出 `BlockingIOError`），而不是把调用线程挂起等待 | `SlotManager` 要做的是"抢不到这个槽位就马上去试下一个槽位"，不能在第一个槽位上傻等——如果不加 `LOCK_NB`，调用会一直阻塞在 `slot_0.lock` 上，哪怕 `slot_1.lock` 这时候是空的也发现不了 |

**为什么是"轮询 + 睡眠"而不是直接阻塞等一个锁**：因为这里本质上是"N 个槽位里只要抢到任意一个就行"，而不是"死等某一个固定的槽位"。如果对某个固定文件用阻塞模式的 `flock`（不带 `LOCK_NB`），等于把自己绑死在一个槽位上，即使另一个槽位先空出来，也不会被唤醒去尝试。所以代码选择的策略是：扫一遍所有候选槽位，只要有一个能抢到就立刻返回；如果一圈下来全部抢不到，就睡一小段时间（`poll_interval`，默认 0.2 秒）再重新扫一遍——这是一种简单的忙等（busy polling），不是真正的"排队"。

**这里有没有公平性保证**：没有。如果 P3 睡眠恰好跟 P4 的下一次扫描错开，先醒的那个总是先遍历 `slot_idx=0`，谁先调用到 `flock` 谁就拿到——不保证"先来后到"。对这套机制而言问题不大：一个 SciAccel episode 通常运行几分钟到几小时（§16.5 诚实说明里提到过 episode 可能长达 4 小时持有 GPU），相比之下 0.2 秒的轮询间隔和"不保证公平"带来的等待时间波动可以忽略不计；如果场景换成"请求以毫秒级到达、对排队顺序敏感"，这种忙等轮询就不是合适的设计了。

**为什么"检查槽位有没有被占用"和"占用这个槽位"之间不需要再加一把锁保护**：这是初学者很容易想岔的一点——如果在用户态自己写逻辑："检查这个文件存不存在，不存在就创建它表示占用"，这中间存在一段时间窗口：两个进程都可能在"检查"那一刻看到"文件不存在"，然后都去执行"创建"，于是都以为自己抢到了——这是经典的 **TOCTOU（Time-Of-Check-To-Time-Of-Use，检查时刻到使用时刻之间的竞态）** 问题。`flock(fd, LOCK_EX | LOCK_NB)` 之所以不会有这个问题，是因为"检查锁状态"和"把锁标记为自己持有"这两步，在内核里是用**同一次系统调用**完成的单一原子操作——内核在处理这次系统调用期间持有内部的锁状态结构，不会被另一个进程的同类调用打断，所以"两个进程同时认为自己抢到了同一把锁"在 `flock` 这一层不可能发生。这也是几乎所有操作系统锁原语（`flock`、`fcntl` 记录锁、互斥锁的 `compare-and-swap` 实现等）的共同设计要求：锁的"尝试获取"接口本身必须是原子的，否则锁这个概念就不成立。

### 19.3 逐行解读抢锁循环的源码

完整源码在 [`psrl/psrl/utils/concurrency/slot.py`](../../../psrl/utils/concurrency/slot.py)。先看锁目录怎么确定：

```python
@staticmethod
def slot_lock_dir(output_dir: str, prefix: str = "psrl_slots") -> str:
    digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{digest}")
```
—— [`slot.py:18-26`](../../../psrl/utils/concurrency/slot.py)

`output_dir`（对 Gate A 而言就是 `config.harbor.jobs_dir`，见 [`gpu_device_gate.py:137`](../../../psrl/utils/concurrency/gpu_device_gate.py)）先取绝对路径再做 SHA1 哈希、截取前 12 位，拼成锁目录名。这样设计的作用：不同的 `output_dir`（比如不同的训练任务用了不同的 jobs 目录）会落到系统临时目录下不同的子目录，各自的槽位池互不相干——同一台机器上两个毫不相关的训练任务，不会因为共用了同一套 `slot_0.lock`...`slot_N.lock` 而互相把对方的 episode 挤占掉。

再看抢锁主循环：

```python
while True:
    for slot_idx in range(max_slots):
        lock_path = os.path.join(lock_dir, f"slot_{slot_idx}.lock")
        fd = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o666,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode())
            return fd, slot_idx
        except BlockingIOError:
            os.close(fd)

    await asyncio.sleep(poll_interval)
```
—— [`slot.py:57-73`](../../../psrl/utils/concurrency/slot.py)

逐行拆开看每一步在做什么、为什么这么做：

| 代码 | 在做什么 | 为什么这么做 |
|---|---|---|
| `os.open(lock_path, os.O_CREAT \| os.O_RDWR \| os.O_CLOEXEC, 0o666)` | 打开（必要时先创建）这个槽位对应的锁文件，可读可写 | `O_CREAT`：第一次运行、锁目录刚建好时文件还不存在，需要能自动创建；`O_RDWR`：后面要往文件里写 pid 调试信息，只读模式不够；`O_CLOEXEC`：如果这个进程之后 `exec()` 替换成另一个程序（比如 fork 出子进程跑 shell 命令），这个 fd 不会被继承到子进程里——避免子进程意外持有一份对同一个锁文件的引用，造成锁状态被不该知情的子进程影响 |
| `fcntl.flock(fd, fcntl.LOCK_EX \| fcntl.LOCK_NB)` | 尝试非阻塞地给这个 fd 对应的文件整体加独占锁 | 如 19.2 节所述，这一步原子地完成"检查 + 占用"；加了 `LOCK_NB` 所以抢不到立刻抛异常而不是卡住 |
| `except BlockingIOError: os.close(fd)` | 加锁失败时关闭这次 `os.open` 打开的 fd，然后外层 `for` 循环自然推进到下一个 `slot_idx` | 必须显式 `close(fd)`——因为 `os.open()` 本身是成功的（文件确实被打开了），只是随后的 `flock` 没抢到锁；如果不关闭，每试一个槽位就泄漏一个文件描述符，在槽位数很多、轮询很多轮之后可能把这个进程的 fd 上限（`ulimit -n`）耗尽 |
| `os.ftruncate(fd, 0)` + `os.write(fd, f"pid={os.getpid()}\n".encode())` | 抢到锁后，把文件内容清空并写入当前进程的 pid | 纯粹是给人类调试用的：运维/开发者在服务器上手动 `cat /tmp/psrl_slots_<hash>/slot_0.lock`，可以直接看到"现在是哪个进程占着这个槽位"，不参与锁逻辑本身——就算这两行删掉，锁的互斥语义不受影响 |
| `return fd, slot_idx` | 把"锁住的文件描述符"和"槽位编号"一起返回给调用方 | `fd` 必须留着不能关：`flock` 加的锁是跟着这个打开的文件描述符（更准确说是跟着它背后的"打开文件描述"/open file description）走的，一旦 `close(fd)`，这把锁会被内核自动释放。调用方后续要靠持有这个 `fd` 才能保证锁一直有效，直到显式调用 `release` |
| `await asyncio.sleep(poll_interval)` | 一整轮都没抢到任何槽位时，让出协程调度权，睡 `poll_interval`（默认 0.2 秒）后重新回到 `while True` 开头 | 用 `asyncio.sleep` 而不是 `time.sleep`：`SlotManager.acquire` 本身是 `async def`，运行在 PSRL 的 asyncio 事件循环里（比如 Harbor episode 的调度协程），`time.sleep` 会同步阻塞整个事件循环、连带卡住这个进程里其他本可以继续推进的协程；`asyncio.sleep` 只挂起当前这一个协程，事件循环在这 0.2 秒里可以去处理同一进程内其他正在跑的 episode |

最后是释放逻辑：

```python
@staticmethod
def release(run_slot: tuple[int, int] | None) -> None:
    if run_slot is None:
        return
    fd, _ = run_slot
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
```
—— [`slot.py:75-87`](../../../psrl/utils/concurrency/slot.py)

`flock(fd, LOCK_UN)` 显式解锁，`finally: os.close(fd)` 保证不管解锁是否抛异常，这个 fd 一定会被关闭。前面提到过"关闭 fd 本身也会让内核自动释放 flock 锁"，这里为什么还要显式 `LOCK_UN` 再 close？——这是防御性写法：显式解锁让意图在代码里一目了然（"这里是在释放锁"，而不是依赖"关闭 fd 顺带解锁"这个不那么直观的副作用），且不依赖具体平台对"关闭即解锁"这条规则的实现细节，两步都做，哪一步出问题都不会让锁永久泄漏（`try/finally` 保证即使 `flock(LOCK_UN)` 抛出异常，`close` 依然会执行，进程退出时未显式关闭的 fd 也会被内核回收并连带释放锁，所以即便某处异常跳过了 `release`，锁最终也不会永久卡死，只是不会立刻释放）。

<a id="194-直接回答现在的-gate-a-是按-episode-分配整卡还是按资源-unit-分配"></a>

### 19.4 分配粒度：整卡与资源单元

**结论：按 episode 分配整卡，不按资源 unit 分配**。完整源码在 [`psrl/psrl/utils/concurrency/gpu_device_gate.py`](../../../psrl/utils/concurrency/gpu_device_gate.py)，核心的 `acquire`：

```python
class GpuDeviceGate:
    def __init__(self, jobs_dir: str, slot_prefix: str = _DEFAULT_SLOT_PREFIX):
        self.jobs_dir = jobs_dir
        self.slot_prefix = slot_prefix

    async def acquire(self) -> AllocatedGpu | None:
        devices = discover_gpu_devices()
        if not devices:
            logger.warning(...)
            return None

        slot = await SlotManager.acquire(len(devices), self.jobs_dir, prefix=self.slot_prefix)
        ...
        _fd, slot_index = slot
        device = devices[slot_index]
        return AllocatedGpu(device=device, index=slot_index, _slot=slot)
```
—— [`gpu_device_gate.py:102-147`](../../../psrl/utils/concurrency/gpu_device_gate.py)

三个关键点能直接证明"一个 episode 独占一整张卡"，而不是"按 unit/按显存切分"：

1. **`max_slots` 直接等于物理卡数量**：`SlotManager.acquire(len(devices), ...)`（[`gpu_device_gate.py:137`](../../../psrl/utils/concurrency/gpu_device_gate.py)）——`devices` 是 `discover_gpu_devices()` glob 出来的 `/dev/nvidia0`、`/dev/nvidia1`…（[`gpu_device_gate.py:69-90`](../../../psrl/utils/concurrency/gpu_device_gate.py)），一张卡对应一个槽位，槽位总数就是卡的总数。这里没有任何地方出现"一张卡划成几个子槽位"或"按显存 MB 切分配额"的逻辑。
2. **`AllocatedGpu` 只记录"哪张卡"，不记录"用了多少显存"**：
   ```python
   @dataclass
   class AllocatedGpu:
       device: str
       index: int
       _slot: tuple[int, int]
   ```
   —— [`gpu_device_gate.py:93-99`](../../../psrl/utils/concurrency/gpu_device_gate.py)
   
   字段里没有 `gpu_memory_mb`、`max_containers` 这类"资源包预算"概念，和 §18.1 介绍的 `EnvUnit`（`2026-09-21-env-service-design.md:429-439`（本地资料，未随仓库发布），带 `gpu_memory_mb`/`max_containers`/`gpu_indices: tuple[int, ...]` 这些字段，一个 unit 可以装下多个容器）在数据结构层面就是两回事。
3. **一把 `flock` 独占锁，天然排斥"多个 episode 同时持有同一个槽位"**：19.1/19.2 节已经讲过，`LOCK_EX` 独占锁决定了 `slot_0.lock` 同一时刻只能有一个 episode 的进程持有——既然"一个槽位"直接映射"一张卡"，这就直接等价于"一张卡同一时刻只服务一个 episode"，中间没有"同一张卡上多个 episode 共存"的空间。

对照 §18.1 讨论过的、**尚未落地的**资源包模型，两者差异一目了然：

| | 现在部署的 Gate A（`GpuDeviceGate`，本文 §16.5/本节） | §18.1 讨论的 `EnvUnit` 模型（`2026-09-21-env-service-design.md`，仅是设计提案） |
|---|---|---|
| 分配的最小单位 | 一整张物理卡（一个 `/dev/nvidiaN`） | 一个 unit（可以包含多张卡，`gpu_indices: tuple[int, ...]`），unit 内部可以塞多个容器 |
| 一张卡能同时服务几个 episode | 恰好 1 个（`LOCK_EX` 独占锁保证） | 可以多个，只要 `max_containers` 和 `gpu_memory_mb` 预算没超（`2026-09-21-env-service-design.md:487-491`（本地资料，未随仓库发布） 的"加总判断"式 Gate A） |
| 准入校验维度 | 只有"这个槽位有没有被占" | CPU、host memory、device memory、container 数，四项加总 |
| 实现状态 | **已部署**，本文 §16.5 给出的实测脚本（`diagnostics/gate_a_before.py`/`gate_a_after.py`）跑的就是这一套 | **未实现**，§18.1 末尾已经指出"一个 unit 塞几个容器合适"连配置超参都还没实测调优，仍在 Roadmap 上 |

所以如果之后训练任务里同时出现多个低显存的 GPU episode，用现在这份代码会观察到的现象是：哪怕 8 张卡总共有几百 GB 显存，一旦并发的 GPU episode 数超过卡数，多出来的 episode 一律在 19.1 节描述的轮询循环里排队等待，不会被塞进同一张还有空闲显存的卡——这正是 §18.1 开头那句"现状把一张物理卡当成不可再分的互斥单位"的直接代码证据。

## 延伸阅读

- 设计动机和角色分工：[`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md) 第 0-10 节。
- 怎么把现有 EnvWorker 接进 PSRL、怎么写 GPU 集成测试：[`onboarding-and-gpu-validation.md`](../knowledge/onboarding-and-gpu-validation.md)。
- 旧 EnvWorker 的修复前审查和后续阅读入口：[`implementation-and-gaps.md`](implementation-and-gaps.md)。
- 完整的重构设计稿（unit 模型、两道 Gate、Harbor 集成深度、Roadmap、Known bugs）：`2026-09-21-env-service-design.md`（本地资料，未随仓库发布）。
- 训练设备 / vLLM 推理设备这两个 Ray 资源池的划分，和 PSRL 整体并发/并行机制（协程/线程/进程三层）的关系：`psrl-concurrency-landscape.md`（本地资料，未随仓库发布）。
