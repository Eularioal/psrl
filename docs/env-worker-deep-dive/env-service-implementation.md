# Env Service 实现讲解：一卡多容器、EnvRouter、训推节点策略（2026-10-12 改动）

## TL;DR

- **看不懂租约（lease）、overlay、槽位 TTL 这些词？直接看 [§2 基础概念](#2-基础概念unitslot-槽位demand租约-lease心跳ttloverlay)**：租约 = 一张"有有效期的门禁卡"，申请成功就发、定期心跳续期、没心跳超过 TTL（默认 1800 秒）就作废回收；overlay = 运行时生成的、追加在任务原 compose 之后的"备注贴纸"，写这次容器具体挂哪几张卡；slot 槽位 = unit 的容器名额（`max_containers`）。配可运行脚本 `scripts/lease_ttl_demo.py`。
- 这次在 `psrl/` 仓库里把 Env Service 设计提案（本地资料，未随仓库发布） 的核心部分真正写了出来：**`EnvUnit`（一张卡多个容器）、`EnvRouter`（Ray actor）、资源加权的 Gate A、每租约 compose overlay（运行时解析驱动库）、TTL 回收、单元感知派发**，外加提案里没有的**训推节点策略**（容器能不能和训练/推理共用节点）。代码在 `psrl/psrl/workers/env_service/`（10 个文件），接入点改了 7 个已有文件。
- **默认关闭**（`psrl.env_service.enable: False`）。关闭时 SciAccel 仍走旧路径（进程信号量 + `GpuDeviceGate`），派发也与旧的轮转算法**逐字等价**（有差分测试证明）。
- 一卡多容器的规则：一个 unit 独占若干整卡，容器共享这些卡；SM 算力靠驱动时间片，**显存是相加的**，所以准入时对显存求和（取"声明值"与"实测值"的较大者），而不是按卡排队。80 GB 的卡、每个任务 24 GB，可以同时跑 3 个容器。
- 训推节点策略 `node_policy` 有四档：`any` / `cpu_only`（默认）/ `avoid` / `prefer_other`。它同时管两件事：哪些节点能放 agent worker（容器落在启动它的 worker 所在节点），以及 router 能给哪些 unit 发租约。
- **验证情况要说清楚**：本机没有 Ray、Harbor、torch、GPU、Docker，所以真实训练、真实容器、真实 `nvidia-smi` 都**没跑过**。能跑的都跑了：新增 7 个测试文件共 **136 个测试全过**；`ledger.py` 做了变异检查（故意改坏 3 处，都被测试抓到）；演示脚本全部期望成立。`ray_trainer.py` 的接入只做了语法检查，没有运行过。
- **没有做的**：借还协议（`LEASE_PENDING`/`RETURNING` 状态、"验证卡已空"后置条件）、Phase 0（合并 Path A 执行后端）、容器复用（L2）、远程代理（L3）、按进程的 `pmon` 遥测、注册时健康探测。见 [§8](#8-没有实现的部分)。

## 目录

- [1. 先回答：这次实现了提案的哪些部分](#1-先回答这次实现了提案的哪些部分)
- [2. 基础概念：unit、slot 槽位、demand、租约 lease、心跳、TTL、overlay](#2-基础概念unitslot-槽位demand租约-lease心跳ttloverlay)
- [3. 一张图：一个 GPU episode 现在怎么走](#3-一张图一个-gpu-episode-现在怎么走)
- [4. 图例：每个部分对应的源码位置](#4-图例每个部分对应的源码位置)
- [5. 一卡多容器：账本怎么算](#5-一卡多容器账本怎么算)
- [6. 训推节点策略：容器要不要和训练、推理共用节点](#6-训推节点策略容器要不要和训练推理共用节点)
- [7. 怎么打开，以及打开后行为有什么变化](#7-怎么打开以及打开后行为有什么变化)
- [8. 没有实现的部分](#8-没有实现的部分)
- [9. 测试与演示的实测输出](#9-测试与演示的实测输出)
- [10. 没有验证的部分与已知风险](#10-没有验证的部分与已知风险)
- [相关文档](#相关文档)

## 1. 先回答：这次实现了提案的哪些部分

| 提案里的内容 | 状态 | 说明 |
|---|---|---|
| `EnvUnit`：一个 unit 多个容器、带显存预算 | **已实现** | [`types.py`](../../psrl/workers/env_service/types.py)；同节点两个 unit 不得声明同一张卡（启动时校验） |
| `EnvRouter`（Ray actor：unit 注册、准入、租约、回收） | **已实现** | [`router.py`](../../psrl/workers/env_service/router.py)。核心逻辑是不依赖 Ray 的 `EnvRouterCore`，actor 只是它的包装 |
| Gate A：按 CPU、主机内存、容器数、**设备显存求和**准入，等多久都行 | **已实现** | [`ledger.py`](../../psrl/workers/env_service/ledger.py) + [`client.py`](../../psrl/workers/env_service/client.py) |
| 实测占用优先于声明（§4.6） | **已实现** | 取 `max(声明, 实测)`；实测由 worker 自己节点的 `nvidia-smi` 读出，随准入请求带给 router |
| 每租约 compose overlay、运行时解析驱动库（修 B1/B2/B3/B6） | **已实现** | [`overlay.py`](../../psrl/workers/env_service/overlay.py)，仅限新路径；旧的静态 `gpu-compose-override.yaml` 没删 |
| `override_gpus: 0` 绕过 Harbor 的 GPU 校验（§7.3） | **已实现** | 有 GPU 租约时在 [`runner.py`](../../examples/sciaccel_rl/runner.py) 里设置 |
| 槽位 TTL，回收时停止容器 | **已实现** | router 侧 reaper 释放容量，容器由该节点的下一次准入顺带清理（按 `psrl.env_lease` 标签） |
| 单元感知派发（§6.4） | **已实现（节点粒度）** | [`placement.py`](../../psrl/workers/env_service/placement.py)；按节点的剩余容量挑 worker，不是"先选 unit 再派发" |
| Gate B 适配共享卡、多设备 | **已实现** | [`session_router.py`](../../psrl/workers/gen/session_router.py)：多设备要全部空闲；共享卡只看利用率 |
| 训推节点策略 | **新增（提案没有）** | [`node_policy.py`](../../psrl/workers/env_service/node_policy.py) |
| 借还协议（`LEASE_PENDING`/`RETURNING`）、Phase 0、L2/L3、`pmon`、健康探测 | **未实现** | 见 [§8](#8-没有实现的部分) |
| `EnvWorker`（每个 unit 一个 Ray actor） | **未实现为独立 actor** | 容器生命周期仍在 agent loop worker 里；原 `psrl/workers/env_worker/` **没有改动** |

## 2. 基础概念：unit、slot 槽位、demand、租约 lease、心跳、TTL、overlay

后面的章节会用这些词，先把它们说清楚。每个词都按同一个套路讲：**它是什么 → 为什么需要它 → 一个可手算的小例子 → 对应到哪段代码**。

### 2.1 一张对照表：这些词各自在解决什么问题

| 术语 | 一句话它是什么 | 类比 | 不用它会怎样 |
|---|---|---|---|
| **unit**（资源单元） | 一个节点上划出来的一块可分配资源：几张整卡 + CPU 配额 + 内存 + 容器数上限 | 一间带固定工位的实验室 | 容器和训练互相抢卡，谁也不知道还剩多少 |
| **slot / 槽位** | "还能再放几个容器"的一个名额，是 unit 的三种容量之一 | 实验室的工位数 | 只能靠"一张卡一个容器"这样粗暴的换算 |
| **demand**（需求） | 一个 episode 申请的资源清单：多少 CPU、多少内存、要不要卡、要多少显存 | 进实验室前填的申请表 | 无法判断"这个任务放不放得下" |
| **lease / 租约** | 一次成功申请的凭证：它记录"哪个 episode、拿到哪个 unit、哪些卡、占了多少资源、什么时候批的" | 一张**有有效期的门禁卡** | 只有"放行/拒绝"两个动作，出事了无法追溯谁占了什么 |
| **心跳 heartbeat** | 持有者定期报"我还活着"，用来续租约 | 每隔一段时间去前台刷一次门禁卡 | 崩掉的 episode 会永远占着位置 |
| **slot TTL** | 租约"多久没心跳就作废"的时长，默认 1800 秒 | 门禁卡的有效期 | 只能靠人工发现卡住了 |
| **overlay** | 一份运行时生成的、追加在任务原有 compose 文件之后的 Docker Compose 片段，写"这个容器具体挂哪几张卡" | 在标准菜单上贴的一张"备注贴纸" | 只能把所有容器都挂到写死的 `/dev/nvidia0` |
| **Gate A / Gate B** | 两道准入的闸门：A 管"开始干活前有没有位置"，B 管"回复模型前卡空不空" | 进门前排队 / 出门前看路 | 见 [§2.7](#27-gate-a-和-gate-b两道闸门各管什么) |

这些都是**账本上的概念**，和 Docker 里真实存在的容器是两套东西：账本里"已经算给某个 episode 了"不等于容器已经起来了。这个区别在 [§2.4](#24-租约-lease一次成功申请的凭证) 末尾会再强调一次。

### 2.2 unit 和 slot 槽位

**是什么。** `EnvUnit` 一次声明一整套资源：

```19:45:psrl/psrl/workers/env_service/types.py
@dataclass(frozen=True)
class EnvUnit:
    """
    One allocatable resource bundle on one node.

    A unit owns whole GPU devices outright and shares them among every container it
    hosts, so several containers may sit on one card as long as their summed device
    memory fits `gpu_memory_mb`. Device memory adds under sharing and scheduling cannot
    reclaim it, which is why admission sums it instead of queueing per card.

    Attributes:
        unit_id (str): Stable, unique cluster-wide id.
        node_ip (str): Node whose Docker daemon hosts the unit's containers.
        cpu_quota (float): CPUs shared by this unit's containers.
        memory_mb (int): Host memory shared by this unit's containers.
        gpu_indices (tuple[int, ...]): Whole devices owned by this unit. Empty means CPU only.
        max_containers (int): Configured ceiling on concurrent containers, not derived.
        gpu_memory_mb (int): Device memory budget across `gpu_indices`. Zero for CPU units.
    """

    unit_id: str
    node_ip: str
    cpu_quota: float
    memory_mb: int
    gpu_indices: tuple[int, ...] = ()
    max_containers: int = 1
    gpu_memory_mb: int = 0
```

**为什么需要。** Ray 自己也有资源的"申请-记账"，但那是**声明式**的：它只知道"我要 1 个 GPU"，然后给你一个 `CUDA_VISIBLE_DEVICES`，既不检查那张卡上有没有别人，也没有跨进程的准入。Harbor 起的容器对 Ray 完全不可见，于是两份记账会各自以为卡是空的。`EnvUnit` 就是把"一张卡上能放几个容器"这件事变成一个**可以配置、可以被检查、有单一出处**的声明。为什么不能共享就不能安心——因为显存是相加的，见 [§5.2](#52-有效占用取声明和实测的较大者)。

**slot 槽位是什么。** 就是 `max_containers` 这个数字本身：这个 unit 最多同时容纳几个容器。它和显存是**两条独立的限制**，谁先到顶谁生效：

| 场景 | 先到顶的是 | 结果 |
|---|---|---|
| 80 GB 卡，每容器 24 GB，`max_containers: 3` | 两条同时到顶 | 放 3 个容器，用了 72 GB |
| 80 GB 卡，每容器 24 GB，`max_containers: 2` | 槽位（容器数） | 放 2 个容器，只用了 48 GB |
| 80 GB 卡，每容器 24 GB，`max_containers: 10` | 显存 | 放 3 个容器，第 4 个超显存 |

所以"槽位"这个词指的就是**容器名额**（`max_containers`），"槽位 TTL"字面意思是"这个名额多久没被确认还活着就收回"。注意它虽然叫 slot TTL，管的其实是**租约**的生死；配置项名和数据结构名在这里没对齐，是本项目自己的历史命名，见 [§2.5](#25-心跳-heartbeat-和-ttl有效期的判定与回收的动作)。

### 2.3 demand 需求：申请单

```110:126:psrl/psrl/workers/env_service/types.py
@dataclass(frozen=True)
class EnvDemand:
    """
    Resources one episode's container set needs while it runs.

    Attributes:
        cpu (float): CPUs charged against the unit.
        memory_mb (int): Host memory charged. Zero means "not accounted".
        needs_gpu (bool): Whether the episode needs a GPU device.
        gpu_memory_mb (int): Device memory charged. Zero on a GPU demand means "use the
            router's configured default", because Harbor declares no such field.
    """

    cpu: float = 1.0
    memory_mb: int = 0
    needs_gpu: bool = False
    gpu_memory_mb: int = 0
```

这些数字从哪来？从任务自己的 `task.toml`：`[environment]` 段里的 `cpus`、`memory_mb`、`gpus`，由 [`demand_from_task`](../../examples/sciaccel_rl/env_demand.py#L23) 读出来。**注意一个缺口**：Harbor 没有"需要多少显存"这个字段，所以 `gpu_memory_mb` 常常是 0，表示"用 router 配的默认值"（`default_gpu_memory_mb`，默认 24000 MB）。这也是为什么它会以 24 GB 这个数字反复出现在例子和演示里。

### 2.4 租约 lease：一次成功申请的凭证

**是什么。** 申请成功不是返回一个"OK"，而是返回一张凭证：

```135:162:psrl/psrl/workers/env_service/types.py
@dataclass
class EnvLease:
    """
    One admitted episode's hold on a unit.

    Attributes:
        lease_id (str): Unique id. Stamped on the containers as the `psrl.env_lease` label.
        session_id (str): The episode's session id.
        unit_id (str): The unit that hosts the episode.
        node_ip (str): The unit's node.
        gpu_indices (tuple[int, ...]): Devices the containers get. Shared with other leases of the unit.
        demand (EnvDemand): What was charged.
        granted_at (float): Ledger clock when granted.
        last_seen (float): Ledger clock of the latest heartbeat. Drives TTL reclaim.
        shared (bool): Whether the unit may host several containers on these devices. Gate B
            then judges idleness on compute utilization alone, because a neighbour's resident
            model weights hold device memory even while it is idle.
    """

    lease_id: str
    session_id: str
    unit_id: str
    node_ip: str
    gpu_indices: tuple[int, ...]
    demand: EnvDemand
    granted_at: float
    last_seen: float
    shared: bool = False
```

逐字段读一遍，就明白它为什么必须存在：

| 字段 | 实际作用 | 去掉会怎样 |
|---|---|---|
| `lease_id` | 会作为 Docker 标签 `psrl.env_lease=<id>` 打在容器上（[`overlay.py:35`](../../psrl/workers/env_service/overlay.py#L35)）。**这是"账本里的租约"和"现实里的容器"之间唯一的绳子** | 超出 TTL 的残留容器没法按标签找到并删掉 |
| `session_id` | 把租约对回 episode；同一个 session 重复申请时直接返回已有租约 | 重复申请会重复占名额 |
| `unit_id` / `node_ip` | 告诉调用方（以及 Gate B）容器将出现在哪个节点的哪张卡上 | 无法正确挂载设备、无法回收 |
| `gpu_indices` | 这份租约能用的卡号，例如 `(0, 1)` | 容器只能挂到写死的卡上 |
| `demand` | 占用了多少资源，释放时按它归还 | 归还的额度对不上，账本会漂移 |
| `granted_at` / `last_seen` | 审批时刻和最后一次心跳时刻，`last_seen` 就是 TTL 判定的依据 | 无法判断租约是否过期 |
| `shared` | 这张卡是否与其他容器共享（Gate B 据此只用利用率判空闲） | 共享卡上邻居的常驻显存会让 Gate B 每次都等到超时 |

`lease_id` 的形状是 `{unit_id}-{序号}-{随机串}`，例如演示脚本里出现的 `g0-1-5c1dd22f`：

```165:165:psrl/psrl/workers/env_service/ledger.py
                lease_id=f"{unit.unit_id}-{next(self._counter)}-{self._id_factory()}",
```

- `g0` 是 unit 名，方便人一眼看出是哪台机器上的哪块资源；
- `1` 是该进程内的递增序号，看到跳号就知道中间发生过别的申请；
- 随机串是 [`id_factory`](../../psrl/workers/env_service/ledger.py#L89) 生成的，避免 router 重启后序号从 1 重来、与残留容器的旧标签撞车。

**为什么叫"租约"而不是"锁"。** 这是个有意的区分，也是这个词唯一的难点：

| | 锁（本项目旧的 `GpuDeviceGate`） | 租约（lease） |
|---|---|---|
| 何时释放 | 只有持有者显式释放 | 显式释放**或**有效期到期 |
| 持有者崩了 | 一直锁着，直到有人人工清理 | 最多多占一个 TTL，之后自动回收 |
| 续期 | 不需要 | 靠心跳续期 |
| 代价 | 实现简单 | 调用方必须记得心跳，且要容忍"续期失败"这种情况 |

之所以要选后者：episode 会跑几十分钟到几小时，而 agent loop worker 可能被 Ray 重启、被抢占、被 kill。只要没释放就永远占着，卡就会慢慢被"幽灵租约"吃光。租约用"必须定期证明我还活着"换取"崩溃能自愈"。设计提案里还规划了更严格的状态机（借卡要验证、还卡要验证卡真的空了），那部分见 [§8](#8-没有实现的部分)。

**一个容易混淆的点**：账本授予租约 ≠ 容器已经起来了。授予发生在 Harbor 起容器**之前**（正因为如此，"等待"才是免费的）。所以如果紧接着起容器就失败，账本上会出现一个"有租约但没容器"的短暂状态——这就是 `finally` 里必须释放租约的原因。

### 2.5 心跳 heartbeat 和 TTL：有效期的判定与回收的动作

这两个词经常被混在一起说，其实是**两件事**：

- **TTL（time to live，存活时长）是一个判定条件**：`现在 - 最后一次心跳 > TTL` 就认为持有者没了。
- **回收 reaper 是一个动作**：真的检查一遍、把过期的租约收回来、把容量还回 unit。

判定条件写在一行里：

```231:238:psrl/psrl/workers/env_service/ledger.py
    def _reap(self, now: float) -> list[EnvLease]:
        expired = [lease for lease in self._leases.values() if now - lease.last_seen > self._lease_ttl_s]
        for lease in expired:
            self.release(lease.lease_id)
            self._reaped.setdefault(lease.node_ip, []).append(lease)
        for sid in [s for s, ts in self._waiting.items() if now - ts > self._wait_ttl_s]:
            del self._waiting[sid]
        return expired
```

谁能触发它？两条路，都在 [`router.py`](../../psrl/workers/env_service/router.py)：

1. **后台循环**：actor 启动后在它自己的事件循环里跑 `_reap_forever`，间隔是 `TTL/10`（默认 1800/10 = 180 秒），所以实际回收发生在 1800~1980 秒之间。这一段在 [`router.py:76`](../../psrl/workers/env_service/router.py#L76) 和 [`router.py:84`](../../psrl/workers/env_service/router.py#L84)。
2. **顺带检查**：每次 `try_admit` / `heartbeat` / `release` 进来时先跑一次 `_reap`（[`ledger.py:127`](../../psrl/workers/env_service/ledger.py#L127)），所以只要有人在申请，过期租约就不会多停留。

所以"TTL 1800 秒"要理解成**上界**：最坏情况多占 1800 秒多一点，而不是精确在 1800 秒那一刻归还。

心跳本身很简单，就是刷新时间戳：

```189:201:psrl/psrl/workers/env_service/ledger.py
    def heartbeat(self, lease_id: str) -> bool:
        """
        Refresh a lease's TTL.

        Returns:
            bool: `False` when the lease is unknown, most likely because it was already
            reclaimed after a missed heartbeat window.
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            return False
        lease.last_seen = self._clock()
        return True
```

**为什么心跳周期必须远小于 TTL。** 这两者的关系是启动时强校验的：`heartbeat_interval_s` 必须 ≤ `slot_ttl_s` 的一半（[`config_check.py:49`](../../psrl/workers/env_service/config_check.py#L49)）。默认 60 秒 vs 1800 秒，留了 30 倍余量。原因是漏掉一次心跳不能有事——网络抖动、GC 停顿、事件循环短暂阻塞都可能让某次心跳迟到，只要下一个周期能补上，租约就不该被判死。

**心跳失败时会发生什么？** 这里有一个刻意的取舍（[`client.py:55`](../../psrl/workers/env_service/client.py#L55)）：如果心跳返回"这个租约已经不存在了"，客户端**只是打个 warning 继续跑**，不会去杀掉正在运行的 episode。理由是账本出错不该毁掉几小时的工作；代价是这一刻可能出现两个 episode 用同一张卡（旧的那个还在跑，新的已经被批准）。这是已知风险之一，见 [§10](#10-没有验证的部分与已知风险)。

**为什么阈值取 1800 秒。** 它要覆盖"一次正常的 env 调用"，又不能长到让死掉的 episode 占卡太久。演示脚本的第一个场景就是给你看这个量级：每隔 60 秒刷一次卡，跑 4 小时也没问题。

### 2.6 overlay：把"这次用哪几张卡"变成一份运行时文件

**是什么。** Docker Compose 允许用多个文件叠加（`-f base.yaml -f extra.yaml`），后者覆盖前者。Harbor 的任务镜像自带一份标准 compose 文件，里面写死了 GPU 段；overlay 就是**运行时生成的那份追加文件**，只写这次租约要改的部分。

生成的函数在 [`overlay.py:145`](../../psrl/workers/env_service/overlay.py#L145)，产物形状是：

```yaml
services:
  main:
    labels: ["psrl.env_lease=g0-1-5c1dd22f", "psrl.env_unit=g0"]
    devices: ["/dev/nvidia0:/dev/nvidia0", "/dev/nvidia1:/dev/nvidia1", "/dev/nvidiactl:/dev/nvidiactl", ...]
    volumes: ["/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro", "/usr/lib64/libcuda.so.535.1:/usr/lib64/libcuda.so.1:ro", ...]
    environment: ["LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64"]
```

**为什么需要它。** 旧的静态文件 `gpu-compose-override.yaml` 只能写死一个设备名和一套驱动库版本。它有三个连带问题，overlay 一次性解决：

| 旧写法的问题 | 具体后果 | overlay 怎么解决 |
|---|---|---|
| 写死 `/dev/nvidia0` | N 个并发 GPU episode 全挂到 0 号卡，1 号卡空着 | `devices` 由 `lease.gpu_indices` 生成，见 [`overlay.py:185`](../../psrl/workers/env_service/overlay.py#L185) |
| 写死驱动库版本（如 535） | 节点换了驱动版本后，容器加载到不匹配的 `libcuda`，报错难查 | 先读 `/proc/driver/nvidia/version`（[`overlay.py:76`](../../psrl/workers/env_service/overlay.py#L76)），再按版本号拼挂载路径 |
| 不标容器是谁的 | 残留容器没有任何线索能定位 | 打上 `psrl.env_lease` 标签，TTL 回收就靠它按标签删容器 |

**为什么文件要临时创建、用完删掉。** 每次申请的卡号、驱动库、租约 id 都不同，所以不能预先生成一份长期文件；而且如果按 `session_id` 命名留一堆文件，多机部署时目录会越积越多。实现用的是 `NamedTemporaryFile(delete=False)` 拿一个原子创建的唯一路径（[`overlay.py:197`](../../psrl/workers/env_service/overlay.py#L197)），episode 结束的 `finally` 里删掉（[`runner.py:521`](../../examples/sciaccel_rl/runner.py#L521)）。这里不能用 `tempfile.mktemp`，它只发明一个名字、不创建文件，在多进程环境里存在竞态。

**一个反直觉的地方**：overlay 里没有一行是"给这个容器几张卡"的算力限制。`devices` 只是把设备节点暴露进容器，能同时用多少 SM 由**驱动的时间片**决定，账本管不到。账本能管的只有"什么可以放进来"，管不了"进来之后各占多少算力"——所以准入时对显存求和（不可回收、会硬失败），对算力则不做限制（可被调度）。

### 2.7 Gate A 和 Gate B：两道闸门各管什么

| | Gate A | Gate B |
|---|---|---|
| 管什么 | 开始干活**之前**有没有位置 | 回复模型**之前**卡有没有算完 |
| 谁在等 | 新的 episode（还没有容器） | 已经在跑、马上要发起下一轮对话的 episode |
| 等在哪里 | 账本准入函数里轮询（[`client.py:156`](../../psrl/workers/env_service/client.py#L156)） | SessionRouter 的回复出口（[`session_router.py:120`](../../psrl/workers/gen/session_router.py#L120)） |
| 判据 | 容器数、CPU、内存、显存四条加起来放不放得下 | `nvidia-smi` 显示的利用率（共享卡只看利用率） |
| 超时 | 默认无限等（等待期间不占资源） | 默认 600 秒后放行 |

为什么"等待免费"这件事值得强调：Gate A 是在**容器还不存在**的时候等的。等到之前，这个 episode 对集群的消耗是零（除了一个 worker 协程），所以它可以等半小时也没关系。这与旧的进程信号量恰好相反——旧信号量是在已经占着名额等卡，所以会出现"等卡的 GPU episode 把 CPU episode 全挡住"的情况（这就是打开 env service 后会绕过它的原因，见 [§7](#7-怎么打开以及打开后行为有什么变化)）。

### 2.8 一条最短的因果链

把这几个词串起来，一个 GPU episode 的生命就是这两句话：

1. **申请**：读 `task.toml` 得到 **demand** → 向 router 申请 → router 在候选 **unit** 里找有空 **slot** 的 → 授予 **lease**（含卡号），并开始**心跳**。
2. **使用与归还**：按 lease 生成 **overlay** 起容器 → 正常结束时显式释放；异常结束时靠 **TTL** 兜底回收，残留容器由同节点按标签清理。

### 2.9 实验脚本：把这条链亲手跑一遍

```bash
cd psrl && PYTHONPATH=. python3 ../docs/psrl/scripts/lease_ttl_demo.py
```

脚本用假时钟驱动真实的 `EnvUnitLedger`，一秒内跑完 30 分钟的租约时间线，四个场景对应上面四个概念：

| 场景 | 演示的机制 | 对应小节 |
|---|---|---|
| 1 活着的 episode 每 60 秒心跳 | 心跳续期，TTL 不会误杀 | [§2.5](#25-心跳-heartbeat-和-ttl有效期的判定与回收的动作) |
| 2 episode 崩了、没有心跳 | TTL 到期后容量归还；容器待同节点按标签清理 | [§2.5](#25-心跳-heartbeat-和-ttl有效期的判定与回收的动作) |
| 3 正常退出时显式释放 | 立即归还，不必等满一个 TTL | [§2.4](#24-租约-lease一次成功申请的凭证) |
| 4 同一个 episode 申请两次 | 返回同一份租约，不重复占名额 | [§2.4](#24-租约-lease一次成功申请的凭证) |

场景 2 的实际输出（节选），能直观看到"判定条件"和"回收动作"是分开的：

```
  t=0      admit episode-A        -> lease g0-1-91ed4019
  t=0      ... episode-A crashes. From here on: no heartbeat, no release.
  t=   900  lease still held?       -> True
  t=  1620  lease still held?       -> True
  t=  1980  lease still held?       -> True      <- TTL(1800) 已过，但回收动作还没跑到
  t=  1981  reaper runs             -> reclaimed ['g0-1-91ed4019']
  t=  1981  episode-B asks again    -> GRANTED
  t=  1981  node collects the job    -> ['g0-1-91ed4019'] (stops by label)
  t=  1981  collect again           -> [] (delivered once only)
```

---

## 3. 一张图：一个 GPU episode 现在怎么走

```
 Driver (ray_trainer)
 │ _init_env_service：建 EnvRouter；算出"训推节点"集合；
 │ 按 node_policy 和"哪些节点有 unit"挑出能放 agent worker 的节点
 ▼
 AgentLoopManager ──每 2s 拉──► EnvRouter.node_capacity()   各节点剩余容器槽
 │ get_dispatch_plan：每个 prompt 组 → 一个 worker（该 worker 的节点有空位）
 ▼
 AgentLoopWorker（节点 X）
 │ demand_from_task：读 task.toml 的 cpus / memory_mb / gpus
 │
 │ ① Gate A：EnvServiceClient.acquire(session, demand)
 │      nvidia-smi 读本节点各卡显存 ──► router.try_admit(demand, node=X, measured)
 │      wait → 退避轮询（此刻没有容器，等多久都不占资源）
 │      granted → EnvLease{unit, gpu_indices, shared}，开始心跳
 │ ② _apply_env_lease：
 │      /proc/driver/nvidia/version → 精确版本的 libcuda 等 → 每租约 overlay
 │      override_gpus=0；POST /control/gpu_bind {devices, shared}
 │ ③ Job.create → docker compose up   容器挂的就是该 unit 的卡，带 psrl.env_lease 标签
 │
 │      每轮模型回复 ──► SessionRouter ──► Gate B（共享卡只看利用率，多卡要全空闲）
 │
 │ ④ finally：force_remove_compose_project（容器确认已删）
 │      → 解绑 Gate B → lease.release() → 删 overlay
 ▼
 EnvRouter 账本：容量回到 unit；若 worker 崩了没释放，TTL 到期由 reaper 回收，
 该节点下一次准入时按标签补删残留容器
```

## 4. 图例：每个部分对应的源码位置

| 图上 | 内容 | 源码 |
|---|---|---|
| 建 router、节点集合 | `_init_env_service` | [`ray_trainer.py:549`](../../psrl/trainer/ppo/ray_trainer.py#L549) |
| 筛选 worker 节点 | `select_worker_nodes` + `restrict_to_unit_nodes` | [`ray_trainer.py:1907`](../../psrl/trainer/ppo/ray_trainer.py#L1907)、[`node_policy.py`](../../psrl/workers/env_service/node_policy.py) |
| 派发 | `get_dispatch_plan` / `_refresh_env_capacity` | [`manager.py:947`](../../psrl/workers/agent_loop/manager.py#L947)、[`:909`](../../psrl/workers/agent_loop/manager.py#L909) |
| demand | `demand_from_task` | [`env_demand.py:23`](../../examples/sciaccel_rl/env_demand.py#L23) |
| ① Gate A（客户端） | `EnvServiceClient.acquire` | [`client.py:156`](../../psrl/workers/env_service/client.py#L156) |
| ① Gate A（账本） | `EnvUnitLedger.try_admit` | [`ledger.py:102`](../../psrl/workers/env_service/ledger.py#L102) |
| ① 接入 runner | env_client 分支 | [`runner.py:305-321`](../../examples/sciaccel_rl/runner.py#L305) |
| ② overlay | `resolve_driver_libs` / `build_lease_overlay` | [`overlay.py:99`](../../psrl/workers/env_service/overlay.py#L99)、[`:145`](../../psrl/workers/env_service/overlay.py#L145) |
| ② 实现租约 | `_apply_env_lease` | [`runner.py:84`](../../examples/sciaccel_rl/runner.py#L84) |
| Gate B | `_wait_for_gpu_idle` / `_parse_gpu_indices` | [`session_router.py:120`](../../psrl/workers/gen/session_router.py#L120)、[`:110`](../../psrl/workers/gen/session_router.py#L110) |
| ④ 释放 | `finally` 末尾 | [`runner.py:514-522`](../../examples/sciaccel_rl/runner.py#L514) |
| TTL 回收 | `_reap` / `pop_reaped` / `reap_local` | [`ledger.py:231`](../../psrl/workers/env_service/ledger.py#L231)、[`:221`](../../psrl/workers/env_service/ledger.py#L221)、[`client.py:213`](../../psrl/workers/env_service/client.py#L213) |
| 启动校验 | `validate_env_service_config` | [`config_check.py:15`](../../psrl/workers/env_service/config_check.py#L15)，由 [`config.py:490`](../../psrl/utils/config.py#L490) 调用 |

这些代码如果有理解上的疑问，可以随时追问。

## 5. 一卡多容器：账本怎么算

### 5.1 规则

一个 unit 独占若干整卡，**它的所有容器共享这些卡**。准入要同时满足四条（任何一条超了就等）：

| 资源 | 判定 | 为什么 |
|---|---|---|
| 容器数 | 已有租约数 + 1 ≤ `max_containers` | 配置出来的上限，不是推导的 |
| CPU | 已用 + 本次 ≤ `cpu_quota` | 容器共享这个 unit 的 CPU |
| 主机内存 | 已用 + 本次 ≤ `memory_mb` | 同上 |
| 设备显存 | **有效占用** + 本次 ≤ `gpu_memory_mb` | 显存相加且无法靠调度回收（设计 §5.3） |

```342:352:psrl/psrl/workers/env_service/ledger.py
    def _fits_now(self, unit: EnvUnit, demand: EnvDemand, now: float) -> bool:
        leases = self._unit_leases(unit.unit_id)
        if len(leases) + 1 > unit.max_containers:
            return False
        if sum(lease.demand.cpu for lease in leases) + demand.cpu > unit.cpu_quota + _EPS:
            return False
        if sum(lease.demand.memory_mb for lease in leases) + demand.memory_mb > unit.memory_mb:
            return False
        return not (
            demand.needs_gpu and self._effective_gpu_mb(unit, now) + demand.gpu_memory_mb > unit.gpu_memory_mb + _EPS
        )
```

### 5.2 "有效占用"取声明和实测的较大者

```333:334:psrl/psrl/workers/env_service/ledger.py
        measured = self._measured_gpu_mb(unit, now)
        return max(float(self._declared_gpu_mb(unit)), measured if measured is not None else 0.0)
```

两个方向各有一个反例，所以取较大者：

- **实测比声明大**：邻居（某个后台 solver、甚至 vLLM）占了没人声明的显存。只信声明会超订。
- **实测比声明小**：一个容器刚声明了 24 GB、还在加载模型，只用了 1 GB。只信实测，等它加载完就超订了。

实测值超过 `measurement_ttl_s`（60 秒）就作废，否则一个过期的高读数能把 unit 永远堵死。实测要覆盖 unit 的**每一张**卡才算数，缺一张就当未知，不拿部分和去猜。

### 5.3 手算一个例子

教学数字，和 `env_service_demo.py` 的输出一致：一张 80 GB 的卡，换算成 81920 MB；每个 GPU 任务记 24 GB，即 24576 MB。

| 已有容器数 | 已声明显存 | 再来一个 24576 MB 后 | 判定 |
|---|---|---|---|
| 0 | 0 | 24576 | 放行 |
| 1 | 24576 | 49152 | 放行 |
| 2 | 49152 | 73728 | 放行 |
| 3 | 73728 | 98304 > 81920 | **等待** |

所以 `max_containers` 配成 3 时，显存和容器数两条限制正好同时到顶。如果 `max_containers` 配成 2，则容器数先到顶，显存只用了 49152 MB。反之把 `max_containers` 配成 10，也只会放进 3 个，因为显存先到顶。

此时如果有人跑 `nvidia-smi` 发现这张卡已经用了 76 GB，而账本里一个租约都没有（别的进程占的），新来的 24 GB 任务会得到 `wait`，演示脚本里这一条也成立。

### 5.4 其他几条规则

- **同一个 session 重复申请返回同一份租约**，所以丢了回复后的重试不会重复扣账。
- **`reject` 和 `wait` 分开**：任务大到连空 unit 也装不下，或者没有任何 unit 允许承载它，会立刻 `reject`（episode 以 `env_admission_failed` 结束，不启动 Harbor）；只有"现在满了"才 `wait`。否则一个永远放不下的任务会无限排队。
- **CPU episode 优先放 CPU-only 的 unit**，即使 GPU unit 看起来更空，因为那会占掉 GPU unit 的容器槽、可能让真正要卡的 episode 白等。没有 CPU unit 有空位时才退到 GPU unit。这一条是我写测试时发现的：按负载比最小来选，CPU episode 会漂到 GPU unit 上，先写了测试确认它在修复前失败，再改的。
- **负载按"最紧的那个资源"算**，不是容器个数（设计 §4.5：env 调用从几秒到几小时，个数是糟糕的负载代理）。

## 6. 训推节点策略：容器要不要和训练、推理共用节点

### 6.1 为什么需要它

Harbor 容器对 Ray 不可见，Ray 的 GPU 记账也只是 `CUDA_VISIBLE_DEVICES` 上的约定，没有强制。vLLM 在 `gpu_memory_utilization=0.85` 的 96 GB 卡上只剩约 14 GB，一次碰撞是间歇性的静默 OOM，不是干净的失败（设计附录 12.1）。所以容器能不能出现在训练或 rollout 的节点上，是个必须能配置的开关。

容器必然落在**启动它的 agent worker 所在节点**（Harbor 调的是该 worker pod 里的嵌套 `dockerd`），所以策略在两处起作用：**挑 worker 放哪些节点**，和 **router 能给哪些 unit 发租约**。

### 6.2 四档

| `node_policy` | 训推节点上的 CPU episode | 训推节点上的 GPU episode | 对 agent worker 放置的影响 |
|---|---|---|---|
| `any` | 允许 | 允许 | 不限制 |
| `cpu_only`（**默认**） | 允许 | **拒绝** | 不限制 |
| `avoid` | **拒绝** | **拒绝** | 训推节点上不放 worker |
| `prefer_other` | 允许，但排在后面 | 允许，但排在后面 | 有别的节点就只用别的节点，没有才回退到训推节点 |

演示脚本在"worker 在训推节点上"的假设下，对同一组 unit 跑出的结果（实测输出）：

```
  policy        CPU episode, hint=train node      GPU episode, hint=train node
  any           granted -> train-cpu              granted -> train-gpu
  cpu_only      granted -> train-cpu              REJECT
  avoid         REJECT                            REJECT
  prefer_other  granted -> train-cpu              granted -> train-gpu
```

`prefer_other` 在不指定节点时，会先用非训推节点的 unit（第 0 档），只有它们都满了才溢出到训推节点的 unit（第 1 档）。

### 6.3 "训推节点"怎么确定

`train_rollout_node_ips` 配了就用配置的；留空就在启动时从各 worker group 的 **placement group 元数据**里读（不调用 actor，所以 worker 还没初始化完也能读），**跳过 PS 组**，因为它是纯 CPU 存储。逻辑在 [`node_ips_from_worker_groups`](../../psrl/workers/env_service/node_policy.py#L220)，已用假 placement 表测过"PS 组被忽略"。

### 6.4 两个容易忽略的约束

1. **只在声明了 unit 的节点上放 agent worker**（`restrict_to_unit_nodes`）。否则落在没有 unit 的节点上的 worker 会让每个 episode 都被拒绝，manager 又会不断补发失败的组。没有任何节点有 unit 时启动直接报错，而不是跑起来再失败。
2. **GPU unit 声明在训推节点上会打警告**（`lint_units`）。`cpu_only` 下它只能承载 CPU episode，`any` 下则明确警告"没有东西阻止容器打开 vLLM 正在用的卡"。它是警告不是错误，因为也可能是 `train_rollout_node_ips` 写错了。

## 7. 怎么打开，以及打开后行为有什么变化

最小配置（教学数字；节点 IP 和显存按你的集群改）：

```yaml
psrl:
  env_service:
    enable: true
    node_policy: cpu_only          # 或 avoid（需要一台专属 env 节点）
    default_gpu_memory_mb: 24000
    units:
      - {unit_id: env-gpu, node_ip: 10.0.0.1, cpu_quota: 16, memory_mb: 65536,
         gpu_indices: [0, 1], max_containers: 6, gpu_memory_mb: 160000}
      - {unit_id: env-cpu, node_ip: 10.0.0.1, cpu_quota: 32, memory_mb: 131072, max_containers: 8}
```

启动时 [`validate_env_service_config`](../../psrl/workers/env_service/config_check.py#L15) 会拦下这些错：没有 unit、unit 字段缺失（报第几个、缺哪个键）、两个 unit 抢同一张卡、心跳间隔超过 TTL 的一半（否则活的租约会在两次心跳之间被回收）、GPU unit 小到连一个默认任务都放不下。

**打开后和关闭时的差别**（这些是行为变化，不是纯加法）：

| 方面 | 关闭 | 打开 |
|---|---|---|
| 并发上限 | 进程级信号量 `max_concurrent_episodes` | **绕过该信号量**，由 router 的按 unit 上限接管（节点级）。保留它会让等卡的 GPU episode 占着名额挡住 CPU episode |
| GPU 分配 | `GpuDeviceGate`：一个 episode 一整张卡 | unit：多个容器共享卡 |
| compose override | 静态 `gpu-compose-override.yaml`（写死 535 版本与 `/dev/nvidia0`） | 每租约生成，驱动库按内核版本解析 |
| 派发 | 轮转 | 按节点剩余容量；快照过期（>10 秒）或 router 不可达时**自动退回轮转** |
| agent worker 位置 | 所有存活节点（或 `agent.node_ips`） | 再按 `node_policy` 与"有无 unit"过滤，并改用硬亲和 |
| 准入失败 | 无此概念 | episode 以 `env_admission_failed` / `env_lease_setup_failed` 结束，**不启动 Harbor**，租约会释放 |

## 8. 没有实现的部分

逐条对照设计提案，没做的和原因：

- **借还协议**（unit 状态机 `FREE → LEASE_PENDING → ENV → RETURNING → FREE`，借卡/还卡时用 `nvidia-smi --query-compute-apps` 验证卡已空）。当前 unit 是**静态声明**的，不从 rollout/训练池里借卡。这与设计 §11 的推荐一致：从活跃的训推池借卡风险高（env 容器可能占卡 4 小时不能抢占），推荐加一台专属 env 节点，用 `node_policy: avoid` 让容器根本不碰训推节点。
- **Phase 0**：把 SciAccel 的 Harbor 调用搬到 `EnvWorker` actor 后面、删掉 Path A 的 `shell.py` 和 agent loop 里的 Harbor 线程。**没有动 `psrl/workers/env_worker/`**，MLGym/AIRS-Bench 路径完全不受影响。设计里的关口（证明 terminus-2 的 tmux 能满足 MLGym）也没有验证。
- **每个 unit 一个 `EnvWorker` actor、容器生命周期归它管**：没有做。容器仍由 agent loop worker 里的 Harbor 起。负载快照由 router 账本（声明值）加 worker 随请求带来的实测显存拼出来，不是各 unit 的独立周期上报。
- **Gate B 的判据**：设计说 Gate B 管"并发活跃的 env 调用数"。这里延续上个提交的做法，仍用整卡 `utilization.gpu`/`memory.used`，只是加了多设备和"共享卡只看利用率"。没有改成按进程的 `pmon`。
- **注册时对 unit 做健康探测**（取代手工维护的 `agent.node_ips`）：没做。router 不探测 Docker 守护进程是否卡死，仍靠 `deny_node_ips` 手工排除。
- **L2（替换 Harbor 的 Environment、容器复用）、L3（远程代理）**：没做，设计也建议先停在 L1。
- **Phase 3**（负载感知放置、带尝试次数上限的重新派发）、**B8**（`retry_limit` 默认 1）：没做，没动。
- **B5**（Harbor verifier 容器拆除不可靠）：没有新增处理。新路径只保证"租约在 `force_remove_compose_project` 之后才释放"，沿用已有的清理链。
- 原 `config.harbor.override_gpus` 这个"死配置"字段**没有接线**：新路径在有 GPU 租约时直接把 `env_kwargs["override_gpus"] = 0`，没有读该字段。

## 9. 测试与演示的实测输出

我在本机运行的结果（没有 pytest，所以测试写成既能被 pytest 收集、也能 `python` 直接跑的纯断言形式）：

```
test_ledger                             36 passed, 0 failed
test_node_policy_overlay_placement      33 passed, 0 failed
test_router_client                      12 passed, 0 failed
test_gate_b                             11 passed, 0 failed
test_runner_env_service                 13 passed, 0 failed
test_manager_dispatch                   13 passed, 0 failed
test_config_and_demand                  18 passed, 0 failed
```

共 136 个。几项值得说明的验证手段：

- **差分测试**：`test_manager_dispatch.py` 里放了一份旧 `get_dispatch_plan` 的逐字副本，随机生成 200 组（worker 数、组大小、起点都随机）参数，要求新代码在 env service 关闭时的输出与旧算法完全一致。
- **性质测试**：`test_ledger.py` 对账本做 3000 步随机的申请/释放/心跳/回收/实测，每步都从租约表重新推导各 unit 的用量，断言没有任何 unit 超过容器数、CPU、内存、显存上限，且 GPU 需求从不落到 CPU unit、`cpu_only` 下从不落到训练节点。分别在 `prefer_other` 和 `cpu_only` 下各跑一遍。
- **变异检查**：我故意把 `ledger.py` 改坏三处，看测试能不能抓到：去掉内存检查（3 个测试失败）、容器上限差一（6 个）、显存忽略声明值（5 个）。三处都被抓到，文件随后已还原。
- **既有测试回归**：仓库原有的 `tests/examples/sciaccel_rl/test_runner.py` 在 Harbor 打桩的环境下 22 过 1 败。那 1 个是 `test_matches_harbors_own_sanitizer`，它要导入真实的 `harbor.environments.docker.docker`，本机没有 Harbor，桩里没有这个模块；`docker_utils.py` 我没有改，所以这与本次改动无关，但它在真实环境下是否通过我无法证明。

演示脚本（`PYTHONPATH=. python3 examples/sciaccel_rl/diagnostics/env_service_demo.py`）的输出：

```
1. 2 card(s) of 80 GB, each GPU task needs 24 GB, 10 episodes arrive at once
  exclusive cards (GpuDeviceGate): 2 episodes run, 8 wait
  shared units (env service):      6 episodes run, 4 wait
    card0: [###] 3/3 containers, 72/80 GB declared
    card1: [###] 3/3 containers, 72/80 GB declared
  a card whose nvidia-smi shows 76 GB already in use (nobody declared it): WAIT OK
...
RESULT: all expectations held
```

同样 2 张卡、10 个并发 episode：整卡独占只能同时跑 2 个，一卡多容器能同时跑 6 个。

## 10. 没有验证的部分与已知风险

**没有验证（本机缺依赖）：**

- **没有真实运行过任何东西**：没有 Ray、Harbor、torch、GPU、Docker。真实训练、真实容器启动、真实 `nvidia-smi` 输出、真实 `/proc/driver/nvidia/version` 与驱动库都没跑过。
- **`ray_trainer.py` 的接入（`_init_env_service`、节点筛选、传参给 manager）只做了语法检查**，没有运行。它依赖 Ray 的 placement group 与节点表，这部分逻辑我只是读代码写出来的。
- **`EnvRouter` 作为真实 Ray actor 没有跑过**。测试里用本地适配器（`handle.method.remote(...)` 变成直接的协程调用）代替 Ray 句柄，所以"Ray 对 `Admission`/`EnvLease` 这些 dataclass 的序列化""在 Harbor 线程的事件循环里 await Ray 的返回值"都没有被真实验证。
- **Harbor 的行为按设计提案转述**：`extra_docker_compose` 叠加在任务自己的 compose 之后；`override_gpus: 0` 能绕过 GPU 校验；多个 compose 文件里的 `labels` 如何合并（新 overlay 与 runner 里已有的 `psrl.actor_id` overlay 各写一份 `labels`）。Harbor 源码不在工作区，我没读过。
- **整套逻辑在 Python 3.14 下跑的**，仓库的目标版本我没有核对。

**已知风险，打开前应该知道：**

1. **打开会绕过 `max_concurrent_episodes`**。并发上限从"每进程 N 个"变成"每 unit 的 `max_containers`"，如果 unit 配得宽，容器总数可能比以前多；该配置值需要按节点容量重新估。
2. **心跳跑在 Harbor 事件循环线程上**，如果那条循环被阻塞很久，租约可能因漏心跳被回收。默认 TTL 1800 秒、心跳 60 秒留了很大余量，但没有压测。
3. **TTL 回收会让账本与现实暂时不一致**：router 立即释放容量，而残留容器要等"该节点的下一次准入"才被清理。这段时间卡上可能仍有旧容器；实测显存取大值的规则会在一定程度上兜住（残留容器的显存会被读到），但这是推断，没有演示过。
4. **Gate B 的"共享卡只看利用率"是个取舍**：它放过了"邻居的模型权重常驻显存"这种情况，但也意味着显存泄漏类的异常不会再被 Gate B 的显存阈值拦住。
5. **`nvidia-smi` 在 worker 所在节点读取，Gate B 的 `nvidia-smi` 在 SessionRouter 所在节点读取**：这是[上个提交讲解 §7 第 3 条](commit-72ca50c-gate-ab-walkthrough.md#7-从代码读出的-6-处注意点推断)指出的跨节点问题，这次没有解决，只是 Gate A 这一侧不再受它影响。多节点部署时 Gate B 仍可能读到别的机器上的卡。

## 相关文档

- Env Service 设计文档（译文）（本地资料，未随仓库发布）：这次实现所依据的提案；译文索引里的"现状核对表"现已更新。
- [提交 72ca50c 代码讲解](commit-72ca50c-gate-ab-walkthrough.md)：上一个提交只做了 Gate A 的整卡互斥与 Gate B，那份文档对**那个提交**的结论依然成立。
- [SciAccel GPU Gate](sciaccel-gpu-gate.md)：整卡互斥 `GpuDeviceGate` 的背景与验证，是关闭 env service 时仍在用的路径。
- [PSRL 笔记总索引](../README.md)
