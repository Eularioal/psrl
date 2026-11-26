# Env历史：72ca50c版本GateAB机制

**这个提交解决了“多个任务同时拿到同一张卡”的问题。** Gate A 是任务开始前的设备闸门：用跨进程文件锁把一张 GPU 暂时交给一个任务。Gate B 是模型回复出口的软节流：卡忙时先等一会儿，达到等待上限后继续返回回复。

例如机器有 3 张卡，4 个任务同时到达。前三个各拿一张，第 4 个等前面的任务结束、容器清理完成，再取得归还的卡。这里的一次完整任务尝试叫 **episode**；锁覆盖整个 episode，包括它等待模型和执行 CPU 工作的时间。

本文固定分析提交 `72ca50c9813276bf7f2fbc714ac2fe2a6d0cd886`（2026-09-29，8 个文件，新增 1068 行、删除 3 行）。改动在 science / Harbor 执行路径；EnvRouter 的共享显存账本属于后续实现。现状见[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)，从整卡独占到一卡多容器的早期变化见[Env Service 实现](02-Env历史_2026年10月12日资源单元与租约.md)。

当次验证包括设备字符串上的并发诊断；真实 Docker、GPU 和跨节点行为的验证范围集中列在[第 10 节](#10-验证范围与复现条件)。源码摘录和标签中的行号属于固定提交，本地文件链接用于查找同名模块。可在仓库根目录执行 `git show 72ca50c:examples/sciaccel_rl/runner.py` 查看当时的完整文件。

## 目录

- [1. 先回答三个问题](#1-先回答三个问题)
- [2. 一张图：一个 GPU episode 经过哪些门](#2-一张图一个-gpu-episode-经过哪些门)
- [3. 图例：每个部分对应的源码位置](#3-图例每个部分对应的源码位置)
- [4. Gate A 代码逐段讲解](#4-gate-a-代码逐段讲解)
  - [4.1 要解决的问题](#41-要解决的问题)
  - [4.2 抢卡：acquire](#42-抢卡acquire)
  - [4.3 SlotManager 抢锁循环](#43-slotmanager-抢锁循环)
  - [4.4 设备发现与"哪些设备算可独占"](#44-设备发现与哪些设备算可独占)
  - [4.5 生成"只挂这张卡"的 override：build_gpu_override](#45-生成只挂这张卡的-overridebuild_gpu_override)
  - [4.6 接入 run_harbor_episode：拿卡、降级、释放](#46-接入-run_harbor_episode拿卡降级释放)
- [5. Gate B 代码逐段讲解](#5-gate-b-代码逐段讲解)
  - [5.1 它想拦的是什么](#51-它想拦的是什么)
  - [5.2 三步：绑定、等待、解绑](#52-三步绑定等待解绑)
  - [5.3 用一个例子串起来](#53-用一个例子串起来)
- [6. 与设计提案的能力对照](#6-与设计提案的能力对照)
- [7. 从代码读出的 6 处注意点（推断）](#7-从代码读出的-6-处注意点推断)
- [8. 跟着图读源码：执行路径](#8-跟着图读源码执行路径)
- [9. 可运行实验与实测输出](#9-可运行实验与实测输出)
- [10. 验证范围与复现条件](#10-验证范围与复现条件)
- [相关文档](#相关文档)

---

## 1. 先回答三个问题

<!-- figure:reply-gate-vs-device-lock -->
这张图沿同一个场景解释：设备准入与回复等待为什么会分别增加时间？

[![两道门：占卡互斥与回复节流控制不同的位置（1/2）](../../_static/img/document-figures/reply-gate-vs-device-lock-screen.png)](../../_static/img/document-figures/reply-gate-vs-device-lock.png)

[![两道门：占卡互斥与回复节流控制不同的位置（2/2）](../../_static/img/document-figures/reply-gate-vs-device-lock-screen-02.png)](../../_static/img/document-figures/reply-gate-vs-device-lock.png)

历史机制例子：GPU 工作已完成，但控制条件尚未满足，调用方还没收到回复。 计算结束与下游能继续是不同事件。控制路径也可能拉长 rollout 的总完成时间。 历史路径阅读图。Gate B 等待不自动表示 vLLM 已睡眠、卸载或释放显存。 [查看矢量图](../../_static/img/document-figures/reply-gate-vs-device-lock.svg)。
<!-- /figure:reply-gate-vs-device-lock -->

| 问题 | 答案 | 证据 |
|---|---|---|
| 实现了 Gate A、Gate B 吗？ | **是**，这就是这个提交的全部内容 | `gpu_device_gate.py` 新增 194 行；`session_router.py` 新增 175 行；`runner.py` 改动 78 行 |
| 改了 EnvWorker 吗？ | **没有**。只新增了 `tests/env_worker/test_worker_gpu_integration.py`（159 行，3 个测试），测的是**已有**的 `EnvWorker._allocate_gpu_indices` 与 `sandbox.build_gpu_argv` | `git show --stat` 的 8 个文件里没有 `psrl/workers/env_worker/` 下的任何文件 |
| 实现了 EnvRouter 吗？ | 这个提交使用整卡文件锁；共享资源 Router 是后续实现 | 固定提交的文件清单与 `GpuDeviceGate.acquire` |

提交说明中的 “env worker session routing” 指此次 science 侧的 GPU 闸门与模型会话绑定。逐文件 diff 显示，既有 EnvWorker 的生产实现保持原样，新增集成测试用于检查其原有 GPU 路径。

## 2. 一张图：一个 GPU episode 经过哪些门

```
AgentLoopWorker 进程（节点 X，Ray Actor）
│
│ ① episode 计数门：asyncio.Semaphore（原有，只数"几个"，不认设备）
│        agent_loop.py:349
▼
run_harbor_episode()                                   ← runner.py
│
│ ② Gate A：GpuDeviceGate.acquire()   ★ 本提交新增
│     discover_gpu_devices() → ["/dev/nvidia0", "/dev/nvidia1", ...]
│     SlotManager.acquire(len(devices), jobs_dir)  → (fd, slot_index)
│     device = devices[slot_index]                  ← 这张卡现在归你独占
│     build_gpu_override(静态文件, device)          ← 现场生成"只挂这张卡"的 compose override
│
│ ③ 告诉 SessionRouter 这个 session 在用哪张卡
│     POST {session_router_url}/control/gpu_bind  [{session_id, gpu_device}]
│
│ ④ Harbor Job.create → docker compose up（容器看见的就是这张卡）
│        │
│        │ agent 多轮：每一轮都要问模型
│        ▼
│   SessionRouter（独立 uvicorn 子进程，在 RolloutGateway 所在节点 Y）
│        │  拿到 vLLM 的回复后，返回给 agent 之前：
│        │ ⑤ Gate B：_wait_for_gpu_idle(device)   ★ 本提交新增
│        │     每 2s 用 nvidia-smi 读 util / mem；util≤5% 且 mem≤256MB 才放行；最多等 600s，超时照样放行
│        ▼
│   agent 拿到回复，在容器里执行命令……
│
│ ⑥ finally：force_remove_compose_project（容器确认删掉）
│     → GpuDeviceGate.release()  → 解绑 gpu_bind(None) → 删临时 override 文件
▼
```

图里 ①②③⑥ 在**节点 X**（容器所在的 agent worker 节点），⑤ 在**节点 Y**（SessionRouter 所在节点）。X 和 Y 是否同一台机器，取决于部署，见 §7 第 3 条。

## 3. 图例：每个部分对应的源码位置

| 图上 | 内容 | 源码 |
|---|---|---|
| ① | 进程内计数信号量（原有，未改动逻辑） | [`agent_loop.py:349`](../../../examples/sciaccel_rl/agent_loop.py) |
| ② 发现设备 | `PSRL_SCIACCEL_GPU_DEVICES` 覆盖，否则 glob `/dev/nvidia[0-9]*` | [`gpu_device_gate.py:69`](../../../psrl/utils/concurrency/gpu_device_gate.py) |
| ② 抢卡 | `GpuDeviceGate.acquire` | [`gpu_device_gate.py:117-147`](../../../psrl/utils/concurrency/gpu_device_gate.py) |
| ② 文件锁 | `SlotManager.acquire`（`flock` 非阻塞 + 0.2s 轮询） | [`slot.py:29-73`](../../../psrl/utils/concurrency/slot.py) |
| ② 生成 override | `build_gpu_override` | [`gpu_device_gate.py:158-194`](../../../psrl/utils/concurrency/gpu_device_gate.py) |
| ②③ 接入点 | `run_harbor_episode` 里的 Gate A 块 | [`runner.py:230-262`](../../../examples/sciaccel_rl/runner.py) |
| ③ 绑定调用 | `_bind_gate_b`（尽力而为，永不抛错） | [`runner.py:41-68`](../../../examples/sciaccel_rl/runner.py) |
| ③ 传参 | agent_loop 把 `session_router_url` 传给 runner | [`agent_loop.py:358`](../../../examples/sciaccel_rl/agent_loop.py) |
| ③ 绑定端点 | `POST /control/gpu_bind` | [`session_router.py:272`](../../../psrl/workers/gen/session_router.py)（注册）、[`:535-572`](../../../psrl/workers/gen/session_router.py)（实现） |
| ③ 状态字段 | `SessionState.gpu_device` | [`session_router.py:210`](../../../psrl/workers/gen/session_router.py) |
| ⑤ 出口钩子 | 回复返回前的 Gate B | [`session_router.py:425-433`](../../../psrl/workers/gen/session_router.py) |
| ⑤ 等空闲 | `_wait_for_gpu_idle` | [`session_router.py:120-172`](../../../psrl/workers/gen/session_router.py) |
| ⑤ 读遥测 | `_read_gpu_utilization`（`nvidia-smi`，1s 缓存，失败即"全空闲"） | [`session_router.py:64-107`](../../../psrl/workers/gen/session_router.py) |
| ⑥ 释放 | `finally` 末尾 | [`runner.py:402-406`](../../../examples/sciaccel_rl/runner.py) |
| 测试 | 3 个 EnvWorker GPU 集成测试 | [`test_worker_gpu_integration.py:63,90,131`](../../../tests/env_worker/test_worker_gpu_integration.py) |

这些代码如果有理解上的疑问，可以随时追问。

## 4. Gate A 代码逐段讲解

### 4.1 要解决的问题

提交前，每个需要 GPU 的 episode 都 `append` 同一份静态 override 文件，里面写死 `/dev/nvidia0`。唯一的准入控制是进程内信号量，它只数"几个 episode"，不认"哪张卡"。所以 4 个并发 GPU episode 会同时拿到同一个设备字符串。`gate_a_before.py` 复现的就是这个：见 [§9](#9-可运行实验与实测输出)。

### 4.2 抢卡：`acquire`

```python
# 历史摘录：提交 72ca50c，psrl/utils/concurrency/gpu_device_gate.py
        slot = await SlotManager.acquire(len(devices), self.jobs_dir, prefix=self.slot_prefix)
        if slot is None:
            # discover_gpu_devices() returned non-empty, so max_slots > 0;
            # SlotManager.acquire only returns None when max_slots <= 0.
            # Defensive: should be unreachable.
            return None

        _fd, slot_index = slot
        device = devices[slot_index]
        logger.info("Gate A: reserved %s (slot %d/%d).", device, slot_index, len(devices))
        return AllocatedGpu(device=device, index=slot_index, _slot=slot)
```

核心思想是一句话：**"第几个锁文件抢到了" = "第几张卡归你"**。`SlotManager` 本来是个"限制并发数"的原语，这里把它的 `slot_index` 直接当设备列表的下标用。

手算（教学例子）：这台机器有 3 张卡，`devices = [nvidia0, nvidia1, nvidia2]`，`SlotManager.acquire(3, ...)` 会依次尝试锁 `slot_0.lock`、`slot_1.lock`、`slot_2.lock`：

| 时刻 | 动作 | 结果 |
|---|---|---|
| t0 | episode A 抢 | `slot_0` 空闲 → 拿到，`device = nvidia0` |
| t0 | episode B 抢 | `slot_0` 被锁 → 试 `slot_1`，拿到，`nvidia1` |
| t0 | episode C、D 抢 | C 拿到 `nvidia2`；D 三个都被锁 → 每 0.2 秒再扫一遍 |
| t1 | A 释放 | D 下一轮扫描拿到 `slot_0` → `nvidia0` |

### 4.3 `SlotManager` 抢锁循环

```python
# 历史摘录：提交 72ca50c，psrl/utils/concurrency/slot.py
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

`flock(LOCK_EX | LOCK_NB)` 是操作系统级的互斥文件锁，"NB"表示拿不到就立刻抛 `BlockingIOError` 而不是阻塞。它跨进程生效，这正是选它而不选 `asyncio.Semaphore` 的原因：同一台节点上可能有多个 `AgentLoopWorker` **独立进程**，互相看不到对方的 Python 内存。背景原理（advisory 锁、为什么进程崩溃锁会自动释放）见 [SciAccel GPU Gate §19](04-Env历史_Science整卡锁与设备账本.md#19-追加问题2026-10-08fcntl-文件锁背景知识与-gate-a-的分配粒度)，这里不重复。

锁目录由 `jobs_dir` 的路径哈希决定：`/tmp/{prefix}_{sha1(abspath(jobs_dir))[:12]}`（[`slot.py:25-26`](../../../psrl/utils/concurrency/slot.py)），所以**同一个 `jobs_dir` 的所有进程共用一个池**。默认 `jobs_dir` 是 `/tmp/sciaccel_jobs`（[`config.py:22`](../../../examples/sciaccel_rl/config.py)）。

### 4.4 设备发现与"哪些设备算可独占"

[`is_exclusive_gpu_device`](../../../psrl/utils/concurrency/gpu_device_gate.py) 只认 `/dev/nvidia` 后面**全是数字**的路径（`/dev/nvidia0`、`/dev/nvidia12`）。`/dev/nvidiactl`、`/dev/nvidia-uvm`、`/dev/nvidia-uvm-tools` 是整机共用的控制设备，不对应某一张卡，可以原样挂进每个容器。

[`discover_gpu_devices`](../../../psrl/utils/concurrency/gpu_device_gate.py) 优先读环境变量 `PSRL_SCIACCEL_GPU_DEVICES`（逗号分隔），否则 glob `/dev/nvidia[0-9]*` 并按数字排序。环境变量的作用是让没有真 GPU 的机器（开发机、CI、诊断脚本）也能跑通这套逻辑，设备只是字符串。

### 4.5 生成"只挂这张卡"的 override：`build_gpu_override`

```python
# 历史摘录：提交 72ca50c，psrl/utils/concurrency/gpu_device_gate.py
    base = yaml.safe_load(Path(base_override_path).read_text()) or {}
    main = dict(base.get("services", {}).get("main", {}))

    rewritten = []
    for entry in main.get("devices", []):
        host_path = str(entry).split(":", 1)[0]
        rewritten.append(f"{device}:{device}" if is_exclusive_gpu_device(host_path) else entry)
    main["devices"] = rewritten or [f"{device}:{device}"]

    override_file = Path(tempfile.mktemp(suffix=".yaml", prefix="sciaccel-gate-a-"))
    override_file.write_text(yaml.dump({"services": {"main": main}}))
```

做法：读静态 [`gpu-compose-override.yaml`](../../../examples/sciaccel_rl/config/gpu-compose-override.yaml)，**只改 `services.main.devices`** 这一项：独占设备那一行（`/dev/nvidia0:/dev/nvidia0`）换成预留的设备，`nvidiactl`/`nvidia-uvm*` 原样保留；其余键（驱动库挂载、环境变量等）一概不动。

注意：这意味着静态文件里写死的驱动库挂载（文件名里带 `535.247.01`）**依然生效**，这个提交没有去解决"驱动库版本写死"的问题，见 [§6](#6-与设计提案对照做了什么没做什么)。

### 4.6 接入 `run_harbor_episode`：拿卡、降级、释放

```python
# 历史摘录：提交 72ca50c，examples/sciaccel_rl/runner.py
        gpu_gate = GpuDeviceGate(jobs_dir=str(config.harbor.jobs_dir))
        allocated_gpu = await gpu_gate.acquire()
        if allocated_gpu is not None:
            gpu_override_path = Path(build_gpu_override(config.harbor.gpu_compose_override, allocated_gpu.device))
            extra_compose_paths.append(gpu_override_path)
            await _bind_gate_b(session_router_url, session_id, allocated_gpu.device)
```

三个分支：

1. **拿到卡**：生成临时 override，追加进 compose 文件列表，并通知 SessionRouter 绑定（Gate B）。
2. **没发现任何设备**（`acquire` 返回 `None`）：回退成旧的静态映射，并打印 WARNING "no per-device exclusivity"（[`runner.py:246-256`](../../../examples/sciaccel_rl/runner.py)）。设计意图是不要悄悄假装提供了互斥。
3. **没有 GPU 需求**：整个块跳过。

释放在 `finally` 的最末尾：

```python
# 历史摘录：提交 72ca50c，examples/sciaccel_rl/runner.py
        if allocated_gpu is not None:
            GpuDeviceGate.release(allocated_gpu)
            await _bind_gate_b(session_router_url, session_id, None)
        if gpu_override_path is not None:
            gpu_override_path.unlink(missing_ok=True)
```

位置在 `force_remove_compose_project` 之后（代码注释的理由：设备号不能在旧容器还没退干净时就被下一个 episode 拿到）。顺序是：释放锁 → 解绑 Gate B → 删临时文件。

## 5. Gate B 代码逐段讲解

### 5.1 它想拦的是什么

代码注释的说法：延迟的是**给 agent 的回复**，而不是下一个请求，这样模型已经要求的一条工具命令，就不会和"即将复用同一张物理卡的 verifier"赛跑。它是**软约束**：后台 `nohup solver &` 在函数返回后仍可能在跑。背景是遥测绊线（`gpu_active` 读整卡利用率）可能被邻居的负载欺骗，见官方设计提案译文 03 §5.4（本地资料，未随仓库发布）。

### 5.2 三步：绑定、等待、解绑

**绑定**：Gate A 拿到卡之后，runner 向 SessionRouter 发一次 HTTP：

下面是 [`runner.py:41-68`](../../../examples/sciaccel_rl/runner.py) 的骨架（省略了 docstring 与日志参数，所以不用行号块格式）：

```python
async def _bind_gate_b(session_router_url: str, session_id: str, device: str | None) -> None:
    if not session_router_url or not session_id:
        return
    try:
        await post(
            f"{session_router_url.rstrip('/')}/control/gpu_bind",
            payload=[{"session_id": session_id, "gpu_device": device}],
            max_retries=1,
        )
    except Exception as exc:  # noqa: BLE001
        psrl_logger.debug(...)   # 只记 debug 日志，不抛出
```

端点 [`control_gpu_bind`](../../../psrl/workers/gen/session_router.py) 解析请求体，找到 `self.states[sid]`，在 `state.lock` 保护下写 `state.gpu_device`；`device` 为空则清空（解绑）。找不到 session 就记进 `missing` 返回。整个绑定调用**失败只记 debug 日志，绝不让 episode 失败**。

**等待**：每个模型回复返回前，钩子读 `state.gpu_device`，非空才进入等待：

```python
# 历史摘录：提交 72ca50c，psrl/workers/gen/session_router.py
        # Gate B: for a session bound to a GPU device (via /control/gpu_bind), hold
        # this reply until that device looks idle, or `_GATE_B_MAX_WAIT_SEC` elapses.
        # `gpu_device` is `None` for every non-GPU session, so this is a no-op unless
        # examples/sciaccel_rl/runner.py's Gate A explicitly bound one.
        async with state.lock:
            gpu_device = state.gpu_device
            gpu_shared = state.gpu_shared
        if gpu_device:
            await _wait_for_gpu_idle(gpu_device, sid, shared=gpu_shared)
```

> **关于行号和这段代码（2026-10-12 校正）：** 本节所有 `session_router.py` 的行号都指向**当前工作区**。当天新增的 Env Service 改动在这个文件里加了 `gpu_shared`、多设备（`_parse_gpu_indices`）等内容，所以上面这段钩子比提交 `72ca50c` 里的版本多一行 `gpu_shared`、`_wait_for_gpu_idle` 多一个 `shared` 参数；提交里原版是 `await _wait_for_gpu_idle(gpu_device, sid)`，行号是 402-409。下面判定表的逻辑对"单设备、非共享"的情况与提交版一致；共享卡只看利用率、多设备要求全部空闲，是工作区新增的行为（见 [Env Service 实现讲解](02-Env历史_2026年10月12日资源单元与租约.md)）。

没有绑定的 session（所有非 GPU 任务、AIRS-Bench/MLGym）完全跳过，所以对它们是零开销。

[`_wait_for_gpu_idle`](../../../psrl/workers/gen/session_router.py) 是一个轮询循环，判定规则可以用一张表说清：

| 情况 | 动作 | 对应代码 |
|---|---|---|
| `nvidia-smi` 读不到这张卡（没装、超时、设备不在输出里） | **立刻放行**（fail open） | [`:146-147`](../../../psrl/workers/gen/session_router.py) |
| `util ≤ 5%` 且 `mem ≤ 256 MB` | 放行 | [`:150`](../../../psrl/workers/gen/session_router.py) |
| 已超过 `PSRL_GATE_B_MAX_WAIT_SEC`（默认 600 秒） | 打 WARNING，**照样放行** | [`:160-170`](../../../psrl/workers/gen/session_router.py) |
| 其他 | 睡 `PSRL_GATE_B_POLL_INTERVAL_SEC`（默认 2 秒）后再读 | [`:172`](../../../psrl/workers/gen/session_router.py) |

阈值与超时都是环境变量（[`session_router.py:43-48`](../../../psrl/workers/gen/session_router.py)）：`PSRL_GATE_B_MAX_WAIT_SEC=600`、`PSRL_GATE_B_POLL_INTERVAL_SEC=2`、`PSRL_GATE_B_IDLE_UTIL_PCT=5`、`PSRL_GATE_B_IDLE_MEM_MB=256`。

**为什么超时要"放行"而不是"失败"**：SciAccel agent 的 LLM 请求超时是 900 秒且不重试（[`runner.py:316-317`](../../../examples/sciaccel_rl/runner.py)），Gate B 在这次请求**内部**等，所以必须小于 900；超时放行是因为"超出软容量目标只损失吞吐，让 episode 失败会毁掉最长四小时的工作"（设计提案 §6.3 的原话意思，见 译文 04（本地资料，未随仓库发布））。

**读遥测**：[`_read_gpu_utilization`](../../../psrl/workers/gen/session_router.py) 执行 `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits`，解析成 `{index: (util, mem)}`。有一个模块级锁和 1 秒缓存，避免几十个 session 同一瞬间各开一个子进程；`nvidia-smi` 不存在或 5 秒超时则返回空字典，即"全部当空闲"。这里用的是**整卡**计数器，而不是设计提案建议的 `pmon` 按进程口径。

**解绑**：episode 结束后 `_bind_gate_b(..., None)`，`gpu_device` 置空，下一次回复就跳过 Gate B。

### 5.3 用一个例子串起来

教学数字：device=`/dev/nvidia1`，agent 第 7 轮刚让模型生成完 `make -j8 && ./solver`，回复到了 SessionRouter 出口：

| 时刻 | `nvidia-smi` 读到 index=1 | 判定 |
|---|---|---|
| 第 0 秒 | util=0%，mem=0 MB | 空闲 → 立刻放行，agent 拿到命令并开始执行 |
| 第 8 轮回复到出口时，solver 正在跑 | util=97%，mem=18000 MB | 忙 → 每 2 秒重读 |
| solver 刚跑完，残留显存 300 MB | util=0%，mem=300 MB | 300 > 256 → 仍判忙，继续等 |

表中使用教学数字。两项条件取 **且**：`util ≤ 5%` 与 `mem ≤ 256 MB` 同时满足才放行。因此利用率降到 0、显存读数仍超过阈值时，代码会继续等待。

<a id="6-与设计提案对照做了什么没做什么"></a>

## 6. 与设计提案的能力对照

设计提案全文见 Env Service 设计文档译文（本地资料，未随仓库发布）。这个提交对应其中"Gate A / Gate B"的一个**子集**：

| 设计提案里的内容 | 这个提交 |
|---|---|
| `EnvRouter`：注册、准入、租约、回收 | 本提交使用 `GpuDeviceGate`；Router 属于后续实现 |
| `EnvUnit`：带显存预算的一卡多容器 | 本提交按整卡独占，每张卡同时发给一个 episode |
| Gate A：按显存、主存、CPU、容器数累计准入 | 本提交仅按设备互斥；会话数量由各进程的信号量限制 |
| Gate A 在 `Job.create` 之前、可无限期等待 | **是**。[`runner.py:240`](../../../examples/sciaccel_rl/runner.py) 先于 `job_dir` 与 `Job.create`（[`:278`](../../../examples/sciaccel_rl/runner.py)、[`:339`](../../../examples/sciaccel_rl/runner.py)）；`SlotManager` 的循环没有超时 |
| Gate B：有界等待、超时放行、仅 GPU unit | **是**。600 秒、fail open、只对被 `gpu_bind` 的 session 生效 |
| Gate B 判据：活跃的环境调用数 | 本提交检查整卡 `util ≤ 5%` 且 `mem ≤ 256 MB` |
| 运行时解析驱动库 | 保留静态 override 的驱动挂载，仅修改设备列表 |
| Harbor GPU 校验覆盖 | 本提交保持原配置；当次未核对 Harbor 内部行为 |
| 槽位有效期与回收 | 文件锁随持有进程退出释放，实际容器由清理链回收 |
| unit 负载快照 | 本提交只有 Gate B 的整卡轮询，尚无资源单元快照 |

所以这个提交解决的是 并发全景 §5④（本地资料，未随仓库发布） 里指出的"SciAccel 设备抢占根因"（所有 episode 读到同一个写死的 `/dev/nvidia0`）的**互斥**部分，而不是设计提案的整套重构。

## 7. 从代码读出的 6 处注意点（推断）

下面每一条都是**读代码结构得出的推断**，我没有在真实多节点 GPU 集群上运行验证。

1. **Gate A 的设备池是"整机所有 `/dev/nvidia[0-9]*`"，不知道谁占着哪张卡。** `discover_gpu_devices` 只做 glob（[`gpu_device_gate.py:69`](../../../psrl/utils/concurrency/gpu_device_gate.py)），不看 Ray 把哪些卡分给了 vLLM 或训练。如果 agent worker 所在节点同时跑训练/推理，Gate A 仍可能把一张正被引擎使用的卡发给某个 episode。这就是 [SciAccel GPU Gate §17.3](04-Env历史_Science整卡锁与设备账本.md#173-对比envworker-路径为什么天然安全sciaccelgate-a路径为什么需要额外小心) 讨论的问题，这个提交没有解决它，只是让"同一个字符串被所有人拿到"变成了"在整机卡里互斥"。
2. **池按 `jobs_dir` 的哈希分。** `/tmp/{prefix}_{sha1(abspath(jobs_dir))[:12]}`（[`slot.py:25-26`](../../../psrl/utils/concurrency/slot.py)）。两个用不同 `jobs_dir` 的任务在同一节点上互相看不到对方的占用。
3. **Gate B 的 `nvidia-smi` 跑在 SessionRouter 子进程所在的节点，不一定是容器所在的节点。** SessionRouter 由 `RolloutGateway` Actor 用 `multiprocessing.Process` 拉起，并绑定该 Actor 所在节点的 IP（[`rollout_gateway.py:112,148,187`](../../../psrl/workers/gen/rollout_gateway.py)）；而 episode 的容器和 Gate A 在 `AgentLoopWorker` 所在节点。示例脚本把 agent worker 限定在 `AGENT_NODE_IPS`（[`fsdp_qwen35_4b.sh:93`](../../../examples/sciaccel_rl/fsdp_qwen35_4b.sh)），而 `RolloutGateway` 是不带节点亲和创建的（[`ray_trainer.py:600-605`](../../../psrl/trainer/ppo/ray_trainer.py)）。如果两者不在同一台机器，Gate B 读的是**别的机器上同名 index 的卡**；那台机器没有 `nvidia-smi` 时则整个 Gate B 悄悄退化为空操作。单节点部署不受影响。
4. **Gate B 多半在等"自己的后台 GPU 任务"。** Gate A 已保证这张卡只有本 episode 在用，而 Gate B 读的是整卡计数器，所以 agent 用 `nohup ./solver &` 启动后台任务后，**下一轮**的模型回复会被扣住，直到 solver 结束或 600 秒到。这是否符合设计意图我没有确认（提交注释强调的是防止与 verifier 赛跑）；它和"agent 可以后台跑任务然后 `tail log`"这个正常用法是有张力的。600 秒加生成时间仍小于 900 秒的 LLM 请求超时，所以不会因此直接超时。
5. **Gate A 在拿到 episode 计数名额之后才排队等卡。** 入口顺序是先 `_acquire_episode_slot`（[`agent_loop.py:349`](../../../examples/sciaccel_rl/agent_loop.py)），再进 `run_harbor_episode` 里等卡。若卡数少于 `max_concurrent_episodes`，等卡的 GPU episode 会一直占着计数名额，可能挡住本可以开跑的 CPU episode。
6. **两处小的实现细节。** ① `build_gpu_override` 用的是 `tempfile.mktemp`（只生成名字不创建文件，标准库文档不推荐，存在竞态），见 [`gpu_device_gate.py:192`](../../../psrl/utils/concurrency/gpu_device_gate.py)；② `_read_gpu_utilization` 在 `wait_for` 超时的分支里没有 `kill` 那个 `nvidia-smi` 子进程（[`session_router.py:89-94`](../../../psrl/workers/gen/session_router.py)），从代码看超时的子进程会被丢下。

## 8. 跟着图读源码：执行路径

| 步骤 | 对应图上 | 调用 |
|---|---|---|
| 1 | ① | [`agent_loop.py:349`](../../../examples/sciaccel_rl/agent_loop.py) 拿计数名额 → [`:358`](../../../examples/sciaccel_rl/agent_loop.py) 把 `session_router_url` 传给 `run_harbor_episode` |
| 2 | ② | [`runner.py:240-241`](../../../examples/sciaccel_rl/runner.py) `GpuDeviceGate(...).acquire()` |
| 3 | ② | [`gpu_device_gate.py:127`](../../../psrl/utils/concurrency/gpu_device_gate.py) `discover_gpu_devices()` → [`:137`](../../../psrl/utils/concurrency/gpu_device_gate.py) `SlotManager.acquire` → [`slot.py:66`](../../../psrl/utils/concurrency/slot.py) `flock` |
| 4 | ② | [`runner.py:243`](../../../examples/sciaccel_rl/runner.py) `build_gpu_override` → 追加进 `extra_compose_paths` |
| 5 | ③ | [`runner.py:245`](../../../examples/sciaccel_rl/runner.py) `_bind_gate_b` → HTTP `POST /control/gpu_bind` → [`session_router.py:567`](../../../psrl/workers/gen/session_router.py) 写 `state.gpu_device` |
| 6 | ④ | [`runner.py:339`](../../../examples/sciaccel_rl/runner.py) `Job.create` 起容器（compose 文件里已含单卡 override） |
| 7 | ⑤ | 每轮回复：[`session_router.py:425-433`](../../../psrl/workers/gen/session_router.py) → [`:120`](../../../psrl/workers/gen/session_router.py) `_wait_for_gpu_idle` → [`:64`](../../../psrl/workers/gen/session_router.py) `_read_gpu_utilization` |
| 8 | ⑥ | [`runner.py:402-406`](../../../examples/sciaccel_rl/runner.py) 释放锁、解绑、删临时 override |

## 9. 可运行实验与实测输出

提交自带两个 CPU 诊断脚本。`gate_a_after.py` 直接导入 `GpuDeviceGate`，`--num-gpus` 提供设备字符串；脚本实际执行并发协程和 `fcntl` 加锁，展示卡的分配顺序。

```bash
cd psrl
python3 examples/sciaccel_rl/diagnostics/gate_a_before.py --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1
python3 examples/sciaccel_rl/diagnostics/gate_a_after.py  --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1 --num-gpus 3
# 跨进程版本（对应生产里多个独立 AgentLoopWorker 进程）：
python3 examples/sciaccel_rl/diagnostics/gate_a_after.py --cross-process --num-workers 4 --num-episodes 3 --hold-seconds 0.3 --num-gpus 3
```

**当次 CPU 诊断输出（节选）**：

```
# BEFORE
  t=  0.00  BEGIN  episode=0   device=/dev/nvidia0
  t=  0.00  BEGIN  episode=1   device=/dev/nvidia0
  t=  0.00  BEGIN  episode=2   device=/dev/nvidia0
  t=  0.00  BEGIN  episode=3   device=/dev/nvidia0
  device='/dev/nvidia0': peak simultaneous holders = 4  -> CONTENTION

# AFTER (--num-gpus 3)
  t=  1.00  BEGIN  episode=4   device=/dev/nvidia0  (Gate A slot 0)
  t=  1.00  BEGIN  episode=5   device=/dev/nvidia1  (Gate A slot 1)
  t=  1.01  BEGIN  episode=3   device=/dev/nvidia2  (Gate A slot 2)
  device='/dev/nvidia0': peak simultaneous holders = 1  -> ok, exclusive
  device='/dev/nvidia1': peak simultaneous holders = 1  -> ok, exclusive
  device='/dev/nvidia2': peak simultaneous holders = 1  -> ok, exclusive
  RESULT: zero contention across 3 device(s) and 6 episodes.
```

读法：BEFORE 里 4 个 episode 在 t=0 同时拿到同一个 `/dev/nvidia0`；AFTER 里 3 张卡各自最多 1 个持有者，第 4~6 个 episode 在 t=1.00 前一批释放之后才开始。

<a id="10-没有验证的部分"></a>

## 10. 验证范围与复现条件

| 项目 | 这份记录中的证据 |
|---|---|
| Gate A 设备分配 | 第 9 节保留进程内诊断输出；设备为字符串，文件锁与协程调度实际执行 |
| 跨进程文件锁 | [早期记录第 16.5 节](04-Env历史_Science整卡锁与设备账本.md#165-修复已部署gate-a设备互斥-gate-b回复节流代码位置与验证脚本)保存另一次运行；本篇当次没有重复执行 |
| Gate B 回复节流 | 当次按源码分析。环境缺 `aiohttp`，没有运行 SessionRouter 或真实遥测 |
| EnvWorker GPU 集成测试 | 运行需同时具备 Docker、`nvidia-smi` 和 `PSRL_GPU_PROBE_IMAGE` 指定的 CUDA / PyTorch 镜像；当次环境不满足，测试未运行 |
| 多节点、设备阈值与 Harbor 行为 | 第 7 节给出依赖部署拓扑的风险分析；当次没有真实集群和 Harbor 源码核验 |

这些条件说明证据覆盖到哪里。需要使用现有系统时，按[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)和[运行前检查](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)选择对应路径。

## 相关文档

- [Gate B 新手版讲解](07-Env历史_GateB回复节流机制.md)：只讲 Gate B，从名词、故事、全局图到逐行讲解，适合先读它再回来看本文 §5。
- [SciAccel GPU Gate（§16.5 是这个提交的文件级概览）](04-Env历史_Science整卡锁与设备账本.md#165-修复已部署gate-a设备互斥-gate-b回复节流代码位置与验证脚本)：本文是它的逐行代码版。
- Env Service 设计文档译文（本地资料，未随仓库发布）：Gate A/B、`EnvRouter`、`EnvUnit` 的完整设计提案。
- 并发/并行全景（本地资料，未随仓库发布）：协程、线程、进程的分层，以及 SciAccel 的 Harbor 线程。
- EnvWorker 代码详解（本地资料，未随仓库发布）：Path A（`EnvWorker` + `EnvWorkerCoordinator`）的现状代码。
- [PSRL 笔记总索引](../../README.md)
