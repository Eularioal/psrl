# 取消安全与可运行实验

> **核心结论**：本篇聚焦「取消安全与可运行实验」及其相邻完整章节。原文的例子、推导和证据保留，历史条件不代表当前部署。

## 目录

- [9. 可运行实验脚本](#9-可运行实验脚本)
- [10. 一句话总结 + 延伸阅读](#10-一句话总结--延伸阅读)

[返回专题索引](../gpu-admission-attribution-ebpf-identity.md)

### 8.3 新模式 B：多持有者引用计数 + 取消安全清理（routing_pause.py）

`RoutingPauseController` 要解决的问题：训练过程中，模型权重同步、副本迁移等好几个独立的操作，都可能需要"暂停往下游派发新的推理/生成请求"，而且这些操作可能**并发**发生（比如正在同步权重的同时，又触发了一次副本迁移）。如果简单地用一个布尔值记录"是否暂停"，第一个操作做完就把它改回 `False`，会错误地提前恢复路由——哪怔第二个操作还没做完。

```python
async def _acquire(self, owner):
    async with self._lock:
        ...
        self._holds.add(owner)
        await self._write_running(False)

async def _release(self, owner):
    async with asyncio.timeout(self._timeout_s):
        async with self._lock:
            if owner not in self._holds:
                return
            self._holds.remove(owner)
            await self._write_running(self._running and not self._holds)
```
—— [`routing_pause.py:24-38`](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L24)

这里的解法是用一个 **集合**（`self._holds`）记录"当前有哪些独立操作正在要求暂停"，而不是一个布尔值——这其实是**引用计数**思想的一种变体（引用计数：记录"有多少个使用者正在持有这份资源"，只有计数归零才真正释放；这里每个 `owner` 就是一个"使用者"）。`_write_running(False)` 只要集合非空就一直传 `False`；只有 `_release` 把最后一个 `owner` 也移出集合（`not self._holds` 变成 `True`）之后，才会真正把"恢复运行"的信号写回去。

更精细的地方在它的上层封装 `pause()`：

```python
@asynccontextmanager
async def pause(self):
    owner = object()
    try:
        async with asyncio.timeout(self._timeout_s):
            await self._acquire(owner)
        yield
    finally:
        cleanup = asyncio.create_task(self._release(owner))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
```
—— [`routing_pause.py:40-59`](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L40)

`@asynccontextmanager` 把一个写成"先做准备工作、`yield` 一下、再做收尾工作"的协程函数，包装成一个可以配合 `async with` 使用的"异步上下文管理器"——`async with controller.pause(): ...` 这种写法，进入代码块前自动跑 `yield` 之前那部分（这里是 `_acquire`，登记一个 hold），离开代码块（无论正常结束还是中途抛异常）自动跑 `finally` 里那部分（这里是 `_release`，释放这个 hold）。

真正精巧（也是本文想重点讲的新知识）的是 `finally` 块里处理"取消"（cancellation）的方式。协程的**取消**是 `asyncio` 里一种特殊的异常注入机制：外部代码可以调用 `task.cancel()`，这会在这个协程当前正挂起等待的那个 `await` 点上，强行注入一个 `asyncio.CancelledError` 异常，让它"看起来像是那个等待操作自己失败了"——协程可以选择捕获这个异常做清理，但默认行为是继续往外传播，最终让这个协程整体终止。

问题是：如果调用方正在执行 `pause()` 的 `finally` 块（也就是正在做"释放 hold、把暂停状态写回去"这件收�020尾工作）的过程中，**又**被取消了一次（比如上层调用方自己也被取消，取消信号层层传递下来），默认情况下这第二次取消会直接中断 `_release` 的执行——可能刚把 `self._holds.remove(owner)` 执行完、还没来得及调用 `_write_running` 把"恢复运行"的信号真正写出去就被打断了，造成"集合里明明已经没有任何 hold 了，但路由状态却一直卡在暂停"的不一致。

`asyncio.shield(cleanup)`（**shield**，字面意思"盾牌"，这里指"保护某个协程不被外部的取消信号直接打断"）就是为了防住这第二次取消：`cleanup = asyncio.create_task(self._release(owner))` 先把清理工作包装成一个独立的 Task（Task 有自己的生命周期，不会因为"等待它的那一方被取消"而自动跟着被取消）；`await asyncio.shield(cleanup)` 等待这个 Task，但即使等待方（当前协程）被取消，`shield` 也只会让"等待"这个动作本身抛出 `CancelledError`，`cleanup` 这个 Task 不受影响，继续在后台跑完。代码用一个 `while not cleanup.done()` 循环反复尝试等待它——哪怕中途再被取消几次（每次把 `cancelled` 标记设为 `True`），也会一直等到 `cleanup` 真正跑完为止，保证清理工作**一定会被完整执行**；最后 `cleanup.result()` 取出结果（如果清理过程本身出了异常，这里会重新抛出），如果期间确实被取消过，再手动 `raise asyncio.CancelledError`——**把"我被取消了"这个事实如实告诉外部**，只是不让这次取消打断必须完成的清理工作。这是协程编程里一个经典但容易被忽略的教训：**取消信号可以、也应该被妥善处理，但绝不能让它打断必须完整执行的收尾逻辑。**

## 9. 可运行实验脚本

下面这份脚本把第 5/7/8 节讲到的四段真实逻辑（显存会计公式、容器预算、健康状态机、多持有者暂停控制器）原样搬过来跑一遍，再加一个纯 Python 手写的"迷你内核"模拟第 4 节的 eBPF 身份追踪状态机（真实的 eBPF 需要 Linux 内核 + root 权限，在这里只能用等价的数据结构和逻辑模拟，方便在任何机器上直接观察状态变化）。只用标准库，可以直接用 `python3` 运行：

```python
"""
自包含教学复刻脚本：对应 docs/psrl/env-worker-deep-dive/gpu-admission-attribution-ebpf-identity.md
第 5/7/8 节的真实逻辑搬运 + 第 4 节 eBPF 身份追踪的纯 Python 模拟。

只用标准库（asyncio + dataclasses + contextlib），不依赖 Linux/BPF/Docker/PSRL 代码。

对照真实代码（仅供对照阅读，不在本脚本中导入）：
  - 显存会计公式 accounted_gpu_memory_mb -> psrl/psrl/workers/env_service/gpu_memory.py:9-28
  - 容器预算 ContainerBudget           -> psrl/psrl/workers/env_service/container_budget.py:6-46
  - 健康状态机 NodeHealth              -> psrl/psrl/workers/env_service/health.py:6-26
  - 暂停控制器 RoutingPauseController   -> psrl/psrl/workers/gen/rollout_coordination/routing_pause.py:7-59
  - eBPF 身份追踪（本脚本用 FakeKernel 模拟其状态机，不是真实 eBPF）
    -> psrl/psrl/workers/env_service/bpf_identity.bpf.c:57-106
"""

import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass


# ---------- 第 5 节：GPU 显存会计公式（与 gpu_memory.py 逐行一致） ----------

def accounted_gpu_memory_mb(measured_mb, reservations, owned_mb=None):
    owned_mb = owned_mb or {}
    values = [measured_mb, *reservations.values(), *owned_mb.values()]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("GPU memory observations and reservations must be finite and nonnegative.")
    if owned_mb.keys() - reservations.keys() or sum(owned_mb.values()) > measured_mb:
        raise ValueError("GPU ownership credits must be disjoint and contained in measured usage.")
    return measured_mb + sum(
        max(0.0, amount - owned_mb.get(lease_id, 0.0)) for lease_id, amount in reservations.items()
    )


def demo_gpu_memory_accounting():
    print("=== 第 5 节：GPU 显存会计公式 ===")
    unknown = accounted_gpu_memory_mb(70, {"a": 15})
    print(f"未知归属: measured=70, reservations={{a:15}} -> {unknown}  (期望 85)")
    unknown_grew = accounted_gpu_memory_mb(84, {"a": 15})
    print(f"实测涨到84仍未知归属:            -> {unknown_grew}  (期望 99，不会被当成是a用的)")
    verified = accounted_gpu_memory_mb(84, {"a": 15}, {"a": 14})
    print(f"核实a占用14MB(已包含在84里):      -> {verified}  (期望 85，14MB不会被收两次)")
    print()


# ---------- 第 6 节：容器预算（与 container_budget.py 逐行一致） ----------

@dataclass(frozen=True)
class ContainerBudget:
    main_cpu: float
    main_memory_mb: int
    max_sidecars: int
    environments: int = 1
    sidecar_cpu: float = 0.25
    sidecar_memory_mb: int = 128

    @property
    def cpu(self):
        return self.environments * (self.main_cpu + self.max_sidecars * self.sidecar_cpu)

    @property
    def memory_mb(self):
        return self.environments * (self.main_memory_mb + self.max_sidecars * self.sidecar_memory_mb)


def demo_container_budget():
    print("=== 第 6 节：容器预算手算 ===")
    # 任务声明 2 核 4096MB，验证器要求独立环境(environments=2)，
    # docker-compose.yaml 里有 2 个 sidecar（+1 预留给 Harbor 可能追加的网络服务）
    budget = ContainerBudget(main_cpu=2.0, main_memory_mb=4096, max_sidecars=2 + 1, environments=2)
    print(f"任务自己声明: 2 核 / 4096MB")
    print(f"实际预留:     {budget.cpu} 核 / {budget.memory_mb}MB  (期望 5.5 核 / 8960MB)")
    print()


# ---------- 第 7 节：健康状态机（与 health.py 逐行一致） ----------

@dataclass
class NodeHealth:
    failures: int = 0
    checked_at: float = float("-inf")
    error: str = ""
    pending_cleanup: bool = False

    @property
    def available(self):
        return not self.error and not self.pending_cleanup

    def success(self, now, pending_cleanup=False):
        self.failures = 0
        self.checked_at = now
        self.error = ""
        self.pending_cleanup = pending_cleanup

    def fail(self, now, error):
        self.failures += 1
        self.checked_at = now
        self.error = f"{type(error).__name__}: {error}"


def demo_node_health():
    print("=== 第 7 节：节点健康熔断 ===")
    max_failures = 3
    state = NodeHealth()
    for t in range(1, 5):
        state.fail(float(t), RuntimeError("node unreachable"))
        tripped = state.failures >= max_failures
        print(f"第{t}次探测失败: failures={state.failures}, available={state.available}, "
              f"熔断(停止继续探测)={tripped}")
    print()


# ---------- 第 8.3 节：多持有者暂停控制器（与 routing_pause.py 逐行一致） ----------

class RoutingPauseController:
    def __init__(self, read_status, write_running, timeout_s=5.0):
        self._read_status = read_status
        self._write_running = write_running
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._holds = set()
        self._running = None

    async def _acquire(self, owner):
        async with self._lock:
            if self._running is None:
                status = await self._read_status()
                self._running = not status["paused"]
            self._holds.add(owner)
            await self._write_running(False)

    async def _release(self, owner):
        async with asyncio.timeout(self._timeout_s):
            async with self._lock:
                if owner not in self._holds:
                    return
                self._holds.remove(owner)
                await self._write_running(self._running and not self._holds)

    @asynccontextmanager
    async def pause(self):
        owner = object()
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._acquire(owner)
            yield
        finally:
            cleanup = asyncio.create_task(self._release(owner))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError


async def demo_routing_pause():
    print("=== 第 8.3 节：多持有者暂停，只有最后一个释放才真正恢复 ===")
    running_log = []

    async def read_status():
        return {"paused": False}

    async def write_running(value):
        running_log.append(value)

    controller = RoutingPauseController(read_status, write_running)

    async with controller.pause():  # 操作A 开始暂停
        print(f"操作A进入(第1次写入): running写入历史={running_log}  (期望末尾是False)")
        async with controller.pause():  # 操作B 并发也要求暂停
            print(f"操作B也进入(第2次写入): running写入历史={running_log}  (期望末尾仍是False)")
        print(f"操作B释放后(第3次写入): running写入历史={running_log}  (期望末尾仍是False，因为A还在持有)")
    print(f"操作A释放后(第4次写入): running写入历史={running_log}  (期望此时末尾才变成True)")
    print()


# ---------- 第 4 节：用纯 Python 模拟 eBPF 的三张 map 和 capture/invalidate ----------

class FakeKernel:
    """
    模拟 bpf_identity.bpf.c 的 registrations / identities 两张 map，
    以及 capture()（对应 fork/exec/sys_enter 钩子）、invalidate()（对应 exit/cgroup move 钩子）。
    真实版本在内核里用 BTF 算出的偏移量读 task_struct，这里直接用 Python 字典里的
    "cgroup" 字段代替，逻辑一一对应，只是省去了内核内存布局这一层。
    """

    def __init__(self):
        self.registrations = {}   # cgroup_id -> cookie
        self.identities = {}      # pid -> {"cgroup": ..., "cookie": ...}

    def bind(self, cgroup_id, cookie):
        """对应 bpf_identity.py:155-159 的 bind()：用户态注册一个要追踪的 cgroup。"""
        self.registrations[cgroup_id] = cookie

    def capture(self, pid, cgroup_id):
        """对应 bpf_identity.bpf.c:57-74 的 capture()：fork/exec/sys_enter 钩子触发时调用。"""
        cookie = self.registrations.get(cgroup_id)
        if cookie is None:
            self.identities.pop(pid, None)
            return
        self.identities[pid] = {"cgroup": cgroup_id, "cookie": cookie}

    def invalidate(self, pid):
        """对应 bpf_identity.bpf.c:93-96 的 invalidate()：exit/cgroup_attach_task 钩子触发时调用。"""
        self.identities.pop(pid, None)

    def owners(self, cookie, pids):
        """对应 bpf_identity.py:187-202 的 owners()：用户态按 cookie 查询一批 pid 还归不归自己。"""
        return {
            pid for pid in pids
            if pid in self.identities and self.identities[pid]["cookie"] == cookie
        }


def demo_ebpf_identity_state_machine():
    print("=== 第 4 节：eBPF 身份追踪状态机模拟 ===")
    kernel = FakeKernel()

    # 任务 A 的 device cgroup 是 9001，bind() 时拿到凭证 cookie_A
    cookie_a = "cookie_A_0x7f3a"
    kernel.bind(cgroup_id=9001, cookie=cookie_a)
    print(f"任务A注册: registrations={kernel.registrations}")

    # 任务A容器里 fork 出 PID=4201，属于 cgroup 9001 -> on_fork 钩子触发 capture()
    kernel.capture(pid=4201, cgroup_id=9001)
    print(f"PID 4201 fork（cgroup=9001）后: identities={kernel.identities}")

    owned = kernel.owners(cookie_a, pids=[4201, 9999])
    print(f"任务A查询 owners([4201, 9999]) -> {owned}  (期望只有4201，9999根本没记录)")

    # 管理员把 4201 挪到了别的 cgroup（9002）-> on_move 钩子触发 invalidate()
    kernel.invalidate(pid=4201)
    print(f"PID 4201 被移出 cgroup（on_move 触发 invalidate）后: identities={kernel.identities}")
    owned_after_move = kernel.owners(cookie_a, pids=[4201])
    print(f"任务A再查询 owners([4201]) -> {owned_after_move}  (期望空集，归属已自动失效)")
    print()


async def main():
    demo_gpu_memory_accounting()
    demo_container_budget()
    demo_node_health()
    demo_ebpf_identity_state_machine()
    await demo_routing_pause()


if __name__ == "__main__":
    asyncio.run(main())
```

**实测跑通的输出**（已在本地实际运行，不是推算）：

```
=== 第 5 节：GPU 显存会计公式 ===
未知归属: measured=70, reservations={a:15} -> 85.0  (期望 85)
实测涨到84仍未知归属:            -> 99.0  (期望 99，不会被当成是a用的)
核实a占用14MB(已包含在84里):      -> 85  (期望 85，14MB不会被收两次)

=== 第 6 节：容器预算手算 ===
任务自己声明: 2 核 / 4096MB
实际预留:     5.5 核 / 8960MB  (期望 5.5 核 / 8960MB)

=== 第 7 节：节点健康熔断 ===
第1次探测失败: failures=1, available=False, 熔断(停止继续探测)=False
第2次探测失败: failures=2, available=False, 熔断(停止继续探测)=False
第3次探测失败: failures=3, available=False, 熔断(停止继续探测)=True
第4次探测失败: failures=4, available=False, 熔断(停止继续探测)=True

=== 第 4 节：eBPF 身份追踪状态机模拟 ===
任务A注册: registrations={9001: 'cookie_A_0x7f3a'}
PID 4201 fork（cgroup=9001）后: identities={4201: {'cgroup': 9001, 'cookie': 'cookie_A_0x7f3a'}}
任务A查询 owners([4201, 9999]) -> {4201}  (期望只有4201，9999根本没记录)
PID 4201 被移出 cgroup（on_move 触发 invalidate）后: identities={}
任务A再查询 owners([4201]) -> set()  (期望空集，归属已自动失效)

=== 第 8.3 节：多持有者暂停，只有最后一个释放才真正恢复 ===
操作A进入(第1次写入): running写入历史=[False]  (期望末尾是False)
操作B也进入(第2次写入): running写入历史=[False, False]  (期望末尾仍是False)
操作B释放后(第3次写入): running写入历史=[False, False, False]  (期望末尾仍是False，因为A还在持有)
操作A释放后(第4次写入): running写入历史=[False, False, False, True]  (期望此时末尾才变成True)
```

对照着看几个现象：显存会计那组数字和第 5 节手算完全一致（第三个例子打印出的是整数 `85` 而不是 `85.0`——这是 Python `max(0.0, 1)` 的一个实现细节：`max`/`min` 返回的是"两个参数里较大/较小的那一个原始对象"，不会做类型统一转换，`1`（int）严格大于 `0.0`（float）时就直接原样返回这个 `int` 对象，不影响数值大小，只是打印形式不同）；容器预算 `5.5核/8960MB` 和第 6 节手算一致；健康熔断在第 3 次失败时 `熔断=True` 开始生效，之后继续失败也保持 `True`（对应 `_health_task` 不再排新检查）；eBPF 模拟部分，`invalidate` 之后归属立刻查不到了（不需要任何"过期时间"判断，状态变化即时生效）。

暂停控制器部分的真实行为比我最初设想的更细致一点，值得纠正：**不是"只有首次加入和末次释放才触发写入"，而是每一次 `acquire`/`release` 都会触发写入，只是写入的那个布尔值是由"当前持有者集合是否为空"实时计算出来的**——B 自己进入时也会触发一次写入（第 2 次，`async with` 嵌套调用 `_acquire`，而 `_acquire` 内部无条件执行 `await self._write_running(False)`，见 [`routing_pause.py:24-29`](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L24)），B 释放时也会触发一次写入（第 3 次，但因为 A 还在持有集合里，`self._running and not self._holds` 算出来仍是 `False`）；只有第 4 次——也就是最后一个持有者（A）释放、集合彻底清空的那一刻，这个表达式才第一次算出 `True`。这仍然是"引用计数"的核心思想：**决定对外生效的状态值，是整个持有者集合的大小（是否为空），而不是任意单次 acquire/release 动作本身**；只是这次代码选择了"每次都写、但写之前先用集合状态算出正确的值"这种更简单直接的实现方式，而不是额外判断"是否需要跳过这次写入"。

脚本保存在 [`scripts/env_admission_attribution_demo.py`](../../env-service-verification/scripts/env_admission_attribution_demo.py)。

## 10. 一句话总结 + 延伸阅读

这组改动的动机：**当"运行 env_service 的节点本身也是一个容器"时，不能再相信容器自己视角里的 `/proc`，所以用 eBPF 直接在内核态拿到不依赖命名空间视角、事件驱动、不可伪造的进程归属真相，再配上一条不会重复计费的显存会计公式、一份调度和创建共用同一个口径的容器预算、一个让坏节点自动从调度候选中消失但不会贸然销毁未清理资源的健康状态机——四件事共同保证"显存和 CPU/内存记账记得准，节点挂了能安全恢复"。**

- [返回专题入口](../../README.md)
- psrl 仓库自带的权威设计文档（本文假设读者已经点开过，如果还没读建议先读这两篇再回来看本文的"零基础展开"部分）：
  - [`process-identity-without-procfs.md`](../process-identity-without-procfs.md)
  - [`gpu-memory-admission.md`](../gpu-memory-admission.md)
  - [`unified-gpu-execution.md`](../unified-gpu-execution.md)
- 协程/线程/进程三层叠加的基础模型：并发/并行全景（本仓库尚未收录）
- cgroup / procfs 更系统的背景知识：cgroup v2 资源控制知识笔记（本仓库尚未收录）、procfs 知识笔记（本仓库尚未收录）
