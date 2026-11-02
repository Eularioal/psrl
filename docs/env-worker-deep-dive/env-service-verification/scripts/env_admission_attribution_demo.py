"""
自包含教学复刻脚本：对应 psrl/docs/env-worker-deep-dive/knowledge/gpu-admission-attribution-ebpf-identity.md
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
    print("任务自己声明: 2 核 / 4096MB")
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
        print(
            f"第{t}次探测失败: failures={state.failures}, available={state.available}, "
            f"熔断(停止继续探测)={tripped}"
        )
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
