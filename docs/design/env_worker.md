# EnvWorker

## TL;DR

Reserve a remotely owned persistent sandbox with CPU and RAM admission. Poll outside the
coordinator, heartbeat its reservation, and return capacity only after confirmed deletion.
A command timeout invalidates the sandbox. AIRS grading uses a separate owned container.

## Contents

- [Motivation](#motivation)
- [Component Triple](#component-triple)
- [Placement Policies](#placement-policies)
- [Sandbox Handle Lifecycle](#sandbox-handle-lifecycle)
- [Persistent Shell Protocol](#persistent-shell-protocol)
- [Observation Truncation](#observation-truncation)
- [GPU Passthrough](#gpu-passthrough)
- [Configuration](#configuration)
- [Extension Points](#extension-points)
- [Relationship to Other PSRL Components](#relationship-to-other-psrl-components)
- [Harbor 的 EnvService 路径](#harbor-的-envservice-路径)

The EnvWorker subsystem provides a managed pool of persistent sandbox containers for
agentic RL tasks that require a stateful execution environment. It is designed for tasks like
AIRS-Bench where the agent must run code across multiple turns and the environment state, such
as installed packages, intermediate files, and trained model checkpoints, must persist for the
duration of the episode.

---

## Motivation

Standard tool-use in PSRL invokes a stateless function per turn. That model works for tools
that look up facts, execute arithmetic, or call external APIs, but it fails for tasks where
the environment accumulates state. An ML training task, for example, expects to call a data
preprocessing script in turn 2 and read its output in turn 3. Stateless exec re-runs the
environment from scratch on every call and cannot satisfy this requirement.

The requirements that drive the design are:

1. **Persistent shell.** The container must run a single long-lived `bash -l` session so that
   the working directory, exported variables, and the active conda environment survive across
   agent turns.
2. **Placement separation.** On heterogeneous clusters some nodes are better suited for
   sandbox workloads than for GPU training. Sandboxes must be pinnable to specific nodes.
3. **Capacity accounting.** Each node has a finite number of concurrent sandboxes it can
   host. The coordinator must admit sandboxes within capacity and queue excess requests.
4. **GPU pass-through without the NVIDIA Container Toolkit.** Clusters that lack
   `nvidia-container-runtime` need a different path to expose GPUs to containers.
5. **Observation safety.** A training run inside a container can emit megabytes of log text.
   The worker must cap output before returning it to the agent.

---

## Component Triple

The subsystem is built from three components with separated concerns.

```
EnvWorkerManager (construction, placement)
    |
    +-- EnvWorkerCoordinator  (routing, capacity accounting)
    |       |
    |       +-- WorkerSlot × N  (per-worker bookkeeping)
    |
    +-- EnvWorker × N  (container lifecycle, shell I/O)
            |
            +-- ContainerShell × M  (one per live sandbox)
```

Source legend, in diagram order:

| Node | Source |
|---|---|
| Manager | [`EnvWorkerManager`](../../psrl/workers/env_worker/manager.py#L49) |
| Coordinator | [`EnvWorkerCoordinator`](../../psrl/workers/env_worker/coordinator.py#L202) |
| Worker slots | [`WorkerSlot`](../../psrl/workers/env_worker/coordinator.py#L28) |
| Container owner | [`EnvWorker`](../../psrl/workers/env_worker/worker.py#L131) |
| Shell | [`ContainerShell`](../../psrl/workers/env_worker/worker.py#L27) |

Ask about any of these code paths when reading the implementation.

### EnvWorker (`worker.py`)

Owns the sandbox containers scheduled onto one placement unit. The class is plain Python with
no Ray dependency, so argument construction and shell mechanics stay unit-testable without a
running cluster. `EnvWorkerManager` wraps it with `ray.remote` at startup using a resource
request built from configuration.

Each `EnvWorker` tracks:

- The `CUDA_VISIBLE_DEVICES` range granted by Ray, from which it draws GPU indices for
  sandboxes. This makes it impossible for a sandbox to touch a GPU Ray did not assign.
- A `ContainerShell` per live sandbox, keyed by `sandbox_id`.
- Owned verifier ids and deletion state for remote cleanup.

Key operations: `create_sandbox`, `exec`, `read_file`, `write_file`, `destroy_sandbox`,
`evaluate`.

### EnvWorkerCoordinator (`coordinator.py`)

Routes sandbox requests to workers and tracks their capacity. The coordinator is a Ray actor
with `max_concurrency` set from configuration to allow concurrent RPCs.

It holds a `WorkerSlot` per registered worker (populated by `EnvWorkerManager` at startup).
Each slot carries the live counts of occupied CPU and GPU slots. The routing logic in
`select_by_method` filters to slots with available capacity, then applies the configured
policy.

`reserve` returns immediately with either a reservation handle or no capacity. The
`acquire_sandbox` client polls outside the actor, starts owner heartbeats, and asks the selected
worker to create the reserved sandbox. Startup failure is reported without an implicit retry.

Reservations account for CPU, RAM, container slots and GPUs. The coordinator watchdog reclaims
expired owners by asking their worker to delete the agent and verifier containers. Capacity is
returned only after deletion succeeds. Failed deletion stays reserved and is retried.

### EnvWorkerManager (`manager.py`)

Constructs the pool at PSRL startup and owns the coordinator and worker Ray actors for one
training job. Its responsibilities are limited to startup and shutdown: it reads the
`psrl.env_worker` config, resolves which nodes receive workers via the placement policy, and
creates one coordinator actor and one or more worker actors per node.

---

## Placement Policies

The `psrl.env_worker.placement` key selects the node assignment policy. Switching between
policies requires only a config change. No code change is needed.

### `colocated`

Workers are created on every alive Ray node. The env node list is the full output of
`ray.nodes()` filtered to `Alive=True`.

Use this when the cluster has homogeneous nodes and sandbox CPU workloads can share resources
with train and rollout GPUs. It is simpler to operate.

### `dedicated`

Workers are created only on the nodes listed in `psrl.env_worker.dedicated_node_ips`. The
manager validates that every listed IP appears in the alive Ray node set and raises
`ValueError` if any are missing.

Use this when the cluster has nodes specialized for sandbox workloads (such as high-CPU, no
GPU), or when you want to fence sandbox I/O away from the training and rollout GPUs. The
AIRS-Bench recipe uses this policy. Configure the environment node explicitly and budget its
resources separately from training and rollout placement.

---

## Sandbox Handle Lifecycle

The caller (an agent loop or environment) never talks to the `EnvWorker` directly. It holds a
`SandboxHandle` that hides which node hosts the sandbox.

```
acquire_sandbox(spec)        client polls coordinator.reserve without holding a waiting RPC
                             coordinator charges CPU, RAM, slots and verifier capacity
                             client starts heartbeat, worker starts container and probes shell
        |
exec(command)               worker serializes shell access and matches a unique command id
read_file(path)             bounded Docker exec snapshot
write_file(path)            bounded Docker exec file write
        |
destroy() or heartbeat TTL  coordinator asks owning worker to delete agent and verifier
                             deletion success returns capacity, failure stays reserved
```

Follow the execution path:

| Step | Function | State change |
|---|---|---|
| 1. Client admission | [`acquire_sandbox`](../../psrl/workers/env_worker/coordinator.py#L163) | Poll, then start heartbeat |
| 2. Reservation | [`reserve`](../../psrl/workers/env_worker/coordinator.py#L236) | Charge CPU, RAM and slots |
| 3. Startup | [`create_sandbox`](../../psrl/workers/env_worker/worker.py#L197) | Register and probe shell |
| 4. Command | [`exec`](../../psrl/workers/env_worker/worker.py#L251) | Serialize and match completion |
| 5. File read | [`read_file`](../../psrl/workers/env_worker/worker.py#L316) | Bounded snapshot |
| 6. File write | [`write_file`](../../psrl/workers/env_worker/worker.py#L334) | Bounded file transfer |
| 7. Explicit release | [`release`](../../psrl/workers/env_worker/coordinator.py#L283) | Delete remotely, then decrement |
| 8. Parallel watchdog | [`reap_expired`](../../psrl/workers/env_worker/coordinator.py#L301) | Retry cleanup for lost owners |

Run the CPU experiment from the repository root:

```bash
python -m pytest tests/env_worker/test_airs_regressions.py -q
```

[`test_airs_regressions.py`](../../tests/env_worker/test_airs_regressions.py) uses real Bash for
step 4 and substitutes Docker ownership for steps 3 and 7. It checks timeout invalidation,
process exit, 80 waiters, owner expiry and failed deletion. It starts no cluster or container.

`SandboxHandle.destroy` is idempotent. The coordinator releases by reservation id, so a lost
reply or repeated cleanup cannot decrement another episode's allocation. Always call it in a
`finally` block. The watchdog covers owner loss.

---

## Persistent Shell Protocol

A sandbox container is started with `docker run -i ... /bin/bash -l`. The `-i` flag keeps
stdin open, and `/bin/bash -l` starts a login shell. The shell never exits. Each command is
submitted by writing to its stdin and command completion is signaled by a sentinel line echoed
immediately after the command.

### Sentinel Design

After writing the command to stdin, the worker appends:

```bash
EXITSTATUS="$?"; printf "\n%s\n" "///PROCESS-DONE:<command-id>:$EXITSTATUS:PROCESS-DONE///"
```

The example substitutes `<command-id>` for the generated id. See
[`wrap_command`](../../psrl/workers/env_worker/shell.py#L22) for the executable form.
`parse_sentinel` accepts only that invocation's id and an integer exit status. Commands on the
same sandbox are serialized. Process exit is detected while reading, and continuous output
does not bypass the wall-clock deadline.

Why not stateless exec? Docker's `docker exec <name> bash -c "<cmd>"` starts a fresh shell for
each command. A fresh shell loses all state: the working directory resets, exported variables
vanish, and the activated conda environment must be re-sourced. MLGym's own environment
contract requires that the shell state accumulates across calls, so stateless exec is
incompatible with MLGym.

### Timeout and Drain

A timed out shell is destroyed before `exec` returns a timed-out result. AIRS discards the
trajectory and reports an infrastructure failure. It cannot run a following command in that
sandbox. `drain` only removes output buffered between successful calls. It is not an interrupt
mechanism and does not establish that a running command has stopped.

---

## Observation Truncation

MLGym does not bound command output. A `pip install` or `python train.py` inside a sandbox
can emit enough text to saturate the model context in one turn.

`truncate_observation` in `shell.py` caps each observation to `max_observation_chars` per
side (head plus tail). The head carries the command echo and early errors, the tail carries
the final result. An elision notice shows the exact number of omitted characters. The total
output is at most `max_observation_chars * 2` characters, which the caller can rely on.

Truncation is applied by the `EnvWorker.exec` method before returning `ExecResult.stdout`.
The `max_observation_chars` value of 0 or less disables truncation entirely.

---

## GPU Passthrough

Some clusters lack `nvidia-container-runtime`, which is required for `docker run --gpus`. The
`build_gpu_argv` function in `sandbox.py` implements an alternative that works without the
toolkit.

The approach is:

1. Resolve each assigned GPU enumeration index to its host minor number, then pass
   the corresponding `--device /dev/nvidiaN`.
2. For each NVIDIA control device (`/dev/nvidiactl`, `/dev/nvidia-uvm`,
   `/dev/nvidia-uvm-tools`), pass `--device` if the path exists on the host.
3. Resolve CUDA and NVML libraries matching the loaded kernel driver and bind them
   to their container library names. Include the PTX compiler when available.
4. Optionally bind-mount `nvidia-smi` from the host.

GPU containers receive a library search path containing the mounted driver libraries.

The `EnvWorker` derives the allowed GPU indices from `CUDA_VISIBLE_DEVICES`, which Ray sets
to exactly the GPUs it granted the worker actor. This partitions the physical GPUs safely:
a worker with indices 4 and 5 can only assign those indices to sandboxes and cannot touch
training GPUs 0 through 3.

---

## Configuration

All settings live under `psrl.env_worker` in the Hydra config tree. The defaults are in
`psrl/trainer/config/psrl/env_worker.yaml`.

| Key | Default | Purpose |
|---|---|---|
| `enable` | `False` | Build the pool at startup. Off by default. |
| `placement` | `colocated` | Node assignment policy. |
| `dedicated_node_ips` | `[]` | Node IPs for dedicated placement. |
| `workers_per_node` | `1` | Worker actors per node. |
| `cpu_slots_per_worker` | `8` | Concurrent sandboxes per worker. |
| `gpu_slots_per_worker` | `0` | GPU slots per worker. `0` for CPU-only sandboxes. |
| `cpu_budget_per_worker` | `null` | Derive CPU budget from Ray node capacity unless explicitly set. |
| `memory_budget_bytes_per_worker` | `null` | Derive RAM budget from Ray node capacity unless explicitly set. |
| `routing.method` | `least_loaded` | Routing policy. `least_loaded`, `round_robin`, or `random`. |
| `exec_default_timeout_s` | `3600` | Default timeout per command. |
| `max_observation_chars` | `8000` | Per-side observation character cap. `0` disables. |
| `coordinator_max_concurrency` | `64` | Max concurrent RPCs to the coordinator. |

---

## Extension Points

The following seams exist for future additions without requiring changes to the existing
components.

**Locality-aware routing.** `select_by_method` receives the full `candidates` list and the
`SandboxSpec`. A future policy can consult `slot.node_ip` to prefer the node closest to the
agent worker issuing the request.

**Dynamic slot rescaling.** The coordinator's `WorkerSlot` counts are set at registration
time. A future mechanism could have workers report their current load and the coordinator
adjust `cpu_slots` based on observed utilization.

**Heterogeneous capacity.** The `SandboxSpec` already carries `gpus`, `cpus`, and `memory`.
Admission filters candidates by CPU and RAM demand as well as slot and GPU counts.

**Per-sandbox network isolation.** `SandboxSpec.network` defaults to `host` for simplicity.
A future policy can set it to a named Docker network or `none` for stricter isolation.

---

## Relationship to Other PSRL Components

The `EnvWorkerManager` is constructed by the PSRL trainer before the agent loop workers
start. Agent loops look up the named coordinator and call the client function `acquire_sandbox` to
obtain a `SandboxHandle` for each episode.

EnvWorker is orthogonal to the Router, PS, and TransferQueue subsystems: it sits entirely
on the agent side of the data flow, between the agent loop and the environment, and produces
no training data itself. The training data path begins when the agent loop calls TITO via
the SessionRouter.

:::{seealso}
- {doc}`architecture`: full PSRL data and control flow
- {doc}`router_tito`: TITO session capture and SMG routing
- {doc}`staleness_control`: staleness buffer and Reserve/Occupy/Consume protocol
:::

## Harbor 的 EnvService 路径

Science 示例通过 `EnvRouter → EnvServiceClient → ManagedDockerEnvironment → Harbor`
管理容器。它与上文的 `EnvWorkerCoordinator → EnvWorker` 是两条独立执行路径。
启用 `psrl.env_service.enable` 不会自动把 Harbor 的命令执行转到 EnvWorker。

`EnvRouter` 负责资源准入及租约。每个参与节点由路由器创建一个 `EnvNodeService`
actor，负责本机 GPU 遥测和过期租约容器回收。节点间须允许访问该服务返回的 HTTP
地址。SessionRouter 使用租约中的节点地址和 GPU 枚举编号检查任务所在节点的设备。
Gate B 的等待有上限，遥测不可用时放行，因此它不是 GPU 互斥锁或显存硬隔离。

配置中的 `gpu_indices` 是宿主机 `nvidia-smi` 的枚举编号。设备挂载通过宿主机清单
转换成真实设备文件的 minor 编号，不能假定编号 1 就对应 `/dev/nvidia1`。
例如枚举编号 1 的 minor 为 3 时，应挂载 `/dev/nvidia3`，遥测仍查询枚举编号 1。

`ManagedDockerEnvironment` 使用 Harbor 的自定义环境接口，将租约资源和归属标签
加入每个环境。标签覆盖独立 verifier 及其网络辅助容器，因此 worker 退出后的
actor 回收进程和租约回收服务都能找到它们。过期租约的清理失败时保留待重试记录，
确认容器删除前不在该节点发放新租约。正常任务仍由 Harbor 先销毁容器，再释放租约。

Qwen 的多维位置编码还要求使用与依赖版本匹配的 verl 补丁。在使用提交
`d613affc` 的独立项目环境中，从仓库根目录执行：

```bash
bash patch/apply_patch.sh verl d613affc.patch
```

补丁按实际 token 轴重建嵌套张量的边界。训练侧不能只修改可变长度维度的标记。
对训练输出启用截断过滤后，整组有效 token 为零会触发补采，连续失败由
`psrl.agentic_rl.refill_failure_threshold` 限制。应给任务配置足够的轮数和上下文预算，
避免持续补采无法完成的任务。
