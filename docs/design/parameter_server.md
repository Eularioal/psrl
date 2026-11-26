# Parameter Server

The Parameter Server (PS) is the central weight-storage and synchronization hub that
makes PSRL's decoupled train/gen architecture possible. Training workers **push**
updated weights to it, and rollout instances **pull** weights when coordinated to
sync. The two compute pipelines can overlap, while parameter transfers still obey
read/write locks and resource limits.

For example, training publishes version 11 while a rollout instance is finishing
work begun at version 10. The instance later loads version 11 before serving new
work under that version. This is a teaching example, not a timing measurement.
Read [the complete flow](../overview/01-PSRL入门_从一道题到模型更新.md) if the roles are unfamiliar.

[![PSRL Parameter Server architecture（1/3）](../_static/img/document-figures/explained-ps-d3858c3-screen.png)](../_static/img/document-figures/explained-ps-d3858c3.png)

[![PSRL Parameter Server architecture（2/3）](../_static/img/document-figures/explained-ps-d3858c3-screen-02.png)](../_static/img/document-figures/explained-ps-d3858c3.png)

[![PSRL Parameter Server architecture（3/3）](../_static/img/document-figures/explained-ps-d3858c3-screen-03.png)](../_static/img/document-figures/explained-ps-d3858c3.png)

PSRL's Parameter Server: a PS Manager coordinates a group of CPU-resident PS Workers
co-located with rollout instances, and weights move over NIXL (UCX-selected RDMA /
IPC / shared memory).

<!-- explained-source:explained-ps-d3858c3 -->
**读图结论：**发布目标版本不等于所有副本已同步完成，启用时刻由完整协议决定。 实际 NIXL 后端与同步范围按配置确认。 [完整解释图（SVG）](../_static/img/document-figures/explained-ps-d3858c3.svg) · [原始机制图或实验图](../_static/img/ps.svg)。
<!-- /explained-source:explained-ps-d3858c3 -->

---

## Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
  - [PS Manager (control plane)](#ps-manager-control-plane)
  - [PS Workers (data plane)](#ps-workers-data-plane)
- [Synchronization Modes (psrl.ps_mode)](#synchronization-modes-psrlps_mode)
- [Push / Pull Path](#push--pull-path)
  - [Push (after every training step)](#push-after-every-training-step)
  - [Pull (when a rollout instance syncs)](#pull-when-a-rollout-instance-syncs)
  - [How push and pull interact](#how-push-and-pull-interact)
- [Broadcast Initialization](#broadcast-initialization)
- [Configuration Reference](#configuration-reference)
- [Compatibility With Other Subsystems](#compatibility-with-other-subsystems)

## Motivation

In tightly-coupled RLHF systems the training step and the generation step share the
same GPU memory, so updating the model is essentially free, but every training
iteration has to wait for the previous generation batch to finish, and vice versa.
PSRL breaks this barrier by keeping training and generation in separate processes
(often on separate nodes), which raises a new question: **where do the weights live,
and how do we move them across the cluster fast enough that neither side stalls?**

The PS answers this with three design decisions:

1. **CPU-resident storage by default.** GPU memory is the scarcest resource in the
   cluster, the PS keeps the canonical model in pinned CPU memory so it never
   competes with vLLM's KV cache or the trainer's optimizer state.
2. **Place PS Workers on participating nodes.** In `nixl_cpu` mode PSRL creates CPU
   storage workers on distinct rollout, actor, and validation nodes involved in
   model transfer. Consumers prefer a local shard source when available.
3. **Separate control plane from data plane.** The PS Manager is a single Ray actor
   that tracks model versions and orchestrates push/pull permissions, the actual
   bytes are streamed by PS Workers over NIXL, never through the manager.

---

## Architecture

PSRL splits the Parameter Server into two cooperating roles, both implemented under
[`psrl/workers/ps/`](https://github.com/psrl-project/psrl/blob/main/psrl/workers/ps).

### PS Manager (control plane)

A single Ray actor (`PSManager`, [`ps_manager.py`](https://github.com/psrl-project/psrl/blob/main/psrl/workers/ps/ps_manager.py))
launched on the head node. It owns:

- The **current model version tag** and a small `ModelStore` per version (either the
  state dict or a Ray `ObjectRef` to it, depending on `ps_mode`).
- The **staleness inventory** (the per-version buffers + Reserve/Occupy/Consume
  protocol described in [Staleness Control](staleness_control.md)), the PS Manager is
  also the request-status tracker, so version assignment and buffer bookkeeping live
  in the same place.
- A **single-writer / multi-reader lock** on push/pull operations
  (`_exclusive_push_locked` vs. `_shared_pull_count`): many rollout instances can
  pull concurrently, but a push from the trainer takes the lock exclusively.
- A NIXL meta-server that PS Workers and train/rollout sides connect to so they can
  discover each other's transport endpoints.

In `nixl_cpu` mode the PS Manager coordinates metadata and bytes flow between
workers. The simpler `cpu` and `cpu_ref` modes store a state dict or its references
through the Manager instead.

### PS Workers (data plane)

A group of CPU-side processes (`PSStorageWorker` +
[`PSWorkerGroup`](https://github.com/psrl-project/psrl/blob/main/psrl/workers/ps/ps_worker_group.py))
placed on relevant rollout, actor, and validation nodes. Each PS Worker holds a shard of the model
weights in pinned CPU memory and exposes NIXL endpoints that:

- The **train side** writes into (push: each train rank streams its share of the
  state dict over NIXL into the appropriate PS Worker).
- The **rollout side** reads from (pull: a rollout instance's GPU ranks pull their
  shard from the local PS Worker, same-node DMA when possible, cross-node RDMA
  otherwise).

Local PS workers make same-node transfers possible, while NIXL/UCX handles
cross-node transfers when local placement is unavailable.

```{mermaid}
sequenceDiagram
    participant TW as Train Worker
    participant PSM as PS Manager
    participant PSW as PS Worker
    participant RI as Rollout Worker

    Note over TW,RI: Training step v -> v+1
    TW->>PSM: acquire exclusive push lock(v+1)
    PSM-->>TW: lock granted
    TW->>PSW: NIXL write(shard), data plane
    TW->>PSM: push_model(version=v+1), control plane
    PSM->>PSM: bump version tag, free old buffer
    PSM->>TW: release push lock

    Note over RI: Sync trigger fires
    RI->>PSM: acquire shared pull lock
    PSM-->>RI: lock granted (concurrent with other rollouts)
    RI->>PSW: NIXL read(shard), data plane
    RI->>PSM: release pull lock
    Note over RI: New version loaded, resume generation
```

---

## Synchronization Modes (`psrl.ps_mode`)

PSRL ships **four** weight-storage / transport combinations, selected by the single
flag `psrl.ps_mode`. They differ in where the bytes live and which transport carries
them.

| `ps_mode` | Where weights live | Train → PS path | PS → Rollout path | Status |
|---|---|---|---|---|
| **`cpu`** | Real state dict in PS Manager process | `push_model_state_dict_cpu` (Ray RPC, serialized) | `pull_model_state_dict_cpu` (Ray RPC) | Implemented. Smallest models / debugging, simplest path, no NIXL setup needed. |
| **`cpu_ref`** | Ray `ObjectRef` to state dict | `push_model_state_dict_cpu_ref_list` (pass references) | `pull_model_state_dict_cpu_ref` (resolve references) | Implemented. Shipped default. References reduce repeated RPC payloads; tensor copies depend on placement and serialization. |
| **`nixl_cpu`** | Sharded across PS Workers in pinned CPU memory | NIXL write from each train rank | NIXL read into each rollout rank | Implemented. **Recommended default**: same-node DMA / cross-node RDMA, no Ray RPC on the hot path. |
| **`nixl_gpu`** | (planned) sharded across PS Workers in GPU memory | (planned) NIXL write, GPU-direct | (planned) NIXL read, GPU-direct | **TBD, not implemented.** Initializing this mode currently raises `NotImplementedError`, the scaffolding exists so the eventual GPU path can drop in without churning the API. |

The main DAPO launch example sets **`nixl_cpu`** for distributed weight transfer.
Use `cpu` / `cpu_ref` for small debugging runs where NIXL setup is unnecessary.
Select the mode explicitly when reproducing a recipe.

:::{admonition} Why CPU, not GPU, in the production default?
:class: tip

A GPU-resident PS would shave one PCIe hop off each pull, but rollout instances need
every GB of GPU memory they can get for vLLM's KV cache and CUDA graphs. Keeping the
canonical weights in pinned CPU memory + same-node DMA on pull is usually the better
trade, pulls are infrequent compared to prefill, and a larger KV cache pays off on
every request.
:::

---

## Push / Pull Path

### Push (after every training step)

1. The train worker (rank 0 in each TP/PP group) calls
   `psrl.ps_manager.acquire_exclusive_push_lock()`, waits until no concurrent pull
   is in flight.
2. Each train rank streams its **shard** of the freshly-computed state dict over
   NIXL into the matching PS Worker. Backend-specific conversion and
   synchronization happen before transfer, according to the parameter-sync plan.
3. Once all shards are received, the train side calls
   `push_model_state_dict_nixl(version_tag=v+1)`, a control-plane Ray RPC that
   simply bumps the PS Manager's version counter and frees `v−1`'s storage.
4. The PS Manager notifies the rollout coordinator that a new version is available.

### Pull (when a rollout instance syncs)

1. The rollout instance, driven by the configured sync strategy (see [Flexible
   Rollout](flexible_rollout.md)), calls
   `psrl.ps_manager.acquire_shared_pull_lock()`. Multiple instances may hold this
   lock simultaneously.
2. Each GPU rank inside the instance issues a NIXL read from its **local** PS
   Worker, UCX picks the best underlying transport (`cuda_ipc` / shared memory
   on the same node, RDMA across nodes).
3. After all shards land, the rollout instance updates its local `version_tag` and
   resumes generation under the new policy.
### How push and pull interact

The single-writer / multi-reader lock on `PSManager` gives the following semantics
(see `_try_acquire_exclusive_push_lock` / `_try_acquire_shared_pull_lock`):

| Operation | Blocks on in-flight push? | Blocks on in-flight pull(s)? |
|---|---|---|
| Push | Yes (one writer only) | Yes (waits for all readers to drain) |
| Pull | Yes (waits for the writer to finish) | **No, multiple pulls run concurrently** |

The lock admits multiple readers at once. They can still contend for memory and
network bandwidth, and a writer waits for active readers to finish. Measure these
costs for the actual model and topology rather than assuming transfer is free.

The lock conditions are in [PSManager](../../psrl/workers/ps/ps_manager.py#L1088).
The training entry [push_model](../../psrl/workers/train/base_train_worker.py#L294)
also shows backend preparation around NIXL transfer. For the full execution path
and a CPU teaching exercise, see [the reading guide](../overview/01-PSRL入门_从一道题到模型更新.md).

---

## Broadcast Initialization

**Config**: `psrl.broadcast_init.*`

At job start, every PS Worker normally reads the same checkpoint from shared
filesystem. With dozens of workers this can hammer the filesystem and add minutes to
startup. `broadcast_init` replaces the parallel disk read with a tree-broadcast over
NIXL:

```yaml
psrl:
  broadcast_init:
    enabled: true        # rank-0 reads disk, everyone else gets bytes over NIXL
    algorithm: binary_tree   # only algorithm currently shipped
```

When enabled, only the **rank-0 PS Worker** reads the checkpoint from disk, it then
fans the state dict out to all other workers in `ceil(log2(N))` rounds of binary
broadcast over the same NIXL endpoints used for steady-state push/pull
([`broadcast.py`](https://github.com/psrl-project/psrl/blob/main/psrl/workers/ps/broadcast.py)).

Use this whenever PS Workers outnumber what your shared filesystem can comfortably
serve in parallel (typically `>= 4` nodes).

---

## Configuration Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `psrl.ps_manager_ip` | str | `127.0.0.1` | IP of the head node where `PSManager` runs. Other workers connect here. |
| `psrl.ps_mode` | str | `cpu_ref` | Storage + transport mode: `cpu` / `cpu_ref` / `nixl_cpu` / `nixl_gpu`. Production: `nixl_cpu`. |
| `psrl.nixl.server_ip` | str | `${psrl.ps_manager_ip}` | NIXL meta-server bind address. |
| `psrl.nixl.server_port` | int | `23456` | NIXL meta-server port. Change if it collides with something on your cluster. |
| `psrl.nixl.max_pinned_temp_memory_slots` | int | `16` | Pinned temp-memory slot count for non-contiguous tensor transfers. Bump if you see "no free slot" warnings during push. |
| `psrl.nixl.enable_tms_for_temp_buffers` | bool | `${psrl.tms.enable_nixl}` | Let TMS reclaim NIXL pinned buffers when not transferring. See [Resource Elasticity](resource_elasticity.md). |
| `psrl.broadcast_init.enabled` | bool | `False` | Rank-0 reads checkpoint, broadcasts to all PS Workers over NIXL instead of parallel disk reads. |
| `psrl.broadcast_init.algorithm` | str | `binary_tree` | Broadcast topology. `binary_tree` is the only one currently shipped (`ring` is reserved). |

PS-Worker count and placement are not configured directly. In `nixl_cpu` mode they
are derived from the distinct nodes used by rollout, actor, and validation workers,
so model consumers can prefer a local CPU storage worker when one is available.

---

## Compatibility With Other Subsystems

- **Staleness Control**: version tags and Reserve/Occupy/Consume buffers are owned
  by the same `PSManager` actor, so weight push and buffer state always advance
  atomically. See [Staleness Control](staleness_control.md).
- **Flexible Rollout**: the sync strategy (`psrl.rollout_coordination.sync_and_mig_strategy.sync`) is
  what decides *when* a rollout instance triggers a pull, the PS only services the
  pull when asked. See [Flexible Rollout](flexible_rollout.md).
- **Resource Elasticity (TMS)**: when `psrl.tms.enable_nixl=True`, NIXL's pinned
  temp buffers are managed by TMS and can be reclaimed between transfers, lowering
  the PS's steady-state memory footprint. See [Resource Elasticity](resource_elasticity.md).
