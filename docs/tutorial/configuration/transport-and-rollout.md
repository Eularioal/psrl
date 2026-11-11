# 传输、生成与环境配置

> **核心结论**：本篇聚焦「传输、生成与环境配置」及其相邻完整章节。原文的例子、推导和证据保留，历史条件不代表当前部署。

## 目录



[返回专题索引](../configuration.md)

### NIXL

Configuration for RDMA-based weight synchronization via NIXL (used when
`ps_mode: nixl_cpu`).

`nixl.server_ip`
: NIXL server IP address.
  **Default:** `${psrl.ps_manager_ip}`

`nixl.server_port`
: NIXL server port.
  **Default:** `23456`

`nixl.max_pinned_temp_memory_slots`
: Number of pinned temporary memory slots for non-contiguous tensor transfers.
  Increase if you hit registration contention with many concurrent PS workers.
  **Default:** `16`

`nixl.enable_tms_for_temp_buffers`
: Manage NIXL temporary buffers with TMS, simplifying re-registration after memory
  is resumed.
  **Default:** `${psrl.tms.enable_nixl}`

### Checkpoint

Controls the Megatron checkpoint save/load strategy. Relevant only when using the
Megatron training backend.

`checkpoint.use_dcp_save`
: Whether to use verl's default DCP (Distributed Checkpointing) for save/load.

  - **`False`**: Use PSRL's per-rank `torch.save` (saves `rank_N.pt` +
    `parallel_config.json` per rank). This path is UCX-safe: it avoids DCP's
    `FullyParallelSaveStrategyWrapper` which calls `all_gather_object` on all shard
    metadata, causing a large temporary allocation that can corrupt NIXL's UCX
    endpoint memory under high memory pressure (manifests as
    `addr_version assertion` SIGABRT). The NIXL background UCX progress thread
    (`enable_prog_thread`) is kept **enabled** in this mode.

  - **`True` (default)**: Use verl's DCP path. Two patches are applied automatically:
    1. **NCCL no-fork patch**: DCP's async writer normally forks child processes
       that inherit NCCL communicators. When the child exits, `ncclCommAbort`
       corrupts the parent's NCCL state, causing a 600-second timeout then SIGABRT. The patch
       replaces the forking multiproc writer with a sequential in-process version.
    2. **NIXL prog_thread disabled**: `enable_prog_thread=False` is passed to the
       NIXL agent to prevent the UCX background thread from racing with DCP's
       `all_gather_object` memory activity.

  **Default:** `True`

### LMCache

KV cache offloading and cross-instance P2P transfer for reducing re-prefill overhead
in multi-turn workloads.

`lmcache.enable`
: Master switch for LMCache KV offloading in vLLM.
  **Default:** `False`

`lmcache.backend`
: Storage backend for offloaded KV blocks.
  - `cpu`: host memory (fast, limited by DRAM)
  - `disk`: filesystem-backed (large capacity, slower)
  - `remote`: forwards a nonempty `remote_url` as `LMCACHE_REMOTE_URL` before
    engine initialization. The shipped YAML still marks this backend TODO.
    Configuration forwarding is implemented, but this does not establish
    end-to-end remote service support. See [the verified boundary](../../design/kv_cache.md#configuration).

  **Default:** `cpu`

`lmcache.offload_size_gb`
: Total offload budget in GiB, divided automatically across TP ranks. Do **not**
  set `LMCACHE_MAX_LOCAL_CPU_SIZE` as an env var, that would apply the full budget
  to every rank.
  **Default:** `100.0`

`lmcache.chunk_size`
: Token chunk size for hash-based KV indexing (must divide the block size).
  **Default:** `256`

`lmcache.clear_on_weight_update`
: Evict all cached KV entries after each model weight pull from the PS. This prevents
  stale-weight KV from being reused in the next generation round, but it is a blunt
  instrument, it discards every reusable prefix in the offload backend once per
  weight update. The default is `False` because the
  shipped configuration relies on `multi_version_kv` instead, which is the
  finer-grained mechanism and the one P2P requires.
  **Default:** `False`

`lmcache.multi_version_kv`
: Tag cached KV entries with the model version that produced them, so a request
  running under version N can never structurally hit an entry produced under version
  M, instead of clearing the whole cache on every update. Stale entries then age out
  naturally through ordinary LRU eviction as new-version entries fill the cache. This
  is the shipped default because under `psrl.staleness > 0` different rollout
  instances can legitimately sit at different model versions at the same time, so clearing the
  whole cache on every pull would throw away prefixes that a still-behind instance
  could still use. Required when `enable_p2p: True` (the shared P2P backend has no
  clear operation and relies entirely on version tags), in which case
  `clear_on_weight_update` must be `False`.
  **Default:** `True`

`lmcache.reserve_local_cpu_size`
: GiB of CPU memory to keep free and never use for KV offloading (headroom for other
  processes on the same node).
  **Default:** `0.0`

`lmcache.save_decode_cache`
: Also cache KV from decode steps (not just prefill). Increases memory usage but
  improves multi-turn prefix reuse.
  **Default:** `True`

`lmcache.save_unfull_chunk`
: Persist a chunk even when it is not completely filled, which helps prompts shorter
  than `chunk_size`.
  **Default:** `False`. Currently has a known bug, do **not** enable.

`lmcache.cache_policy`
: Eviction policy: `LRU` or `FIFO`.
  **Default:** `LRU`

`lmcache.enable_async_loading`
: Overlap KV cache retrieval with prefill computation to reduce time-to-first-token.
  **Default:** `False`. Currently has a known bug, do **not** enable.

`lmcache.config_file`
: Path to a full LMCache YAML config. When set, **overrides all individual fields
  above**.
  **Default:** `null`

**Disk backend (when `backend: disk`)**

`lmcache.disk_path`
: Filesystem path for disk-backed KV storage. Required when `backend: disk`.
  **Default:** `null`

`lmcache.max_disk_size_gb`
: Maximum disk usage for KV storage (GiB).
  **Default:** `1000.0`

**Remote backend (when `backend: remote`)**

`lmcache.remote_url`
: URL of the remote LMCache server, for example `redis://host:6379`. The remote
  backend is not implemented yet.
  **Default:** `null`

**P2P cross-instance transfer**

`lmcache.enable_p2p`
: Enable cross-instance KV cache transfer via a shared LMCache Controller process.
  Required when `routing_strategy.kv_transfer.enable: True`.
  **Default:** `False`

`lmcache.p2p_transfer_channel`
: Transport for P2P KV transfer.
  - `nixl`: UCX-based (RDMA on multi-node, shared memory on same node). Recommended.
  - `tcp`: fallback when UCX is unavailable.

  **Default:** `nixl`

`lmcache.controller_host`
: Host where the LMCache Controller runs.
  **Default:** `${psrl.ps_manager_ip}`

`lmcache.controller_base_port`
: Base HTTP port for the LMCache Controller's REST API (`/move`, `/lookup`, etc.).
  The actual port is selected via `find_available_port()` starting here.
  **Default:** `9000`

`lmcache.controller_pull_port`
: ZMQ PULL port where the Controller listens for worker registrations and heartbeats.
  **Default:** `8300`

`lmcache.controller_reply_port`
: ZMQ REPLY port for Controller → worker task dispatch.
  **Default:** `8400`

`lmcache.controller_health_timeout_s`
: Seconds to wait for the Controller's HTTP API to become healthy before failing
  init. The Controller imports torch and vLLM at startup and runs on the busy
  `ps_manager` node, so under cluster CPU or filesystem contention it can take
  considerably longer than a standalone launch.
  **Default:** `3000`

`lmcache.gpu_pin_block_budget`
: Max number of GPU KV blocks PSRL may hold pinned simultaneously, used by
  `routing_strategy.kv_transfer.transfer_mode == "pin_sync"`. When exceeded, the
  oldest-pinned trajectory is unpinned (PSRL-side LRU). `0` means no limit.
  **Default:** `0`

```{seealso}
{doc}`../design/kv_cache`, KV cache management architecture, LMCache Controller
process, and cache eviction behavior.
```

### TMS (torch_memory_saver)

GPU memory management that transparently swaps idle tensors to CPU, enabling colocated
workloads to share GPU memory.

`tms.range`
: Scope of TMS management.
  - `null`: disabled
  - `train`: manage training worker memory only
  - `all`: manage both rollout and training worker memory

  **Default:** `null`

`tms.enable_cuda_graph`
: Release CUDA graphs via TMS when not in use. Requires `range: all`.
  **Default:** `False`

`tms.enable_nixl`
: Manage NIXL temporary buffers with TMS (simplifies re-registration after resume).
  **Default:** `False`

### Agentic RL

Settings for multi-turn agent training loops (tool-use, code generation, SWE-agent).

`agentic_rl.refill_failure_threshold`
: How many rollout groups may fail back to back, with none completing in between,
  before training aborts. A train buffer entry is all-or-nothing, so a failed group
  is purged and replaced by one fresh prompt. When the fault is deterministic (an
  unreachable registry, a broken task image) every replacement fails the same way
  and the run churns refills at zero progress instead of failing. On reaching the
  threshold the manager latches a diagnosis naming the dominant `TerminateReason`
  and raises it from `wait_for_training_batch` and `wait_for_training_chunk`, which
  are the calls the driver blocks on. Any group that reaches `OCCUPIED` clears the
  count, so sporadic failures over a long run never accumulate into a trip. A refill
  that dispatches nothing aborts immediately regardless of the count, because the
  slot is already lost at that point.
  **Default:** `32`

`agentic_rl.trajectory_output.enable`
: Whether every agent loop writes a per-trajectory text dump via `TrajectoryWriter`.
  Files land at `<dir>/v{version}/{uid}.txt`, one per rollout trajectory.
  **Default:** `True`

`agentic_rl.trajectory_output.dir`
: Output directory for the per-trajectory dumps. Empty string falls back to
  `<psrl.logging_path>/trajectories`.
  **Default:** `""`

### Broadcast Init

When loading a large model checkpoint, each PS worker normally reads from disk
independently. `broadcast_init` instead has rank-0 read the checkpoint and broadcast
weights to other PS workers via NIXL, reducing filesystem load at scale.

`broadcast_init.enabled`
: Enable rank-0 broadcast initialization.
  **Default:** `False`

`broadcast_init.algorithm`
: Broadcast algorithm. Currently only `binary_tree` is supported.
  **Default:** `binary_tree`

### Group & Buffer Post-Processing

Post-processors can filter, re-weight, or transform trajectory groups before they are
submitted to the staleness buffer.

`group_post_process.enable`
: Enable streaming group-level post-processing.
  **Default:** `False`

`group_post_process.name`
: Registered post-processor name. Options: `dynamic_sampling_filter`, `no_filter`.
  When using `dynamic_sampling_filter`, requires `algorithm.filter_groups.metric` to
  be set.
  **Default:** `null`

`buffer_post_process.enable`
: Enable batch-level buffer post-processing (applied when a full buffer is ready).
  **Default:** `False`

`buffer_post_process.name`
: Same options as `group_post_process.name`.
  **Default:** `null`

### Log Probability

`log_prob.enable_rollout_engine_log_prob`
: Whether to request token log-probabilities from the vLLM rollout engine (used for
  importance sampling corrections). Disable to reduce generation overhead when
  log-probs are not needed.
  **Default:** `True`

### Server Rollout

An optional HTTP gateway that exposes PSRL's rollout service externally (useful for
serving agent loops from non-PSRL clients).

`server_rollout.enable`
: Enable the server rollout HTTP gateway.
  **Default:** `False`

`server_rollout.gateway.router_ip`
: Bind address for the gateway process.
  **Default:** `${psrl.ps_manager_ip}`

`server_rollout.gateway.router_port`
: HTTP port for the gateway.
  **Default:** `18080`

`server_rollout.server_concurrency`
: Max concurrent HTTP connections per rollout server.
  **Default:** `64`

### Rollout Gateway (SMG)

The rollout gateway is the mandatory online request path. A Ray `RolloutGateway` actor
starts SMG and SessionRouter subprocesses, and rollout replicas register as gRPC workers.

`rollout_gateway.server_max_concurrency`
: Maximum HTTP generation concurrency per active rollout server. The shared client
  budget is this value multiplied by active rollout and colocated validation
  instances.
  **Default:** `256`

`rollout_gateway.use_distributed_post`
: Route AgentLoopWorker POST requests through a round-robin Ray actor pool to spread
  HTTP client work across nodes.
  **Default:** `False`

`rollout_gateway.post_actor_num_per_node`
: Number of distributed POST actors placed on each alive Ray node when the pool is
  enabled.
  **Default:** `8`

`rollout_gateway.rust_log_filter`
: Per-module Rust log filter for the SMG gateway process, in `RUST_LOG` tracing
  directive syntax. Overrides the gateway's default `warn` level. An empty string
  means no override. The shipped value keeps SMG quiet while leaving PSRL's
  `route_trace` and `score_trace` targets at `info` so routing decisions stay
  visible.
  **Default:**
  `"warn,smg::routers::grpc::kv_transfer=warn,smg::routers::grpc::common::stages::worker_selector::psrl=warn,route_trace=info,score_trace=info"`

`rollout_gateway.grpc_registration_health_timeout_s`
: Total seconds a replica waits for its local `VllmEngine.HealthCheck` to pass
  before registering itself with the SMG gateway.
  **Default:** `300`

`rollout_gateway.grpc_registration_health_poll_interval_s`
: Interval between those health-check polls (seconds).
  **Default:** `1.0`

`rollout_gateway.grpc_registration_health_rpc_timeout_s`
: Per-RPC timeout for a single health-check call (seconds).
  **Default:** `5.0`

`rollout_gateway.enable_kv_event_replay`
: Serve `SubscribeKvEvents` through a buffering `KvEventReplayHub` that can replay
  missed sequence numbers after a gap, instead of the inline per-subscription ZMQ
  loop.
  **Default:** `False`. Keep it disabled: the gateway's KV-event monitor accepts
  sequence gaps monotonically, so it does not depend on the hub's replay guarantee,
  and the inline path is simpler with no ingester thread and measures at roughly
  100% cache-overlap routing. The hub path is opt-in and currently exhibits a
  zero-overlap indexing bug under investigation.

SMG uses `worker_selection_strategy=psrl`, gRPC worker connections, the
routing loop, and TITO. See {doc}`../design/router_tito`.

### TransferQueue

TransferQueue configuration is a top-level block in `ppo_trainer.yaml`.

`transfer_queue.enable`
: Runtime integration flag. `main_ppo.py` enables it for the current PSRL training
  flow.
  **Default in YAML:** `False`

`transfer_queue.metrics.enabled`
: Expose Prometheus-style metrics on an HTTP `/metrics` endpoint. When `False`,
  metrics are reported through the logger only.
  **Default:** `False`

`transfer_queue.metrics.port`
: Port for that endpoint. `0` auto-assigns a free port.
  **Default:** `0`

`transfer_queue.controller.sampler`
: Metadata sampling strategy.
  **Default:** `SequentialSampler`

`transfer_queue.controller.polling_mode`
: Enable polling-mode controller behavior.
  **Default:** `False`

`transfer_queue.backend.storage_backend`
: Storage implementation: `SimpleStorage` or experimental `MooncakeStore`.
  **Default:** `SimpleStorage`

`transfer_queue.backend.SimpleStorage.total_storage_size`
: Maximum number of experience samples across storage units.
  **Default:** `100000`

`transfer_queue.backend.SimpleStorage.num_data_storage_units`
: Distributed storage-unit count. Use at least twice the node count for larger
  deployments.
  **Default:** `8`

`transfer_queue.backend.MooncakeStore.*`
: Experimental Mooncake metadata/master addresses, local host, TCP/RDMA protocol,
  memory sizes, and NIC selection. See {doc}`../design/transfer_queue`.
