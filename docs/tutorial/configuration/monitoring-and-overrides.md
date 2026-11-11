# 监控配置与覆盖示例

> **核心结论**：本篇聚焦「监控配置与覆盖示例」及其相邻完整章节。原文的例子、推导和证据保留，历史条件不代表当前部署。

## 目录

- [Overrides](#overrides)
- [Example: Advanced 7B FSDP Config](#example-advanced-7b-fsdp-config)

[返回专题索引](../configuration.md)

### Memory Logger

`memory_logger.enable`
: Enable periodic GPU memory logging for debugging memory pressure.
  **Default:** `False`

`memory_logger.interval_seconds`
: Logging interval (seconds).
  **Default:** `30`

### Profile

Switches for profiling and analysis. `disable_attn` and `fix_weight` deliberately
break training correctness. Keep them at `False` for real runs. The `time_split`
and `prefill_composition` switches are observability-only and safe to enable during
normal training.

`profile.disable_attn`
: Disable attention in the rollout engine, propagated to vLLM as `VLLM_DISABLE_ATTN`
  and to `rollout.disable_attn`. Useful for isolating attention cost when profiling.
  **Default:** `False`

`profile.fix_weight`
: Skip the weight-load step after a parameter pull, so rollout instances keep serving
  their initial weights. Useful for measuring sync overhead without the load cost.
  **Default:** `False`

`profile.time_split.enable`
: Enable periodic per-replica prefill/decode time-split logging to
  `<psrl.logging_path>/TimeSplit_I{replica_idx}.log`. Each line reports cumulative
  prefill and decode wall-time, prefill fraction, token counts, and wall time since
  engine start.
  **Default:** `True`

`profile.time_split.interval_in_s`
: Interval between time-split log entries, in seconds.
  **Default:** `60.0`

`profile.prefill_composition.enable`
: Enable per-step prefill composition logging to
  `<psrl.logging_path>/Prefill_I{replica_idx}.log`. Every engine step that runs a
  prefill kernel emits one entry: which sequences were batched (new request, chunked
  prefill, or co-scheduled decode), how many tokens each contributed as cache hit vs.
  actual compute, and the host-side step duration. No sampling. High log volume on
  long runs (potentially GB per replica).
  **Default:** `False`

---

## Overrides

Override any parameter from the command line using Hydra syntax:

```bash
python -m psrl.trainer.main_ppo \
    +psrl.staleness=3 \
    psrl.rollout_coordination.routing_strategy.method=throughput_optimal \
    psrl.deployment.n_rollout_instances=4 \
    psrl.lmcache.enable=True \
    transfer_queue.backend.storage_backend=SimpleStorage
```

Key syntax rules:

- `key=value`: Override an existing key
- `+key=value`: Add a new key not present in the default config
- `~key`: Remove a key from the config
- Use dot notation for nested keys: `psrl.rollout_coordination.routing_strategy.method=...`

:::{tip}
For complex experiments, create a separate YAML file with your overrides and pass it
with `--config-path`:

```bash
python -m psrl.trainer.main_ppo \
    --config-path=/path/to/my_overrides \
    --config-name=my_experiment
```
:::

## Example: Advanced 7B FSDP Config

Here is a representative override pattern for 4-node DAPO training from
`examples/dapo_trainer/advanced_qwen2.5_7b_fsdp.sh`:

```bash
python -m psrl.trainer.main_ppo \
    --config-path=./config --config-name='ppo_trainer' \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=64 \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=8 \
    psrl.deployment.n_rollout_instances=16 \
    psrl.deployment.train_nnodes=2 \
    psrl.deployment.total_nnodes=4 \
    psrl.rollout_coordination.partial_rollout.enable=True \
    psrl.rollout_coordination.redundant_rollout.enable=True \
    psrl.rollout_coordination.routing_strategy.method=throughput_optimal \
    psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue=True \
    psrl.rollout_coordination.sync_and_mig_strategy.method=status_based \
    psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator=kv_cache \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.enable=True \
    psrl.rollout_coordination.proactive_filter_strategy.method=retry \
    psrl.rollout_coordination.proactive_filter_strategy.threshold=4
```

```{seealso}
- {doc}`quickstart`: Minimal working example with DAPO
- {doc}`../design/staleness_control`: Staleness control design
- {doc}`../design/flexible_rollout`: Routing and rollout coordination
- {doc}`../design/kv_cache`: KV cache management
- {doc}`../design/router_tito`: SMG, SessionRouter, and TITO
- {doc}`../design/transfer_queue`: sample data plane
```
