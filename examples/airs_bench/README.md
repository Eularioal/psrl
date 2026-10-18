# AIRS-Bench RL Recipe

## TL;DR

- Run MLGym's unchanged agent against a remote persistent sandbox and capture LLM calls through TITO.
- Only public data splits are mounted into the agent. Trusted grading uses a separate container.
- Command timeout discards the episode. Admission waits consume the episode deadline but create no TITO session.
- APPS keeps private answers in a trusted judge and executes candidate programs in separate containers without datasets.

## Contents

- [Prerequisites](#prerequisites)
- [Prepare and validate](#prepare-and-validate)
- [Run](#run)
- [Resource and lifecycle settings](#resource-and-lifecycle-settings)
- [Behavior and limits](#behavior-and-limits)
- [Small CPU checks](#small-cpu-checks)

## Prerequisites

Run commands from the PSRL repository root in the project environment.

| Variable | Meaning |
|---|---|
| `PSRL_ENV_SCRIPT` | Project environment activation script |
| `AIRS_REPO` | AIRS-Bench checkout |
| `AIRS_DATA_ROOT` | Shared prepared dataset root |
| `MLGYM_REPO` | Upstream MLGym checkout accessible to agent workers |
| `HF_MODEL_PATH` | Policy model weights |
| `ENV_NODE_IP` | Ray address of the environment node |
| `AIRS_SPLIT` | Explicitly reviewed split JSON, including the pinned split when evaluating all tasks |
| `SANDBOX_IMAGE` | Prebuilt image, default `psrl/airs-bench-agent:latest` |

Export these variables before launching. MLGym and its Python dependencies must be available in
the controller environment. All task dependencies must also be installed in the sandbox image.
AIRS containers have no network by default and inherit no controller proxy environment.

## Prepare and validate

```bash
source "${PSRL_ENV_SCRIPT}"
python -m examples.airs_bench.prepare.prepare_airs_data \
    --airs-repo "${AIRS_REPO}" --data-root "${AIRS_DATA_ROOT}"
python -m examples.airs_bench.prepare.build_parquet \
    --airs-repo "${AIRS_REPO}" --data-root "${AIRS_DATA_ROOT}" \
    --split-path "${AIRS_SPLIT}" --out-dir examples/airs_bench/data
python -m examples.airs_bench.prepare.validate_runtime \
    --data-root "${AIRS_DATA_ROOT}" \
    --dataset examples/airs_bench/data/train.parquet examples/airs_bench/data/val.parquet
```

`test_with_labels` remains in prepared host data for the trusted grader. The agent receives
separate read-only mounts for `train`, optional `validation`, and `test`. It never receives
the parent dataset directory.

The pinned `prepare/split.json` contains 14 training tasks and 6 validation tasks, including
`CodeGenerationAPPSPassAt5`. APPS uses the built-in isolated judge instead of running task-local
evaluation code with submitted programs. Its CSV still contains exactly `code1` through `code5`.
Each candidate gets a fresh container with no dataset, proxy credentials or Docker socket.
Only one test input and candidate output cross its channel. Expected answers stay in the judge.
Do not report a reduced split as the full AIRS benchmark.

Build and load the image on every selected environment node before running episodes:

```bash
bash examples/airs_bench/docker/build_image.sh
docker image inspect "${SANDBOX_IMAGE:-psrl/airs-bench-agent:latest}"
```

The image has no passwordless package-manager privilege. Runtime also drops capabilities and
sets `no-new-privileges`. Images are never implicitly pulled during episode startup.

## Run

With the Ray cluster already running:

```bash
bash examples/airs_bench/run_qwen3-4b.sh
```

The launcher validates prepared task assets before starting training workers. Set `ENV_SLOTS`
to the maximum admitted episodes per environment worker. It is a ceiling, not a promise that
CPU and memory can support that many episodes.

For a fixed-action container and grading check without an LLM:

```bash
python -m examples.airs_bench.scripted_episode --task-id TextualClassificationSickAccuracy
```

For time-series tasks, provide `--prediction-length` from the public submission specification.
The script does not derive prediction shapes from hidden test labels. This checks sandbox I/O
and grading, not the MLGym agent or the training pipeline.

## Resource and lifecycle settings

| Setting | Owner and behavior |
|---|---|
| `runtime.sandbox_cpus`, `runtime.sandbox_memory` | Agent container limits, default 8 CPUs and 32 GiB |
| `runtime.verifier_cpus`, `runtime.verifier_memory` | Additional reserved grader capacity, default 1 CPU and 4 GiB |
| `runtime.apps_candidate_cpus`, `runtime.apps_candidate_memory` | Additional APPS execution budget, default 1 CPU and 512 MiB |
| `runtime.apps_case_timeout_s`, `runtime.apps_candidate_timeout_s` | APPS deadlines, default 4 seconds per input and 11 seconds across one candidate's inputs |
| `runtime.episode_timeout_s` | Deadline for admission, startup, agent execution and session processing |
| `runtime.per_action_timeout_s` | Maximum duration of one command or grading action |
| `runtime.startup_timeout_s` | Shell readiness deadline, default 60 seconds |
| `runtime.lease_timeout_s` | Owner heartbeat TTL, default 180 seconds |
| `psrl.env_worker.max_observation_chars` | Observation head and tail budget, default 8000 characters per side |
| `psrl.env_worker.cpu_budget_per_worker` | Admission CPU budget, default derived from Ray node capacity |
| `psrl.env_worker.memory_budget_bytes_per_worker` | Admission RAM budget, default derived from Ray node capacity |

The scheduler accounts for agent and verifier capacity together. Eight default episodes reserve
72 CPUs and 288 GiB. A request larger than an empty worker's budget is rejected immediately.
On shared nodes, set explicit budgets that leave capacity for other workloads.
APPS adds one serial candidate container, so its default episode reserves 10 CPUs and 36.5 GiB.

Waiters poll from the client and do not hold coordinator RPC calls. Heartbeats keep a live episode's
reservation current even during a long command. The watchdog checks every five seconds and deletes
containers on their owning node before releasing slots. Failed deletion retains capacity and is retried.
Cleanup attempts run with bounded concurrency and a 30 second RPC deadline, so one unreachable
worker does not hold up every other expired reservation.

## Behavior and limits

- **Command timeout destroys the sandbox and discards its trajectory.** It does not resume a possibly corrupt shell.
- **Shell I/O is bounded.** Commands are limited to 1 MiB and captured output to 8 MiB per command. Redirect verbose training logs to a file. Overflow discards the episode. Pipe backpressure consumes the same command deadline without blocking other sandboxes.
- **Runner timeout terminates and joins its child process.** There is no persistent runner thread pool.
- **Grading trusts only a fresh verifier and a bounded CSV snapshot.** Changing agent-side `evaluate.py` does not change reward.
- **Invalid final submissions cannot reuse a validation score.** Validation feedback remains available, but a missing or invalid final CSV earns zero. Malformed verifier metrics and unknown episode statuses discard the trajectory.
- **APPS candidate stdout cannot supply a score.** The private judge compares outputs and computes Pass@5. Invalid code, timeout and output overflow fail that candidate. Judge or executor startup failure discards the episode.
- **APPS execution is bounded.** Candidates use the image's non-root `agent` user, a read-only root, 64 process slots, a 64 MiB `/tmp`, no network and no private mounts. Each test forks a fresh Python child. Source is limited to 256 KiB per candidate and output to 1 MiB per input.
- **Preinstall APPS dependencies.** Both judge and candidate use the image's `mlgym_generic` Python. `datasets` and NumPy must be importable. Match the task's Python version before reporting benchmark-comparable scores. Serial code grading can dominate episode time on large test sets.
- **Submission snapshots are capped at 64 MiB.** Oversized files fail explicitly.
- **Task dependencies are preinstalled.** Reset activates `mlgym_generic`, skips per-episode package installation and automatic baseline execution. `runner.py` sets `cache_baseline_scores=False`.
- **Use `DefaultHistoryProcessor`.** Rewriting previous messages breaks TITO prefix alignment.
- **Preload image and dataset assets.** A stopped Docker CLI is detected immediately. A live but unresponsive daemon remains subject to bounded timeouts.
- **Worker or coordinator loss needs operational recovery.** Heartbeat recovery covers a lost episode owner while the container owner and coordinator remain reachable.

## Small CPU checks

```bash
python -m pytest tests/env_worker/test_airs_regressions.py \
    tests/examples/airs_bench/test_grading_isolation.py \
    tests/examples/airs_bench/test_apps_isolation.py \
    tests/examples/airs_bench/test_process_runner.py -q
PYTHONPATH="${MLGYM_REPO}:${PYTHONPATH:-}" \
    python -m pytest tests/examples/airs_bench/test_upstream_mlgym.py -q
```

The first command uses real local Bash and injected RPC/container ownership operations.
The second uses the actual upstream `BaseAgent` with a fixed replay, local Bash, and a tiny
CSV grading substitute. Neither starts Docker, Ray, model inference, or training.
