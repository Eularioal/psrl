# SWE Agentic RL

> **Core idea:** follow the runtime path, prepare the required data and execution environment, then launch and interpret the task-specific reward. The complete recipe is maintained on this page.

## Contents

- [Current Runtime Path](#current-runtime-path)
- [Supported Data Paths](#supported-data-paths)
- [Main Components](#main-components)
- [Reward Semantics](#reward-semantics)
- [Data Preparation](#data-preparation)
  - [Data Paths Summary](#data-paths-summary)
  - [Path A: Toy Dataset](#path-a-toy-dataset)
  - [Path B: SWE-smith-py](#path-b-swe-smith-py)
  - [Path C: SWE-Gym](#path-c-swe-gym)
  - [Docker Image Workflow](#docker-image-workflow)
- [Training](#training)
  - [Choose a valid data path](#choose-a-valid-data-path)
  - [SWE-specific knobs](#swe-specific-knobs)
  - [Reward function](#reward-function)
  - [Follow the implementation](#follow-the-implementation)

> **Core idea:** an agent changes a repository in one container, and a fresh grader checks the submitted patch. A teaching example is fixing a broken sum function: the agent inspects code and runs tests, then the grader applies only the patch and reruns the required tests.

This recipe trains a policy to solve software-engineering tasks with
[mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent), Docker sandboxes,
fresh-container grading, and PSRL's asynchronous GRPO/DAPO training path.

This recipe treats mini-SWE-agent as a **black-box agent**: it uses its normal Python
bindings and OpenAI-compatible model client, which talks to PSRL through a
session-scoped OpenAI endpoint served by SessionRouter.

---

## Current Runtime Path

<!-- figure:software-patch-and-fresh-grader -->
这张图沿同一个场景解释：工作容器里测试通过，为什么还要在新容器里重新评分？

[![软件修复：只把提交补丁交给新的评分环境（1/2）](../../../_static/img/document-figures/software-patch-and-fresh-grader-screen.png)](../../../_static/img/document-figures/software-patch-and-fresh-grader.png)

[![软件修复：只把提交补丁交给新的评分环境（2/2）](../../../_static/img/document-figures/software-patch-and-fresh-grader-screen-02.png)](../../../_static/img/document-figures/software-patch-and-fresh-grader.png)

Agent 在原始仓库上反复编辑与测试，提交物是代码补丁，评分从干净基准重新开始。 新评分容器隔开探索状态与最终成果，避免依赖一次偶然留下的文件或进程。 SWE 类任务机制示意。测试协议与允许提交内容按具体 benchmark 确认。 [查看矢量图](../../../_static/img/document-figures/software-patch-and-fresh-grader.svg)。
<!-- /figure:software-patch-and-fresh-grader -->

```{mermaid}
sequenceDiagram
    participant ALW as AgentLoopWorker
    participant MSL as MiniSWEAgentLoopV1
    participant SR as SessionRouter
    participant SMG as SMG TITO
    participant V as vLLM
    participant A as miniSWEagent
    participant D as Docker sandbox
    participant G as Fresh grader container
    participant TQ as TransferQueue

    ALW->>MSL: run(request)
    MSL->>SR: create TITO session and routing headers
    MSL->>A: run_agent(session-scoped API URL)
    A->>D: inspect, edit, and test repository
    A->>SR: OpenAI chat completion
    SR->>SMG: session request
    SMG->>V: route generation
    V-->>A: assistant turn
    Note over A,D: repeat until submit / max turns / timeout
    MSL->>G: apply patch and execute F2P/P2P tests
    MSL->>SR: fetch TITO session
    SR-->>MSL: tokens, masks, logprobs, turn records
    MSL->>TQ: finalized trajectory, patch, grader result
    MSL->>SR: delete session
```

Per episode:

1. `MiniSWEEnvironment.reset()` validates dataset metadata and builds the
   per-problem sandbox configuration.
2. `MiniSWEAgentLoopV1` creates one SessionRouter/TITO session containing request,
   prompt, trajectory, validation, and model-version metadata.
3. `examples/mini_swe/runner.py` starts mini-SWE-agent in a worker thread. The agent
   owns its Docker interaction loop and calls the session-scoped OpenAI endpoint.
4. SessionRouter pins later turns to the selected rollout instance. SMG TITO records
   exact model-side training data while requests pass through vLLM.
5. After submission, the runner grades the patch in a fresh container when
   `swe_grader=swebench_fresh_container`.
6. The loop fetches TITO once, builds `prompt_ids`, `response_ids`,
   `response_mask`, rollout log-probabilities, and optional routed-expert tensors,
   then writes the finalized trajectory to TransferQueue.

The environment's command outputs are included in `response_ids` with mask `0`, while
assistant-generated tokens use mask `1`. This preserves the full multi-turn context
while training only on policy tokens.

---

## Supported Data Paths

| Path | Dataset | Sandbox | Grading |
|---|---|---|---|
| SWE-smith-py | `SWE-bench/SWE-smith-py` | Per-problem `swebench/swesmith.*` image | Fresh container, generated F2P/P2P test spec |
| SWE-Gym | `SWE-Gym/SWE-Gym` | Per-problem `xingyaoww/sweb.eval.*` image | Fresh container, dataset-provided eval script |
| SWE-bench Verified | `SWE-bench/SWE-bench_Verified` | Per-problem SWE-bench image | Fresh container, usually used for validation |

:::{warning}
The current `MiniSWEEnvironment` requires `swe_grader`, `swe_problem`, and
`swe_problem_image` in every row. `prepare_simple_data.py` does not currently emit
that contract, so the toy launch path is not a valid end-to-end path until its
dataset schema or environment handling is updated.
:::

---

## Main Components

| Component | Responsibility |
|---|---|
| `MiniSWEAgentLoopV1` | Owns SessionRouter/TITO lifecycle and converts the black-box run into a PSRL trajectory |
| `MiniSWEEnvironment` | Parses parquet metadata, applies per-instance overrides, and performs safety-net Docker cleanup |
| `examples/mini_swe/runner.py` | Runs mini-SWE-agent through standard Python bindings and invokes fresh grading |
| `SessionRouter` | Preserves PSRL headers, session affinity, and session close/drain semantics |
| SMG TITO | Captures canonical model-side tokens, log-probabilities, masks, and turn boundaries |
| `MiniSWEAgentData` | Adds patch, grader result, turn count, and resolve-rate metadata |
| `examples/mini_swe/reward.py` | Converts grader output into binary or shaped training rewards |

---

## Reward Semantics

For SWE-smith, SWE-Gym, and Verified, grading uses a clean container created from the
same per-problem image:

1. Apply the submitted patch.
2. Reject policy-violating changes when configured by the grader.
3. Run FAIL_TO_PASS and PASS_TO_PASS tests.
4. Attach the structured grader result to the trajectory.

`compute_score` supports `binary`, `partial_credit`, `test_ratio`, and `shaped`
reward modes. `acc` is always the binary resolve indicator and should be used as the
primary evaluation metric.


## Data Preparation

> **Goal:** produce data rows with problem metadata, grader identity and image references, then make those images available on the allocated execution nodes. Use the supported SWE-smith or SWE-Gym path for training.

Each training path requires converting raw datasets into PSRL training parquets and warming up Docker image caches on all cluster nodes.

---

### Data Paths Summary

| Path | Dataset | Script | Docker Images | Disk Budget |
|------|---------|--------|---------------|-------------|
| **Toy (schema work only)** | `simple_cases_train.json` (40 tasks) | `prepare_simple_data.py` | `python:3.11-slim` (baked as `psrl-mini-swe:latest`) | Minimal |
| **SWE-smith** | [SWE-bench/SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) (50k) | `prepare_swebench.py` | `swebench/swesmith.x86_64.*` (~3 GB each) | 500 GB-1 TB shared FS |
| **SWE-Gym** | [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) (2438) | `prepare_swe_gym.py` | `xingyaoww/sweb.eval.x86_64.*` | 500-800 GB shared FS |

---

### Path A: Toy Dataset

**Schema-development path.** The synthetic tasks and single Docker image are retained for local development. The generated rows lack the grader/problem/image contract required by the current `MiniSWEEnvironment`. Use Path B or C for a complete training recipe; the commands below only prepare the old toy artifacts.

```bash
# 1. Bake repos into Docker image
bash examples/mini_swe/prepare/docker_scripts/bake_simple_repos.sh python:3.11-slim

# 2. Generate parquets
python examples/mini_swe/prepare/prepare_simple_data.py \
    --mode simple --train_size 64 --test_size 16 \
    --output_dir examples/mini_swe/data/mini_swe_agent
```

---

### Path B: SWE-smith-py

Real-world GitHub bugs with per-problem Docker images. Supports repo-balanced subsampling.

```bash
# 1. Generate training parquet (1k subset)
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --total 1000 --per-repo-k 10 \
    --output-dir examples/mini_swe/data/swe_smith_py_1k

# 2. Generate validation parquet
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified --split test \
    --total 80 --repo-balanced \
    --output-dir examples/mini_swe/data/verified_subset_80 \
    --output-filename train.parquet

# 3. Pre-fetch Docker images to shared FS
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe --workers 4

# 4. Fan out to all cluster nodes
bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe
```

---

### Path C: SWE-Gym

Real-world bugs with pre-computed eval scripts. No `git checkout HEAD~1` needed.

```bash
# Quick start (100 instances, no Fork dependency)
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym-subset \
    --output-dir examples/mini_swe/data/swe_gym_subset_100

# Full dataset (2438 instances, requires SWE-Bench-Fork 2.0.13)
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym \
    --output-dir examples/mini_swe/data/swe_gym_2438

# Pre-fetch images
bash examples/mini_swe/prepare/docker_scripts/swe_gym.sh
```

---

### Docker Image Workflow

The two-step workflow avoids hitting Docker Hub `N × nodes` times:

1. **Prefetch** (once): `prefetch_images.sh` pulls each unique image via `skopeo` to a shared-FS tar.
2. **Fan-out** (per cluster): `load_all_nodes.sh` does `docker load` on every node via `pssh`.

Both steps are idempotent, already-cached tars and already-loaded images are skipped automatically.

---

```{seealso}
Full instructions including mirror configuration, retry strategies, and disk planning: [`examples/mini_swe/prepare/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/mini_swe/prepare/README.md)
```


## Training

> **Goal:** run the SWE-smith reference recipe with task-specific Docker images and fresh-container grading. Prepare the data, model, agent configuration and allocated cluster before launching.

### Choose a valid data path

The current `MiniSWEEnvironment` expects a grader identity, problem metadata and a
problem image. Use the [SWE-smith preparation path](#path-b-swe-smith-py)
or SWE-Gym preparation. The old toy data generator does not emit this full contract.
Its retained scripts are useful for schema work, but are not a ready training path.

The checked-in SWE-smith FSDP recipe uses **SWE-agent-LM-7B**, **4 nodes × 8 GPUs**,
and staleness 1. Its filename retains the historical `qwen_7b` label, so read its
actual model path rather than inferring the model from the filename.

```bash
bash examples/mini_swe/fsdp_qwen_7b_swe_smith.sh
# Optional: change the version-age bound to 2.
bash examples/mini_swe/fsdp_qwen_7b_swe_smith.sh 2
```

The script expects `${PSRL_WORKSPACE}/env/psrl.sh`, the model at
`${PSRL_WORKSPACE}/models/SWE-agent-LM-7B`, prepared parquets and per-problem images.
Check the [launch script](../../../../examples/mini_swe/fsdp_qwen_7b_swe_smith.sh)
before allocating resources. Other scripts may use different models, prompt
formats, layouts and reward settings.

### SWE-specific knobs

The [common agentic knobs](../index.md#common-launch-script-knobs) select the loop
and cap interaction length. This recipe then chooses:

| Knob | SWE-smith value or role |
|---|---|
| `*.rollout.agent.env.name` | `mini_swe_env`, parses each problem and owns cleanup metadata |
| `*.rollout.agent.data.name` | `mini_swe_agent_data`, converts run and grading results into training fields |
| `*.rollout.agent.agent_loop_config_path` | `swebench_agent_config_full_sweagent.yaml`, the prompt/command format for this model |
| `*.rollout.agent.num_workers` | Number of AgentLoopWorker actors, not a per-instance Docker-container limit |
| Reward manager settings | Select `examples/mini_swe/reward.py:compute_score` and `reward_mode=binary` |

For example, a correct YAML prompt format lets the model produce a command the
agent parser can execute. Selecting an incompatible format can stop interaction
before any patch is tested, even when Docker and model inference are healthy.

### Reward function

For SWE-smith, SWE-Gym and Verified, the grader applies the patch in a fresh
container from the problem image and runs **FAIL_TO_PASS** tests (the bug should be
fixed) and **PASS_TO_PASS** tests (existing behavior should still work).

With the script's `reward_mode=binary`, the actual reward function returns:

| Condition | `score` | `acc` |
|---|---|---|
| Zero turns, such as an aborted run | `0.0` | `0.0` |
| Grader reports resolved | `+1.0` | `1.0` |
| Other unresolved run with at least one turn | `-1.0` | `0.0` |

For a teaching example, a submitted patch that fixes two failing tests and keeps
all existing tests passing can be marked resolved and receive `+1`. A patch that
fails the full resolution condition receives `-1` in binary mode, even if some
tests pass. Other modes (`partial_credit`, `test_ratio`, `shaped`) expose more
partial feedback and have their own policy-violation handling.

#### Toy path

The reward module retains patch-overlap scoring for toy data. This is a separate
reward branch, and does not make the old toy dataset compatible with the current
environment schema.

### Follow the implementation

- [Environment metadata validation](../../../../psrl/environments/mini_swe_env.py#L143): inspect the per-problem input contract.
- [MiniSWEAgentLoopV1.run](../../../../psrl/workers/agent_loop/loops/mini_swe_agent_loop_v1.py#L79): follow session creation, execution and trajectory conversion.
- [SWE reward implementation](../../../../examples/mini_swe/reward.py#L135): check the exact scoring mode before comparing scores.
- [Complete workflow and CPU exercise](../../../overview/01-PSRL入门_从一道题到模型更新.md): understand how the resulting samples reach training.

This page's launch values and reward branches were checked against local `6aa328d`
plus existing uncommitted changes on 2026-11-16. No training was rerun. The
[recipe overview](index.md) provides the full interaction path and the
[repository recipe](../../../../examples/mini_swe/README.md) covers operational details.
