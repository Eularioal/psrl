# SWE Training

> **Goal:** run the SWE-smith reference recipe with task-specific Docker images and fresh-container grading. Prepare the data, model, agent configuration and allocated cluster before launching.

## Contents

- [Choose a valid data path](#choose-a-valid-data-path)
- [SWE-specific knobs](#swe-specific-knobs)
- [Reward function](#reward-function)
  - [Toy path](#toy-path)
- [Follow the implementation](#follow-the-implementation)

## Choose a valid data path

The current `MiniSWEEnvironment` expects a grader identity, problem metadata and a
problem image. Use the [SWE-smith preparation path](prepare.md#path-b-swe-smith-py)
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

## SWE-specific knobs

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

## Reward function

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

### Toy path

The reward module retains patch-overlap scoring for toy data. This is a separate
reward branch, and does not make the old toy dataset compatible with the current
environment schema.

## Follow the implementation

- [Environment metadata validation](../../../../psrl/environments/mini_swe_env.py#L143): inspect the per-problem input contract.
- [MiniSWEAgentLoopV1.run](../../../../psrl/workers/agent_loop/loops/mini_swe_agent_loop_v1.py#L79): follow session creation, execution and trajectory conversion.
- [SWE reward implementation](../../../../examples/mini_swe/reward.py#L135): check the exact scoring mode before comparing scores.
- [Complete workflow and CPU exercise](../../../overview/reading-guide.md): understand how the resulting samples reach training.

This page's launch values and reward branches were checked against local `6aa328d`
plus existing uncommitted changes on 2026-11-16. No training was rerun. The
[recipe overview](index.md) provides the full interaction path and the
[repository recipe](../../../../examples/mini_swe/README.md) covers operational details.
