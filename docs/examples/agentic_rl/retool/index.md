# ReTool: Code Interpreter RL

> **Core idea:** follow the runtime path, prepare the required data and execution environment, then launch and interpret the task-specific reward. The complete recipe is maintained on this page.

## Contents

- [Architecture](#architecture)
- [Key Concepts](#key-concepts)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
  - [1. Data](#1-data)
  - [2. SandboxFusion (code-execution service)](#2-sandboxfusion-code-execution-service)
- [Training](#training)
  - [ReTool-specific knobs](#retool-specific-knobs)
  - [Reward function](#reward-function)

> **Core idea:** the model can ask a Python service to calculate an intermediate result, read the output, and continue its answer. This page shows the model → environment → tool → service path.

Train language models to **strategically invoke a Python code interpreter** while solving hard math problems, using reinforcement learning. This recipe is inspired by [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) (ByteDance-Seed, 2025).

---

## Architecture

<!-- figure:rollout-batch-training -->
这张图沿同一个场景解释：一条 episode 结束后，为什么模型不一定马上开始训练？

[![异步采样与训练：一条轨迹完成后，先进入批次准备（1/2）](../../../_static/img/document-figures/rollout-batch-training-screen.png)](../../../_static/img/document-figures/rollout-batch-training.png)

[![异步采样与训练：一条轨迹完成后，先进入批次准备（2/2）](../../../_static/img/document-figures/rollout-batch-training-screen-02.png)](../../../_static/img/document-figures/rollout-batch-training.png)

固定批次教学：四条轨迹分别在第 3、5、6、8 秒完成，更新需要这四条。 优化单条延迟需要连到批次关键路径。完成一条轨迹与执行一次训练更新不是一一对应。 数字为教学假设。分块、冗余与异步配置可能改变消费条件，按实际配置说明。 [查看矢量图](../../../_static/img/document-figures/rollout-batch-training.svg)。
<!-- /figure:rollout-batch-training -->

```
┌─────────────────────────────────────────────────────┐
│               PSRL GRPO / DAPO Trainer              │
│  (actor, vLLM rollout, reward scoring, TIS)         │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │ MultiTurnAgentLoop.run()│
          │   (tool_env + hermes)   │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────────┐
     │                 │                     │
     ▼                 ▼                     ▼
┌──────────┐    ┌────────────────┐   ┌─────────────────────────┐
│   vLLM   │    │ ToolEnvironment│   │ CustomSandboxFusionTool │
│ generate │    │  .step(action) │──►│   .async_forward(code)  │
└──────────┘    └──────┬─────────┘   └───────────┬─────────────┘
                       │                         │ HTTP POST
                       │                         ▼
                       │              ┌──────────────────────┐
                       │              │ SandboxFusion service│
                       │              │  (Docker Swarm, 8080)│
                       │              └──────────────────────┘
                       │
               ┌───────┴──────────────┐
               │ compute_score        │
               │ (math_dapo +         │
               │  tool-call shaping)  │
               └──────────────────────┘
```

The call direction is environment → tool → execution service. In a teaching
example, the model asks the tool to run `print(2 + 3)`, the service returns `5`,
and that observation becomes input to the next model turn.

| Diagram part | Implementation |
|---|---|
| Multi-turn loop | [MultiTurnAgentLoop.run](../../../../psrl/workers/agent_loop/loops/multi_turn_agent_loop.py#L41) |
| Environment | [ToolEnvironment.step](../../../../psrl/environments/tool_env.py#L176) |
| Tool adapter | [CustomSandboxFusionTool.async_forward](../../../../examples/retool/retool.py#L38) |
| HTTP service address | [Tool configuration](../../../../examples/retool/sandbox_fusion_tool_config.yaml) |
| Final scoring | [compute_score](../../../../examples/retool/retool.py#L178) |

Follow those entries in table order to trace input code, tool output and reward.
The actual loop can execute tool calls concurrently. For the surrounding training
pipeline and a CPU exercise, see [the beginner flow](../../../overview/01-PSRL入门_从一道题到模型更新.md).
You can explore any one of these steps further without reading every backend first.

---

## Key Concepts

**SandboxFusion**: A Docker Swarm service that runs code on every cluster node (port 8080). Each rollout worker POSTs Python code to `localhost:8080/run_code` and receives stdout/stderr/return_code.

**Hermes tool format**: The model emits tool calls in Hermes format (`<tool_call>...</tool_call>`), which PSRL's `HermesToolParser` routes to the registered `code_interpreter` tool.

**Reward shaping**: Correct answers get `+1.0`. Incorrect answers get a base of `-1.0` with a small tool-call bonus proportional to `num_turns`, capped at `-0.6`. This encourages the model to use the code interpreter even when heading toward a wrong answer.

ReTool uses PSRL's native multi-turn `Environment` + `AgentData` integration. Model
requests still use the default SMG RolloutGateway, and completed trajectories are
written to TransferQueue. It does not require the SessionRouter/TITO black-box-agent
path used by the recommended mini-SWE-agent v1 loop.

---

## Prerequisites

- Docker daemon on every worker node
- SandboxFusion Swarm service deployed (port 8080 on all nodes)

```{seealso}
Full setup instructions including Docker image baking, Swarm deployment, and troubleshooting are in [`examples/retool/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/retool/README.md).
```


## Data Preparation

> **Goal:** prepare the math data and verify that this job’s rollout nodes can reach the configured Python execution service. Preparing data and deploying shared services are separate operations.

Before launching training you need to prepare two things: the **DAPO math parquets**
and the **SandboxFusion code-execution service** that every rollout worker will call
into over HTTP.

---

### 1. Data

Download the DAPO math parquets into `${PSRL_WORKSPACE}/data/dapo/`:

```
${PSRL_WORKSPACE}/data/dapo/
├── dapo-math-17k.parquet    # training set (BytedTsinghua-SIA/DAPO-Math-17k)
├── aime-2024.parquet        # validation set (optional)
└── aime-2025.parquet        # validation set (default)
```

`CustomRLHFDataset` (in `examples/retool/retool.py`) handles tokenization, caches the
post-mapping dataset under `cache_dir/processed_datasets/<md5>.parquet`, and routes
each row through the right `(problem, answer)` extractor based on the parquet name.

:::{tip}
The download/conversion pattern follows veRL's data-prep conventions, see
**→ [veRL: Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)**
for the underlying parquet schema.
:::

---

### 2. SandboxFusion (code-execution service)

Every rollout worker emits Python `code_interpreter` tool calls that hit
`http://localhost:8080/run_code`. That endpoint is served by a cluster-wide
**SandboxFusion** Swarm service. Bringing it up is a two-step + smoke-test process.

#### Step 1: Bake the image onto every node

```bash
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_install.sh

DOCKER_NODE_IPS="${NODE_IPS}" \
DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_copy.sh
```

`docker_install.sh` pulls `code_sandbox:server` via `skopeo` into a shared-FS tar.
`docker_copy.sh` fans the tar out to every node via `pssh` and runs `docker load`.

`PSRL_WORKSPACE` must point to the shared workspace. Set `NODE_IPS` to this job's
allocated node list and replace `DOCKER_NODE_NUM=8` with its actual node count.
`DOCKER_IMAGE_TAG` must name an image accessible from your registry. Set
`DOCKERHUB_MIRROR` only when your deployment explicitly uses a mirror.

#### Step 2: Deploy the SandboxFusion Swarm service

This helper changes Swarm membership and replaces its named sandbox service/network.
Use it for a dedicated sandbox deployment whose resources you own. For an existing
shared service, configure its URL in the tool YAML and validate connectivity instead.

```bash
SANDBOX_NODE_IPS="${NODE_IPS}" \
SANDBOX_NODE_NUM=8 \
  bash examples/retool/sandbox_fusion/launch_service.sh
```

This will `docker swarm init` on the first host, `swarm join` everyone else, create
an overlay network, and `docker service create --publish published=8080,target=8080,mode=host`
so every node has a sandbox replica bound to `localhost:8080`.

#### Step 3: Smoke test

On any node:

```bash
curl http://localhost:8080/run_code \
    -H 'Content-Type: application/json' \
    --data-raw '{"code": "print(2+3)", "language": "python"}'
```

Expected: `"status": "Success"`, `"stdout": "5\n"`. If this fails from a rollout node,
tool calls cannot receive valid execution results. Inspect service errors separately from task-answer scores.

---

```{seealso}
Full bake / Swarm / troubleshooting docs:
[`examples/retool/docker_scripts/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/retool/docker_scripts/README.md)
and
[`examples/retool/sandbox_fusion/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/retool/sandbox_fusion/README.md).
```


## Training

> **Goal:** launch the ReTool recipe after data and execution-service checks pass, then interpret its task reward and tool-use shaping separately.

Once the [data and SandboxFusion service](#data-preparation) are in place, kick off training
with the smallest reference recipe, FSDP2 on Qwen2.5-7B, 4 nodes × 8 GPU:

```bash
bash examples/retool/fsdp_qwen_7b_dapo.sh          # default staleness=2
bash examples/retool/fsdp_qwen_7b_dapo.sh 3        # override staleness
```

Two Megatron paths (`megatron_qwen_32b_dapo.sh`, `megatron_qwen3_30b_dapo_small.sh`)
follow the same structure, switching paths only changes the cluster layout and
checkpoint backend, not the agentic-RL knobs documented below.

---

### ReTool-specific knobs

On top of the [common agentic-RL knobs](../index.md) (`multi_turn.enable`,
`multi_turn.max_turns`, `agent.env.name`, `agent.data.name`, `data.return_raw_chat`),
the launch script pins the agent loop to ReTool's tool-using environment and tells
vLLM to parse the model's output as Hermes-style tool calls.

| Knob | Value in `fsdp_qwen_7b_dapo.sh` | Role |
|---|---|---|
| `*.rollout.agent.env.name` | `tool_env` | Selects `ToolEnvironment` (`psrl/environments/tool_env.py`) |
| `*.rollout.agent.data.name` | `tool_agent_data` | Selects `ToolAgentData` (`psrl/workers/agent_loop/agent_data/tool_agent_data.py`) |
| `*.rollout.multi_turn.tool_config_path` | `examples/retool/sandbox_fusion_tool_config.yaml` | Registers `CustomSandboxFusionTool` as `code_interpreter` |
| `*.rollout.multi_turn.format` | `hermes` | Parse model output as Hermes `<tool_call>...</tool_call>` blocks |
| `data.custom_cls.path` / `.name` | `examples/retool/retool.py` / `CustomRLHFDataset` | Custom dataset class that appends the `\boxed{}` answer-format instruction |
| `custom_reward_function.path` / `.name` | `examples/retool/retool.py` / `compute_score` | Math-answer verification + tool-call reward shaping |

---

### Reward function

`compute_score` (in `examples/retool/retool.py`) verifies the model's `\boxed{...}`
answer via veRL's `math_dapo.compute_score` with `strict_box_verify=True`, then
shapes the reward:

| Outcome | Score |
|---|---|
| Correct `\boxed{answer}` | `+1.0` |
| Wrong answer (no tool calls) | `-1.0` |
| Wrong answer (with tool calls) | `-1.0` + `(num_turns - 2) / 2 * 0.1`, capped at `-0.6` |
| Malformed / no boxed answer | `0.0` |

The tool-call bonus increases a negative score toward `-0.6`; the source uses
`min(-0.6, base_score + bonus)`. For a teaching example with base score `-1`,
`num_turns=6` gives bonus `0.2` and score `-0.8`; `num_turns=14` gives bonus `0.6`
and score `-0.6` after the cap. The exact base score and malformed-answer behavior
come from the pinned veRL `math_dapo` checker. Read the local adapter at
[compute_score](../../../../examples/retool/retool.py#L178) before changing shaping.

---

```{seealso}
Full launch-script parameter reference, tool registry schema, and troubleshooting:
[`examples/retool/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/retool/README.md).
```
