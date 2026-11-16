# ReTool: Code Interpreter RL

> **Core idea:** the model can ask a Python service to calculate an intermediate result, read the output, and continue its answer. This page shows the model → environment → tool → service path.

Train language models to **strategically invoke a Python code interpreter** while solving hard math problems, using reinforcement learning. This recipe is inspired by [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) (ByteDance-Seed, 2025).

---

## Contents

- [Architecture](#architecture)
- [Key Concepts](#key-concepts)
- [Prerequisites](#prerequisites)

## Architecture

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
pipeline and a CPU exercise, see [the beginner flow](../../../overview/reading-guide.md).
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

```{toctree}
:maxdepth: 1
:hidden:

prepare
training
```
