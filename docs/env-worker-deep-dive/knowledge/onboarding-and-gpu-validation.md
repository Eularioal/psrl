# EnvWorker 接入实操：把现有实现接进 PSRL + 验证 GPU 真的分出去了

## TL;DR

> 本文是 [`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md)（EnvWorker 设计详细版）原第 11、12 节拆分出来的独立文档。章节编号沿用原文档编号（11、12），没有从 1 重新编号，方便和速查表、其他文档里已经写好的"详细版 §11/§12"引用对应。
>
> 读本文前建议先读完详细版的第 0-10 节（问题背景、角色分工、unit、流程、GPU 难点），了解"为什么需要 EnvWorker"；本文只讲"现在这份代码怎么接、怎么验证"，不重复讲设计动机。

## 目录

- [11. 现在就能落地：把现有 EnvWorker 接入 PSRL 的最短路径](#11-现在就能落地把现有-envworker-接入-psrl-的最短路径)
  - [第一步：打开开关，配置资源池](#第一步打开开关配置资源池)
  - [第二步：知道它在训练启动的哪一刻被创建](#第二步知道它在训练启动的哪一刻被创建)
  - [第三步：写自己的 Environment 子类，只负责"这次任务长什么样"](#第三步写自己的-environment-子类只负责这次任务长什么样)
  - [第四步：写自己的 AgentLoop 子类，负责"拿号、执行、还号"](#第四步写自己的-agentloop-子类负责拿号执行还号)
  - [第五步：dataset 行里指名用哪个 agent/env](#第五步dataset-行里指名用哪个-agentenv)
  - [第六步：先脱离 LLM 跑通链路，再接训练](#第六步先脱离-llm-跑通链路再接训练)
- [12. 构造 GPU 推理测试 case：验证 unit 真的用上了 GPU](#12-构造-gpu-推理测试-case验证-unit-真的用上了-gpu)
  - [12.1 为什么"分到配额"不等于"能用 GPU"](#121-为什么分到配额不等于能用-gpu)
  - [12.2 三层测试，从纯逻辑到真实推理](#122-三层测试从纯逻辑到真实推理)
  - [12.3 新增的集成测试怎么用](#123-新增的集成测试怎么用)
  - [12.4 如果要接成真实的训练数据集：一条 GPU 推理任务的行长什么样](#124-如果要接成真实的训练数据集一条-gpu-推理任务的行长什么样)
- [延伸阅读](#延伸阅读)

---

## 11. 现在就能落地：把现有 EnvWorker 接入 PSRL 的最短路径

第 9 节说过，代码里现在能跑的只有"旧 EnvWorker"（`psrl/psrl/workers/env_worker/`），Env Service 提案里的 EnvRouter、Gate A、Gate B 都还没有实现。这一节不再讨论提案，只讲：照着现在这份代码，怎样把一个新任务接进去、真正跑起来一个用容器的 episode。全部步骤照抄的是 AIRS-Bench 配方（`examples/airs_bench/`）已经跑通的接入方式。

### 第一步：打开开关，配置资源池

编辑训练配置里的 `psrl.env_worker`（默认值见 [`env_worker.yaml`](../../../psrl/trainer/config/psrl/env_worker.yaml)）：

```yaml
psrl:
  env_worker:
    enable: True                # 默认 False，必须手动打开
    placement: colocated        # 或 dedicated + dedicated_node_ips
    workers_per_node: 1
    cpu_slots_per_worker: 8
    gpu_slots_per_worker: 0     # 需要 GPU 沙箱时改成 >0，见第 13 节
```

`enable: True` 是唯一的开关；其余字段决定资源池的形状（第 4 节讲的 unit 概念，在这份实现里就是"一个 EnvWorker 管理的 CPU/GPU 额度"）。

### 第二步：知道它在训练启动的哪一刻被创建

PPO trainer 在拉起 rollout gateway 和 session router 之后、创建 agent loop worker 之前，检查这个开关：

```python
# psrl/psrl/trainer/ppo/ray_trainer.py:1812-1817（可运行代码，非伪代码）
if self.config.psrl.env_worker.enable:
    from psrl.workers.env_worker import EnvWorkerManager

    self.env_worker_manager = EnvWorkerManager(self.config)
    psrl_logger.info("Env worker pool built.")
```

`EnvWorkerManager.__init__`（[`manager.py:52-104`](../../../psrl/workers/env_worker/manager.py#L52)）会：

1. 用 `ray.nodes()` 找出存活节点，按 `placement` 策略筛出目标节点（第 4/9 节讲过的 `resolve_placement_node_ips`）。
2. 创建一个 `EnvWorkerCoordinator` Ray actor，并注册成**全局可按名字查到的 actor**：`.options(name="env_worker_coordinator")`。这一行是接入的关键——任何 agent loop 都能靠这个固定名字拿到它，不需要手动传引用。
3. 在每个目标节点上创建 `workers_per_node` 个 `EnvWorker` actor，把 `num_gpus=gpu_slots_per_worker` 作为 Ray 的资源请求传给 `ray.remote(...).options(...)`，然后调用 `coordinator.register_worker.remote(...)` 把它登记进容量账本。

### 第三步：写自己的 Environment 子类，只负责"这次任务长什么样"

参考 [`MLGymEnvironment`](../../../psrl/environments/mlgym_env.py)。核心分工：`Environment.reset()` 不创建沙箱，只把"创建沙箱需要的原始参数"整理成一个 dict 返回给 agent loop：

```python
# 简化伪代码，改写自 mlgym_env.py:235-263
@Environment.register("your_task_env")
class YourTaskEnvironment(Environment[dict, None]):
    async def reset(self, task: dict, **kwargs) -> tuple[dict, dict]:
        extra_info = task["extra_info"]          # 来自 dataset 行，见第四步
        observation = {
            "task_id": extra_info["task_id"],
            "image": extra_info["image"],
            "gpus": extra_info.get("gpus", 0),
            "command": extra_info["command"],     # 要在沙箱里跑的命令
        }
        return observation, {}
```

`Environment.get_environment(name, ...)`（[`base.py:138-158`](../../../psrl/environments/base.py#L138)）按名字从注册表里取实例，`name` 通常来自 dataset 行的 `env_class` 字段或 rollout 配置里的默认值。

### 第四步：写自己的 AgentLoop 子类，负责"拿号、执行、还号"

参考 [`MLGymAgentLoop`](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py)。三个必须做的事：

```python
# 简化伪代码，改写自 mlgym_agent_loop.py:70-136
@register("your_agent_name")                      # utils.py 的注册表，dataset 行用同名字符串选中它
class YourAgentLoop(SessionAgentLoop):
    def __init__(self, context, **kwargs):
        super().__init__(context=context)
        if not context.config.psrl.env_worker.enable:
            raise ValueError("该 agent loop 需要 psrl.env_worker.enable=True。")
        self._coordinator = ray.get_actor("env_worker_coordinator")   # 靠固定名字找到协调器

    async def run(self, request: dict):
        env = Environment.get_environment("your_task_env", self.config, self.reward_manager, ...)
        observation, _ = await env.reset(task=request)

        spec = SandboxSpec(image=observation["image"], gpus=observation["gpus"], network="none")
        handle = await self._coordinator.create_sandbox.remote(spec)   # 拿号：创建并登记容量
        try:
            result = await handle.exec(observation["command"], timeout_s=300.0)  # 执行
            # ... 根据 result.stdout / result.exit_code 构造 reward，与训练框架对接的部分省略 ...
        finally:
            await handle.destroy()                                     # 还号：销毁容器 + 归还容量
```

`coordinator.create_sandbox` 内部会重试一次、选一个有空位的 worker（[`coordinator.py:185-217`](../../../psrl/workers/env_worker/coordinator.py#L185)），`handle.destroy()` 销毁容器后自动调用 `coordinator.release`（[`coordinator.py:111-120`](../../../psrl/workers/env_worker/coordinator.py#L111)）归还名额——这一步放进 `finally`，是为了保证哪怕执行中抛异常，名额也不会被卡死不还。

### 第五步：dataset 行里指名用哪个 agent/env

模仿 AIRS-Bench 的 `build_row`（[`build_parquet.py:51-85`](../../../examples/airs_bench/prepare/build_parquet.py#L51)），每一行至少要有：

```python
{
    "prompt": [{"role": "user", "content": "……"}],
    "data_source": "your_task",
    "agent_name": "your_agent_name",              # 对应第四步注册的名字
    "extra_info": {
        "task_id": "...",
        "image": "...",
        "gpus": 0,
        "command": "...",
    },
}
```

`agent_name` 字段被 rollout 侧用来选择第四步注册的 AgentLoop 类；`extra_info` 会原样传到 `Environment.reset(task=request)` 里，也就是第三步读取的 `task["extra_info"]`。

### 第六步：先脱离 LLM 跑通链路，再接训练

不要第一次就带着模型一起调。仿照 [`scripted_episode.py`](../../../examples/airs_bench/scripted_episode.py) 的做法：手写固定的沙箱命令序列（不经过任何 LLM 推理），直接调用 `EnvWorkerManager` 和 `coordinator_handle()` 跑一次，确认容器能起、能执行、能销毁、日志里能看到 `Registered env worker ... with N cpu slot(s)` 这类字样。链路通了之后，再把 `psrl.env_worker.enable=True` 和你的 `agent_name` 接进真实训练配置。

## 12. 构造 GPU 推理测试 case：验证 unit 真的用上了 GPU

上一节的第五、六步是"跑通一个 CPU 任务"。这一节专门解决用户最容易踩坑的地方：**怎么确认某个 sandbox unit 不只是"分到了 GPU 配额"，而是容器里跑的程序真的能在这张 GPU 上做推理计算**。

### 12.1 为什么"分到配额"不等于"能用 GPU"

回顾 GPU 直通的实现（[`sandbox.py::build_gpu_argv`](../../../psrl/workers/env_worker/sandbox.py)）：它给容器传 `--device /dev/nvidiaN` 加上几个 NVIDIA 控制设备，再把宿主机的驱动库（`libcuda.so.1`、`libnvidia-ml.so.1`）和 `nvidia-smi` 只读挂载进容器。这条路径不依赖 `nvidia-container-runtime`，但也意味着它对"容器操作系统的 libc 版本、CUDA 用户态库版本是否与宿主机驱动匹配"更敏感——文档第 11 节末尾提到的"`nvidia-smi` 正常但 CUDA 初始化失败"就是这一类问题的真实记录。也就是说：

```text
分到 --device /dev/nvidia0     ≠     容器里的 torch.cuda.is_available() 一定是 True
```

前者是资源分配是否成功，后者是运行时库匹配是否成功，必须用真实命令分别验证，不能只看 sandbox 创建没有报错。

### 12.2 三层测试，从纯逻辑到真实推理

| 层次 | 要不要 Docker/GPU | 验证什么 | 对应文件 |
|---|---|---|---|
| 单测（已存在） | 不需要 | GPU 编号从 `CUDA_VISIBLE_DEVICES` 派生、分配互斥、超额报错 | `tests/env_worker/test_worker.py` 里 `test_gpu_slot_indices_come_from_cuda_visible_devices` 等 4 个测试 |
| 集成（本次新增） | 需要真机 GPU + Docker | 容器里能看见分到的那张卡、能跑一次真实 CUDA 矩阵乘法、两个沙箱不会拿到同一张卡 | 新增 `tests/env_worker/test_worker_gpu_integration.py` |
| 数据集层（本节给出设计，未落盘为真实训练数据） | 视你是否要接真训练 | 把"GPU 推理探针"包装成一条可被 `Environment.reset` 消费的 dataset 行 | 见 12.4 |

单测那一层已经在仓库里（第一行列出的 4 个测试），不需要新写；本节新增的是集成层，因为原有的 `test_worker_integration.py` 全部用 `network="none"` + 任意本地镜像，从未真正请求 `gpus=1`，也没有在容器里跑过任何 CUDA 指令，所以现在这份实现的 GPU 直通路径其实还没有被任何测试跑过一次。

### 12.3 新增的集成测试怎么用

新文件 [`tests/env_worker/test_worker_gpu_integration.py`](../../../tests/env_worker/test_worker_gpu_integration.py) 提供三个测试：

1. `test_gpu_sandbox_sees_the_assigned_device`：在容器里跑 `nvidia-smi --query-gpu=index --format=csv,noheader`，确认 exit_code 为 0 且真的输出了一个设备号——只验证"设备直通"这一步，不涉及深度学习框架。
2. `test_gpu_inference_runs_on_the_assigned_device`：在容器里跑一段 `python -c "..."`，用 `torch.cuda.is_available()` + 一次 1024×1024 矩阵乘法作为最小的"GPU 推理"样例，脚本打印一行 `GPU_PROBE_RESULT cuda_ok=True matmul_sum=...`，测试断言这一行真的出现且 `cuda_ok=True`。这是最接近用户所说"使用 GPU 推理的那些任务"的一条测试。
3. `test_two_gpu_sandboxes_never_share_a_device`：需要至少 2 张可见 GPU，验证两个并发沙箱各自拿到的设备号不同（对 `_allocate_gpu_indices` 互斥逻辑的真实容器级回归检查，单测层用的是假 shell，这里换成真容器）。

这三个测试全部用 `pytest.mark.skipif` 做自动跳过，跳过条件是**同时**满足：宿主机没有 `docker`、没有 `nvidia-smi`、或者没有设置环境变量 `PSRL_GPU_PROBE_IMAGE`。也就是说没有 GPU 的机器上跑 `pytest` 完全不受影响，会自动跳过而不是报错。要真正跑起来，需要一台有 NVIDIA GPU、装好 Docker 的机器，并指定一个装有 CUDA 版 PyTorch 的本地镜像：

```bash
# 可运行命令。镜像需要提前 docker pull，或换成你们自己已装 torch 的沙箱镜像。
docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
PSRL_GPU_PROBE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \
    pytest tests/env_worker/test_worker_gpu_integration.py -v
```

第 3 个测试额外要求宿主机可见至少 2 张 GPU，否则会在测试内部自我跳过（不是收集阶段跳过），这一点和前两个测试略有不同。

### 12.4 如果要接成真实的训练数据集：一条 GPU 推理任务的行长什么样

12.3 的三个测试只验证 EnvWorker 本身，不产生任何训练样本。如果你想让"GPU 推理任务"变成第 11 节第五步里说的那种 dataset 行，参照 AIRS-Bench `build_row` 的形状（[`build_parquet.py:51-85`](../../../examples/airs_bench/prepare/build_parquet.py#L51)），可以这样设计（下面是教学示例，不是仓库里已经存在的真实文件）：

```python
def build_gpu_probe_row(task_id: str, image: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": "Run the assigned GPU workload and report its result."}],
        "data_source": "gpu_probe",
        "agent_name": "gpu_probe_agent",           # 对应第 11 节第四步注册的 AgentLoop
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "task_id": task_id,
            "image": image,
            "gpus": 1,
            "command": (
                "python -c \"import torch; ok=torch.cuda.is_available(); "
                "print(f'GPU_PROBE_RESULT cuda_ok={ok}')\""
            ),
        },
    }
```

对应的 reward 函数只需要解析 `result.stdout` 里的 `GPU_PROBE_RESULT` 那一行：出现 `cuda_ok=True` 记 1 分，否则记 0 分或标记为基础设施错误（呼应第 6 节 `ROLLOUT_ERROR` 与真实任务失败的区分——探针失败大概率是环境问题，不该当成模型的错）。这只是把 12.3 的探针脚本套进了 dataset 行的壳子，`Environment.reset` 和 `AgentLoop.run` 仍是第 11 节第三、四步写的那两个类，没有新概念。

需要强调：这里给出的是**设计和最小可运行示例**，用来验证"EnvWorker 能不能把 GPU 真正交给一个做推理的容器"；它不是一个已经验证过分数分布、可直接拿去训练出效果的正式任务，真实任务的 reward 通常要比"是否检测到 GPU"复杂得多（参考 AIRS-Bench 的归一化分数设计，[`reward.py`](../../../examples/airs_bench/reward.py)）。

## 延伸阅读

- 设计动机和角色分工：[`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md) 第 0-10 节。
- 现状代码深读（账本结构、崩溃回收、Coordinator/Worker 为什么要拆）：[`implementation-and-gaps.md`](implementation-and-gaps.md)。
- SciAccel 接 GPU Gate 的实测与部署：[`sciaccel-gpu-gate.md`](sciaccel-gpu-gate.md)。
