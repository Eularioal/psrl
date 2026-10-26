# GPU 统一执行后端：episode 与 command 的第一阶段实现

> **核心结论**：两种模式现在共用受管理 GPU 作业的提交、状态、等待、执行超时、取消和收尾流程。episode 保留会话 GPU 预算，command 每次执行后确认物理归还。此次没有实现自动 offload、运行时模式切换、逐任务 TOML 模式选择或同容器多 GPU 命令并行。
>
> 模式仍由 EnvRouter 的 YAML 资源单元配置确定，任务的 task.toml 主要声明 CPU、主存和 GPU 数量。本文是[通用接入设计](env-router-current-flow.md#11-通用-gpu-执行服务的具体设计)的第一阶段落地记录，不能将完整提案视为已实现。

## 目录

- [1. 完整流程与源码](#1-完整流程与源码)
- [2. 模式在哪里选择](#2-模式在哪里选择)
- [3. 已实现接口与状态语义](#3-已实现接口与状态语义)
- [4. 真实 Docker GPU 任务验证](#4-真实-docker-gpu-任务验证)
- [5. 复现与验收边界](#5-复现与验收边界)
- [6. 后续扩展及已知限制](#6-后续扩展及已知限制)

## 1. 完整流程与源码

```text
[U1] Science runner 获得 EnvLease
      │ lease 携带固定的 gpu_reservation
      ↓
[U2] GPU 工具 start/status/wait/cancel → GpuJobManager
      │ 同一套作业身份、状态、观察超时与取消
      ↓
[U3] ManagedDockerEnvironment → GpuExecutionBackend
      │ 同容器串行、总准入期限、执行期限、结果与清理
      ├─ [U4a] EpisodeGpuReservation：使用会话已获准的预算
      └─ [U4b] CommandGpuReservation：向 Router 申请命令预算和设备权限
      ↓
[U5] Docker 内受监督的前台进程组运行真实 GPU 程序
      ↓
      程序结束 → 输出就绪 → 按持有策略收尾
      ├─ episode：作业结束，GPU 会话预算继续保留
      └─ command：节点确认物理归还，再释放命令预算
      ↓
[U6] episode 结束：结束所属作业、销毁所属环境、归还会话租约
```

图例：图里各部分对应的源码位置。

| 节点／箭头 | 源码与函数 | 输入／输出或状态变化 |
|---|---|---|
| U1 | [runner](../../examples/sciaccel_rl/runner.py#L387) | 对独立环境 GPU 的 episode 和 command 都注入 gpu_execution，并启用标准 GPU 工具 |
| U2 | [dispatch_gpu_tool](../../examples/sciaccel_rl/gpu_tools.py#L28)、[GpuJobManager.start](../../psrl/workers/env_service/gpu_jobs.py#L60) | 创建独立 job_id，立即返回，不把等待观察等同于取消 |
| U3 | [exec_gpu](../../psrl/workers/env_service/docker.py#L92)、[GpuExecutionBackend.execute](../../psrl/workers/env_service/gpu_execution.py#L102) | 共用执行、超时和异常收尾，后端接受 Docker 提供的执行与容器身份回调 |
| U4a | [EpisodeGpuReservation](../../psrl/workers/env_service/gpu_execution.py#L9) | 不重复申请预算，不在单条作业完成时释放整次会话 |
| U4b | [CommandGpuReservation](../../psrl/workers/env_service/gpu_execution.py#L34) | 保留原 Router 命令准入、异步归还及取消协议 |
| U5 | [_run_gpu_process](../../psrl/workers/env_service/docker.py#L208)、[gpu_job_process.run](../../psrl/workers/env_service/gpu_job_process.py#L35) | 在容器内建立受监督进程组，以 PID 和进程出生身份执行取消 |
| U6 | [environment.stop](../../psrl/workers/env_service/docker.py#L238)、[runner finally](../../examples/sciaccel_rl/runner.py#L554) | 汇合所属执行和归还，再销毁容器，最后释放租约 |

这些代码如果有理解上的问题，可以随时追问。

**跟着图读源码。** U1 从已获准租约构造绑定，并非由任务随意提交一个模式绕过准入。U2 把文本请求变成作业身份，等待者先在作业管理器排队。U3 再进入共享后端，调用 U4 的持有策略，获准后运行 U5。执行结果可以先就绪，command 的物理归还可能在后台继续，此时预算仍有效，后续 GPU 命令不能跳过确认。episode 不走逐命令物理归还，保留整次会话的承诺。取消正在执行的受管作业时，先等待其进程清理，再调用策略的取消操作。最终 U6 处理会话级资源。

这个抽取统一了执行层，没有把 Ledger 的会话准入和 CommandScheduler 的逐命令准入重写成一套混合范围账本。二者通过持有策略复用公共执行生命周期，后续动态策略可以继续在此边界演进。

## 2. 模式在哪里选择

**当前不是在每个 task.toml 中选择模式。** 配置来自 [Env Service YAML](../../psrl/trainer/config/psrl/env_service.yaml#L56) 中的 units，解析为 [EnvUnit.gpu_reservation](../../psrl/workers/env_service/types.py#L49)。Ledger 分配环境时把它写入租约，runner 再把租约模式传到执行后端。

下面是资源单元配置片段，地址和 GPU 身份需在执行节点现场核验：

```yaml
units:
  - unit_id: environment-gpu
    node_ip: 192.168.1.10
    cpu_quota: 8
    memory_mb: 16384
    gpu_indices: [0]
    gpu_memory_mb: 8192
    max_containers: 2
    gpu_reservation: command
```

在同一片段将 command 改为 episode，就改成整次会话持有。默认值是 episode。它是服务／资源池配置，不是任意在运行中的作业上改字串便完成切换。既有租约仍需遵守已经获准的资源承诺。

[task.toml 资源读取](../../examples/sciaccel_rl/env_demand.py#L23)主要读取 cpus、memory_mb、gpus。显存需求来自调用方覆盖或 Router 默认值，此次没有新增 TOML 字段。仅在 task.toml 增加一个 gpu_reservation 键不会让调度器按该字段选择模式。

可以在不同、不重叠的物理资源单元配置不同策略。但当前需求匹配没有新增“这个任务必须落到 command 池”的选择字段，因此不要把配置两种池理解为已经实现按任务自主选择。同一物理资源混用长短范围、持有期间动态卸载、再次恢复都需要后续协议。

## 3. 已实现接口与状态语义

对已接入独立 GPU 租约的任务，两种模式都可以使用以下同一个工具协议。此处 ./solver 是任务在容器内提供的实际前台程序路径。

```text
psrl-gpu {"op":"start","command":"./solver","cwd":"/workspace"}
psrl-gpu {"op":"status","job_id":"返回的作业编号"}
psrl-gpu {"op":"wait","job_id":"返回的作业编号","timeout_sec":1}
psrl-gpu {"op":"cancel","job_id":"返回的作业编号"}
```

start 立即返回编号，wait 的短观察超时不会终止程序。需要确认执行及持有策略收尾完成时，wait 可以传 released=true。这是兼容原接口的命名，**episode 下不代表整份会话的显存预算被释放**。状态新增 reservation 元数据，避免同名字段造成歧义。

| 观察位置 | episode | command |
|---|---|---|
| reservation_mode | episode | command |
| 单条作业完成后 capacity_released | false，会话仍持有 | true，收到物理归还确认后 |
| released_at | 不伪造逐命令释放时间 | 真实命令归还确认时间 |
| 结束一个作业，环境是否继续存在 | 是 | 是 |
| 单条作业 host telemetry | 明确返回不可用原因 | 保留原命令归属遥测 |

例如，一份 episode 租约申报 4096 MiB，先完成训练 A，再完成训练 B，两次都可以返回 completed，但账本仍保留 4096 MiB，直到该环境销毁。command 则在两次训练各自归还后降为零，下次命令重新申请。completed 表示程序及策略收尾结束，仍需检查 exit_code 是否为零，不能把非零退出码当训练成功。

GPU 作业依赖容器内可执行的 python3 和 bash，用于受监督进程组启动和取消。任务镜像若通过挂载独立 Python 环境提供解释器，应把对应 bin 目录加入容器 PATH，不能只保证训练命令中的绝对 Python 路径存在。它与任务模型是否使用 Python 无关，是当前执行适配器自身的运行依赖。

原 gpu_commands 参数保持兼容，转换为 command 策略。若新旧绑定同时提供但不一致，明确报错。未绑定受管理 GPU 租约时，不开放 start_gpu_job。普通 exec 的兼容行为和 episode 持久程序能力没有被改成逐命令强制归还，只有通过受管理作业入口提交的工作使用上述状态与取消契约。

## 4. 真实 Docker GPU 任务验证

本次验收是统一接口与持有语义，**不是 OFF／ON 性能基准**。两组都是开启 EnvRouter 的路径，episode 不是固定 OFF。没有注入额外显存占位，也不要求这次证明吞吐加速。

使用原 MLGym CIFAR 分类模型与 640 张真实训练图片的已有固定输入，Docker 中执行两条训练命令，每条训练 10 步，使用相同输入、种子、Python、依赖副本和镜像。驱动经过 Science runner、真实 Harbor 环境、EnvRouter 与新公共后端。验证代理是固定脚本，不调用 vLLM，不产生模型探索轨迹，也不是完整 PSRL 强化学习训练。

执行预算：单个 GPU 命令 180 秒，队列 120 秒，观察 240 秒，两组外层启动有界。GPU 使用一张已检查空闲的 H20。每组同一环境顺序执行两次任务，并取消一条排队中的同任务请求，验证取消不会影响活动作业。未验证同容器同时执行多个 GPU 程序，也未验证运行中 GPU 程序的实机强制取消，后者由针对性 CPU 回归覆盖。

本次结果目录为 env-service-verification/results/unified-gpu-20261026，包含源文件哈希及快照、启动配置、CUDA 预检、GPU 遥测、任务产物、作业事件和最终清理证据。

初次 episode 验证暴露任务镜像 PATH 缺少 python3，监督器退出 127，训练尚未开始。已保留 episode-missing-python 失败记录并确认资源清理，随后给两组相同容器配置补充 Python 环境的 bin 路径。该修复不改变任何 GPU 预算或调度语义。

**Results：两组均通过。**

| 本次观测 | episode | command |
|---|---:|---:|
| 完成的真实 GPU 训练命令 | 2 / 2 | 2 / 2 |
| 每条训练步数 | 10 | 10 |
| 两条退出码 | 0、0 | 0、0 |
| 任务验证评分 | 1.0 | 1.0 |
| 两次作业结束后的声明 GPU 预算，MiB | 4096、4096 | 0、0 |
| 排队请求取消 | 成功，活动任务正常完成 | 成功，活动任务正常完成 |
| 最终活动租约／声明 GPU 预算 | 0 / 0 | 0 / 0 |
| 最终所属容器／GPU 计算进程 | 0 / 0 | 0 / 0 |

训练产物报告 CUDA 设备为 H20，每次 10 步训练的平均 loss 为 2.303740954399109，进程内峰值张量分配约 142.68 MiB。宿主遥测峰值为 693.06 MiB，包含 CUDA 运行时等额外占用，两者统计口径不同。两组清理后 NVML v2 used 均为 0.1875 MiB、没有 GPU 计算进程，不把驱动少量常驻开销判成任务泄漏。

Ray NodeManagerAddress 已核对为指定执行节点，预检记录 driver 源码及依赖来源，真实任务通过新的作业状态与预算断言。原始 summary.json 汇总训练输出、逐作业 reservation 和每次执行后的账本快照。源文件快照及哈希保留在 source/ 与 source.json，实机结束后补充一处超时边界兼容：公共后端允许 prepared 接口在 3600 秒任务预算之外增加既有的 5 秒收尾余量，并通过针对性 CPU 回归。本次实机命令使用 180 秒，此修改不改变其执行条件。另将实验脚本长字符串断行。post-run-changes.json 单独记录这两处差异，未覆盖运行时快照。

这组结果验证统一后端可承载真实任务、两种预算持有范围保持差异、排队取消不干扰活动任务，并完成最终资源回收。没有高显存压力、跨容器并发或模型调用，因此不声称加速比、原生高压收益或完整 RL 训练成功。

## 5. 复现与验收边界

[verify_unified_gpu.py](env-service-verification/scripts/verify_unified_gpu.py)运行真实 Science 任务，[unified_gpu_agent.py](env-service-verification/scripts/unified_gpu_agent.py)使用统一工具并检查结果。脚本参数沿用固定数据准备产物，执行前先阅读[集群运行约束](../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)，不得照抄历史节点空闲状态。

需要传入的路径与参数：

| 参数 | 内容 |
|---|---|
| --mode | episode 或 command |
| --output | 全新结果目录，禁止覆盖历史组 |
| --workload | 含原模型 baseline.py、真实 CIFAR 图片 cifar640.npz 和有界 train.py 的目录 |
| --dependencies | 已验证的隔离 Transformers 依赖父目录，容器中只读挂载 |
| --python | 容器可访问的 Python 解释器绝对路径，其环境目录被只读挂载 |
| --image | 执行节点已有且核验的镜像 |
| --node-ip、--gpu | 实际执行节点地址及现场核验的 GPU index |
| --overlay | 历史静态 overlay 参数，本次两种 ON 模式均使用新租约生成实际设备配置 |

脚本命令以仓库根目录运行，变量需先设为上表中的真实值：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/verify_unified_gpu.py \
  --mode episode --output "$RESULT_EPISODE" --workload "$WORKLOAD" \
  --dependencies "$DEPENDENCIES" --python "$TASK_PYTHON" --image "$IMAGE" \
  --node-ip "$NODE_IP" --gpu "$GPU_INDEX" --overlay "$OVERLAY" \
  --count 2 --gpu-budget-mib 8192 --command-memory-mib 4096
```

确认本组容器和 GPU 物理释放后，再使用 command 和新的结果目录执行另一组。验收看两条训练产物是否各完成 10 步、退出码、作业模式、每次执行后的预算及最终资源释放，不用进程存活代替成功，不把冷启动时间差写成加速结论。

CPU 回归入口包括 [test_gpu_execution.py](../../tests/env_service/test_gpu_execution.py) 与 [runner 接入测试](../../tests/env_service/test_runner_env_service.py)，覆盖 episode 作业完成保留预算语义、取消等待清理、排队超时不干扰活动任务、模式拒绝及标准代理接入。既有 command、作业、prepared 执行回归共同验证兼容性。

## 6. 后续扩展及已知限制

- 静态模式仍来自 unit 配置，没有任务 TOML 模式字段、自动选择、运行时切换或常驻状态 offload／恢复策略。
- 本次统一执行后端及标准 Science 工具接入，没有全面统一所有任务初始化、独立 GPU verifier 或旧 EnvWorker 后端。
- episode 的普通工具仍可使用会话设备权限，逐作业归属遥测目前没有补齐。不能用缺失遥测表示 GPU 消耗为零。
- 同容器受管理 GPU 作业仍串行。此扩展单列 [ENV-GPU-SAME-CONTAINER TODO](gpu-jobs-and-task-telemetry.md#7-尚未完成的目标)。
- 预算仍取自租约，逐执行不同显存需求和混合持有范围账本尚未实现。此前讨论的外部占用与未兑现承诺合成问题未在此次执行层抽取中修复。
- 模型同卡借还仍沿原专门协议，本次公共作业后端明确拒绝与该绑定任意组合。容器暂停或观察超时均不意味着物理释放。
- 本次验证只说明真实任务可以通过两种策略的共同执行入口完成，不能据此声称高压力收益、动态卸载有效或新任务全部零适配接入。

本次定向回归累计覆盖 51 个不同测试并通过，新增执行层、episode runner 接入与既有 command／作业／prepared 路径均有覆盖。9 个相关 Python 文件 Ruff 与格式检查通过。全仓 pre-commit 在隔离副本仍受其他文件的既有 Ruff／格式及 prose-style 问题阻塞，自动修订没有回写主目录。生产代码、测试及文档保持主目录未提交状态，未推送或合并。
