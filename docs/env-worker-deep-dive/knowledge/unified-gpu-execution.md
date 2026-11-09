# GPU 统一执行后端：EnvRouter、容器提供者与预算策略

> **核心结论**：两种模式现在共用受管理 GPU 作业的提交、状态、等待、执行超时、取消和收尾流程。episode 保留会话 GPU 预算，command 每次执行后确认物理归还。自动 offload、运行时模式切换和逐任务 TOML 模式选择尚未实现。显式启用 MPS 后的同容器并行见 [MPS 并行后端](mps-parallel-execution.md)，本文的串行描述限定于 direct 路径。
>
> 模式仍由 EnvRouter 的 YAML 资源单元配置确定，任务的 task.toml 主要声明 CPU、主存和 GPU 数量。本文是[通用接入设计](env-router-current-flow.md#11-通用-gpu-执行服务的具体设计)的第一阶段落地记录，不能将完整提案视为已实现。

> **当前接入**：原生 EnvWorker 与 Harbor Compose 共用 `ManagedGpuExecution`，原生路径已接入 GPU 租约和 command 设备权限，创建与网络实现仍不同。当前结构读第 10 节，固定日期的实验分别按第 4、8、9 节说明的任务与版本理解，不把旧预检失败扩大成当前整个 GPU 接入尚未完成。

## 目录

- [1. 完整流程与源码](#1-完整流程与源码)
- [2. 模式在哪里选择](#2-模式在哪里选择)
- [3. 已实现接口与状态语义](#3-已实现接口与状态语义)
- [4. 真实 Docker GPU 任务验证](#4-真实-docker-gpu-任务验证)
- [5. 复现与验收边界](#5-复现与验收边界)
- [6. 后续扩展及已知限制](#6-后续扩展及已知限制)
- [7. 模式选择与 Docker 和 vLLM 的资源关系](#7-模式选择与-docker-和-vllm-的资源关系)
- [8. 原生 science 验收与下一步优先级](#8-原生-science-验收与下一步优先级)
- [9. 两种容器创建方式共用托管执行层](#9-两种容器创建方式共用托管执行层)
- [10. 当前结构与前两阶段的对照](#10-当前结构与前两阶段的对照)

## 1. 完整流程与源码

```text
[U1] Science runner 获得 EnvLease
      │ lease 携带固定的 gpu_reservation
      ↓
[U2] GPU 工具 start/status/wait/cancel → GpuJobManager
      │ 同一套作业身份、状态、观察超时与取消
      ↓
[U3] 两种容器提供者 → ManagedGpuExecution → GpuExecutionBackend
      │ direct 同容器串行、总准入期限、执行期限、结果与清理
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
| U1 | [runner](../../../examples/sciaccel_rl/runner.py#L412) | 对独立环境 GPU 的 episode 和 command 都注入 gpu_execution，并启用标准 GPU 工具 |
| U2 | [dispatch_gpu_tool](../../../examples/sciaccel_rl/gpu_tools.py#L32)、[GpuJobManager.start](../../../psrl/workers/env_service/gpu_jobs.py#L61) | 创建独立 job_id，立即返回，不把等待观察等同于取消 |
| U3 | [exec_gpu](../../../psrl/workers/env_service/managed_execution.py#L53)、[GpuExecutionBackend.execute](../../../psrl/workers/env_service/gpu_execution.py#L102) | 共用执行、超时和异常收尾，后端接受 Docker 提供的执行与容器身份回调 |
| U4a | [EpisodeGpuReservation](../../../psrl/workers/env_service/gpu_execution.py#L9) | 不重复申请预算，不在单条作业完成时释放整次会话 |
| U4b | [CommandGpuReservation](../../../psrl/workers/env_service/gpu_execution.py#L34) | 保留原 Router 命令准入、异步归还及取消协议 |
| U5 | [_run_gpu_process](../../../psrl/workers/env_service/managed_execution.py#L229)、[gpu_job_process.run](../../../psrl/workers/env_service/gpu_job_process.py#L35) | 在容器内建立受监督进程组，以 PID 和进程出生身份执行取消 |
| U6 | [environment.stop](../../../psrl/workers/env_service/docker.py#L76)、[runner finally](../../../examples/sciaccel_rl/runner.py#L610) | 汇合所属执行和归还，再销毁容器，最后释放租约 |

这些代码如果有理解上的问题，可以随时追问。

**跟着图读源码。** U1 从已获准租约构造绑定，并非由任务随意提交一个模式绕过准入。U2 把文本请求变成作业身份，等待者先在作业管理器排队。U3 再进入共享后端，调用 U4 的持有策略，获准后运行 U5。执行结果可以先就绪，command 的物理归还可能在后台继续，此时预算仍有效，后续 GPU 命令不能跳过确认。episode 不走逐命令物理归还，保留整次会话的承诺。取消正在执行的受管作业时，先等待其进程清理，再调用策略的取消操作。最终 U6 处理会话级资源。

这个抽取统一了执行层，没有把 Ledger 的会话准入和 GpuCommandScheduler 的逐命令准入重写成一套混合范围账本。二者通过持有策略复用公共执行生命周期，后续动态策略可以继续在此边界演进。

## 2. 模式在哪里选择

**当前不是在每个 task.toml 中选择模式。** 配置来自 [Env Service YAML](../../../psrl/trainer/config/psrl/env_service.yaml#L56) 中的 units，解析为 [EnvUnit.gpu_reservation](../../../psrl/workers/env_service/types.py#L49)。Ledger 分配环境时把它写入租约，runner 再把租约模式传到执行后端。

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

[task.toml 资源读取](../../../psrl/workers/env_service/harbor_resources.py#L22)主要读取 cpus、memory_mb、gpus。显存需求来自调用方覆盖或 Router 默认值，此次没有新增 TOML 字段。仅在 task.toml 增加一个 gpu_reservation 键不会让调度器按该字段选择模式。

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

[verify_unified_gpu.py](../env-service-verification/scripts/verify_unified_gpu.py)运行真实 Science 任务，[unified_gpu_agent.py](../env-service-verification/scripts/unified_gpu_agent.py)使用统一工具并检查结果。脚本参数沿用固定数据准备产物，执行前先阅读[集群运行约束](../../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)，不得照抄历史节点空闲状态。

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

CPU 回归入口包括 [test_gpu_execution.py](../../../tests/env_service/test_gpu_execution.py) 与 [runner 接入测试](../../../tests/env_service/test_runner_env_service.py)，覆盖 episode 作业完成保留预算语义、取消等待清理、排队超时不干扰活动任务、模式拒绝及标准代理接入。既有 command、作业、prepared 执行回归共同验证兼容性。

## 6. 后续扩展及已知限制

- 静态模式仍来自 unit 配置，没有任务 TOML 模式字段、自动选择、运行时切换或常驻状态 offload／恢复策略。
- 两种容器提供者已共用 GPU 作业层，任务初始化和所有 verifier 仍未自动通用化。KernelBench 已有显式 [command verifier](../../../examples/kernel_bench/verifier.py#L8)，要求专用单卡 direct、每 unit 一个会话，不能把这一特例扩大到所有任务。
- episode 的普通工具仍可使用会话设备权限，逐作业归属遥测目前没有补齐。不能用缺失遥测表示 GPU 消耗为零。
- direct 下同容器受管理 GPU 作业仍串行，MPS 下可按容量并行。同容器并发的实现与边界见 [MPS 后端](mps-parallel-execution.md)，已不再是整体未实施的 TODO。
- direct 命令预算取自租约，MPS 作业可以显式提交 `gpu_memory_mb`，未提供自动峰值预测或任意混合持有范围账本。外部占用与未兑现承诺的合成不属于本次执行层抽取，但当前已由 [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 修复为实测占用加未兑现预约，不能继续列为当前未修复项。
- 模型同卡借还仍沿原专门协议，本次公共作业后端明确拒绝与该绑定任意组合。容器暂停或观察超时均不意味着物理释放。
- 本次验证只说明真实任务可以通过两种策略的共同执行入口完成，不能据此声称高压力收益、动态卸载有效或新任务全部零适配接入。

执行层抽取阶段的定向回归累计覆盖 51 个不同测试并通过，新增执行层、episode runner 接入与既有 command／作业／prepared 路径均有覆盖。当时 9 个相关 Python 文件 Ruff 与格式检查通过，全仓 pre-commit 在隔离副本受其他文件的既有 Ruff／格式及 prose-style 问题阻塞，自动修订没有回写主目录。这是原验收记录，2026-11-05 文档审计未重跑该组回归。

## 7. 模式选择与 Docker 和 vLLM 的资源关系

**当前提前配置的是资源池的策略，单次任务拿到租约后继承它。** 不是每次 rollout 开始时都弹出选项要求人选，也不是模型运行一半决定模式。第 4 节实验由脚本分别以 --mode episode 和 --mode command 创建 Router 配置，两组是手动确定的策略。

需要分开看三条轴：持有 GPU 多久、在哪台机器哪张卡上执行、是否与 vLLM 协作交接。同一个 episode 或 command 名称不能同时回答三者。

```text
[P1] Science 任务声明 CPU、主存、GPU 需求
                 ↓
[P2] Router 选择 unit → 租约继承 episode / command
                 │                    │
                 │                    └→ 决定整次会话或逐次命令持有
                 ↓
[P3] node_policy + 节点/IP/GPU 配置 → 决定环境可放在哪里
         ├─ 不同节点
         ├─ 同节点不同卡，CPU／主存等仍可能共享
         └─ 同卡：必须另判断是否启用协作交接
                 ↓
[P4] 独立环境 GPU → 第 1 节统一执行后端
     显式绑定推理副本的 lending → 环境归还确认 ↔ vLLM sleep/wake
```

图例：图里各部分对应的源码位置。

| 节点／箭头 | 当前源码与作用 |
|---|---|
| P1 | [demand_from_task](../../../psrl/workers/env_service/harbor_resources.py#L22) 读取资源需求，没有读取任务模式字段 |
| P2 | [EnvUnit](../../../psrl/workers/env_service/types.py#L22) 设置模式，[Ledger.try_admit](../../../psrl/workers/env_service/ledger.py#L110) 选择可用 unit，[租约赋值](../../../psrl/workers/env_service/ledger.py#L181) 继承模式 |
| P3 | [NodeRules.tier](../../../psrl/workers/env_service/node_policy.py#L104) 决定节点是否允许，[Trainer 初始化](../../../psrl/trainer/ppo/ray_trainer.py#L549) 从模型 worker 放置得到训练／推理节点列表 |
| P4 独立分支 | [runner](../../../examples/sciaccel_rl/runner.py#L412) 将非 lending 的 GPU 租约交给统一执行适配器 |
| P4 同卡分支 | [Trainer._init_gpu_lending](../../../psrl/trainer/ppo/ray_trainer.py#L608) 校验完整推理副本设备及其他模型角色重叠，[GpuLendingBinding._transition](../../../psrl/workers/env_service/lending.py#L51) 执行阶段交接 |
| 配置拒绝边界 | [EnvUnit.__post_init__](../../../psrl/workers/env_service/types.py#L54) 禁止 command 绑定 rollout_replica_rank，[validate_units](../../../psrl/workers/env_service/types.py#L160) 禁止环境 unit 之间重复声明同一张卡 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，P1/P2 先说明需求怎样成为租约，再看 P3 的节点筛选。P4 的两条路径是不同配置分支，不是 command 自动进入 lending。独立 GPU 执行的收尾沿第 1 节 U5/U6；lending 先确认环境归还，再恢复模型，最终结束仍需清理环境。若模型和环境在不同卡上，它们可以同时执行，GPU 持有模式本身不会让整台机器互斥。

**本次实际拓扑。** 最近的统一后端实验只启动了环境 Docker GPU 工作，没有启动 vLLM。使用一张空闲 H20，独立 Ray 的 num_gpus=0 是为了不将 GPU 调度声明给 Ray，实际 GPU 由 EnvRouter 和 Docker 设备映射管理，不表示任务没用 GPU。因此没有本次 Docker 与 vLLM 同卡交叉或交接的证据，也不存在一套本次实际运行的 vLLM 卡表可供比较。此次只读预检查询到目标节点 8 张 GPU 都没有计算进程，不把该时刻的空闲推广成未来实验的保证。

| 资源关系 | 当前是否可能 | 应怎样解释 |
|---|---|---|
| Docker 和 vLLM 在不同机器 | 可以通过节点配置实现 | GPU 不交叉，网络／共享存储仍可能共享 |
| 同机器、不同 GPU | 可以，但需允许节点共置并显式分配不重叠的卡 | GPU 可以并行，CPU、主存和 I/O 并不隔离 |
| 同机器、同 GPU，绑定 lending | 有专门协作路径 | 物理卡相同，模型与环境使用权按阶段交接，不能随意同时使用 |
| 同机器、同 GPU，仅设置 node_policy=any | 配置可能放行 | 不会自动获得互斥或安全借还，可能争抢资源 |

默认 Env Service 关闭。开启后默认 node_policy=cpu_only，对被标记为训练／推理的节点，普通 GPU 环境请求不允许进入，CPU 任务可以，显式 lending 绑定有例外。这个规则是节点粒度，同节点 GPU 0 给模型、GPU 7 给环境也会被普通 cpu_only 规则拒绝。若配置 any 允许同机不同卡，必须核对实际设备映射。

环境 unit 之间禁止重复卡的检查，不等于已经将所有 Docker GPU 与所有模型 worker 的设备统一登记。普通 any 路径没有自动提供这样的全局互斥保证。此前默认“command 使用独立环境卡”的要求是接入约束，不能理解为仅配置一个 command 字串就已向平台取得物理独占。

**可运行 CPU 配置检查。** [explain_gpu_policy_axes.py](../env-service-verification/scripts/explain_gpu_policy_axes.py)用教学地址调用真实 NodeRules、EnvUnit 和 Ledger，不启动容器或 GPU：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/explain_gpu_policy_axes.py
```

脚本验证 P2 中租约继承模式、P3 中 cpu_only 拒绝同模型节点的普通 GPU 请求而 any 允许，以及 P4 中环境单元的重叠拒绝。它把模型 GPU 0、环境 GPU 7 设为教学例子，说明节点筛选不会逐卡区分，不是机器上实际部署了这些角色。此次实际运行通过，输出保存在 env-service-verification/results/science-next-step-20261026/policy-axes.jsonl。

## 8. 原生 science 验收与下一步优先级

**用户明确要求后续用原生 science 任务验证。** 采用 laps-accel-cuda 等原任务定义、原始输入和评分约束，不能将重封装 MLGym CIFAR、随机矩阵或 CPU science 加 GPU 占位替代。固定脚本能验证执行组件，真实模型驱动的多 rollout 还要验证模型工具使用和原任务评分，两个层次分别记录。

上一轮的 CIFAR 结果保留为真实 GPU 组件证据：它证明新后端可运行实际训练，以及两种预算持有范围有区别。它不证明原生 science 任务端到端、原生 reward_gpu 或与 vLLM 的协作已经通过。

**Results：本次重新执行原生 science 所需的生产预检。** 在首选节点检查路由身份、Docker info 与 ping、GPU 清单，再运行 [gpu_job_probe.py --preflight-only](../env-service-verification/scripts/gpu_job_probe.py#L50)。Docker 和设备控制通过，GPU 进程归属能力被拒绝，错误为 GPU attribution needs host PID procfs。结果明确 workload_started=false、node_service_closed=true，未启动任务容器、CUDA 工作或模型。原始日志与 result.json 位于 env-service-verification/results/science-next-step-20261026/native-preflight。

这个 /proc 视图用于把 GPU 驱动报告的进程对应到实际任务容器。当前缺少宿主机 PID 视图，所以即使 GPU 可以计算，也不能按该任务的资源归属要求确认“确实是这个任务产生了 GPU 活动”。这是本次重新发现的部署条件，不是沿用历史状态。原生 CUDA command 的 [runner 预检](../../../examples/sciaccel_rl/runner.py#L402)会在任务启动前检查，不能关闭检查或改任务名绕过。

原生任务的 reward_gpu 还要求真实 GPU 工作及原始数值正确性。原 oracle 的 CPU 正确解、只在独立进程分配显存、或仅看到整卡利用率都不足以通过。需要合法可运行的 GPU 求解器候选、受管理入口执行原检查，以及任务所属 GPU 遥测和独立 verifier 结果相互对应。若缺少 GPU 候选实现，也应记录为前置条件，不能把成功的 CPU oracle 当 GPU 验收。

**建议的实施顺序如下，属于后续计划，本次未实施这些调度策略。**

| 顺序 | 工作 | 为什么优先 | 最小验收 |
|---|---|---|---|
| 1 | 打通原生 science 的运行条件与完整入口 | 先建立真正目标任务的证据，避免在替代负载上持续优化 | 可核实的 host procfs／遥测条件，原任务 GPU 检查与评分通过；少量真实模型 rollout 使用标准工具 |
| 2 | 容量合成已修复，继续联合验收设备归属与共享边界 | 高压力下正确的准入比自动模式猜测更基础 | 外部占用加未兑现预约不漏算，逐卡容量可核实，普通环境卡与模型卡意外交叉被拒绝 |
| 3 | 任务显式生命周期与预算契约 | 让调度器知道哪些工作可以结束归还、哪些必须保留状态 | 任务声明受支持的持有方式与可信需求，不能误落入不兼容池，覆盖启动及 GPU 评分入口 |
| 4 | 多独立 rollout 的公平排队与受限回填 | 在已有合法边界内减少空闲，通常比状态搬运容易验证 | 同题多个独立环境，容量足够重叠、不足等待、大请求不会无限饿死，取消独立回收 |
| 5 | 基于能力声明的自动选择，再考虑运行期动态卸载 | 自动选择需要可靠输入；动态状态迁移需要更强协议 | 合法模式可解释选择，卸载可恢复正确状态并计入保存、恢复和重排队成本 |

第 2 项的旧公式反例已经修复：16 GiB 卡，外部占用 8 GiB，A 获准 6 GiB 但尚未分配，B 又申请 6 GiB，旧 max(6,8)+6=14 会错误放行。当前 [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 将实测占用与未兑现预约相加，此时 8+6+6=20，B 等待。有可信归属时只抵扣相应任务已经实际分配的部分，归属缺失时保守计费。该修复有后续高压力训练证据，但不改变本文早期组件实验的版本与结果。

第 4 项的回填是让暂时能放下的小请求利用空余容量。例如教学预算 16 GiB、已有 12 GiB，队首 A 需 8 GiB，后面的 B 需 2 GiB。当前 command [先进先出队首检查](../../../psrl/workers/env_service/command_gpu.py#L73)使 B 也等，剩余 4 GiB 暂时用不上。允许 B 有限次绕过可能提高利用率，但若 A 所需时间不确定、B 可能运行很久，就不能保证不推迟 A。需要等待时长保护、恢复预约或受限绕过规则，不能简单“谁小谁先跑”。这是候选优化，不是本次实测收益，也不依赖同容器并发。

自动选择还需要拆成两个不同目标：启动时根据任务声明选择合法模式，和运行中决定是否保存状态、腾出显存再恢复。前者可以先用确定性规则实现，但在目前模式绑定 unit、任务没有模式需求字段的状态下，需要先完成第 3 项。后者还要考虑剩余模型等待时间、释放量、状态正确性、恢复排队及公平性，沿用第 13 节的[成本分析](env-router-current-flow.md#13-常驻-gpu-状态与释放恢复成本)。它不能用 GPU 利用率为零代替“状态已释放”。

后续多 rollout 验收保持用户原意：同一道 science 题、多份独立会话与环境，先固定模式验证共享容量和完整流程，再扩大自动决策范围。同容器并发作为可选 MPS 后端提供，与多 rollout 对照的验收范围分别记录。

本次补充的 CPU 配置检查实际运行通过，新增脚本 Ruff／格式检查通过，本文及总索引 83 个本地链接、行号范围和章节锚点通过。全仓 pre-commit 在隔离副本仍因其他文件既有 Ruff／格式与 prose-style 问题失败，日志保存在本节结果目录。此次未修改生产调度代码，未绕过原生任务预检。


## 9. 两种容器创建方式共用托管执行层

**可以统一，但需要分清“怎样创建环境”和“怎样执行作业”。** 前者包括镜像构建、网络白名单、多个服务及卷配置；后者包括 GPU 准入、命令执行、等待、取消和确认归还。原生单容器与 Compose 的创建能力不同，不能为了只剩一个类名而删掉现有任务需要的能力。

当前实现把执行层收敛为一份代码，保留两个容器创建提供者。它不是把两个名字指向两段重复 GPU 逻辑，也不是宣布全部环境系统已经统一。

```text
[1] science runner / 其他 Harbor 调用方
                │ 同样的 exec_gpu / start_gpu_job / wait / cancel
                ↓
[2] ManagedGpuExecution：唯一托管 GPU 作业实现
    ├→ GpuJobManager：作业状态、输出、等待与取消
    ├→ GpuExecutionBackend：episode / command 准入和归还协议
    ├→ gpu_job_process：容器内进程组启动与取消
    │
    ├→ [3a] Compose 提供者：解析 main 容器 ID，Harbor exec
    └→ [3b] 原生提供者：解析所属容器 ID，EnvWorker exec_process
                │
                ↓
[4] 同一 EnvRouter / 节点设备授权与物理归还协议
                │
                ↓
[5] 停止新作业 → 汇合取消与归还 → 提供者删除所属容器
```

图例：图里每个部分对应的源码位置。

| 编号 | 实现与职责 |
|---|---|
| 1 | [science runner](../../../examples/sciaccel_rl/runner.py#L275) 保留整次任务准入；[_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L100) 为两种提供者生成相同来源的设备映射与归属 |
| 2 | [ManagedGpuExecution](../../../psrl/workers/env_service/managed_execution.py#L14) 统一直接执行、异步作业、预准备命令、遥测入口与关闭；[GpuExecutionBackend](../../../psrl/workers/env_service/gpu_execution.py#L84) 仍负责预算协议 |
| 3a | [Compose 的执行与身份钩子](../../../psrl/workers/env_service/docker.py#L66) 只处理其创建方式特有的传输与容器定位 |
| 3b | [原生容器身份](../../../psrl/workers/env_worker/harbor.py#L196)、[原生命令传输](../../../psrl/workers/env_worker/harbor.py#L259) 复用 EnvWorker，返回真实完整容器 ID |
| 4 | [command 准入](../../../psrl/workers/env_service/command_gpu.py#L58) 与 [归还](../../../psrl/workers/env_service/command_gpu.py#L215) 沿用同一调度器，不增加第二份 GPU 账本 |
| 5 | [公共关闭流程](../../../psrl/workers/env_service/managed_execution.py#L192) 先拒绝新作业并等待所属作业结束，各提供者随后销毁环境 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，先从 1 的租约生成进入 2 的 `exec_gpu`。2 向 3 查询容器身份，再调用 4 获取准入，授权后由 3 执行命令。异步作业是另一条进入同一个 2 的入口，输出可先于物理归还完成。取消时由 2 的进程组监督结束所属命令，再等待 4 确认。最后 5 汇合工作，创建提供者才删除容器，不能把“拿到命令输出”理解成“GPU 已归还”。

**一个具体例子。** 环境先写入 `/tmp/state`，再提交一个需要 512 MiB 预约的 GPU 作业。无论容器来自 Compose 还是原生创建，调用方都先得到 job_id，查询到 running，再等待 completed。取消另一个 `sleep 30` 作业后，容器中的 `/tmp/state` 仍能读回。这里的 512 MiB 是准入申报，CUDA 探针实际分配 64 MiB，不能把申报当作硬显存配额。`sleep 30` 用于验证进程组取消与容器保留，不证明能抢占任意长 CUDA 内核。

**原生 GPU 的安全边界。** 映射由已准入租约生成，NVML index 经库存查询转成真实设备路径，不把 index 拼成 minor number。原生 `SandboxSpec.devices` 接收已有租约的映射，不再次分配本地 GPU 槽位。command 模式的设备初始权限必须为 `m`，即不包含读写授权，同时保留禁用提权和设备节点创建能力的约束。执行绑定和设备映射的 lease_id 不一致，或含额外未支持服务，都会被拒绝。

**Results。** 本轮后台实验先检查 GPU 0，发现它已被其他进程占用，因此在创建实验资源前退出。重新核验后使用空闲 GPU 1，既有进程未被停止或干扰。

| 验收项 | Compose | 原生 EnvWorker |
|---|---|---|
| 未取得命令授权时直接运行 CUDA | 拒绝 | 拒绝 |
| `exec_gpu` 执行真实 CUDA | 内核读回 42，UUID 匹配 | 内核读回 42，UUID 匹配 |
| 异步 GPU 作业等待归还 | completed | completed |
| 取消持有 GPU 预约的 sleep 作业 | cancelled | cancelled |
| 取消后容器文件状态 | 保留 | 保留 |
| 归还后再次直接运行 CUDA | 拒绝 | 拒绝 |
| 所属容器及 GPU 进程 | 已清理 | 已清理 |

GPU 结束时 NVML 记录设备使用约 0.19 MiB、无计算进程；独立收尾再次核验设备和所属容器。四个核心实现文件的 SHA256 与实验记录一致。公共回归 61 项通过，补充原生安全专项 27 项通过，二者有重叠，不能相加写成 88 个不同测试。

真实 science 回归通过 PSRL runner 运行 `sciaccel/laps-repair-coef-2d-rktmod-l18k4` 标准答案与独立 verifier，两算例和等价性检查通过，`reward_repair=1`，任务异常为空，所属容器已清理。仍使用上一轮公开说明的缓存镜像、禁网 oracle 部署，不改变科学代码与评分。该 CPU 任务验证集成未回退，**不是原生 GPU science 任务，也不是模型训练或本轮性能对照**。任务调用约 25.82 秒仅作日志记录，不与其他时刻的结果相减宣称提速。

**可运行验证。** 在项目已配置的 Python 环境、仓库根目录运行 CPU 检查：

```bash
python -m pytest tests/env_service/test_gpu_execution.py tests/env_service/test_gpu_jobs.py tests/env_worker/test_harbor_backend.py tests/env_worker/test_sandbox.py tests/env_service/test_runner_env_service.py -q
```

真实脚本 [shared_execution_check.py](../env-service-verification/scripts/shared_execution_check.py) 内部预算 240 秒，顺序运行两个提供者，不启动模型。命令所需的节点、空闲 GPU、已缓存镜像和新输出目录均由调用者明确填写：

```bash
read -r -p 'Verified node IP: ' ENV_NODE_IP
read -r -p 'Free GPU index: ' ENV_GPU_INDEX
read -r -p 'Cached Bash/Python image: ' ENV_IMAGE
read -r -p 'New result directory: ' ENV_RESULT
python docs/env-worker-deep-dive/env-service-verification/scripts/shared_execution_check.py \
  --node-ip "$ENV_NODE_IP" --gpu "$ENV_GPU_INDEX" --image "$ENV_IMAGE" --output "$ENV_RESULT"
```

本次任务使用 nohup、setsid 和外层 timeout 脱离连接运行。启动、仅恢复未执行阶段的脚本、逐阶段退出码、GPU/科学任务证据和回归日志保存在本地 `env-service-verification/results/unified-execution-20261029/`。先读退出码与 evidence，不能重复运行已经成功的阶段。断开后已经启动的脚本可以继续，代码推理修改本身不具有脱离 Codex 会话继续执行的保证。

**仍然保留的差异。** 原生路径仍不支持 Compose 构建、多服务、allowlist 网络以及旧模型睡眠所有权钩子；普通 CPU 命令的终端语义与文件传输也尚未合并为一个传输实现。GPU 的公开托管作业协议已经统一，本轮没有验证新的 vLLM 同卡共存、模型干扰性能或同容器多个 GPU 作业并行。science 默认提供者仍保持显式选择，未偷偷改为自动回退。


## 10. 当前结构与前两阶段的对照

**仍然走 EnvRouter，但要区分“受管理路径”和“默认配置”。** 开启 `psrl.env_service.enable` 后，science 在创建容器前申请租约。租约是带有身份、资源额度和保活周期的使用许可，不是预先分配好的显存。选择原生 `environment_backend: env_worker` 时，如果没有 EnvServiceClient，runner 会直接拒绝运行。选择 `docker` 则仍允许关闭 EnvRouter 的路径。

下面是当前 direct 后端受管理路径的全局图，两个创建分支每次只选择一个。episode/command 决定预算持有期，MPS 则有独立的节点作业并发路径，见 [MPS 专题](mps-parallel-execution.md)。旧模型睡眠借卡协议另见第 7 节。

```text
[1] PSRL trainer 创建 EnvRouter
    AgentLoopManager 读取容量 → 派发到合适节点的 AgentLoopWorker
                                      │
                                      ↓
[2] science runner → EnvServiceClient → EnvRouter + Ledger
    提交 CPU/主存/容器数/GPU 需求       │
    获得 EnvLease ←───────────────────┘
        │ 后台心跳保活，租约携带节点、设备和预算策略
        ↓
[3] Harbor Job/Trial 组织 Agent、环境、verifier 评分
        │ 根据 environment_backend 二选一
        ├─ docker → ManagedDockerEnvironment → Harbor Compose 创建容器
        └─ env_worker → EnvWorkerEnvironment → 本地 EnvWorker 创建容器
        │ 两者均使用第 2 步的租约映射
        ↓
[4] GPU 工具 → ManagedGpuExecution → GpuJobManager / GpuExecutionBackend
        │ 公共提交、状态、排队期限、执行期限、取消、停止流程
        ├─ episode → 沿用第 2 步已经持有的 GPU 预算
        └─ command → [5] EnvRouter → GpuCommandScheduler → EnvNodeService
                         逐命令准入、设备授权、物理归还确认
        │ 获准后经对应提供者的命令传输，执行容器内进程
        ↓
[6] 汇合所属作业及归还 → 提供者删除容器 → runner 释放会话租约
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与实际函数 | 源码 |
|---|---|---|
| 1 | trainer 创建 Router，manager 根据容量组织 worker 派发 | [创建入口](../../../psrl/trainer/ppo/ray_trainer.py#L591)、[_refresh_env_capacity](../../../psrl/workers/agent_loop/manager.py#L947)、[get_dispatch_plan](../../../psrl/workers/agent_loop/manager.py#L984) |
| 2 | runner 在创建前申请，client 传入当前节点，Router 执行准入并返回租约 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275)、[EnvServiceClient.acquire](../../../psrl/workers/env_service/client.py#L159)、[RayEnvRouter.try_admit](../../../psrl/workers/env_service/router.py#L558) |
| 2 心跳 | 后台保活与最终释放 | [EnvLeaseHandle](../../../psrl/workers/env_service/client.py#L35) |
| 3 任务组织 | Harbor Job 的环境配置由 runner 指定，租约转为实际映射 | [后端选择](../../../examples/sciaccel_rl/runner.py#L349)、[_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L100) |
| 3 Compose | 继承 Harbor DockerEnvironment，保留其容器创建能力 | [ManagedDockerEnvironment](../../../psrl/workers/env_service/docker.py#L25) |
| 3 原生 | 继承 Harbor BaseEnvironment，直接持有本地 EnvWorker 实例 | [EnvWorkerEnvironment](../../../psrl/workers/env_worker/harbor.py#L19)、[EnvWorker.create_sandbox](../../../psrl/workers/env_worker/worker.py#L224) |
| 4 | 共享托管执行接口及 direct 单环境执行锁 | [ManagedGpuExecution](../../../psrl/workers/env_service/managed_execution.py#L14)、[GpuExecutionBackend.execute](../../../psrl/workers/env_service/gpu_execution.py#L102) |
| 5 | command 再次调用同一 Router，由命令调度器和节点服务落实授权 | [CommandGpuReservation.acquire](../../../psrl/workers/env_service/gpu_execution.py#L45)、[Router.acquire_gpu_command](../../../psrl/workers/env_service/router.py#L308)、[GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L58)、[EnvNodeService](../../../psrl/workers/env_service/node.py#L21) |
| 6 | 停止新工作并收尾，提供者删除，runner 最后释放租约 | [_close_gpu_execution](../../../psrl/workers/env_service/managed_execution.py#L257)、[原生 stop](../../../psrl/workers/env_worker/harbor.py#L182)、[Compose stop](../../../psrl/workers/env_service/docker.py#L76)、[runner finally](../../../examples/sciaccel_rl/runner.py#L610) |

这些代码如果有理解上的问题，可以随时追问。

**跟着图读源码。** 先读 1，理解 worker 的节点选择，再读 2 的 acquire，理解真正的准入点。容量快照用于派发，不等于资源已经被占住，最终仍须 Router 批准。当前 client 传入本节点，因此不是在任意 worker 中拿到远端租约就能跨节点执行 Docker。跨节点利用依赖上游先把任务派到对应节点。原生适配器创建的是本地普通 Python EnvWorker 对象，没有把远端 EnvWorkerManager 池完整接入这条 science 路径。

接着读 3，看租约如何变成容器的设备、挂载和归属标签。Harbor 仍负责 Job/Trial、Agent 与 verifier，改变的是它使用的 Environment 实现。原生路径也使用 Docker，只是通过 EnvWorker 调用 Docker 创建单容器，不经 Compose 编排。这里的“双提供者”不是两个互相竞争的调度器。

然后读 4 和 5。异步 start 先返回作业编号，执行任务另行排队，图中的 direct 后端将同一环境的受管 GPU 作业串行化。command 向 Router 重新申请，episode 则使用已经持有的预算。输出可先就绪，物理归还可在后台继续，所以取到结果不能直接视为容量已经释放。最后读 6，正常完成和异常取消都会进入收尾，物理清理失败需要保留占用或阻止重新准入，不能只清空客户端状态。

**结构对照。** 下表中的“本轮统一前”指提交 `e8ab9bf` 的结构。统一执行层已包含在 2026-11-05 核对的本地提交 `0326d46` 中，表中只比较 direct 主链路，不表示远端合并或集群部署状态。

| 维度 | 固定 baseline `0eeead861ffafed6bf31c962d259fd847775dcf9` | 本轮统一前 | 当前主目录实现 |
|---|---|---|---|
| science 资源入口 | 进程内 episode 信号量，GPU 静态 Compose 映射，无整卡锁 | 已有 EnvRouter 租约、账本及节点服务 | 保留同一 EnvRouter 控制路径 |
| Harbor 的角色 | 组织任务并使用 Docker/Compose 环境 | 组织任务，支持 Compose 和原生环境适配器 | 仍组织任务，不重写 Agent 和评分系统 |
| 原生 EnvWorker 接入 | 未有本轮这套接入 | 已接 CPU science、资源预算及生命周期 | 增加租约 GPU 设备映射及托管作业 |
| GPU 公共执行 | 没有当前托管协议 | episode/command 已共用 GpuExecutionBackend，但外围作业接口集中在 ManagedDockerEnvironment | 外围接口进一步提到 ManagedGpuExecution，两个提供者复用 |
| 容器创建 | Compose | Compose 或原生单容器 | 仍保留两种提供者 |
| 优化目的 | 静态运行 science | 集中资源控制、故障收尾、命令预算 | 避免两种创建方式各自复制 GPU 调度与取消逻辑 |

当前代码关闭 EnvRouter 时还可能走 `GpuDeviceGate` 整卡锁。它不是固定 baseline，后续做性能对照仍须检出上述固定提交，不能拿今天的 OFF 冒充。

**为什么还保留两个创建提供者。** 可以把整个系统分成三层理解。

- EnvRouter 决定能不能运行、用哪个资源单元、何时归还。这一层集中记账，防止各个 worker 只看自己的任务。
- Harbor 决定一次 science 任务如何组织，何时让 Agent 操作环境、何时评分、如何收集结果。切换底层容器实现不要求重写这些任务规则。
- 创建提供者负责把镜像变成容器并传输命令。Compose 支持更复杂的构建、网络和多服务定义，原生提供者目前覆盖预构建镜像、禁网和单容器，便于直接控制其生命周期。

共同 GPU 执行层位于这些容器提供者之上，统一作业状态和资源协议，但底层命令传输、文件传输和创建过程仍有差异。它带来的直接收益是减少重复实现和修复遗漏，让原生环境获得相同的 GPU 命令契约。**不能从代码层数减少直接推导训练更快，也没有把取消任务变成 GPU 硬件抢占。**

“GPU 策略”和“创建提供者”是两个独立选项。`environment_backend` 选择 docker/env_worker，资源单元的 `gpu_reservation` 选择 episode/command。前者回答“容器怎么创建”，后者回答“GPU 预算持有多久”。默认提供者仍是 [docker](../../../examples/sciaccel_rl/config.py#L22)，框架 Env Service 总开关默认 [False](../../../psrl/trainer/config/psrl/env_service.yaml#L6)，特定实验配置可以显式打开。不能只看选了 docker 就判断没有 Router，也不能只看有 EnvWorker 类就判断走了远端环境池。

**具体例子，以下数字是教学假设。** 假设某资源单元允许环境使用 8 GiB 显存，任务申报 3 GiB。Agent 先修改源文件，再执行两次 GPU 检验，中间还会调用模型。

| 阶段 | episode | command |
|---|---|---|
| 获准并创建环境 | GPU 预算记 3 GiB | 会话持有 CPU/主存/容器额度，此时 GPU 命令预算记 0 |
| 执行第一次 GPU 检验 | 沿用 3 GiB | 向同一个 Router 申请 3 GiB，获准才运行 |
| 命令输出已返回 | 仍持有 3 GiB | 归还尚未确认时仍计 3 GiB |
| 物理归还确认，Agent 继续改文件 | 仍持有 3 GiB，可供其他任务的声明余量为 5 GiB | 命令预算归零，声明余量恢复 8 GiB，容器和文件保留 |
| 第二次 GPU 检验 | 沿用预算 | 重新申请 3 GiB，期间可能排队 |
| 整次任务清理完成 | 删除所属容器后释放会话预算 | 汇合命令归还并删除容器，再释放会话预算 |

这里假设没有其他占用、遥测缺失和安全预留，实际准入仍需检查物理显存和模型安全余量。3 GiB 是准入承诺，不是显存硬隔离上限。更换 Compose 或原生创建不应改变表中的预算语义，这正是本轮统一的目标。普通 CPU exec 也不会因此全部经过逐命令 GPU 准入。

**可运行的 CPU 阅读实验。** [architecture_walkthrough.py](../env-service-verification/scripts/architecture_walkthrough.py) 使用 Python 标准库，先解析当前源码打印两种提供者的真实继承关系，然后按上述数字模拟四种组合。没有连接 Ray、Docker 或 GPU，也不验证调度器正确性。它假设所有申请都获准且物理归还成功，省略真实遥测、队列竞争、失败隔离和模型交互。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/architecture_walkthrough.py
```

预期先看到两个类都继承 ManagedGpuExecution。其后 Compose 与原生的预算轨迹一致，episode 在两次命令之间保持 3 GiB，command 在每次确认归还后回到 0，同时文件状态仍为 edited。两种模式最终删除环境后都归零。

**Results 与验证边界。** 本次梳理实际运行了上述 CPU 脚本，观察符合预期，没有追加昂贵任务。第 9 节的真实 CUDA 和 CPU science 检查是前一轮结果，证明已测试的执行与集成路径可用，不证明当前原生 GPU science 完整训练、训推共存性能或跨节点容器池都已经验证。当前仍缺统一的镜像构建、复杂网络与多服务生命周期，旧模型睡眠借卡也未接到原生提供者。对已有模型常驻余量的共享应继续复用 Router 的准入机制，不能因为换了创建方式就跳过模型余量检查或主动打断 vLLM。
