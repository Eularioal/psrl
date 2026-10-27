# EnvRouter 命令阶段配额：保留任务状态，归还空档中的 GPU

> **核心结论**：`gpu_reservation: command` 将容器生命周期与 GPU 使用期分开。容器中的文件和 CPU 状态继续保留，显式 `exec_gpu` 命令完成后，节点先禁止新 GPU 访问、确认已有 GPU 使用结束，Router 才释放显存预算。其他任务可以使用这段空档，原任务回来时重新排队。本模式面向独立环境 GPU，要求可写的 cgroup v1 设备控制器，不与 vLLM 借卡模式叠加。
>
> 本文承接[阶段租约设计提案](rollout-gpu-lending.md#15-普通任务的阶段租约与预测回填)，说明已经实现的部分。预测器、后台 CUDA 进程的强制抢占及 cgroup v2 后端不在本实现中。真实实验的条件、结果和限制见第 7 节。
>
> 第 8 节的异步归还已经实现：显式设置 `overlap_return=True` 后，模型规划可以与物理归还检查重叠，下一条 GPU 命令仍必须获得安全授权。第 9 节跨节点调度仍是提案，当前没有活容器迁移能力。第 10 节是 CPU 教学模拟，第 11 节是实际 CUDA 加模拟规划的对照，验证了提前规划，但未证明总体提速。真实 vLLM 两次启动均达到预算，没有实际 rollout 对照结果。
>
> 第 13 节用分段计时定位了授权路径的慢查询，改为 Docker 本机接口和 NVML 精确查询。修复后的真实 GPU 小实验中，异步组预算可用后等待授权的均值从 5.473 秒降至 0.330 秒。该结果属于调度控制路径验证，不代表 science 训练加速或模型质量提升。
>
> 修复后与指定历史 OFF 的同卡、等完成量复验单独维护在[单卡基线对照](../experiments/router-baseline-control-validation.md)，区分低压力额外开销和高压力补采成本。该实验使用 science runner 的合成验证任务，不是完整训练。后续[低负载诊断与 NVML 连接复用](../experiments/router-baseline-control-validation.md#7-低负载为何变慢与连接复用)解释无容量竞争时的授权等待，并记录优化前后实测。

## 目录

- [1. 全局流程与源码图例](#1-全局流程与源码图例)
- [2. 为什么整任务模式保留配额](#2-为什么整任务模式保留配额)
- [3. 三项协议与一个数字例子](#3-三项协议与一个数字例子)
- [4. 配置与调用方式](#4-配置与调用方式)
- [5. 失败清理和适用边界](#5-失败清理和适用边界)
- [6. 跟着图读源码与可运行实验](#6-跟着图读源码与可运行实验)
- [7. Results 与验证边界](#7-results-与验证边界)
- [8. 异步归还与模型规划并行](#8-异步归还与模型规划并行)
- [9. 下一条命令留在本机还是跨节点](#9-下一条命令留在本机还是跨节点)
- [10. 教学模拟与验收边界](#10-教学模拟与验收边界)
- [11. Results：异步归还小实验](#11-results异步归还小实验)
- [12. Results：A 下一条 GPU 调用在等什么](#12-resultsa-下一条-gpu-调用在等什么)
- [13. Results：慢查询定位与直接接口修复](#13-results慢查询定位与直接接口修复)
- [14. 本机 Docker 接口到底属于哪台机器](#14-本机-docker-接口到底属于哪台机器)

## 1. 全局流程与源码图例

下面是当前实现的调用链。实线代表已经存在的调用，排队中的任务与正在进行 CPU 工作的容器并发运行。

```text
[1] science runner 申请会话 lease
         │ 预留 CPU、主存、容器数，确定所属 unit 和 GPU 身份
         v
[2] Harbor 创建持久容器，GPU 设备初始不可读写
         │ 普通 exec 继续运行 CPU 命令
         │ 需要 GPU 时显式调用 exec_gpu
         v
[3] ManagedDockerEnvironment.exec_gpu
         │ request_id，lease_id，container_id
         v
[4] EnvRouter → GpuCommandScheduler：进入 unit 的 FIFO 队列
         │ 队首预算足够：先记显存配额，生成新 epoch
         v
[5] EnvNodeService：核对容器归属，允许设备访问
         │ 确认成功才执行 GPU 命令
         v
[3] 等待前台命令结束
         │ 发起归还
         v
[5] 先禁止新设备打开，再检查旧设备句柄、映射、CUDA 进程
         │ 物理归还确认
         v
[4] GPU 配额归零，CPU / 主存 / 容器配额保留
         ├── 下一个任务可以获得 GPU
         └── 原容器继续 CPU / 模型交互，再用 GPU 时回到 [3]

[6] 整个任务结束或 lease 过期：清理所属容器，再释放剩余资源
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责 | 实现 |
|---|---|---|
| 1 | 将会话配额交给 Harbor，注入命令接口配置 | [runner.run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L397) |
| 2 | 挂载正确的设备，初始只保留设备节点权限，禁用读写 | [build_lease_overlay](../../../psrl/workers/env_service/overlay.py#L184) |
| 3 | 排队、执行、归还、取消，记录各阶段时间 | [ManagedDockerEnvironment.exec_gpu](../../../psrl/workers/env_service/docker.py#L73) |
| 4 | 显存预算、先到先服务队列、每次执行的代号 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L50)、[release](../../../psrl/workers/env_service/command_gpu.py#L165) |
| 5 | 实际设备访问控制及 GPU 使用结束检查 | [grant_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L168)、[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L182) |
| 6 | 正常结束、超时回收与物理清理确认 | [EnvRouterCore.release](../../../psrl/workers/env_service/router.py#L206)、[EnvUnitLedger._reap](../../../psrl/workers/env_service/ledger.py#L240)、[GpuCommandScheduler.finish](../../../psrl/workers/env_service/command_gpu.py#L236) |

这些代码如果有理解上的问题，可以随时沿某个编号追问。

## 2. 为什么整任务模式保留配额

**先区分三个概念。** episode 是一条任务轨迹从开始、多轮交互、评分直到清理的完整生命周期，不是单轮模型回复、一个训练 step 或整个训练作业。command 是一次显式 GPU 执行阶段，可以是一条启动 Python 程序的命令，其中包含很多 CUDA 运算。`exec_gpu` 是工具适配层申请这一阶段使用权的接口，不是新的容器类型，也不等于每次都创建和销毁容器。

两种模式共用 EnvRouter、账本、会话租约和 Harbor 持久容器。区别是 GPU 使用权和预算的有效期。CPU、主存和容器槽位在两种模式下都随会话持续保留。配置位于 [EnvUnit.gpu_reservation](../../../psrl/workers/env_service/types.py#L49)，默认 `episode`，不是每条 shell 命令临时任选一种模式。

| 模式分支与源码 | 账本如何变化 |
|---|---|
| 共同入口 [try_admit](../../../psrl/workers/env_service/ledger.py#L105) → [_fits_now](../../../psrl/workers/env_service/ledger.py#L380) | 都检查宿主预算和容器数量。episode 还要立即容纳当前 GPU 声明，command 只先确认单条需求不超过 unit 的总容量 |
| [租约初始状态](../../../psrl/workers/env_service/ledger.py#L171) | episode 初始 `gpu_active=True`，command 为 `False`。这表示账本是否承诺 GPU 额度，不代表实测 GPU 是否正在计算 |
| command 的 [acquire](../../../psrl/workers/env_service/command_gpu.py#L50) | GPU 命令获准时设为活动，加入声明额度，并授权设备访问 |
| command 的 [_return](../../../psrl/workers/env_service/command_gpu.py#L139) | 物理归还确认成功后清除活动状态，会话租约继续存在 |
| 两种模式共同的 [runner 收尾](../../../examples/sciaccel_rl/runner.py#L550) → [Router.release](../../../psrl/workers/env_service/router.py#L206) | 按归属清理容器，再归还剩余会话资源。命令模式还需汇合尚未完成的 GPU 归还与异常 |

**为什么两种策略都有用途。** 常驻 GPU 任务可能在容器里启动一个加载了模型的 Python 进程，在多轮 CPU／模型交互之间保留 GPU 上的权重、缓存和运行状态。此时命令返回或 GPU 利用率为零，都不表示资源可以交给别人。episode 提供较粗但明确的长期承诺，工具无需逐次申请。另一类任务每次启动一个前台 CUDA 程序，结束后完整释放 GPU，仅把文件留在容器中，后面还有较长 CPU 工作。这类任务可以用 command 交还空档，代价是每段的排队、授权和确认开销。

因此，episode 不是为了兼容旧代码而唯一存在，也不是固定 OFF baseline 的同义词。它与 command 都可以启用 EnvRouter。两者应共用资源管理基础设施，保留不同的持有策略，不能为了统一接口就假定所有任务都能释放 GPU 状态。未来可将长期和短期使用权抽象成统一租约接口，但这是设计方向，不是本次实现改动。

**几个容易误解的边界。** episode 不等于独占整卡，例如预算 80 GiB、两条各申报 20 GiB，宿主预算也够时仍可并发。预留 20 GiB 是账本承诺，不是提前在 CUDA 中分配了 20 GiB。command 也不是自动抢占，不能把还在运行的任意 GPU 进程暂停并搬走。当前每次命令计入的额度仍取自 `lease.demand.gpu_memory_mb`，没有逐命令传入不同显存需求的接口，细化的是持有时间，而非自动预测实际峰值。

默认 Science 代理现已通过显式 psrl-gpu 工具接入受管理 GPU 作业，但普通终端调用、任意自定义代理、任务初始化及所有 verifier 尚未统一自动转换成命令协议。通用化需要明确资源与生命周期契约，详见[当前流程第 10 至 12 节](env-router-current-flow.md#10-episode-与-command-的区别及通用性边界)。反过来，仅调用一个名为 `exec_gpu` 的方法也不能证明命令模式已开启：[实现](../../../psrl/workers/env_service/docker.py#L82)在未配置 `gpu_commands` 时直接执行普通命令，必须核对 unit 配置与 runner 注入。设备、节点身份当前仍随会话固定，归还额度不会迁移容器。模型同卡借还另有 sleep/wake 协议，不能将当前 command 模式与该分支任意组合。

`episode` 模式把“这个任务可能再用 GPU”作为资源声明的有效期。即使某个 Python 进程退出、显存已经释放，容器仍可能在下一轮对话里执行 CUDA 命令。如果 Router 此时仅仅把账本数字清零，而容器仍能随时访问 GPU，就会出现 A、B 同时认为自己有资源的情况。

CUDA 是 NVIDIA 的 GPU 编程接口。一次 CUDA 命令结束，不代表这个容器以后不再使用 GPU，也不一定代表它留下的后台进程已经退出。因此，保留配额是缺少细粒度执行约束时的保守做法，不是 GPU 驱动要求一直预留，也不是 Ray 强制要求如此。

新模式补上的关键是**执行权限与配额一起转移**。A 不用 GPU 时，设备访问控制阻止它直接重新打开 GPU。它下次调用 `exec_gpu` 必须通过 Router 获得新许可，才会恢复访问。

| 状态 | 整任务模式 `episode` | 命令模式 `command` |
|---|---|---|
| 容器建立 | 同时预留主机与 GPU 资源 | 预留主机资源，GPU 初始禁止访问 |
| GPU 命令执行期间 | 持续计费 | 先获得预算和设备权限，再执行 |
| GPU 命令结束，仍在 CPU 或模型阶段 | GPU 预算继续保留 | 确认归还后 GPU 预算释放 |
| 原任务再次需要 GPU | 直接使用一直保留的卡 | 重新申请，允许等待其他任务 |
| 整任务结束 | 容器清理后释放全部资源 | 容器清理后释放仍保留的主机资源 |

这里的“主机资源”指 CPU 配额、系统内存和容器数量，系统内存不是显卡显存。把 GPU 预算释放掉，并不会让仍然运行的容器不再占 CPU 或主存。

## 3. 三项协议与一个数字例子

显存准入按“整卡实测占用 + 每个任务尚未兑现的预约”计算，只有同一次观测中可信归属的当前任务显存才能抵扣该任务预约。缺少可信宿主进程视图时保守保留完整预约，可能降低并发。公式推导、归还边界和验证见 [GPU 显存准入](gpu-memory-admission.md)。

**第一项：任务状态保留，GPU 使用权按阶段申请。** 会话 lease 是账本对容器资源的一份记录。它可以持续存在，同时 `gpu_active=False`，表示任务活着但当前不能使用 GPU。GPU 阶段开始才把声明的显存加入占用量。

**第二项：必须先撤销访问、确认实际释放，再归还预算。** 节点通过 Linux cgroup 的设备访问控制禁止新的 GPU 设备打开。cgroup 可以理解为内核维护的一组进程及其资源规则。撤销访问不会自动收回已经打开的设备，所以还要检查容器进程的文件描述符、GPU 映射和 NVIDIA 驱动报告的计算进程。全部通过后才将 `gpu_active` 清零。内核的设备控制语义见[官方文档](https://docs.kernel.org/admin-guide/cgroup-v1/devices.html)。

**第三项：原任务回来时重新排队，过期许可不能使用。** FIFO 表示先到先服务，同一 unit 的命令按请求顺序排队。`epoch` 是同一 lease 每次获得 GPU 时递增的代号。比如 A 的第一条命令是第 1 代，下一条是第 2 代，第 1 代延迟到达的归还消息不能释放第 2 代配额。不同任务的代号分别计数，不是整个 GPU 的全局编号。

以下是教学假设，不是训练测量值。假设一个 unit 有 80 GiB 显存预算，A、B 各声明 51 GiB，所以同一时刻只能运行一个 GPU 阶段。A 先用 GPU 5 秒，CPU 工作 20 秒，再用 GPU 5 秒。B 的 GPU 工作需要 10 秒。暂时忽略交接开销。

```text
整任务持有：
A GPU  0..5       A CPU 5..25       A GPU 25..30
B 等待 0...................................30    B GPU 30..40
全部完成：40 秒

命令阶段归还：
A GPU  0..5       A CPU 5..25       A GPU 25..30
                 B GPU 5..15
全部完成：30 秒
```

在 A 做 CPU 工作时，A 的主机配额仍在，GPU 配额为零。B 用 51 GiB GPU 配额执行，完成后再归还。如果 A 提前回来，例如第 8 秒就需要 GPU，它会等 B 到第 15 秒结束，随后运行到第 20 秒。A 单条任务延迟增加了 7 秒，但两条任务整体仍可以受益。

这符合训练场景关注总体完成时间的目标，不过不是无条件更快。短空档可能被设备切换、遥测和清理开销吃掉，长 GPU 命令也可能堵住队首。这里实现的是命令边界上的安全复用，没有强行中断已经在执行的 CUDA 计算。

## 4. 配置与调用方式

仍然按 unit 管理物理 GPU 身份、主机资源和显存上限。改变的是**显存配额的持有时间**，不是把一个 unit 永久分配给某个 worker，也不是把剩余显存自动切成新的小 unit。

配置结构示例，节点地址必须替换成实际执行节点的地址，设备索引必须现场核对：

```yaml
psrl:
  env_service:
    enable: true
    node_policy: any
    units:
      - unit_id: environment-gpu
        node_ip: 192.168.1.10
        cpu_quota: 16
        memory_mb: 32768
        gpu_indices: [0]
        gpu_memory_mb: 81920
        max_containers: 4
        gpu_reservation: command
```

`gpu_memory_mb` 是 unit 的准入预算，`memory_mb` 是主存预算，`max_containers` 是持续存在的容器数量上限。任务仍需声明自己的显存需求。本例 4 个容器可以保留状态，但每个任务若声明 51 GiB，只有一个 GPU 阶段可以获得这张 80 GiB 预算卡。

工具适配层中使用下面的调用形式，示例中的脚本由任务提供：

```python
# CPU work can continue without acquiring GPU capacity.
await environment.exec("python prepare_inputs.py", timeout_sec=60)
# Queue time and actual command execution have separate deadlines.
await environment.exec_gpu(
    "python gpu_step.py",
    wait_timeout_sec=300,
    timeout_sec=120,
)
await environment.exec("python summarize_outputs.py", timeout_sec=60)
```

这是接口示例，不是随仓库提供了这三个任务脚本。`wait_timeout_sec` 限制等待许可的时间，`timeout_sec` 限制实际命令的执行时间。外围 Harbor 的整任务超时仍然生效，并不会自动扣除排队时间。

**普通 `exec` 不会自动识别 CUDA。** 在没有获得 GPU 许可时，通过普通命令直接启动 CUDA 会失败。现成工具必须把 GPU 工作接到 `exec_gpu`，本次 science 验证把明确注入的 CUDA 阶段接到了这个入口。不能只改配置，就声称任意模型生成的 GPU shell 命令都已经透明适配。

## 5. 失败清理和适用边界

正常归还顺序是 `禁止新访问 → 检查旧使用结束 → 账本归还`。反过来先归还预算再等待 GPU 真正释放，会给下一个任务制造竞争窗口。

- **后台 CUDA 未退出**：归还检查失败，unit 保持隔离状态，其他任务不能获得该 unit 的 GPU。需要清理所属容器后才能解除隔离，不会把“命令返回”直接等同于“物理资源归还”。
- **队列请求取消**：请求被标记为不可再用，迟到的申请不能重新开启 GPU。取消队首后，下一个有效请求可以推进。
- **lease 心跳过期**：先将 unit 标记为等待清理，再由节点回收本 lease 的容器。账本 TTL 到期不等于物理容器已经消失。
- **节点或设备操作失败**：保留配额或隔离 unit，不能在不确定的情况下放行。只清理本 lease 标签归属的容器，不进行全局 Docker 清理。
- **平台限制**：目前仅有 cgroup v1 设备访问控制实现，要求节点进程有写权限。容器化的执行节点需根据实际挂载根解析 cgroup 路径，不能把 `/proc/PID/cgroup` 中的路径直接拼接到挂载目录。
- **资源声明限制**：显存预算是准入控制，不是每个进程的显存硬隔离。活动阶段里超额申请显存仍可能失败，无法防止其他未接入此账本的外部作业占卡。
- **独立环境卡限制**：命令模式拒绝设置 `rollout_replica_rank`。Docker 与 vLLM 的睡眠借卡协议是另一层状态机，本次没有证明这两套机制可以直接组合。

预测器可以在未来帮助选择更合适的短命令，但不是安全归还的前提。本实现允许原任务回来后等待，不要求预测 GPU 空档一定多长。若要控制某些任务的延迟，可继续加入优先级、等待时长权重或协作式检查点，不能仅修改队列就声称已经支持任意 CUDA 抢占。

## 6. 跟着图读源码与可运行实验

按真实执行顺序阅读：

| 步骤 | 图中编号 | 函数与源码 | 输入、输出或状态变化 |
|---|---|---|---|
| 申请会话 | 1 | [EnvUnitLedger.try_admit](../../../psrl/workers/env_service/ledger.py#L105) | 任务 demand 转为 lease，命令模式初始不计 GPU 活动配额 |
| 建立容器 | 2 | [build_lease_overlay](../../../psrl/workers/env_service/overlay.py#L139) | lease 的设备身份转为容器设备映射和初始禁止读写规则 |
| 请求 GPU | 3 → 4 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73)、[acquire_gpu_command](../../../psrl/workers/env_service/router.py#L218) | 唯一 request_id 进入 unit 队列，调用者异步等待 |
| 准入和启用 | 4 → 5 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L50)、[grant_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L168) | unit 锁内预留显存，节点确认设备权限，返回 epoch |
| 实际执行 | 3 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73) | 取得许可后在容器运行命令，执行过程不持有 unit 锁 |
| 归还 | 3 → 5 → 4 | [revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L182)、[GpuCommandScheduler.release](../../../psrl/workers/env_service/command_gpu.py#L165) | 节点确认无遗留 GPU 使用后，配额归零，会话继续存在 |
| 结束或异常 | 6 | [finish](../../../psrl/workers/env_service/command_gpu.py#L236)、[EnvNodeService.cleanup](../../../psrl/workers/env_service/node.py#L68) | 清理容器，解除隔离，再释放会话资源 |

这里不是单线程串行系统。多个容器的 CPU 阶段并发进行，同一 unit 的申请和归还操作用异步锁保护，GPU 命令在锁外运行，节点过期回收则是独立任务。

CPU 回归可在项目环境中运行，不使用 GPU：

```bash
python -m pytest tests/env_service/test_gpu_commands.py tests/env_service/test_runner_env_service.py -q
```

前者使用真实账本和命令调度器，以测试替身模拟节点物理确认，覆盖“会话保持、GPU 配额归零、B 先于重新排队的 A 获得 GPU、旧代号归还无效、失败隔离、过期回收”。它不能证明操作系统真的阻止 CUDA。后者覆盖 science runner 是否把命令配置传到 Harbor，以及模型回复不再被空闲阶段的旧 GPU gate 阻塞。

自包含 CPU 教学模拟：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/phase_backfill_demo.py
```

该脚本使用人工给定时间，输出整任务持有 37 秒、允许回填 28 秒，以及原任务提前回来时增加等待 6 秒。模拟包含一个简单的预测判断，而生产命令调度器采用 FIFO，并未使用预测器。不要把这组秒数写成训练性能。

真实 GPU 两次交接脚本是 [gpu_command_handoff.py](../env-service-verification/scripts/gpu_command_handoff.py)。它直接调用生产账本、调度器和节点设备控制，省略 Ray 通信，在已有 science 镜像中执行实际 50 GiB CUDA 分配。参数 `--image` 指定已经准备好的镜像，`--gpu` 指定现场确认空闲的 GPU，`--output` 指定独立结果目录。该脚本总是清理自己创建的 lease 容器，并保存最终 GPU 状态。

完整训练通过 [run_science_command_work.sh](../env-service-verification/scripts/run_science_command_work.sh) 运行，变量 `PSRL_ENV_SCRIPT`、`VERIFICATION_DEPS`、`BASELINE_REPO`、`NODE_IP`、`MODEL_PATH`、`DATASET_PATH`、`RESULT_DIR` 必须对应已核实的环境。脚本固定 4 个任务 × 4 条 rollout、最多 4 轮交互、1 个训练 step、每组 1800 秒上限。没有通过这一路径的证据，不应把直接节点测试写成 Harbor/Ray 完整训练验证。

## 7. Results 与验证边界

真实 GPU 协议探针已完成。每组两个任务，各执行两次 50 GiB、3 秒的 CUDA 命令，中间做 12 秒 CPU 工作。整任务模式总耗时 45.98 秒，命令模式 43.63 秒，约减少 5.1%。命令模式额外执行了归还前后的权限拒绝探测，因此这组数字主要验证机制，不适合作为低开销 GPU 调度基准。

命令模式中两个容器保留各自文件，CPU 阶段继续执行，GPU 阶段按 A、B、A、B 顺序运行。初始状态及每次归还后的直接 CUDA 尝试均被拒绝，许可下的 CUDA 分配和内存写回检查成功。两个模式结束后实验 GPU 显存均为 0 MiB，没有遗留所属容器。

原始时间线：[episode.json](../env-service-verification/results/gpu-command-handoff-20261019/episode.json)、[command.json](../env-service-verification/results/gpu-command-handoff-20261019/command.json)。首次实机尝试暴露了嵌套 cgroup 挂载根问题，修复后完成命令模式验证，未重新运行已完成的整任务模式。

真实 Harbor 集成探针也已通过：使用实际 Compose、持久容器和 `exec_gpu`，通过进程内适配器连接生产调度器及节点，连续执行两轮 50 GiB CUDA 分配，详见[Harbor 探针结果](../env-service-verification/results/gpu-command-handoff-20261019/harbor/result.json)。首次完整训练在 Compose 验证时遇到重复 `security_opt`，修复为仅由最后的 ownership 覆盖层写入安全选项，未改共享 Harbor 依赖。

真实故障注入故意遗留一个后台 CUDA 进程。归还检查失败，unit 被隔离且继续计入 52224 MiB GPU 配额，容器清理后显存回到 0，详见[后台进程拒绝归还结果](../env-service-verification/results/gpu-command-handoff-20261019/harbor-leak/result.json)。这证明归还确认不只依赖命令退出状态。

Science 对照采用 Qwen3.5-4B、4 个不同 LAPS repair 任务，每题 4 条有效 rollout，每条最多 4 轮模型交互，训练目标为 1 个 GRPO step。GRPO 是按同一道题的多条采样结果比较奖励并计算策略更新的方法。本次每条任务都额外注入同样的 50 GiB、45 秒 CUDA 阶段，任务原本是 CPU 工作负载，不能把注入的显存需求当作原生 science 需求。

历史 OFF 使用指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9` 加共同运行修复，静态映射环境 GPU 6。命令 ON 使用当前实现，环境卡为 GPU 6、7，模型训练与推理在另外三张卡上。两组都允许最多 4 个活动尝试，OOM 不算完成，采用同一有界补采策略。每条任务超时 360 秒，子训练进程预算 1800 秒，Ray 建立和清理时间另外记录。

| 指标 | 历史 OFF | 命令 ON |
|---|---:|---:|
| 有效 rollout，总数及每题数量 | 16，每题 4 | 16，每题 4 |
| 全部尝试，包含失败补采 | 46 | 16 |
| CUDA OOM | 30 | 0 |
| 采齐全部有效 rollout，包含重试和等待 | 1231.27 秒，20.52 分钟 | 751.14 秒，12.52 分钟 |
| 有效 rollout 的实际模型交互 | 每条 4 轮，合计 64 轮 | 每条 4 轮，合计 64 轮 |
| Harbor 统计的生成 token | 19945 | 22815 |
| 有效任务超时 | 0 | 0 |
| Science repair reward 均值 | 0 | 0 |
| 完成的优化器更新 | 0，达到预算时尚在更新 | 1，正常退出 |
| 从 arm 启动到清理的观测时间 | 30.97 分钟，预算终止 | 24.74 分钟，正常完成 |

**本次等完成量采样时间减少 38.99%，吞吐约为历史 OFF 的 1.64 倍。** 新模式并没有靠少采样、少交互或者任务超时来缩短时间，生成 token 也更多。不过它与历史 OFF 的差别同时包括安全地使用两张环境卡及命令阶段归还，不能把全部 39% 单独归因于阶段协议。没有再增加第三组完整训练去拆分每一项收益，单独协议效应由前面的同卡两任务探针提供有限证据。

OFF 采样量达标，但优化器更新未在预算内完成。不能把它的 30.97 分钟截断时间当作“完成同一个训练 step 的时间”，也不能填写不存在的 OFF loss 或 entropy。预算到达后已停止，未自动延长或补跑完整 OFF 训练。

![等完成量与失败次数对比](../env-service-verification/results/science-command-work-20261019/completion-comparison.png)

账本检查结果：同时保留的会话最多 4 个，每个 unit 活动 GPU 阶段最多 1 个，声明显存、CPU、主存和容器数量均无超预算样本，GPU 配额与活动阶段计数的核对差异为 0。遥测中有 471 个 unit 样本处于“容器仍活着，GPU 配额为零”的状态。16 次 GPU 阶段均有完整申请及归还记录，平均排队 10.03 秒，最长 58.89 秒，命令结束后的归还确认平均 2.98 秒，最长 6.73 秒。

共有 14 对相邻同卡 GPU 阶段满足“前一条任务还没结束，后一条已经开始使用它释放的 GPU”。一个实测例子：任务 1 的 CUDA 已释放，约 10.94 秒后任务 3 在同一卡开始分配，此时任务 1 距整任务结束还有约 79.83 秒。这就是协议允许的安全空档复用，数据保存在结果 JSON 的 `same_gpu_reuse_before_previous_task_end`。

![显存、活动会话与 GPU 配额时间线](../env-service-verification/results/science-command-work-20261019/resource-timeline.png)

图上方是实际环境卡显存占用，不能解释为 GPU 计算利用率。中间图展示活着的容器数可以超过活动 GPU 阶段数。下方虚线表示“把本次所有活着的会话都按 51 GiB 持有”的假设预算，不是旧模式实测值。虚线超过两张卡的合计预算时，整任务模式实际上会阻止部分会话启动。图展示的是时间上的配额闲置得到消除，不是 CUDA 内存分配器碎片减少的测量。

命令 ON 完成更新的记录为：4 个训练组，22815 个有效训练 token，masked sample fraction 为 0，entropy 为 0.43555，policy loss 和 grad norm 都为 0。Entropy 表示策略输出分布的不确定程度，不能直接用来衡量调度方案好坏。这里每组奖励都是 `[0, 0, 0, 0]`，相对组均值的优势均为 0，因此没有来自奖励差异的策略梯度。完成了训练执行链路，但没有证据证明模型学得更好。

![可用训练指标，OFF 未完成处明确留空](../env-service-verification/results/science-command-work-20261019/training-metrics.png)

代码回归最终为 174 项通过，覆盖本次增加的 Compose 安全选项去重。修改文件的 Ruff 检查通过。仓库级 `pre-commit run --all-files` 存在既有文档脚本格式及 prose 检查问题，不能称为全仓检查通过，日志保留在[precommit-all.log](../env-service-verification/results/gpu-command-handoff-20261019/precommit-all.log)。其自动产生的无关格式改动已撤回。

两组的 GPU 释放记录均已确认，训练退出后现场曾全部回到 0 MiB，最后的清理记录为 0 至 3 MiB，无遗留 Docker 容器或本次实验进程，见[cleanup.json](../env-service-verification/results/science-command-work-20261019/cleanup.json)。首次 Compose 失败保存在 `command-compose-failed/`，不算作完成对照，也未隐藏该次初始化成本。

原始结果与图表再生：

- [比较数据 comparison.json](../env-service-verification/results/science-command-work-20261019/comparison.json)，包含逐次尝试、训练指标和账本核对。
- [命令 ON 实际启动、退出和 GPU 回收记录](../env-service-verification/results/science-command-work-20261019/command/launch.json)。
- [历史 OFF 实际启动、预算退出和 GPU 回收记录](../env-service-verification/results/science-command-work-20261019/off/launch.json)。
- [CPU 回归日志](../env-service-verification/results/gpu-command-handoff-20261019/cpu-tests.log)。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_command_work.py \
  docs/env-worker-deep-dive/env-service-verification/results/science-command-work-20261019 \
  --micro docs/env-worker-deep-dive/env-service-verification/results/gpu-command-handoff-20261019
```

这条命令只分析已有结果并生成 PNG、SVG，不会启动 Docker 或训练。

## 8. 异步归还与模型规划并行

可以并行。`exec_gpu(..., overlap_return=True)` 等待 Router 确认接管归还后就返回稳定输出，agent 可以立即调用模型。节点在后台继续撤销访问和检查物理归还，确认成功后才释放配额。默认 `overlap_return=False` 仍等待整个归还过程，适用于不能保证提前观察输出安全的命令。

下图的异步分叉已经实现，编号 4 的跨节点分支仍是第 9 节的提案。

```text
[1] A 的 GPU 前台命令完成，获得稳定的输出
     │ Router 确认接管归还请求，记录 request_id / epoch / RETURNING
     ├───────────────────────────────────┐
     v                                   v
[2] 输出交给模型，生成下一步计划       [3] 节点撤销访问，确认物理归还
     │ 此时不预占下一次 GPU 配额          │ 确认前保留 A 的 GPU 账目
     │                                   │ 成功后释放预算，B 可以获准执行
     │                                   │ 失败则隔离并清理
     └──────────────┬────────────────────┘
                    v
[4] 下一条 GPU 命令：前次归还成功 + 计划就绪 + 新授权
     ├── 当前实现：原 unit 排队
     └── 提案：可移植命令比较本地与远端成本，再选执行节点
                    v
[5] 执行、提交输出、归还。会话结束时等待或接管未完成的清理
```

图例：图里每个部分对应的源码位置。远端命令执行器尚未实现。

| 编号 | 当前源码 | 当前行为 |
|---|---|---|
| 1 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73)，分支位于 [90 行](../../../psrl/workers/env_service/docker.py#L90) | 显式选择提前返回，客户端保留请求编号和授权代号 |
| 2 | [模型请求路径](../../../examples/sciaccel_rl/agent.py#L103) | 独立模型卡可提前规划，保留会话顺序与输出版本 |
| 3 | [Scheduler.begin_return](../../../psrl/workers/env_service/command_gpu.py#L124)、[_return](../../../psrl/workers/env_service/command_gpu.py#L139)、[节点 revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L182) | 服务持有任务，锁外做归还检查，确认后在锁内改账 |
| 4 | [Scheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L50)、[EnvLease](../../../psrl/workers/env_service/types.py#L150) | 当前绑定原 unit。远端候选选择和独立命令租约需要新增 |
| 5 | [Scheduler.finish](../../../psrl/workers/env_service/command_gpu.py#L236)、[science 结果验收](../../../examples/sciaccel_rl/runner.py#L498) | 清理前等待后台归还，评分结果接受前检查延迟失败 |

这些代码如果有理解上的问题，可以随时沿图中编号追问。

**必须保留的顺序是“物理归还成功 → 预算释放 → 别的命令获得相应容量”。** 可以移出这条等待链的是模型规划，以及不依赖未完成输出的 CPU 工作。不能把“发出归还请求”当成“归还已经完成”。后台 CUDA 进程可能继续持有显存甚至写文件，前台 shell 返回不保证所有副作用已结束。

因此，提前交给模型的必须是已经稳定的输出或不可变快照。若任务无法提供这个保证，应保留同步行为，或明确将模型结果视为推测结果，归还失败时禁止提交依赖它的操作。不能先让下一轮任意修改同一目录，再期待失败后自动回滚。

实现协议如下：

1. Router 接受带命令编号和 `epoch` 的归还意图，确认自己接管其生命周期。`epoch` 是每次授权递增的代号，避免旧消息释放新授权。记录状态 `RETURNING`，继续计入显存预算。
2. Router 持有后台任务，客户端得到接管确认后可提交模型请求。归还 I/O 总超时为 30 秒，节点遗留句柄检查上限仍为 10 秒。状态可通过 `gpu_command_return_status` 查询，失败记录在请求上。客户端取消等待不会取消服务持有的归还任务。这里没有增加 Router 故障重启后的持久账本恢复能力。
3. 归还期间可以接收其他 worker 的排队请求。服务在 unit 锁内更新状态，锁外等待设备检查，再在锁内提交结果。原账目继续占用容量，设备操作和清理仍按所有者协调，避免取消、重复归还和新授权竞态。若账本中的已声明配额已经放不下新命令，直接返回等待，不发起重复的 GPU 遥测。只有账本允许尝试准入时才读取物理显存，再做最终检查，避免排队请求的轮询干扰归还。
4. 模型先返回时，可以存储下一步计划或排队，但不能越过前次归还成功的屏障。前次成功后仍要取得下一条命令自己的授权。归还失败时保留账目、隔离对应资源、上报失败并清理，不允许静默继续执行依赖链。
5. 下一条 `exec_gpu`、显式 `wait_for_gpu_return` 和容器 `stop` 都会检查前次归还。science runner 在接受 Harbor 结果前再次检查，避免 Harbor 将清理错误记作日志后仍返回成功奖励。`finish` 先禁止该 lease 新授权，等归还任务结束，再做物理清理。租约过期不代表容器已经消失。

接口示意如下，`query_model` 表示调用方已有的模型请求函数，不是新增的生产函数。命令必须是输出稳定的前台工作，不能用此选项绕过后台 CUDA 进程检查。

```python
output = await environment.exec_gpu(command, overlap_return=True)
plan = await query_model(output.stdout)
await environment.wait_for_gpu_return()
# Execute a dependent command only after a successful return and a fresh grant.
next_output = await environment.exec_gpu(next_command)
```

其中显式等待是结果提交屏障，确保推测中的模型计划不会掩盖前次执行的归还失败。后续 `exec_gpu` 自身也会执行这个检查。CPU 命令仍由调用方保证只操作已稳定的数据，不提供任意文件写入的自动回滚。

这里的异步主要隐藏归还检查的等待，并不要求预测模型多久返回。预测器将来可以辅助选择队列或预热环境，但不能代替授权、物理确认和资源账本。

**独立模型卡与共用同一卡必须区分。** 第 7 节的 science 实验使用独立的模型 GPU 和环境 GPU，所以模型规划可以和环境卡归还检查真正同时进行。若 Docker 借的是同一份 vLLM 实例的卡，提前提交模型请求不代表该实例能立即执行。它仍须等 Docker 释放资源后唤醒，或将请求路由到另一份已经就绪的模型实例。当前借卡模式的交接见 [GpuLendingBinding._transition](../../../psrl/workers/env_service/lending.py#L51)，命令模式尚未与该借卡模式组合。

跟着图读源码：

| 步骤 | 图中编号 | 输入、当前状态变化和未实现分支 |
|---|---|---|
| 入口 | 1 | `exec_gpu` 接收命令，申请授权，得到输出。开启异步选项时，等待服务接管后返回 |
| 规划分支 | 2 | agent 的模型请求接收稳定观察，与编号 3 并发 |
| 归还分支 | 3 | `begin_return` 校验请求代号并接管任务，`revoke_gpu_command` 禁止新设备访问并检查旧句柄。确认后账目归零 |
| 汇合与重申请 | 4 | 下一条命令带资源需求，等待前次成功和本次授权。当前只能进入原 unit 队列 |
| 执行到清理 | 5 | 新命令重复上述流程，会话退出进入 `finish`。取消和超时属于并发分支，必须与后台归还协调 |

## 9. 下一条命令留在本机还是跨节点

用户提出的“模型返回时发现本机被占用，就等或换机器”是合理的方向。这里把 PD 理解为模型推理中的预填充与解码分离：预填充处理输入，解码逐个生成输出，阶段之间传递注意力计算使用的键值缓存，即 KV cache。vLLM 的对应机制见[官方说明](https://docs.vllm.ai/en/latest/features/disagg_prefill/)。可以借鉴阶段分离和状态交接，但 Docker 命令的状态不只有一种缓存。

当前 `EnvLease` 固定记录 `unit_id`、`node_ip` 和 `gpu_indices`，节点授权还核验实际 `container_id`。修改一个节点地址不能把运行中的容器搬过去。任务可能依赖容器本地文件、当前目录、环境变量、tmux 会话、CPU 内存和 GPU 上下文。同机换卡也不能只改账本，因为设备映射和权限仍绑定原卡。

建议将持久会话与 GPU 命令执行器分开：原容器保留 agent 的工作目录和交互状态，每条可移植 GPU 命令声明镜像版本、输入文件版本、参数、所需显存和输出位置，由调度器分配独立执行器。完成后把退出码、日志和结果文件提交回原会话。共享文件系统能减少复制，但仍需要输出版本和原子提交，不能让两个重试同时覆盖同一输出。授权重试也不意味着可以重复任意外部副作用。

| 命令类别 | 合适的调度方式 |
|---|---|
| 只依赖已声明文件和参数，结果写入独立目录 | 可以跨节点执行，核对镜像、数据和 GPU 能力兼容 |
| 依赖原容器的 tmux、常驻进程或未保存的内存状态 | 原 unit 等待，除非任务显式提供状态保存和恢复协议 |
| 活着的 CUDA 进程还在计算 | 不属于命令边界调度，不能直接搬迁或提前归还 |

决策发生在模型已经生成具体命令、资源需求可知时。模型生成期间可以预热候选镜像，但不必为猜测中的下一步长期占 GPU。先估计：

```text
本地预计完成耗时 = 本地排队 + 执行
远端预计完成耗时 = 远端排队与准备 + 数据传递 + 执行 + 结果提交
```

能够并行的准备阶段按关键路径计算，不能一律相加。只有远端收益超过估计误差和切换成本才转发，同时限制重试与转发次数，避免在机器之间来回跳。等待时间随排队增长而提高优先级，可以减少长任务被不断插队的问题。单条命令最早完成是启发式规则，不保证整个训练批次的总完成时间最优，还需关注其他任务的排队与公平性。

一个可手算的教学例子：t=0 时 A 的前一条 GPU 命令结束，归还确认需要 3 秒，模型规划需要 8 秒，A 下一条 GPU 命令需要 4 秒。B 排在前面。假设模型使用独立 GPU，命令输出稳定，忽略归还意图接管开销与末尾共同清理时间。

| 条件 | 同步规划 | 异步规划或跨机后的结果 |
|---|---|---|
| B 只用 2 秒 GPU | B 在 3 至 5 秒执行，A 在 11 至 15 秒执行 | A 在 8 至 12 秒执行，全部完成从 15 秒降到 12 秒 |
| B 使用 10 秒 GPU | B 在 3 至 13 秒执行，A 在 13 至 17 秒执行 | 规划虽然第 8 秒完成，仍须等到第 13 秒，总时间仍是 17 秒 |
| B 使用 10 秒，远端空闲，准备与传输共 2 秒 | 本地全部完成为 17 秒 | A 第 8 秒准备远端，第 10 至 14 秒执行，B 第 13 秒完成，全部完成为 14 秒 |
| 上一行的远端准备成本改为 9 秒 | 本地全部完成为 17 秒 | 远端要到第 21 秒才能完成，所以选择本地等待 |

这说明异步归还与跨节点执行解决两个不同瓶颈。前者减少不必要的串行等待，后者缓解原节点排队。两者都不是所有负载下必然加速。

## 10. 教学模拟与验收边界

运行 [async_gpu_planning_demo.py](../env-service-verification/scripts/async_gpu_planning_demo.py)：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/async_gpu_planning_demo.py
```

脚本只用 Python 标准库，以逻辑秒计算事件，没有睡眠、Docker、Ray 或 GPU 调用。注释对应第 8 节图中编号，固定模拟 9 个场景并检查授权顺序。它假设远端空闲、数据可版本化且模型使用独立 GPU，不覆盖真实竞态、网络故障或文件迁移。实际运行输出保存于 [async-planning-teaching.json](../env-service-verification/results/async-planning-teaching.json)。

本次教学模拟得到：短队列 15 秒变 12 秒，长队列两种方式都是 17 秒，低成本跨节点为 14 秒，高成本或不可移植时选择本地。模型在第 1 秒先返回的场景仍要到第 3 秒才允许执行下一条 GPU 命令。归还失败的场景保持预算，不启动 B 或 A 的下一条 GPU 命令。这些是给定假设下的计算结果，不能当作新协议已经实机验证。

已有实测的归还确认均值约 2.98 秒，说明值得检查是否位于关键路径，但不能将每次确认耗时求和直接当作端到端可节省时间，也不能在第 7 节约 39% 的结果上直接叠加预测收益。其他任务排队、模型推理和跨机传输可能掩盖或抵消它。

异步归还验收覆盖规划提前、预算不提前释放、快速模型返回仍等待授权，以及归还失败和取消后的清理。真实实验见下一节。跨节点命令的输入输出一致性与收益仍未验证。

## 11. Results：异步归还小实验

**异步机制已经实现，验证到了提前开始规划，但本次没有验证到总体完成时间收益。** 必须区分三类证据：CPU 并发回归、真实 Ray / Docker / CUDA 加模拟规划等待，以及未能就绪的真实 vLLM 启动尝试。本节没有完整 science rollout、优化器更新或模型效果对比。

实机脚本是 [async_return_gpu_probe.py](../env-service-verification/scripts/async_return_gpu_probe.py)。最终对照只使用一张环境 GPU，创建 A、B 两个 Harbor 容器。每条 GPU 命令分配 1 GiB，执行 CUDA 内存写入和读取验证，持有 0.2 秒后释放。每个会话声明 4096 MiB，unit 预算为 6144 MiB，因此同一时刻最多允许一条命令持有 GPU 配额。

每回合工作量相同：A 执行第一条 GPU 命令，进入固定 8 秒的模拟规划，再执行第二条 GPU 命令。B 在 A 第一条命令结束时申请同一卡，执行一条 GPU 命令。计时从 A 第一条命令申请前开始，到 A、B 的三条命令及归还均完成为止，不包含 Ray、容器初始化和最终销毁。两个方式唯一的实验开关是 A 第一条命令的 `overlap_return`。

固定等待使用 `asyncio.sleep(8)`，不是 vLLM 推理。这让两个方式的规划耗时相同，但不代表真实模型的计算、排队或 token 生成行为。最终只跑同步、异步各一回合，因此结果是有界探针，不是有统计把握的吞吐基准。

| 最终修复后的指标 | 同步归还 | 异步归还 |
|---|---:|---:|
| CUDA 命令数 | 3 | 3 |
| 模拟规划时间 | 8 秒 | 8 秒 |
| A 第一条命令结束到开始规划 | 2.467 秒 | 0.012 秒 |
| A 第一条命令归还检查耗时 | 2.464 秒 | 2.607 秒 |
| 规划是否与归还重叠 | 否 | 是，重叠约 2.595 秒 |
| B 归还到 A 第二条命令获准 | 2.882 秒 | 8.948 秒 |
| A、B 全部完成 | 23.393 秒 | 28.540 秒 |

异步提前启动规划的机制生效，但本次总时间约增加 22.0%。不能只展示“2.467 秒变 0.012 秒”，就声称训练更快。

![最终两回合完成时间，异步本次更慢](../env-service-verification/results/async-return-gpu-20261019/controlled-fixed/completion.png)

按异步回合的时间线解释：A 第一条命令在第 4.068 秒结束，第 4.080 秒开始模拟规划，第 6.675 秒确认归还。B 在第 10.600 秒获准，第 15.693 秒确认归还。A 虽在第 12.082 秒规划完，但第二条命令直到第 24.641 秒才获准，最后第 28.540 秒全部结束。因此，后续排队与节点控制路径的等待抵消了提前规划带来的节省。

本次没有为节点控制路径的每次 Docker 查询、遥测和设备授权分别打细粒度时间戳，不能从这一个样本断言 8.948 秒具体由哪个子步骤造成，也不能据此断言异步方案长期必然更慢。当前证据只支持“可以安全重叠，端到端收益未证明”，不继续补跑以挑选更快的样本。

![真实 CUDA 与模拟规划的时间线](../env-service-verification/results/async-return-gpu-20261019/controlled-fixed/overlap.png)

图中蓝色表示获准后的 GPU 命令阶段，包括命令启动开销，不是 GPU 计算利用率。橙色是命令结束后的归还检查，绿色是明确模拟的规划等待。竖虚线是 A 第一条命令的实际归还确认。B 的授权在该确认之后，异步输出返回时账本仍保留 A 的 4096 MiB 配额。两回合均通过“不提前授权”和 CUDA 结果校验。

**实验发现并修复的两个问题：**

1. 配额已经满时，申请轮询仍发起 GPU 遥测。归还改成锁外执行后，这些无意义的查询会与归还争用节点控制路径。现在先检查账本，只有声明预算足够时才做物理遥测，真正授权前的检查仍保留。[快速判断](../../../psrl/workers/env_service/command_gpu.py#L76)有回归覆盖，连续十次满额申请不再读取遥测。该修复消除了已确认的多余查询，但不能把所有实机耗时变化都归因于它。
2. 探针的工作预算可能在最终清理时到期，中断剩余容器回收。现在清理由受保护任务执行，另有 60 秒上限。首轮留下的两个容器已按本次标签清理，相关 Compose 网络也已清除，没有操作其他任务。

首轮完整记录仍保留：同步两回合为 24.507、32.192 秒，异步为 31.863、25.328 秒，均值分别为 28.350、28.595 秒，未显示加速。四个计时回合完成后，清理触及 180 秒预算，因此不能把其 `passed` 字段单独解释成整次无错误退出，应同时查看 `error` 和[后续清理记录](../env-service-verification/results/async-return-gpu-20261019/controlled/cleanup-followup.log)。最终修复回归只有两回合，正常退出且没有清理错误。

真实 vLLM 尝试使用本地 Qwen3.5-4B 和独立模型 GPU。两次都在 300 秒启动预算内未就绪，没有进入计时对照。第一次启用多进程执行器，日志显示模型初始化及首次 GDN 内核编译。第二次改用单进程执行器、仅语言模型和 Triton GDN 预填充后仍未在预算内就绪。关闭这些路径不足以保证冷启动落入当前预算，不能将启动超时写成推理成功。模型进程已按本次进程组结束，没有继续扩大启动预算。

验证记录与原始数据：

- [最终对照与 GPU 回收记录](../env-service-verification/results/async-return-gpu-20261019/controlled-fixed/result.json)、[数值汇总](../env-service-verification/results/async-return-gpu-20261019/controlled-fixed/comparison.json)、[最后的物理资源检查](../env-service-verification/results/async-return-gpu-20261019/cleanup-final.log)。最终环境 GPU 回到 3 MiB 驱动基线，无计算进程和本次容器。
- [首轮未显示收益及清理超时的完整记录](../env-service-verification/results/async-return-gpu-20261019/controlled/result.json)。
- [第一次 vLLM 启动超时](../env-service-verification/results/async-return-gpu-20261019/result.json)、[第二次 vLLM 启动超时](../env-service-verification/results/async-return-gpu-20261019/measured/result.json)。
- [资源服务完整回归 180 项](../env-service-verification/results/async-return-gpu-20261019/cpu-tests.log)、[增加结果验收防线后的相关 31 项](../env-service-verification/results/async-return-gpu-20261019/cpu-targeted-tests.log)、[快速准入修复后的 15 项](../env-service-verification/results/async-return-gpu-20261019/cpu-fast-admission-tests.log)均通过。取消、过期、后台失败、提前返回后再次申请等异常边界来自 CPU 测试，不冒充本次 GPU 故障注入。
- 最终补充了归还遥测格式异常的检查，必须先成功读取并记入遥测，再清除 GPU 配额，异常时继续保留配额。[命令协议最终回归](../env-service-verification/results/async-return-gpu-20261019/cpu-final-tests.log)覆盖这条异常路径。
- 修改文件的 Ruff 检查通过。[全仓 pre-commit](../env-service-verification/results/async-return-gpu-20261019/precommit-all.log)仍因已有格式与注释问题失败，其自动生成的无关改动已撤回。

复现受控 GPU 探针前，按[集群检查指南](../../psrl/psrl-run-guide/cluster-preflight.md)激活环境并固定源码。配置以下环境变量，然后从仓库根目录运行。脚本拒绝已占用的目标 GPU，不操作共享 Ray 集群。

| 环境变量 | 含义 |
|---|---|
| `GPU_PROBE_IMAGE` | 当前节点已缓存、提供 Python 3 的实验镜像 |
| `NODE_IP` | 经路由与节点身份核对的实际执行 IP |
| `PROBE_OUTPUT` | 本次独立的新输出目录，不能复用已有容器目录 |

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/async_return_gpu_probe.py \
  --image "$GPU_PROBE_IMAGE" --node-ip "$NODE_IP" --env-gpu 7 \
  --controlled-planning-s 8 --modes sync async --output "$PROBE_OUTPUT"
```

去掉模拟参数并提供 `--model` 才会尝试真实 vLLM，不能把两种模式混算。重新画图只读取已有结果：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_async_return.py \
  docs/env-worker-deep-dive/env-service-verification/results/async-return-gpu-20261019/controlled-fixed
```

## 12. Results：A 下一条 GPU 调用在等什么

**A 第二次申请到获准共 12.128 秒。其中前 3.180 秒处于 B 的配额尚未归还的窗口，后 8.948 秒已经不能归因于 B 保留配额。后半段的具体慢点尚未定位。** 本节重新分析第 11 节已有数据，没有启动新 GPU 实验。图中的 `A next GPU` 开始位置表示客户端收到完整授权，不表示 Router 刚把 A 从队列取出。

本节保留分段计时加入前的分析，源码链接指向修复前快照。新增实测与当前修复见第 13 节。下面是修复前实现的协作链，方括号编号用于图例和源码阅读路径。B 的归还与 A 的申请并发发生。

```text
[1] A 的 exec_gpu 记录 requested_at，向 Router 申请
                   │
                   v
[2] Router 检查队列、显存预算 ← [6] B 禁止新 GPU 访问并确认旧使用结束
       │ 预算不足：返回 wait         │ Router 清除 B 的 GPU 配额
       │ A 等 0.1 秒后重试          └─────────────┘
       │ 预算足够且轮到 A
       v
[3] 节点查询 GPU 实际显存与 CUDA 进程，Router 再判断能否准入
       │ 通过
       v
[4] 节点核对 A 容器身份及 GPU 设备，打开并核验设备权限
       │ 授权完成，经 Router 返回 granted
       v
[5] A 记录 admitted_at → 执行 GPU 命令 → 完成后进入 [6] 归还
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与对应函数 | 源码 |
|---|---|---|
| 1 | `exec_gpu` 记录申请时间并等待远程调用 | [docker.py:76](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/docker.py#L76) |
| 2 | `acquire_gpu_command` 转交 `acquire`，在 unit 锁内检查队首和预算 | [router.py:218](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/router.py#L218)、[command_gpu.py:47](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu.py#L47) |
| 3 | `gpu_lending_snapshot` 顺序执行两次 `nvidia-smi`，查询显存和计算进程 | [lending_node.py:46](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/lending_node.py#L46) |
| 4 | `grant_gpu_command` 调用 `_command_container`，执行 `docker inspect`、设备清单查询和权限检查 | [command_gpu_node.py:111](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu_node.py#L111)、[probe.py:36](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/probe.py#L36) |
| 4 → 5 | `set_access` 打开设备权限，客户端在整个授权调用返回后记录时间 | [command_gpu_node.py:61](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu_node.py#L61)、[docker.py:85](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/docker.py#L85) |
| 6 | `revoke_gpu_command` 撤权并确认无 GPU 使用，`_return` 随后释放配额 | [command_gpu_node.py:136](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu_node.py#L136)、[command_gpu.py:124](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu.py#L124) |

**按已有数据拆时间。** 以下时间从异步回合开始计算，与第 11 节图中“从 A 第一条命令结束计时”的横轴原点不同，时间差相同。

| 时间 | 事件 | 对 A 下一条命令的影响 |
|---|---|---|
| 6.675 秒 | A 第一条命令已确认归还 | A 自己的旧配额已释放 |
| 12.082 秒 | A 的固定 8 秒模拟规划结束 | 已具备申请下一条命令的条件 |
| 12.246 秒 | B 前台 GPU 命令结束 | B 还需要物理归还检查 |
| 12.513 秒 | A 记录第二次申请时间 | 开始计入 12.128 秒申请耗时 |
| 15.693 秒 | B 确认归还，Router 清除 B 配额 | B 不再构成声明预算阻塞 |
| 24.641 秒 | A 收到授权完成的回复 | 才能进入 GPU 命令执行 |

unit 是 Router 维护的一组资源及其预算。本次 unit 有 6144 MiB 的显存准入预算，每条命令声明 4096 MiB。这是实验设置的预算，不是显卡物理总容量，也不等于命令实际只分配的 1 GiB。

在 12.513 到 15.693 秒，B 的 4096 MiB 尚未归还。允许 A 同时持有配额需要 `4096 + 4096 = 8192 MiB`，超过预算，所以 A 不能获准。B 的前台命令虽然结束，但系统还要禁止容器再次打开 GPU 设备，检查残留的设备句柄、内存映射和 CUDA 进程，才能确认“可以安全交给另一个任务”。这 3.180 秒是预算不可用的时间窗口，不是通过逐次调用日志测得的纯队列睡眠时间。

在 15.693 到 24.641 秒，B 已经释放配额。单看声明预算，A 的 4096 MiB 可以放入 6144 MiB，不能再用“B 仍占着卡”解释这 8.948 秒。申请函数还有一条实际执行链：远程调用到达 Router，获得 unit 锁，查询节点遥测，检查实际占用，调用节点授权，再把结果返回 A。RPC 即远程过程调用，调用者要等待另一个进程执行方法并回复，即使进程位于同一台机器也有调度和通信环节。

节点授权也不只是改一个 Python 布尔值。修复前实现会执行 `docker inspect` 验证容器身份，执行 `nvidia-smi -q -x` 查 GPU 身份与设备映射，再通过 cgroup 设备控制器修改并核验容器权限。cgroup 是 Linux 管理一组进程资源和权限的机制，这里用它保证没有授权的容器不能打开 GPU 设备。加上前面的两次 GPU 遥测，一次成功的完整准入路径包含三次 `nvidia-smi` 和一次 `docker inspect`，以及文件读写、线程调度和 RPC。这些是源码确认存在的操作，**不是已经测出各自耗时的结论**。

修复前 `acquire` 在等待遥测和设备授权时仍持有 unit 锁，所以这些外部操作会延长同一 unit 的控制操作等待。这是可检查的性能风险，但现有数据不足以认定本次 8.948 秒主要消耗在锁、Docker、GPU 查询或 RPC 的哪一个环节。日志也没有逐次准入拒绝原因和连续物理显存采样，不能断言整个窗口 GPU 都空闲，或只有一次授权尝试。

**可以排除的误解：**

- 这不是等待 A 的第一条命令归还。它在 6.675 秒已完成，规划结束后显式加入归还任务只用了约 0.002 秒。
- 这不是 vLLM 继续生成了 12 秒。本次规划是固定等待模拟，12.082 秒已结束，未执行真实 vLLM 推理。
- 这不是协议要求固定等待 9 秒。重试间隔是每次收到 `wait` 后 0.1 秒，单次 RPC 的执行和排队时间另算。归还循环的 10 秒是超时上限，不是必等时长，而且 B 的该阶段已经结束。
- 申请前的 `docker compose ps` 发生在 `requested_at` 之前，只可能影响规划结束到申请之间的约 0.431 秒，不能拿它解释申请后的 8.948 秒。授权内部的 `docker inspect` 则在申请之后。

**跟着图读源码。**

1. 入口 → [1] → `exec_gpu` [docker.py:67](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/docker.py#L67)：加入自己的旧归还任务，查询主容器，生成请求身份，记录申请时间。
2. 申请 → [2] → `acquire` [command_gpu.py:50](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu.py#L50)：获取 unit 锁，进入 FIFO 队列，即先申请者优先的队列，检查声明预算。返回 `wait` 时客户端稍后重试，此时 B 可并发归还。
3. 物理核验 → [3] → `acquire` [command_gpu.py:78](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu.py#L78)：等待节点遥测，再结合实际占用决定继续等待或授权。此处仍在 unit 锁内。
4. 授权 → [4] → `grant_gpu_command` [command_gpu_node.py:125](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu_node.py#L125)：核对容器和设备，开启设备权限。Router 收到成功回复后返回 `granted`。
5. 执行 → [5] → `exec_gpu` [docker.py:85](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/docker.py#L85)：记录完整授权完成时间，再启动 GPU 命令。这个时间边界把上述控制操作全部包含在“申请到获准”中。
6. 归还与清理 → [6] → `_return` [command_gpu.py:124](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu.py#L124)：节点撤权和检查在 unit 锁外进行，随后持锁提交配额释放。整会话结束由 `finish` [command_gpu.py:194](../env-service-verification/results/async-admission-diagnosis-20261020/before-source/psrl/workers/env_service/command_gpu.py#L194)清理所属容器和状态。

这些代码如果有理解上的问题，可以随时按图中编号追问。

**只用 CPU 复算现有日志。** [analyze_async_gpu_wait.py](../env-service-verification/scripts/analyze_async_gpu_wait.py)使用 Python 标准库读取原始结果，不模拟新数据，也不连接 Ray、Docker 或 GPU。从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_async_gpu_wait.py \
  docs/env-worker-deep-dive/env-service-verification/results/async-return-gpu-20261019/controlled-fixed/result.json
```

已执行上述命令复算原始日志。异步结果：`request_to_admission_s ≈ 12.128`，`B_budget_unavailable_window_s ≈ 3.180`，`remaining_admission_interval_s ≈ 8.948`。同步回合分别约为 2.753、0、2.753 秒。同步的“B 归还到 A 获准”约 2.882 秒，其中约 0.129 秒发生在 A 提交申请之前，所以它不等于申请耗时。分析脚本的 Ruff 检查通过，文档本地链接和目录锚点检查通过。

后续若要定位性能，应给同一请求记录 RPC 到达、锁等待、每轮准入结果、遥测开始结束、容器检查、设备查询、授权完成及客户端收包的时间。先定位后半段，再决定是否需要缓存稳定设备信息、缩短锁覆盖范围或调整等待者唤醒。以上是诊断建议，不能写成已经实现或已经验证的收益。这些旧结果支持提前规划，但不足以证明整体更快。

## 13. Results：慢查询定位与直接接口修复

**原因已通过新增的小实验定位，控制路径已修改并在 GPU 上回归。** 复现中一次异步请求在 B 归还后仍等待 8.254 秒，主要是 `docker inspect` 子进程耗时 4.224 秒、完整 GPU 设备查询耗时 3.317 秒。修改后四回合的对应授权等待均为 0.226～0.537 秒，归还后的两个容器都不能继续调用 CUDA。

这里的“控制路径”指决定谁能使用 GPU、查询占用、调整容器权限的程序流程，与 GPU 正在执行计算不同。异步归还只允许规划提前开始，后续 GPU 请求仍要走这条流程，因而慢查询可以抵消并行节省。

```text
[1] A 调用 exec_gpu，Router 检查队列和显存预算
                     │ 预算足够
                     v
[2] EnvNodeService → NVML：读取当前显存和 CUDA 进程
                     │ Router 检查实际占用
                     v
[3] 节点 → 本机 Docker Unix socket：读取容器身份、状态和设备配置
                     │
[4] 节点 → NVML：读取 GPU UUID 与 minor → cgroup：开启并检查设备权限
                     │ Router 返回完整授权
                     v
[1] A 执行 CUDA 命令，获得稳定结果
         ├── overlap_return=True：收到后台归还确认后，继续规划
         └── [5] 后台撤权 → 检查设备句柄、映射、CUDA 进程
                              │ 确认结束后才释放配额
                              └── 其他申请可进入 [1]
```

图例：图里每个部分对应的源码位置。

| 编号 | 对应实现 | 作用 |
|---|---|---|
| 1 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73)、[acquire](../../../psrl/workers/env_service/command_gpu.py#L50) | 保持命令阶段预算、队列和异步归还语义 |
| 2 | [gpu_lending_snapshot](../../../psrl/workers/env_service/lending_node.py#L52)、[read_gpu_snapshot](../../../psrl/workers/env_service/probe.py#L85) | 在线程中读取实时显存与计算进程，出错不伪造空闲 |
| 3 | [inspect_container](../../../psrl/workers/env_service/docker_api.py#L23)、[_command_container](../../../psrl/workers/env_service/command_gpu_node.py#L109) | 使用本机 Docker 接口，继续核对归属与运行状态 |
| 4 | [read_gpu_devices](../../../psrl/workers/env_service/probe.py#L74)、[grant_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L168) | 精确读取设备身份，不再查询全套 GPU 健康信息 |
| 5 | [revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L182)、[_return](../../../psrl/workers/env_service/command_gpu.py#L139) | 物理确认之后才清除配额，允许独立规划并发进行 |

**实验边界与停止条件。** 使用真实 Ray、Harbor Docker 和单张 GPU。A 执行两条 CUDA 命令，中间固定模拟规划 8 秒，B 回填一条 CUDA 命令。每条命令分配 1 GiB、写入读取校验 42、持有 0.2 秒。unit 预算为 6144 MiB，每条命令仍申报 4096 MiB。修复前后分别运行 `sync async async sync`，共八回合，每种实现每种方式各两个样本。每批工作预算 240 秒，清理独立预算 60 秒，没有自动补跑。模型加载、真实 vLLM 推理、science rollout、训练 loss 和 entropy 均不在本实验范围。

这是当前代码的修复前后与同步/异步局部对照，不是指定历史提交 `0eeead861ffafed6bf31c962d259fd847775dcf9` 的 EnvRouter ON/OFF 对照。两种实现保留相同资源预算、队列和物理归还要求。

**计时如何确定原因。** [gpu_command_trace.py](../env-service-verification/scripts/gpu_command_trace.py)在实验 actor 中记录申请、unit 锁等待、外部查询和设备权限操作。记录先存于有界内存队列，回合计时结束后才写出，避免共享存储写日志干扰关键路径。节点和 Router 位于同一实验机器，事件先后用墙钟对齐，各段耗时使用单调时钟计算。

复现中第一个异步回合，A 第二次申请到获准共 10.630 秒。B 配额尚未归还的窗口为 2.377 秒，归还后还等 8.254 秒。后半段进一步分解为：

| 分段 | 实测耗时 | 能得出的结论 |
|---|---:|---|
| 归还后到成功准入调用开始 | 约 0.077 秒 | 包含客户端重试唤醒和 RPC 到达 |
| 查询实际显存和 CUDA 进程 | 0.629 秒 | 两次 `nvidia-smi` 子进程 |
| 核验容器 | 4.224 秒 | `docker inspect` 子进程的整体耗时 |
| 获取 GPU 身份和设备映射 | 3.317 秒 | `nvidia-smi -q -x` 查询了远超所需字段的信息 |
| 实际修改并检查设备权限 | 0.000157 秒 | 设备权限写入本身没有秒级延迟 |
| 成功准入中的其他控制处理 | 约 0.0065 秒 | 包含未单列的调度、通信及处理 |

整个 A 请求的累计 unit 锁等待约 0.000038 秒。所以这个样本没有显示“在锁外排队等待其他任务”是主要瓶颈。成功授权期间确实持有 unit 锁，但消耗时间的是锁内等待外部查询。

这个证据定位到了具体查询边界，没有继续区分 `docker inspect` 的 4.224 秒中客户端初始化、CPU 调度和 daemon 处理各占多少。第 11 节原始 8.948 秒没有这些分段记录，不能事后给它填上本次分段数值。本次是重新复现相同现象并定位，两个样本不能混写。

**实际修改。**

1. Docker 查询直接走本机 Unix socket，也就是连接本机服务的文件形式通信端点。使用已有 `aiohttp` 库读取同一容器 JSON，每次调用都有独立会话、超时和关闭流程。仍核验租约标签、运行状态及设备配置，HTTP 错误继续阻止授权。仅接受本机 Unix socket，不允许远程 daemon 的容器去控制本机设备权限。
2. GPU 查询改为 NVML，即 NVIDIA Management Library，NVIDIA 提供的 GPU 管理接口。通过 `nvidia-ml-py` 精确读取 UUID、minor number、显存和计算进程，不再每次启动命令行程序或获取完整健康报告。minor number 是设备文件编号，不能直接拿 GPU 显示序号替代。读数每次重新获取，没有缓存“显存空闲”结论。
3. 显存读取显式使用 NVML v2 接口。该机器实测旧接口显示 589.5 MiB，新接口区分应用占用 3.4375 MiB 和驱动保留 586.0625 MiB，`nvidia-smi` 的对应整数显示是 3 和 586。选 v2 保持应用显存计费口径，避免错误阻塞空闲卡。原始核对记录见 [memory-semantics.json](../env-service-verification/results/async-admission-diagnosis-20261020/memory-semantics.json)。

**全部结果。** 表内均值来自两个样本，括号中保留全部总时间，不挑选较快回合。

| 实现 | 方式 | 三条命令及归还全部完成 | A 下一条 GPU 在预算可用后的授权等待均值 |
|---|---|---:|---:|
| 修复前命令行查询 | 同步 | 24.239 秒（24.185、24.293） | 3.112 秒 |
| 修复前命令行查询 | 异步 | 24.542 秒（26.113、22.971） | 5.473 秒 |
| 修复后直接接口 | 同步 | 16.067 秒（18.349、13.786） | 0.429 秒 |
| 修复后直接接口 | 异步 | 12.233 秒（12.296、12.170） | 0.330 秒 |

授权等待定义为 `admitted_at - max(requested_at, B.released_at)`，避免把 A 尚未申请的时间也算入等待。A 第二次成功准入调用的平均执行耗时，合并同步和异步四回合后，从 4.256 秒降至 0.377 秒。四次修复后的容器查询均约 0.9 毫秒，设备身份查询约 0.138～0.362 秒，实际显存/进程查询约 0.076～0.166 秒。

![控制路径修复前后的全部样本及耗时分解](../env-service-verification/results/async-admission-diagnosis-20261020/control-comparison.png)

图中黑点代表每个实际回合，柱高代表均值。第三幅只分解已获得预算后的成功准入调用，不把 B 尚未归还的等待算进去。

**为什么 async 会看起来更慢，又为什么不能把本次总时间差全部归功于 async？** 修复前一个异步样本完成于 26.113 秒，另一个完成于 22.971 秒，两者跨在同步样本两侧，说明没有证据支持“异步固定增加约 9 秒”。原实现仍需经过慢查询，提前规划节省的时间可能被某次查询抖动抵消。直接接口同时加速同步与异步，主要收益来自控制路径缩短。

修复后异步均值小于同步，但每组只有两个样本，Harbor 命令阶段仍包含启动和执行抖动。例如同步首回合 A 第一条命令获准后到完成用了 3.162 秒，两个异步回合分别为 1.534 和 1.407 秒。异步回合实际隐藏的第一条归还检查平均约 0.375 秒，不能把两组总均值差 3.835 秒全部解释成异步重叠收益。本次能确认的是慢查询导致的授权空档显著缩短，不能推导长期训练吞吐百分比。

**跟着图读源码。**

1. [1] `exec_gpu` → `acquire`：申请、等待、预算检查与原协议相同。预算不足仍等待，不提前释放上一条命令的资源。
2. [2] `gpu_lending_snapshot` → `read_gpu_snapshot`：线程执行实时 NVML 查询，异常向上传播，不用零值冒充查询成功。
3. [3] `_command_container` → `inspect_container`：读取本机容器 JSON，检查归属与状态。
4. [4] `read_gpu_devices` → `grant_gpu_command`：用真实 minor 和 UUID 验证映射，开启权限后返回完整授权。
5. [5] 命令完成后，后台 `revoke_gpu_command` 与规划并发，`_return` 只在物理确认完成后改账。整会话结束仍由 [finish](../../../psrl/workers/env_service/command_gpu.py#L236)清理所属容器和状态。

上面每一步的具体源码链接都在图例中，代码有不清楚的地方可以继续追问。

**验证与清理。** 相关 CPU 回归 93 项通过，校准 NVML v2 后的五项查询回归通过，修改文件 Ruff 检查和格式检查通过。全仓 pre-commit 仍有已有格式和注释错误，其自动修改的 14 个文件已恢复，仅修正本次触及文件的日志标点，记录见 [precommit-all.log](../env-service-verification/results/async-admission-diagnosis-20261020/precommit-all.log)。GPU 回归的三条命令每回合均返回正确 CUDA 结果，B 获准时间没有早于 A 的物理归还。修复后额外绕过 `exec_gpu`、从已归还容器直接运行 CUDA，两次返回码均为 1，证明这两个容器的设备访问权限确实被撤销。两批实验都正常结束，清理错误列表为空。

按两批实验的 `psrl.actor_id` 标签核验容器及 Compose 网络，核对 GPU 计算进程和实验 Ray 进程。目标 GPU 的最终 NVML 应用占用为 3.4375 MiB，命令行显示 3 MiB，无计算进程。证据见 [最终归属清理记录](../env-service-verification/results/async-admission-diagnosis-20261020/ownership-cleanup-final.log)和[进程与显存检查](../env-service-verification/results/async-admission-diagnosis-20261020/after-cleanup.log)。未启动额外训练。

**复算与材料。**

- [修复前原始结果及分段记录](../env-service-verification/results/async-admission-diagnosis-20261020/before/result.json)、[修复后原始结果及分段记录](../env-service-verification/results/async-admission-diagnosis-20261020/after/result.json)、[完整数值汇总](../env-service-verification/results/async-admission-diagnosis-20261020/analysis.json)。
- [相关 CPU 回归](../env-service-verification/results/async-admission-diagnosis-20261020/cpu-tests.log)、[NVML v2 回归](../env-service-verification/results/async-admission-diagnosis-20261020/cpu-memory-v2-tests.log)。
- [分段计时脚本](../env-service-verification/scripts/gpu_command_trace.py)、[分析与画图脚本](../env-service-verification/scripts/analyze_gpu_control.py)、[GPU 探针](../env-service-verification/scripts/async_return_gpu_probe.py)。

只复算现有结果不需要 GPU，从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_async_gpu_wait.py \
  docs/env-worker-deep-dive/env-service-verification/results/async-admission-diagnosis-20261020/after/result.json
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_gpu_control.py \
  docs/env-worker-deep-dive/env-service-verification/results/async-admission-diagnosis-20261020
```

第一条仅需 Python 标准库，第二条使用实验环境已有的 Matplotlib 生成上图。两条命令均已实际执行。若需要重新进行 GPU 验证，先按集群指南检查节点，并沿用第 11 节环境变量定义，给探针增加 `--trace-control --work-budget-s 240 --modes sync async async sync`，同时将脚本所在目录加入 worker 的 `PYTHONPATH`。不要把未包含分段计时的旧图当作当前实现的结果。

## 14. 本机 Docker 接口到底属于哪台机器

**这里的“本机”是运行 EnvNodeService 的任务执行节点。** 容器 A 在节点 E 上，负责它的节点服务就部署在 E 上，并访问 E 的 Docker 服务、GPU 和设备权限控制器。当前工作区节点 W 的 Docker 是否响应，是另一件事。接口优化没有把远端容器交给工作区的 Docker 管理，也没有修复工作区 Docker 服务本身。

下面是通用部署关系。Router 可以与 E 同机，也可以在另一个 Ray 集群节点上。第 13 节的小实验将 Router、节点服务、Harbor 客户端和任务容器都放在同一个远端执行节点 E，工作区 W 只负责发起实验和读取共享结果。

```text
工作区节点 W：发起远端实验，读取共享源码与结果
              │ 不通过 W 的 Docker 创建或检查远端实验容器
              v
执行节点 E：[1] 管理 A 的 Harbor 客户端
              │ 申请 GPU 预算、发送 lease_id 和 container_id
              v
           [2] EnvRouter（可与 E 同机，也可在其他集群节点）
              │ 用 lease.node_ip 找到 E，发送 Ray RPC
              v
执行节点 E：[3] EnvNodeService
              ├── [4] E 的 Unix socket → E 的 Docker daemon：检查容器 A
              └── [5] E 的 NVML / cgroup：查询 GPU、控制 A 的设备权限

授权成功后：[1] Harbor → E 的 Docker daemon → 在 A 内执行 CUDA 命令
命令结束后：[3] 撤权、核实归还 → [2] 释放配额
整任务结束：[1] / [3] 清理 E 上归属本任务的容器
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与函数 | 源码 |
|---|---|---|
| 1 | `exec_gpu` 发送申请，获准后调用 Harbor 的 `exec` | [docker.py:52](../../../psrl/workers/env_service/docker.py#L73) |
| 1 → 2 | 正常 worker 申请资源时携带自身节点，`EnvServiceClient.acquire` | [client.py:188](../../../psrl/workers/env_service/client.py#L188) |
| 2 → 3 | `acquire` 使用租约节点找节点服务，`_start_node` 把服务固定到该节点 | [command_gpu.py:77](../../../psrl/workers/env_service/command_gpu.py#L77)、[router.py:296](../../../psrl/workers/env_service/router.py#L296) |
| 3 | `_command_container` 调用检查接口，继续核验容器租约标签和状态 | [command_gpu_node.py:109](../../../psrl/workers/env_service/command_gpu_node.py#L109) |
| 4 | `inspect_container` 委托 `_local_get` 使用 `aiohttp.UnixConnector`，Docker daemon 是外部服务 | [docker_api.py:9](../../../psrl/workers/env_service/docker_api.py#L9) |
| 5 | `read_gpu_devices` / `read_gpu_snapshot` 读本节点 GPU，`DeviceCgroupFence` 操作本节点设备权限 | [probe.py:74](../../../psrl/workers/env_service/probe.py#L74)、[probe.py:85](../../../psrl/workers/env_service/probe.py#L85)、[command_gpu_node.py:59](../../../psrl/workers/env_service/command_gpu_node.py#L59) |
| 归还与清理 | `revoke_gpu_command` 确认物理归还，`finish` 清理所属资源 | [command_gpu_node.py:131](../../../psrl/workers/env_service/command_gpu_node.py#L182)、[command_gpu.py:194](../../../psrl/workers/env_service/command_gpu.py#L236) |

**为什么不会把请求发到工作区机器？** Ray RPC 是远程过程调用，Router 将方法调用发送给指定的节点服务进程。`_start_node` 先通过 Ray 的节点表将节点 IP 映射成 NodeID，再设置 `NodeAffinitySchedulingStrategy(..., soft=False)`。这里的节点亲和调度是明确要求服务部署到那个节点，`soft=False` 表示不能为了能运行就换到其他节点。实验用的计时子类也使用同样的[节点固定逻辑](../env-service-verification/scripts/gpu_command_trace.py#L128)。

例如，A 的租约属于 E，Router 就向 E 上的服务发起查询。E 上的服务打开自己运行环境中的 `docker.sock`，读取 E 上的 A。如果错误地去 W 查询，W 不响应会导致超时，即使 W 的 Docker 正常，也不会因为共享了源码目录就自动拥有 E 的容器。共享工作区文件不等于共享 Docker daemon、GPU 或进程。

正常 worker 的资源准入带上自身节点，因为 Harbor 在调用它的进程所在节点启动容器。本实现没有依靠“把 Docker 地址换成另一个节点”来迁移容器。需要跨节点执行任务，必须另外解决工作进程放置和任务状态交接，第 9 节对此仍按提案说明。

**谁访问 Docker？** 是执行节点上的管理进程访问它。容器 A 执行用户的 CUDA 工作，不要求业务容器自身持有 Docker 管理接口。节点服务还需要访问与该 Docker 容器一致的 GPU 设备与 cgroup。cgroup 是 Linux 对一组进程实施资源和权限控制的机制，在这里用于启用或撤销容器对 GPU 设备的访问。

如果节点服务本身运行在基础设施容器中，它所看到的 socket、GPU 和 cgroup 挂载也必须对应同一执行节点。代码拒绝用远程 TCP Docker 地址来操作本地设备权限，但这不等于自动证明任意用户挂载的 socket 都正确，部署仍需保证上述一致性。

**为什么地址写着 `http://localhost`？** `inspect_container` 调用的 `_local_get` 创建的是 `aiohttp.UnixConnector(path=host.path)`。连接器决定实际传输走 Unix socket 文件，默认路径是当前节点的 `/var/run/docker.sock`。请求 URL 中的 `localhost` 提供 HTTP 请求的主机字段和路径结构，不会因此连接工作区机器，也不是通过 TCP 访问本节点 80 端口。

Unix socket 是同一台主机上进程间通信的端点，可以承载 HTTP 协议。HTTP 规定请求和响应的格式，底层传输不一定是 TCP。明确设置的 `DOCKER_HOST=unix://...` 可以选择另一个本地 socket。当前命令阶段设备控制路径拒绝非 Unix 地址以及 `DOCKER_CONTEXT`，避免隐式选择远端 daemon。

**为什么直接接口会更快？** 默认本地 Docker 配置下，改动前后最终都访问执行节点的同一个 Docker 服务。减少的是客户端中间步骤：

```text
修复前：E 上的 Python → 启动 docker inspect 子进程
        → Docker 客户端初始化、读取配置、请求 E 的 daemon
        → 输出 JSON 到 stdout → Python 收集并解析

修复后：E 上的 Python → 经 Unix socket 请求 E 的 daemon
        → 直接解析响应 JSON
```

CLI 是命令行客户端，也就是这里执行的 `docker` 程序。直接接口减少了创建子进程、客户端初始化、标准输出收集等工作，不是把一次远程网络访问变成本地访问。当前 helper 每次请求都会创建并关闭自己的 HTTP 会话，所以本次也不能把收益归因于长期复用连接池。

第 13 节的旧路径存在一次 4.224 秒的 Docker 检查，新路径四次 A 第二条命令授权中的检查约 0.9 毫秒。但旧路径其他检查也有约 80 毫秒的记录，不能说 Docker 命令行每次都需要四秒，也不能把那四秒全部认定为进程启动时间。旧计时没有拆分客户端与 daemon 内部耗时，实测支持的是这次改动缩短了整个查询路径。

另外，GPU 检查从 `nvidia-smi` 命令行改成 NVML 直接查询，同样减少了中间程序和不需要的查询字段。整体授权提速包含这两部分，不能全部算作 Docker 接口优化。

**能不能顺便解决工作区 Docker 不响应？不能。** Docker daemon 是实际管理容器的后台服务，命令行和直接接口都依赖它。如果该服务本身不响应，两条路径都会失败。换客户端不会修复服务、恢复丢失的容器状态或让远端容器出现在工作区机器。

本节做了限时只读复查：工作区节点的直接 `/_ping` 请求在 3 秒内没有收到响应，`docker info` 在 6 秒内未返回，说明当前仍不适合在这里启动 Docker 实验。它没有确定服务不响应的根因，也没有重启 daemon。第 13 节的 GPU 实验在另一执行节点完成，两者不矛盾。[本次节点路由与 Docker 探活记录](../env-service-verification/results/async-admission-diagnosis-20261020/workspace-docker-health.json)保留了节点区分和退出状态。

**跟着图读源码。**

1. [1] `exec_gpu` 记录容器 ID 和租约 ID，通过 Router 申请，客户端等待完整授权。
2. [2] `acquire` 从租约读取 `node_ip`。预算足够时调用 `_gpu_command_node`，复用或创建固定到该节点的服务。
3. [3 → 4] 节点服务在自己的进程环境里执行 `inspect_container`，通过本地 socket 读取容器状态。
4. [5] 在同一节点核验 GPU 与 cgroup，完成授权后将结果经 Router 返回 [1]。
5. [1] 调用 Harbor 的 `exec` 执行任务。当前安装的 Harbor 将其转为 Compose exec，这部分没有被替换成新接口。仓库内可从 [docker.py:87](../../../psrl/workers/env_service/docker.py#L87)看到委托调用，下游 Harbor 实现属于安装依赖。
6. 命令完成后，[3] 在后台撤权、检查物理归还，[2] 随后释放配额。整任务结束时由 [finish](../../../psrl/workers/env_service/command_gpu.py#L236)清理该租约归属的资源。规划可以与归还检查并发，下一条 GPU 命令不能跳过授权。

这些代码有不清楚的地方，可以按图中编号继续追问。

**不依赖 Docker 和 GPU 的可运行检查。** 复用[接口回归脚本](../../../tests/env_service/test_gpu_control_queries.py#L68)，在临时 Unix socket 上启动模拟 Docker HTTP 服务，再调用真实的 `inspect_container`。它验证真实连接器的传输行为，但后台服务是测试替身，不代表工作区 Docker 已恢复，也不是跨物理机实验。

```bash
python -m pytest -q \
  tests/env_service/test_gpu_control_queries.py::test_inspect_uses_local_socket_and_propagates_missing_container \
  tests/env_service/test_gpu_control_queries.py::test_remote_daemon_cannot_control_local_devices
```

预期：本地模拟服务中的 `owned` 容器信息能返回，不存在的容器得到 404 错误，远程 TCP daemon 配置被拒绝。两项检查均已实际通过，见 [CPU 检查日志](../env-service-verification/results/async-admission-diagnosis-20261020/docker-locality-cpu-tests.log)。这说明健康本地服务上的直接接口可用，同时本机真实 Docker 探活仍失败，明确区分了接口代码与 daemon 健康状况。

