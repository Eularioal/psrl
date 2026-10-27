# GPU 显存准入：实测占用与尚未兑现的预约

> **核心结论**：容量应按 `实测总占用 + Σ max(每任务预约 − 已核实的当前任务显存, 0)` 计算。它等价于“外部占用 + 每个任务的预约与实际占用取大值”。不能把整卡实测和全部预约直接取大值，也不能用历史峰值抵扣当前预约。代码已实现，Env Service 回归 257 项通过，短 Docker/CUDA 验证通过。实机缺少可信宿主进程视图时保留完整预约，可靠归属路径的并发能力由 CPU 回归验证，未重新跑训练。

## 目录

- [1. 全局流程与源码](#1-全局流程与源码)
- [2. 为什么两个总量取 max 不够](#2-为什么两个总量取-max-不够)
- [3. 新公式与可手算例子](#3-新公式与可手算例子)
- [4. 抵扣必须来自当前可信归属](#4-抵扣必须来自当前可信归属)
- [5. 缺少归属时的取舍与适用边界](#5-缺少归属时的取舍与适用边界)
- [6. 跟着图读源码与运行示例](#6-跟着图读源码与运行示例)
- [7. Results与验证边界](#7-results与验证边界)

## 1. 全局流程与源码

本文补充[命令级预约](command-gpu-reservations.md)，修复触发条件来自[science 常驻压力短测](../experiments/science-single-gpu-training-control.md#11-results常驻显存压力四组短测)。原实验结果保留为修复前数据。

```text
[1] GPU 命令请求 → 调度器锁住该 unit，检查队首与已有预约
                         ↓
[2] 节点按活动命令的 lease / container / epoch 查询新显存快照
                         ↓
[3] NVML 读取整卡占用、进程显存
    宿主 procfs 核验 PID 出生时间和容器 cgroup 归属
    ├─ 归属可靠：给对应任务本次实际显存抵扣
    └─ 无法核验：抵扣为零，保留完整预约并记录原因
                         ↓
[4] 账本计算 M + Σ max(Rᵢ − Aᵢ, 0)
    加上新请求额度，与配置预算 / 物理容量比较
    ├─ 超预算 → 保持等待，尚未授权 CUDA
    └─ 可容纳 → 先记预约，再授权容器访问 GPU
                         ↓
[5] 任意 CUDA 程序分配、释放、再次分配
    后续准入重查当前占用，不使用历史峰值
                         ↓
[6] 命令结束 → 禁止新设备访问 → 确认进程 / 句柄退出
    → 调度器清除活动预约 → 下一条重新检查容量
```

图例：图里每个部分对应的源码位置。

| 编号 | 责任与源码 |
|---|---|
| 1、4 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L50) 串行处理该 unit 的准入，在授权前记账 |
| 2 | [gpu_command_memory_snapshot](../../../psrl/workers/env_service/command_gpu_node.py#L111) 在观测前后检查命令身份，丢弃已经归还或换代的抵扣 |
| 3 | [NvmlProbe.read_gpu_snapshot](../../../psrl/workers/env_service/probe.py#L85) 读取当前设备与进程显存；[read_command_memory](../../../psrl/workers/env_service/task_telemetry.py#L30) 核验归属并生成本次抵扣 |
| 4 | [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 集中实现公式和输入约束；[EnvUnitLedger._effective_gpu_mb](../../../psrl/workers/env_service/ledger.py#L357) 只纳入仍活动的 GPU 预约 |
| 5 | [真实 CUDA 验证程序](../env-service-verification/scripts/cuda_pressure.py#L10) 展示实际分配与释放，不是生产程序必须采用的接口 |
| 6 | [调度器 _return](../../../psrl/workers/env_service/command_gpu.py#L139) 在节点确认归还后释放额度；[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L182) 核验物理归还 |

NVML 是 NVIDIA 的设备管理接口，用于读取显存和进程信息。procfs 是 Linux 的进程信息文件系统，cgroup 是进程所属的控制组，用于判断进程是否属于指定容器。PID 是进程编号，可能被回收复用，因此还需要检查进程出生时间。epoch 是同一个租约第几次 GPU 授权，避免上一条命令的数据被下一条误用。这些代码如果有理解上的问题，可以随时追问。

## 2. 为什么两个总量取 max 不够

预约是调度器已经答应将来提供的容量，实测是当前已经分配出去的容量。两者既可能重叠，也可能互不重叠。

以原实验为例，外部进程常驻 70 GiB，任务 A 申报 15 GiB，但获准后尚未调用 CUDA 分配。此时整卡实测 70 GiB 与 A 的 15 GiB 完全不同源，应合计 85 GiB。`max(70, 15)` 却只算 70 GiB，等于让 A 的预约没有增加占用。

下一条 B 也申报 15 GiB 时，错误算法仍检查 `70 + 15 <= 90`，于是继续放行。两条任务各实际需要 14 GiB，合计 `70 + 14 + 14 = 98 GiB`，超过原环境卡容量，最终 OOM（显存分配失败）。这不是调度器内部忘了加锁，而是授权与真正的 CUDA 分配之间有延迟，锁住公式并不能使公式正确。

简单把实测与申报全部相加也不合适。A 已经分配 14 GiB 后，整卡实测 84 GiB 已经包含它。再加完整 15 GiB 会算成 99 GiB，把 A 的同一部分需求算了两次。

## 3. 新公式与可手算例子

令所有数值采用同一种显存单位：

- `M`：本次整卡实测总占用。
- `Rᵢ`：仍活动的任务 i 的预约额度。
- `Aᵢ`：本次观测中，已可靠核实属于任务 i 的当前占用。
- `R_new`：新请求需要的额度。

准入计算为：

```text
U = M + Σᵢ max(Rᵢ − Aᵢ, 0)

只有 U + R_new <= min(配置预算, 物理容量)，才能放行。
```

`max(Rᵢ − Aᵢ, 0)` 表示“已经答应，但还没有被本次实际分配覆盖”的额度。当所有归属都已准确核实时，设外部占用为 E，有 `M = E + ΣAᵢ`，公式等价于：

```text
U = E + Σᵢ max(Rᵢ, Aᵢ)
```

因此这里并没有放弃 max，而是把它从两个混合总量之间，移到每个任务自己的预约和占用之间。

以下是使用原实验配置的手算示例，未计小量 CUDA 上下文开销，不冒充新实测：

| 状态 | M | A 的预约 R | 核实占用 A | 新公式 U | 再申请 15 GiB |
|---|---:|---:|---:|---:|---|
| 外部 70，任务尚未分配 | 70 | 15 | 0 | 85 | 100 > 90，等待 |
| 任务已经分配 14 | 84 | 15 | 14 | 85 | 100 > 90，等待 |
| 任务只分配了 6 | 76 | 15 | 6 | 85 | 100 > 90，等待 |
| 任务释放显存但命令尚未归还 | 70 | 15 | 0 | 85 | 仍保留它再次分配的权利 |
| 命令已确认归还 | 70 | 无活动预约 | 无抵扣 | 70 | 85 <= 90，可放行 |

还必须**逐任务**计算，不能将实测任务显存先汇总再抵扣预约总额。例如 A 预约 10、实际 20，B 预约 30、实际 0，外部占用 0。正确结果是 `20 + 30 = 50`。`max(总预约40, 总占用20)` 只算40，相当于用 A 超用的10抵掉B尚未使用的10，仍会少算。

## 4. 抵扣必须来自当前可信归属

实现分成三层，公式本身没有 Docker、Ray 或 NVML 依赖：

1. `gpu_memory.py` 只处理算术，校验有限非负数、抵扣归属和总量一致性。
2. 节点同时读取设备与进程显存，用宿主进程视图核验容器归属，在观测前后核对进程出生标识。进程被复用、多个任务声明同一进程、GPU UUID 改变，都不能获得不可信抵扣。
3. 命令调度器在 unit 锁内消费这次结果，并在观测后重新检查租约是否过期。节点对 lease、container、epoch 的检查阻止旧命令结果抵扣新预约。

抵扣不会跨准入查询缓存，也不从 `peak_memory_mb` 历史峰值取值。任务可能分配 14 GiB 后释放，接着又分配14。如果继续拿历史峰值抵扣，就会在中间空档提前卖掉仍然承诺给它的显存。

归还阶段保留已有原则：只有节点确认设备句柄和 CUDA 进程退出，调度器才清除活动预约。物理确认失败仍隔离该 unit，不能因为测到显存降低就提前释放预约。取消、超时、租约过期的资源回收仍走原有路径。

## 5. 缺少归属时的取舍与适用边界

仅看“整卡用了84 GiB”，无法判断这是“外部70 + 本任务14”，还是“外部84 + 本任务尚未分配”。这两种状态需要的预约余量不同。没有可靠归属信息时，不存在既精确装箱又保证不漏算的纯总量公式。

因此，无法确认的部分取 `Aᵢ = 0`，**这是不给预约抵扣，不是声称任务实际用了零显存**。上述两种状态都会保守按 `84 + 15 = 99 GiB` 计入。错误原因保存在准入记录的 `attribution_errors` 中。

这会带来明确的吞吐代价：低压力下若已有三条各分配14 GiB、各预约15 GiB，且完全不能归属，那么第四条请求会看到 `42 + 45 + 15 = 102 GiB` 而等待；归属可靠时只需 `42 + 3×1 + 15 = 60 GiB`，仍能放行。不能宣称修复在缺少归属能力的部署上完全没有并发损失。应部署可信的宿主 procfs 视图，使精确抵扣可用，而不是用不可靠 PID 或固定延时绕过核验。

两个模式的范围也要分清：当前 command 路径每次获取带归属的新快照；普通 episode 路径只收到整卡占用，因此采用保守计算，没有在本次添加新的 episode 进程归因服务。绑定推理副本的 GPU lending（同卡借还）继续由其物理交接协议管理，仅按借用任务的预约记账，不能把尚未 sleep 的模型显存混入普通 command 计算。

公式修复不等于显存硬隔离。未参与调度的外部进程仍可能在采样后增加占用，任务也可能超出自己申报的峰值。NVML 多项读取不是一个原子的硬件事务。新公式解决的是本次可复现的未兑现预约少算，并在可验证观测范围内进行准入，不能保证任意外部干扰下绝不 OOM。多卡 unit 的预算仍是总量，也不自动解决某一张卡单独过载。

## 6. 跟着图读源码与运行示例

按真实调用路径阅读：

| 步骤 → 图中编号 | 函数入口 | 输入、输出或状态变化 |
|---|---|---|
| 请求 → 1 | [acquire:50](../../../psrl/workers/env_service/command_gpu.py#L50) | 队首请求携带租约，收集本 unit 活动 / 归还中命令的身份 |
| 观测 → 2、3 | [gpu_command_memory_snapshot:111](../../../psrl/workers/env_service/command_gpu_node.py#L111) | 请求节点当前显存及每个已验证任务的抵扣，检查跨查询换代 |
| 算术 → 4 | [accounted_gpu_memory_mb:9](../../../psrl/workers/env_service/gpu_memory.py#L9) | 整卡实测加逐任务未兑现预约，返回可与预算比较的 U |
| 授权 → 4、5 | [grant_gpu_command:168](../../../psrl/workers/env_service/command_gpu_node.py#L168) | 已先记账，再打开容器的 GPU 设备权限，任务随后自行分配显存 |
| 归还 → 6 | [_return:139](../../../psrl/workers/env_service/command_gpu.py#L139) | 节点确认回收后清除预约，等待者重新取样；后台归还与 CPU 工作可并发 |

在仓库根目录，激活项目环境后运行 CPU 示例：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/explain_resident_admission_window.py
```

它调用真实调度器和账本，节点遥测模拟为外部常驻70 GiB、任务还没分配。预期三个请求依次为 `granted / wait / wait`，有效占用为85 GiB。这是明确标注的 CPU 教学模拟，不需要 GPU。修复前的脚本和输出保存在原实验的快照中，不用改写历史结果。

针对性回归：

```bash
PYTHONPATH=. python -m pytest -q tests/env_service/test_gpu_memory.py tests/env_service/test_task_telemetry.py
```

真实短验证脚本为 [verify_resident_admission.py](../env-service-verification/scripts/verify_resident_admission.py)。先检查目标节点 GPU、Docker 和设备映射，准备含 `python3`、`sleep` 的本地镜像，用环境变量提供实际值：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/verify_resident_admission.py \
  --gpu "$TEST_GPU" --image "$TEST_IMAGE" --output "$TEST_OUTPUT"
```

`TEST_GPU` 是现场空闲 GPU 的 index，`TEST_IMAGE` 是已验证的本地镜像 ID，`TEST_OUTPUT` 是未使用的新结果目录。脚本固定使用70 GiB常驻、三条14 GiB请求、各持有2秒，执行阶段上限120秒，常驻分配180秒到期。它直接调用生产节点类，不启动 Ray 或模型训练；先获准第一条而不执行分配，再检查后两条必须等待，从而确定性覆盖这次出错窗口。

## 7. Results与验证边界

- 第一轮针对性回归79项通过，随后包含命令代次交错用例的全部 Env Service 回归257项通过。
- CPU 示例修复后为 `granted / wait / wait`。准确归属下四路并发、未知归属、部分分配、超申报、进程复用和重叠归属均有回归覆盖。
- 真实 Docker/CUDA 验证约10.43秒完成。70 GiB常驻贯穿三条命令，首次分配前两个等待者被阻挡，三条14 GiB分配均返回42且无OOM，归还后队首正常获准。
- 实机没有可信宿主 procfs，因此本次真实验证覆盖保守路径。没有把它写成已验证精确归属装箱，也没有据此推算修复后的训练提速。
- 三个所属容器已回收，常驻进程退出码0，最终节点快照无CUDA进程、设备占用回到初始水平。本次没有追加训练。

定向 Ruff、格式检查通过。全仓 pre-commit 在隔离副本执行，仍因其他文件的已有风格问题失败，未将自动格式化带回工作目录。

证据：[CPU 示例输出](../env-service-verification/results/gpu-memory-admission-fix-20261026/cpu-example.json)、[物理释放核验](../env-service-verification/results/gpu-memory-admission-fix-20261026/physical-cleanup.log)、[全仓检查日志](../env-service-verification/results/gpu-memory-admission-fix-20261026/precommit-all.log)、[全量回归日志](../env-service-verification/results/gpu-memory-admission-fix-20261026/env-service-tests.log)、[真实验证结果](../env-service-verification/results/gpu-memory-admission-fix-20261026/docker-cuda/result.json)。本次代码与文档保留为本地未提交修改。
