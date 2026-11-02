# GPU 显存准入：实测占用与尚未兑现的预约

> **核心结论**：unit 的配置容量与实时准入余量是两层。独立环境 GPU 应按整卡允许使用的容量建池，实时占用不能自动扩大显式低配的池。当前记账为 `实测总占用 + Σ max(每任务预约 − 已核实的当前任务显存, 0)`，command 与普通 episode 路径均支持可信归属抵扣。无法确认归属时仍保守叠加实测与预约。episode 的整卡压力验证见第 9 节，历史验证与 CPU 教学模拟分别标注。

## 目录

- [1. 全局流程与源码](#1-全局流程与源码)
- [2. 为什么两个总量取 max 不够](#2-为什么两个总量取-max-不够)
- [3. 新公式与可手算例子](#3-新公式与可手算例子)
- [4. 抵扣必须来自当前可信归属](#4-抵扣必须来自当前可信归属)
- [5. 缺少归属时的取舍与适用边界](#5-缺少归属时的取舍与适用边界)
- [6. 跟着图读源码与运行示例](#6-跟着图读源码与运行示例)
- [7. Results与验证边界](#7-results与验证边界)
- [8. unit 总容量、动态余量与装箱气泡](#8-unit-总容量动态余量与装箱气泡)
- [9. episode 归属记账修复](#9-episode-归属记账修复)

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
| 1、4 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L58) 串行处理该 unit 的准入，在授权前记账 |
| 2 | [gpu_command_memory_snapshot](../../../psrl/workers/env_service/command_gpu_node.py#L135) 在观测前后检查命令身份，丢弃已经归还或换代的抵扣 |
| 3 | [NvmlProbe.read_gpu_snapshot](../../../psrl/workers/env_service/probe.py#L85) 读取当前设备与进程显存；[read_command_memory](../../../psrl/workers/env_service/task_telemetry.py#L10) 核验归属并生成本次抵扣 |
| 4 | [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 集中实现公式和输入约束；[EnvUnitLedger._effective_gpu_mb](../../../psrl/workers/env_service/ledger.py#L432) 只纳入仍活动的 GPU 预约 |
| 5 | [真实 CUDA 验证程序](../env-service-verification/scripts/cuda_pressure.py#L10) 展示实际分配与释放，不是生产程序必须采用的接口 |
| 6 | [调度器 _return](../../../psrl/workers/env_service/command_gpu.py#L215) 在节点确认归还后释放额度；[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L207) 核验物理归还 |

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

命令抵扣不使用跨查询的历史值。episode 将同一次观测的占用与抵扣成对保存，超过观测 TTL（有效期）后同时失效，也不从 `peak_memory_mb` 历史峰值取值。任务可能分配 14 GiB 后释放，接着又分配14。如果继续拿历史峰值抵扣，就会在中间空档提前卖掉仍然承诺给它的显存。

归还阶段保留已有原则：只有节点确认设备句柄和 CUDA 进程退出，调度器才清除活动预约。物理确认失败仍隔离该 unit，不能因为测到显存降低就提前释放预约。取消、超时、租约过期的资源回收仍走原有路径。

## 5. 缺少归属时的取舍与适用边界

仅看“整卡用了84 GiB”，无法判断这是“外部70 + 本任务14”，还是“外部84 + 本任务尚未分配”。这两种状态需要的预约余量不同。没有可靠归属信息时，不存在既精确装箱又保证不漏算的纯总量公式。

因此，无法确认的部分取 `Aᵢ = 0`，**这是不给预约抵扣，不是声称任务实际用了零显存**。上述两种状态都会保守按 `84 + 15 = 99 GiB` 计入。错误原因保存在准入记录的 `attribution_errors` 中。

这会带来明确的吞吐代价：低压力下若已有三条各分配14 GiB、各预约15 GiB，且完全不能归属，那么第四条请求会看到 `42 + 45 + 15 = 102 GiB` 而等待；归属可靠时只需 `42 + 3×1 + 15 = 60 GiB`，仍能放行。不能宣称修复在缺少归属能力的部署上完全没有并发损失。应部署可信的宿主 procfs 视图，使精确抵扣可用，而不是用不可靠 PID 或固定延时绕过核验。

两个模式的范围也要分清：command 路径按命令获取带归属的新快照；episode 路径现已按租约覆盖整个容器集合，见第 9 节。原先 episode 只有整卡占用的限制已经修复，无法核验归属的部署仍回退到保守计算。绑定推理副本的 GPU lending（同卡借还）继续由其物理交接协议管理，仅按借用任务的预约记账，不能把尚未 sleep 的模型显存混入普通 command 计算。

公式修复不等于显存硬隔离。未参与调度的外部进程仍可能在采样后增加占用，任务也可能超出自己申报的峰值。NVML 多项读取不是一个原子的硬件事务。新公式解决的是可复现的未兑现预约少算，并在可验证观测范围内进行准入，不能保证任意外部干扰下绝不 OOM。当前多卡 unit 还要求逐卡预算并逐卡检查，不能只把各卡剩余容量相加就放行，见 [_effective_gpu_by_device](../../../psrl/workers/env_service/ledger.py#L487) 与 [_gpu_fits_now](../../../psrl/workers/env_service/ledger.py#L530)。

## 6. 跟着图读源码与运行示例

按真实调用路径阅读：

| 步骤 → 图中编号 | 函数入口 | 输入、输出或状态变化 |
|---|---|---|
| 请求 → 1 | [acquire:58](../../../psrl/workers/env_service/command_gpu.py#L58) | 队首请求携带租约，收集本 unit 活动 / 归还中命令的身份 |
| 观测 → 2、3 | [gpu_command_memory_snapshot:135](../../../psrl/workers/env_service/command_gpu_node.py#L135) | 请求节点当前显存及每个已验证任务的抵扣，检查跨查询换代 |
| 算术 → 4 | [accounted_gpu_memory_mb:9](../../../psrl/workers/env_service/gpu_memory.py#L9) | 整卡实测加逐任务未兑现预约，返回可与预算比较的 U |
| 授权 → 4、5 | [grant_gpu_command:192](../../../psrl/workers/env_service/command_gpu_node.py#L192) | 已先记账，再打开容器的 GPU 设备权限，任务随后自行分配显存 |
| 归还 → 6 | [_return:215](../../../psrl/workers/env_service/command_gpu.py#L215) | 节点确认回收后清除预约，等待者重新取样；后台归还与 CPU 工作可并发 |

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

以下是早前 command 记账修复的验证记录，本次 episode 修复与整卡压力实验见第 9 节。

- 第一轮针对性回归79项通过，随后包含命令代次交错用例的全部 Env Service 回归257项通过。
- CPU 示例修复后为 `granted / wait / wait`。准确归属下四路并发、未知归属、部分分配、超申报、进程复用和重叠归属均有回归覆盖。
- 真实 Docker/CUDA 验证约10.43秒完成。70 GiB常驻贯穿三条命令，首次分配前两个等待者被阻挡，三条14 GiB分配均返回42且无OOM，归还后队首正常获准。
- 实机没有可信宿主 procfs，因此本次真实验证覆盖保守路径。没有把它写成已验证精确归属装箱，也没有据此推算修复后的训练提速。
- 三个所属容器已回收，常驻进程退出码0，最终节点快照无CUDA进程、设备占用回到初始水平。本次没有追加训练。

定向 Ruff、格式检查通过。全仓 pre-commit 在隔离副本执行，仍因其他文件的已有风格问题失败，未将自动格式化带回工作目录。

证据：[CPU 示例输出](../env-service-verification/results/gpu-memory-admission-fix-20261026/cpu-example.json)、[物理释放核验](../env-service-verification/results/gpu-memory-admission-fix-20261026/physical-cleanup.log)、[全仓检查日志](../env-service-verification/results/gpu-memory-admission-fix-20261026/precommit-all.log)、[全量回归日志](../env-service-verification/results/gpu-memory-admission-fix-20261026/env-service-tests.log)、[真实验证结果](../env-service-verification/results/gpu-memory-admission-fix-20261026/docker-cuda/result.json)。本次代码与文档保留为本地未提交修改。

## 8. unit 总容量、动态余量与装箱气泡

针对常驻推理实验的疑问，先串联当前实现。这里的自动准入，是自动计算已经配置的资源池还能接纳多少任务，不是自动发现任意空闲 GPU 并把所有 unit 扩大到整卡。

```text
[1] 部署或启动器选定 GPU，建立 unit 的设备、总容量、CPU/主存和并发槽
        |
        v
[2] Router 读取 units，节点用现场 NVML 总容量校验预算没有超过物理卡
        |
        v
[3] 到达新任务 -> 节点更新实际显存占用 -> ledger 合并未兑现预约
        |
        +-- 容量、CPU、主存和槽位足够 -> [4] 给同一 unit 增加一条任务租约
        |                                      |
        |                                 多个任务可以共享同一张 GPU
        |
        +-- 不足 -> 等待或选择其他符合条件的已配置 unit
                                               |
[5] 任务清理与资源回收确认 -> 解除预约 -> 等待者重新检查容量
```

图例：图里每个部分对应的源码位置。

| 编号 | 函数与职责 |
|---|---|
| 1 | [EnvUnit](../../../psrl/workers/env_service/types.py#L23) 定义 unit，`gpu_memory_mb` 是池容量，`max_containers` 是独立并发上限。普通 benchmark [run.py](../../../examples/gpu_benchmarks/run.py#L116) 从现场 `total_mb` 填容量 |
| 2 | [EnvRouterCore.__init__](../../../psrl/workers/env_service/router.py#L48) 读取显式配置，[验证逐卡容量](../../../psrl/workers/env_service/router.py#L509) 拒绝超过物理总量，不自动放大低配预算 |
| 3 | [RayEnvRouter.try_admit](../../../psrl/workers/env_service/router.py#L658) 查询节点健康与资源，[实际设备采样](../../../psrl/workers/env_service/probe.py#L85) 返回总容量和占用，[ledger 计算有效占用](../../../psrl/workers/env_service/ledger.py#L432) 保留预约 |
| 4 | [EnvUnitLedger.try_admit](../../../psrl/workers/env_service/ledger.py#L112) 挑选能容纳任务的 unit 并建立租约，[逐资源检查](../../../psrl/workers/env_service/ledger.py#L508) 同时检查槽位、CPU、主存和 GPU |
| 5 | [Router.release](../../../psrl/workers/env_service/router.py#L241) 区分 episode、command 和借卡回收路径，[节点清理](../../../psrl/workers/env_service/node.py#L182) 处理物理资源回收责任 |

跟着图读源码：从启动器提供给 Router 的 `units` 开始，确认选中的设备和总预算。Router 创建节点服务时校验物理容量，后续准入会刷新观测并进入 ledger。ledger 先按节点与 GPU 使用策略筛选，再检查各维资源，写入一条 lease 后，才由 runner 创建环境。lease 是某个任务已获准使用资源的记录，一个 unit 可以同时持有多条 lease。清理时 episode 先进入待回收责任，不能仅删除逻辑租约就假定物理资源空闲。节点探测和后台清理可以并发进行，不应把图读成单一同步线程。相关源码如果有理解上的问题，可以随时继续追问。

**总容量、余量、单任务需求是三个不同数。**

| 数值 | 含义 | 当前来源 |
|---|---|---|
| unit 总预算 C | 这组设备允许给池记账的容量上限 | 部署配置，或启动器探测整卡容量后生成配置 |
| 实测占用 M | GPU 当前已经被进程分配的显存 | 节点 NVML 采样，存在采样间隔 |
| 任务预约 R | 给一个任务保留的需求预算，包含尚未分配但可能需要的部分 | 任务声明、profile 后设置或默认需求值 |

因此，“自动根据余量决定是否放行”与“必须给 unit 一个容量上限”并不矛盾。当前 Router 不会猜测一个显式小配额是测试设置、资源隔离要求还是误配，也没有自动扩大它的逻辑。配置中的 `default_gpu_memory_mb: 24000` 是未声明需求的**单任务默认预约**，不是整卡 unit 的固定容量，更不是此前实验的 `24576 MiB` 池配额。

独立、专用于环境的 GPU，应将其允许使用的整卡容量纳入 unit，可按部署政策保留明确余量。本例启动前 GPU 空闲，没有任务外保留要求，给它只配置 24 GiB 没有利用率方面的必要性。实际卡约 95.6 GiB，只入账 24 GiB，相当于约 71.6 GiB 未纳入这个池。这是实验入口人为造成的容量空洞。不要把 unit 总预算设成采样到的 `free`，然后又在准入时减一次实测占用，否则同一外部占用会被扣两次。应明确使用“总预算减有效占用”的口径。

当前 [validate_units](../../../psrl/workers/env_service/types.py#L160) 禁止同一节点的同一 GPU 被两个 unit 重复声明。因此，不能用两个 unit 分别声明同一卡的 24 GiB 和 72 GiB 来补齐这个空洞。常见配置是“一张环境 GPU 对应一个 unit，多个任务在 unit 内共享容量”。未配置的其他 GPU 也不会自动成为候选资源。多卡任务可以用多设备 unit，但仍须逐卡校验，不能把不同卡的零散余量当成一张大卡的连续显存。

**覆盖整卡之后，还不等于没有气泡。** 气泡在这里指已有物理容量没有被有效利用，至少有以下不同原因：

1. 池容量配置过小。把整卡纳入预算即可消除这类配置空洞。
2. 任务申报保守。一次短请求用了 14 GiB，不代表其所有允许的上下文和批量配置永远只用 14 GiB。需要按实际工作范围校准预约，不能用瞬时空闲覆盖尚未兑现的承诺。
3. 归属不明导致保守记账。修复前普通 episode 只有整卡观测，采用 `M + ΣR`。同一任务已分配的部分同时出现在 M 和 R 中。第 9 节补齐了 episode 归属传递，无法核验的部署仍会出现这种保守气泡。
4. 槽位、CPU、主存或任务尺寸限制。即使还有显存，`max_containers=2` 仍只允许两个任务。剩余 15 GiB 不能放入需求 20 GiB 的任务，但可能可以运行更小任务，前提是其他限制满足。
5. 计算空窗。模型驻留但等待下一次请求时，GPU 计算单元可能空闲，显存却没有释放。提高显存装箱率与填补计算空窗是两项不同目标。

**教学例子，不是新的 GPU 实测。** 假设一张 96 GiB 卡，槽位、CPU、主存都足够，每任务预约 20 GiB，并在下一请求到达前实际分配 14 GiB。

| 状态 | 普通 episode 无归属抵扣 | 有可信当前归属时的记账公式 |
|---|---|---|
| 已有两个任务，再接纳第三个 | `28 + 40 + 20 = 88`，可接纳 | `28 + 2×(20−14) + 20 = 60`，可接纳 |
| 已有三个任务，再接纳第四个 | `42 + 60 + 20 = 122`，不能接纳 | `42 + 3×(20−14) + 20 = 80`，可接纳 |

这解释了为什么“已经读了实时显存”仍可能留下空洞。安全地减少这类重复记账，需要当前、可信、逐任务的占用归属，不能直接把整卡占用全部认作本系统任务，也不能改回 `max(整卡实测, 总预约)`。普通 episode 在归属失败时仍保留保守路径，第 9 节新增了可信归属成功时的生产抵扣路径。

unit 选择策略也不是全局最优装箱器。当前 [选择逻辑](../../../psrl/workers/env_service/ledger.py#L170) 同时考虑能否即时用 GPU、资源优先级及负载，[负载函数](../../../psrl/workers/env_service/ledger.py#L545) 使用最紧张资源的比例。它不会承诺把任意任务组合的剩余空间压到最小。

**此前已做的默认值纠正。** [resident_ab.resolve_pool_gpu_mb](../../../examples/inference_bench/resident_ab.py#L22) 默认读取所选 GPU 的现场总容量，不再默认 24 GiB，仍要求选择空闲独立 GPU。只有显式传 `--pool-gpu-mb 24576` 才执行小配额压力情形，超过物理容量或非正数会被拒绝。审计器默认不再要求 ON 必须只有一个 GPU 进程，`--expect-serialized` 仅用于这组历史压力数据。原始结果 JSON 与运行哈希保持原样，修改前脚本保存在原始结果的 `executed-source/`，不能将代码默认值纠正说成已经完成新 GPU 对照。

可运行 CPU 教学脚本使用真实 `EnvUnitLedger`，但显存观测、96 GiB 总量和 14 GiB 分配均为模拟，不创建 Ray、容器或 GPU 进程：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/explain_unit_capacity.py
```

本次实际运行输出：24 GiB 池只有第一条 `granted`，其余 `wait`。96 GiB 池前三条 `granted`，第四条 `wait`。脚本还核实三个活动任务的有效占用在无归属时为 102 GiB、在提供可信归属的算术示例中为 60 GiB，以及两个 unit 重复声明同一卡会被拒绝。清理步骤显式模拟物理回收确认，再释放预约并核验等待者可进入。脚本现还增加可信归属分支，96 GiB 池可接纳第四条。这证明账本计算路径，本身不能作为实际装箱吞吐或新版 ON 加速结果。


## 9. episode 归属记账修复

episode 是一项任务从环境创建到清理的完整生命周期。模型服务可以跨多条命令一直驻留，所以预约不能在某条 CPU 命令结束时消失。这次修复的是已兑现预约的重复计算，不是提前取消尚有效的预约。

```text
[1] Router 刷新节点，收集仍有效的 episode lease
             ↓
[2] 节点按 lease/unit 标签发现 agent、verifier 等容器
    Docker inspect 核对容器 ID、启动时间和 cgroup
             ↓
[3] procfs 或 eBPF 核验 GPU 进程归属与出生身份
    NVML 读取当前逐设备、逐进程显存，前后重查身份
             ↓
[4] 容器抵扣汇总为 lease × GPU 抵扣
    Router 原子提交“这次测量 + 这次抵扣”到 ledger
             ↓
[5] 总量和逐卡都计算 M + Σ max(Rᵢ − Aᵢ, 0)
    准入 → 创建环境 → 执行多条命令期间保持预约
             ↓
[6] 清理容器、释放租约，旧归属失效，关闭身份注册
```

图例：图里每个部分对应的源码位置。

| 图中编号 | 函数与职责 |
|---|---|
| 1 | [RayEnvRouter._check_node](../../../psrl/workers/env_service/router.py#L515) 收集租约，消费同一次显存观测 |
| 2 | [EnvNodeService.episode_gpu_memory_snapshot](../../../psrl/workers/env_service/node.py#L131) 选择节点身份后端；[EpisodeMemory._snapshot](../../../psrl/workers/env_service/episode_memory.py#L60) 遍历所属容器 |
| 2、3 | [container_identity](../../../psrl/workers/env_service/episode_memory.py#L13) 解析实际 cgroup，拒绝特权容器和不匹配的容器路径；[BpfIdentity](../../../psrl/workers/env_service/bpf_identity.py#L137) 检查内核归属后端 |
| 3 | [read_command_memory](../../../psrl/workers/env_service/task_telemetry.py#L10) 复用经过前后身份核对的当前显存采样，新增逐设备抵扣返回值 |
| 4 | [EnvUnitLedger.report_measurement](../../../psrl/workers/env_service/ledger.py#L277) 校验每个 lease 的设备范围和抵扣总量，再成对替换观测 |
| 5 | [_effective_gpu_mb](../../../psrl/workers/env_service/ledger.py#L432)、[_effective_gpu_by_device](../../../psrl/workers/env_service/ledger.py#L487) 分别检查总量与每张卡，避免跨卡挪用余量 |
| 6 | [Router.release](../../../psrl/workers/env_service/router.py#L241)、[节点 cleanup](../../../psrl/workers/env_service/node.py#L182)、[EpisodeMemory.close](../../../psrl/workers/env_service/episode_memory.py#L51) 保留物理清理责任并关闭归属注册 |

跟着图读源码：请求到达后先走 1 刷新健康状态。2 将租约 ID 映射到整个容器集合，而非只看 agent。3 的 eBPF 是 Linux 内核可加载的受约束观测程序，这里用它识别宿主 GPU PID 的真实出生标识和 cgroup 归属，适用于当前进程看不到宿主 `/proc` 的部署。它不是从容器进程名猜 PID，也不通过采样历史峰值估算当前占用。4 将结果交给账本时再次检查 lease 仍然有效。5 写入新预约后才启动任务。6 与后台观测可以并发，因此采样结束还会重查容器是否重启，账本也会丢弃已释放租约的抵扣。这些函数可以按图逐个追问。

**手算例子，数字为教学假设。** 整卡容量 96 GiB，三项任务各预约 20 GiB、实际各分配 14 GiB，没有其他程序占用。修复前记为 `42 + 60 = 102 GiB`，第四项 20 GiB 请求无法进入。修复后为 `42 + 3×(20−14) = 60 GiB`，再加第四项得到 80 GiB，可以进入同一个 unit。若另有外部程序占 30 GiB，结果就变成 `30 + 60 + 20 = 110 GiB`，仍须等待。归属抵扣不会把外部程序的占用当作自己的预约。

**保护边界。** 观测过期时不继续使用旧抵扣。收到仅有整卡占用的新观测时清除旧抵扣，不能混用新实测与旧归属。容器被替换、进程 PID 被复用、归属冲突或显存数据不一致时保守处理。单个 episode 允许多个已核验容器的占用相加，但不允许同一进程给多个容器抵扣。默认身份后端仍为 `procfs`，部署需提供可信宿主进程视图；本次私有实验节点使用已现场验证支持的 `gpu_identity_backend: ebpf`。当前新增的容器解析对应 cgroup v1 devices 布局，其他布局没有被本次证明支持，不支持时不给抵扣并保留错误。

**可运行 CPU 示例。** 在仓库根目录设置本地源码路径后执行：

```bash
PYTHONPATH="$PWD" python docs/env-worker-deep-dive/env-service-verification/scripts/explain_unit_capacity.py
```

它调用真实 ledger，但用小整数模拟 GPU 观测，不启动 Docker 或 GPU。输出应同时包含：24 GiB 配额只能放行第一项；96 GiB、无归属抵扣时放行前三项；96 GiB、带可信归属时四项全部放行。脚本还模拟清理后重新准入，并检查两个 unit 不能重复声明同一 GPU。真实整卡高压力数据与代码版本记录在 [GPU benchmark 实验记录](../experiments/gpu-benchmarks/FINDINGS.md)，不能把该 CPU 教学结果当作真实模型吞吐。
