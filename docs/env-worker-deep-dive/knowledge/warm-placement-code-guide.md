# 从零读懂热卡等待、冷卡重建与异步执行代码

> **核心结论**：调度先回答“能否放下”，再比较“等已有模型”与“另加载一份模型”的完整成本。`warm_dispatch.py` 只计算建议，驱动和队列执行器负责提交，容器提供者与 GPU 服务负责执行和清理。新增[可预测性门控](command-predictability.md)约束何时允许时间预测改变放置。Python 协程让多个等待过程重叠，不会自动使同一张 GPU 上的训练并行。
>
> **核对范围**：本文的矩阵驱动与 GPU 服务说明对应 2026-11-13 的 `6aa328d` 加当时未提交内容，GPU 性能以[系统对照报告](../experiments/gpu-benchmarks/warm-placement-matrix.md)及[公开证据](../experiments/gpu-benchmarks/warm-placement-matrix-evidence.json)为准，完成 53/57 个区块，48 个核心区块完整。2026-11-16 新增命令门控、受管理队列及固定步数 GPU 服务接线，CPU 验证通过，新入口尚未实机运行，详见[独立教程](command-predictability.md)。生产 Router 未默认启用通用预测调度，历史冻结证据未改。


## 目录

- [1. 先看完整调用图](#1-先看完整调用图)
- [2. 一个请求为什么可能要等待或重建](#2-一个请求为什么可能要等待或重建)
- [3. 协程线程进程与 GPU 并行](#3-协程线程进程与-gpu-并行)
- [4. await 与任务怎样组织执行](#4-await-与任务怎样组织执行)
- [5. 类型化成本与三种放置策略](#5-类型化成本与三种放置策略)
- [6. 进度怎样变成剩余时间预测](#6-进度怎样变成剩余时间预测)
- [7. 失败取消与资源生命周期](#7-失败取消与资源生命周期)
- [8. 矩阵实验究竟验证什么](#8-矩阵实验究竟验证什么)
- [9. 运行 CPU 协程教学实验](#9-运行-cpu-协程教学实验)
- [10. 跟着图读源码](#10-跟着图读源码)

## 1. 先看完整调用图

这里的 command 是已经确定代码、输入和更新步数的一条 GPU 工作命令。实验回放固定训练程序，没有让新的 Agent 临时生成下一条命令，也没有执行 PSRL 策略训练。

下图解释原 GPU 矩阵的真实调用链。新的可复用接口在 W2 与 W3 之间增加进度完整性、稳定性及完成误差校准，在 W4 处由 `ManagedCommandDispatcher` 管理逐设备队列并调用提供者。首次样本不足时先走容量安全放置，同时积累影子预测。完整新图和源码入口见[命令可预测性教程](command-predictability.md#1-从一个真实问题看完整流程)。

```text
[W0] matrix_suite：冻结源码与镜像，启动子进程，用总截止时间监督
           │
           ▼
[W1] matrix：固定案例、先做画像，准备目标命令 A 和阻塞命令 B
           │ 先启动 B，主协程等待 B 已完成 64 步的进度
           ▼
[W2] 进度预测 + ModelCost + QueueSnapshot
           │ 只有决策时已经观察到的成本与状态
           ▼
[W3] WarmPlacementController.decide：纯计算，返回建议
           ├─ 保持亲和 → A 请求发送到热卡服务，等待 B
           └─ 复制模型 → A 请求发送到冷卡服务
           ▼
[W4] 驱动 TaskGroup 管理 B/A 两个请求任务
           │ request → EnvWorker 或固定旧 runner → client → 本机 socket
           ├───────────────────────────┐
           ▼                           ▼
[W5] 热卡：串行服务先做 B，再接 A    冷卡：串行服务可重建 A 并执行
           │                           │
           └──── 原子发布 progress ─────┘ → 回到 W2 的可见状态
           ▼
[W6] 两个请求都返回 → 记录工作块时间 → 清空冷卡复制状态
           ▼
[W7] finally：取消/汇合客户端 → 停所属容器 → 释放租约 → 核实物理释放
           │ 父进程确认本组释放后，才允许下一组
           └──────────────────────────────────────→ W0

[T] 父进程采样线程：每隔一段时间只读查询 GPU → 写观测日志
    它是 W0 启动的旁路观察者，不替 W3 作决定，也不替 W5 训练。
```

图中的“热”指已经驻留需要的模型，“冷”指尚未驻留，和设备温度没有关系。两张卡上的服务是不同进程。同一服务使用串行 socket 处理，A、B 都送到热卡时不会因为驱动创建了两个协程就同时训练。Unix socket 是同一台机器上进程之间的通信入口。客户端发一条 JSON 文本请求，服务接收并执行，再返回 JSON 结果。JSON 是保存字段名和数值的文本格式。

EnvWorker 是当前容器执行实现。runner 是启动和组织任务的程序，固定旧 runner 由 [load_old_runner](../env-service-verification/scripts/warm_router_study/baseline.py#L119)核验，来自提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，它与当前 EnvWorker 路径分别启动，不把当前代码简单关一个开关当成旧版本。租约是带身份的资源许可，记录本环境获准使用的设备和预算。

图例：图里每个部分对应的源码位置。

| 节点或箭头 | 实际职责与函数 | 源码 |
|---|---|---|
| W0 | `freeze` 固定脚本与纯模块，`run` 组织后端，`supervise_backend` 监督所属进程组，`stop_owned` 只按本次标签回收 | [freeze](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L37)、[run](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L161)、[监督](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L138)、[标签回收](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L88) |
| T | `GpuSampler` 用独立线程采样，停止时有限等待 | [GpuSampler](../env-service-verification/scripts/warm_router_study/gpu_sampler.py#L14) |
| W1 | `plan_cells` 固定工作量与策略顺序，`study` 做画像并逐单元执行 | [matrix.py](../env-service-verification/scripts/warm_router_study/matrix.py#L21) |
| W1 → W2 | `observe` 等待指定 B 命令的进度，不读取其他命令的旧报告 | [observe](../env-service-verification/scripts/warm_router_study/matrix.py#L91) |
| W2 | `predict_remaining` 估计固定步数剩余时间，`ModelCost` 和 `QueueSnapshot` 保存显式输入 | [预测器](../../../psrl/workers/env_service/step_predictor.py#L33)、[成本](../../../psrl/workers/env_service/warm_dispatch.py#L79)、[队列快照](../../../psrl/workers/env_service/warm_dispatch.py#L98) |
| W3 | `decide` 检查容量、累计队列成本并比较策略，不执行 GPU 操作 | [控制器](../../../psrl/workers/env_service/warm_dispatch.py#L237) |
| W4 | `study` 的 TaskGroup 创建两个请求任务，提供者的 `request` 调用容器命令 | [matrix.study](../env-service-verification/scripts/warm_router_study/matrix.py#L108)、[native request](../env-service-verification/scripts/warm_router_study/experiment.py#L129)、[旧 runner request](../env-service-verification/scripts/warm_router_study/baseline.py#L228) |
| W4 → W5 | 客户端连接 Unix socket，发送 JSON 并等待一条结果 | [client.py](../env-service-verification/scripts/warm_router_study/client.py) |
| W5 | `Service.execute` 恢复参数并执行固定训练，`Handler.handle` 分派服务操作 | [训练循环](../env-service-verification/scripts/warm_router_study/server.py#L59)、[服务入口](../env-service-verification/scripts/warm_router_study/server.py#L225) |
| W5 → W2 | `progress` 先写临时文件，再替换正式报告 | [原子发布](../env-service-verification/scripts/warm_router_study/server.py#L23) |
| W6 | `study` 在 TaskGroup 正常退出后取结果，单列冷卡清空耗时 | [matrix.study](../env-service-verification/scripts/warm_router_study/matrix.py#L108) |
| W7 | 提供者处理所属物理资源，清理辅助函数控制等待时限 | [native finally](../env-service-verification/scripts/warm_router_study/experiment.py#L366)、[runtime_guard](../env-service-verification/scripts/warm_router_study/runtime_guard.py#L13) |

这些代码如果有理解上的问题，可以随时按 W1 至 W7 追问。源码表区分计算建议、提交工作、真实执行和物理回收，不能把控制器函数返回理解成 GPU 资源已经分配。

## 2. 一个请求为什么可能要等待或重建

先用教学数字说明。热卡已经有目标模型 A，但正在执行 B。预计 B 还要 12 秒，A 本身执行 4 秒。冷卡空闲，却需要先用 5 秒加载 A，再执行 4 秒。

| 放置 | 排队 | 模型重建 | A 执行 | 从此刻到 A 完成 |
|---|---:|---:|---:|---:|
| 热卡 | 12 秒 | 0 秒 | 4 秒 | 16 秒 |
| 冷卡 | 0 秒 | 5 秒 | 4 秒 | 9 秒 |

复制让 A 提前 7 秒完成。若控制器要求至少超过 2 秒的切换门槛，9 + 2 < 16，仍可选择冷卡。如果 B 只剩 1 秒，热卡总计 5 秒，冷卡仍需 9 秒，就应该保留模型亲和。

模型亲和指优先把请求交给已经有相同模型的设备。重建包括重新读取权重、建立模型对象和把张量送入 GPU，不能只理解为显卡之间的一次内存复制。CUDA 是 NVIDIA GPU 的编程与执行接口。活 CUDA 状态包括进程在设备上持有的模型张量和运行状态。当前服务没有把 B 的这些状态搬走，B 会继续在原卡运行。

比较的是固定请求的完成成本，不是“让每张卡都有工作”。GPU 空闲时复制一个大模型，可能比等待一个很短的命令更慢。反过来，长命令足够摊薄重建成本时，两张独立卡并行才可能值得。

这里的 12、4、5 秒都是教学假设。真实输入由画像与当前进度给出，实际命令完成后再检查误差，不能拿事后看到的准确时长回填原来的决策。

## 3. 协程线程进程与 GPU 并行

协程可以理解为“能在等待点暂停、以后接着执行的函数”。驱动调用容器后，大部分时间在等另一个进程、socket 或设备工作完成。暂停这个等待者，事件循环就能处理另一个请求。

事件循环是当前 Python 线程中负责推进异步任务的调度器。它不是 GPU 调度器。它只决定哪个 Python 等待任务现在有机会继续执行。

| 名称 | 谁安排它执行 | 本实验中是什么 | 不能推出什么 |
|---|---|---|---|
| 协程 coroutine | Python 事件循环在等待点切换 | 等待 B 结果、轮询进度、等待 A 结果 | 创建两个协程不代表两份 Python CPU 代码同时运行 |
| 线程 thread | 操作系统安排线程运行 | 可把阻塞的清理调用移到线程中等待 | 把函数移到线程不等于可以强制终止它 |
| 进程 process | 操作系统管理独立进程和地址空间 | 驱动、容器客户端、两张卡的服务 | 客户端进程结束不表示服务进程已经退出 |
| GPU 执行 | CUDA 运行时、驱动和硬件安排 | 服务向选定 GPU 提交模型计算 | Python `await` 不会直接停止正在执行的 CUDA 工作 |

两张卡可能真正同时计算，是因为各自服务进程向不同设备提交了工作。驱动协程的作用是同时保持这些请求在执行或等待，不必先等热卡完成，才有机会向冷卡发请求。

同一张卡是否并行，取决于服务后端。本研究的 `UnixStreamServer` 一次处理一个请求，没有使用线程版服务器。因此同一卡上的串行性来自服务实现。其他并行执行后端的能力不能直接套到这个串行服务上。

采样线程是另一层并发。[GpuSampler.start](../env-service-verification/scripts/warm_router_study/gpu_sampler.py#L30)启动一个后台线程，[_sample](../env-service-verification/scripts/warm_router_study/gpu_sampler.py#L60)调用有超时的 `nvidia-smi`，查询利用率、显存、频率与功耗。线程是为了不让监控的阻塞等待卡住主实验流程，不负责提交模型工作。采样器中的 `threading.Lock` 是线程互斥锁，只保护启动和停止时的共享状态，不管理 GPU 训练准入。采样超时记为缺失或失败，不补成 GPU 利用率零。

父监督进程还会同步等待后端子进程退出，采样线程仍能独立周期运行。后端子进程里才是组织 A/B 请求的事件循环。

所以这里有三个不同事实：请求协程可以同时等待，两张 GPU 可以各自执行，监控线程可以同时采样。第三个事实不能证明前两个，也不能把监控缺样当作训练失败。

CPU 教学脚本使用 `asyncio.sleep` 表示“外部工作尚未完成”。它能证明异步等待重叠，不能证明两个 CPU 计算核或两张真实 GPU 同时计算。

## 4. await 与任务怎样组织执行

`async def request(...)` 定义协程函数。直接写 `request(...)` 得到的是协程对象，它还没有自动成为一个独立执行的任务。

- `await request(...)`：当前协程执行并等待这次调用，结束后才继续下一行。如果内部等待还没完成，事件循环可以推进其他任务。如果内部正在做同步计算，则不会仅因为写了 `await` 就自动让出 CPU。
- `create_task(request(...))`：把协程登记成独立任务，事件循环可在后续调度它。返回的 Task 是任务句柄，可等待结果、查询完成状态或请求取消。
- `gather(task_a, task_b)`：同时等待所列任务并收集结果，返回顺序按传入顺序。它本身不是 GPU 并行开关。
- `TaskGroup`：把一组子任务放在同一个生命周期范围中。正常退出范围前等待它们完成。一个子任务出现普通异常时，会请求取消其他子任务并汇合异常。

这些语义可对照 [Python 3.12 asyncio 文档](https://docs.python.org/3.12/library/asyncio-task.html)。TaskGroup 从 Python 3.11 开始提供，本文 CPU 脚本要求 Python 3.11 或更高版本。

教学改写，展示 W4 的结构，不是逐字源码摘录：

```python
async with asyncio.TaskGroup() as tasks:
    blocker = tasks.create_task(request(hot, blocker_command))
    progress = await observe(blocker)
    decision = controller.decide(cost, snapshots_from(progress))
    target = tasks.create_task(request(decision.device_id, target_command))
# 正常离开这里，表示这两个 Python 请求任务都结束了。
```

`observe` 是父协程直接等待的调用，没有因为函数名独立就自动生成第三个 Task。等待进度时，B 的 Task 仍在运行。拿到进度后才创建 A 的 Task。两者可能有一段重叠时间，但 A 没有在输入和成本还未知时提前执行。

若连续写 `await request(B)` 再 `await request(A)`，就要等 B 返回后才提交 A。这会抹掉本来可利用的双卡重叠，即使放置算法认为冷卡更好也没有用。

`gather` 和 TaskGroup 的故障语义不同。普通 `gather` 遇到首个异常会向等待者传播，不自动保证所有兄弟任务已经停止。CPU 演示先用 TaskGroup 拥有这些任务，再用 gather 收集正常结果。清理路径使用 `gather(..., return_exceptions=True)` 时，异常是待检查的结果，不是“全部成功”的凭据。

TaskGroup 可能把多个子任务错误合成一个 `ExceptionGroup`，即一个包含多份异常的对象。CPU 故障演示只接受预期的 RuntimeError，其他异常继续抛出，避免把任何程序错误都误报为“取消验证通过”。

## 5. 类型化成本与三种放置策略

类型化输入指把不同含义的值放进具名对象，并验证字段。例如加载时间与执行时间都以秒计，但不能把两者交换后仍期待调度正确。

先读一小段真实定义，后面的容量例子就不用把对象当作黑盒。这是 [ModelCost 的类定义](../../../psrl/workers/env_service/warm_dispatch.py#L78)，只省略了英文类说明与空行，其余逐行摘录：

```python
@dataclass(frozen=True)
class ModelCost:
    model_id: str
    model_bytes: int
    rebuild: TimeEstimate | None
    execution: TimeEstimate | None
    workspace_bytes: int = 0
```

`class ModelCost` 定义一种保存模型成本的对象。`@dataclass` 是 Python 标准库的“数据类”装饰器：装饰器会在类定义完成时加工这个类。这里它根据具名字段自动生成初始化函数 `__init__`，让我们可以写 `ModelCost(model_id="A", model_bytes=100, ...)`，也自动生成便于打印和比较对象的代码。它负责整理数据，不是自动训练一个预测模型。[Python 数据类说明](https://docs.python.org/3.12/library/dataclasses.html)

`model_id: str` 表示模型身份预计为字符串，`model_bytes: int` 表示容量预计为整数。冒号后的内容叫类型注解，帮助读者、编辑器和静态检查工具理解预期输入。**Python 不会仅因为写了 `: int` 就在运行时拒绝字符串**，所以本项目还要显式校验。[Python 类型注解说明](https://docs.python.org/3.12/library/typing.html)

`TimeEstimate | None` 表示这个字段可以装一份时间估计，也可以装 `None`。`None` 是“这里没有已知值”，不等于耗时零秒。`rebuild=None` 表示不知道加载要多久，不能把冷卡完成成本算成“只剩执行时间”。最后一行的 `= 0` 是默认值：调用时不填工作区字段才取零。它不保证真实任务确实没有临时显存，需要调用者提供正确预算。

`frozen=True` 使对象创建后不能直接重新赋值字段。譬如 `cost.model_bytes = 200` 会报 `FrozenInstanceError`，避免计算完容量后另一段代码悄悄修改同一份成本。它不是任意嵌套容器的深度冻结。本地队列快照另把模型集合转成只读集合、队列转成元组，见 [QueueSnapshot.__post_init__](../../../psrl/workers/env_service/warm_dispatch.py#L128)。

实际输入验证在紧随其后的 [ModelCost.__post_init__](../../../psrl/workers/env_service/warm_dispatch.py#L88)。下面是该方法的完整摘录：

```python
def __post_init__(self) -> None:
    _identifier(self.model_id, "Model identity")
    _nonnegative_integer(self.model_bytes, "Model bytes")
    _nonnegative_integer(self.workspace_bytes, "Workspace bytes")
    for estimate in (self.rebuild, self.execution):
        if estimate is not None and not isinstance(estimate, TimeEstimate):
            raise ValueError("Model durations must be TimeEstimate values or unknown.")
```

自动生成的初始化函数填好字段后，会调用 `__post_init__`。`self` 就是正在创建的对象。第一行检查模型身份，接着两行要求模型和工作区字节是非负整数。两个检查函数的真实判断见 [身份检查](../../../psrl/workers/env_service/warm_dispatch.py#L37)和[容量检查](../../../psrl/workers/env_service/warm_dispatch.py#L32)。因此负数、字符串乃至 Python 中可参与整数运算的 `True` 都不能冒充容量。

`for` 逐一检查重建和执行时间，`isinstance` 判断对象是否为指定类型：已知时间必须是 `TimeEstimate`。未知则允许 `None`。不满足时 `raise ValueError` 抛出输入错误，使本次对象构造失败。`TimeEstimate` 自己还会检查有限非负数，以及点估计是否位于经验区间内，见 [TimeEstimate.__post_init__](../../../psrl/workers/env_service/warm_dispatch.py#L50)。时间字段类型正确，不代表预测一定准确。

把它串回教学输入，可以这样构造。这是便于阅读的示例，数字不是实测：

```python
load = TimeEstimate(5, DurationInterval(4, 6))
execute = TimeEstimate(4, DurationInterval(3, 5))
cost = ModelCost("A", 100, load, execute, workspace_bytes=10)
```

第一行保存加载点估计 5 秒、经验范围 4 至 6 秒。第二行保存执行成本。第三行把它们与模型 100 字节、临时空间 10 字节组合。`cost.model_bytes` 和 `cost.rebuild.point_s` 分别读取容量和加载点估计，字段名说明数值含义，比一串没有名字的数字更容易核对。

策略名称同样有两层约束。源码的 [DispatchStrategy](../../../psrl/workers/env_service/warm_dispatch.py#L12) 写作 `Literal["affinity", "always_copy", "predictive"]`，表示类型检查工具只应接受这三个精确字符串。`Literal` 的意思是“只允许列出的具体值”。真正执行时，[decide 的分支检查](../../../psrl/workers/env_service/warm_dispatch.py#L256)还会拒绝未知名称。只有注解而没有这段检查，Python 不会自动拦住拼错的字符串。

最后，[matrix 的证据记录](../env-service-verification/scripts/warm_router_study/matrix.py#L237)使用 `dataclasses.asdict(decision)`，将数据类及其嵌套数据类展开为按字段命名的字典。点、经验区间、容量和原因码因此能一起写入 JSON 证据，而不是只剩“选了冷卡”。`asdict` 做结构转换，不执行放置，也不验证预测准确性。研究决策中的字段可以序列化，含任意对象的其他数据类并不因此自动变成合法 JSON。

| 对象 | 装的是什么 | 为什么要分开 |
|---|---|---|
| `TimeEstimate` | 一个点估计 `point_s` 与一个经验范围 `interval` | 做点决策时仍保存不确定性，不能把范围抹成 `[point, point]` |
| `ModelCost` | 固定模型身份、模型字节、重建成本、执行成本、额外 workspace 字节 | 模型已驻留仍需要执行临时空间 |
| `QueueSnapshot` | 设备、可保留模型、可用预算、当前剩余和所有前序服务 | 不能只看当前命令，漏掉已经排队的命令 |
| `DeviceCostEstimate` | 每张卡的容量检查、等待、重建、执行、完成成本 | 能追查“为什么选它”，避免只有一个最终分数 |
| `SchedulingDecision` | 建议动作、目标卡、原因码、模式与所有候选估计 | 决策和实际准入、执行分开 |

workspace 是执行需要的额外临时空间，例如中间激活和算子工作区。本实现要求调用者提供预算，算法没有从模型参数量自动推导峰值。

容量的教学例子：模型 100 字节、workspace 10 字节。热卡模型已经存在，新命令还需 10 字节。冷卡模型不存在，还需 110 字节。若可用预算只有 109 字节，冷卡不合格，排队多长都不能让时间收益绕过这个条件。

真实实验的单位当然远大于 100 字节。这里的公式由 [required_gpu_bytes](../../../psrl/workers/env_service/warm_dispatch.py#L140) 和 [fits_capacity](../../../psrl/workers/env_service/warm_dispatch.py#L147)共同实现，三种策略都走相同检查。`available_gpu_bytes` 是预约预算扣掉已驻留及其他长期状态后剩余的容量，尚未扣本命令 workspace，不是随意读一次 `nvidia-smi` 的空闲显存。

快照中的模型必须保证在前序队列完成后仍然可用。若前面的命令会淘汰或修改模型，不能把“现在存在”直接当作“轮到 A 时还存在”。当前研究服务不隐式淘汰模型，固定参数版本也由实验约束，不能据此声称解决了任意 Agent 修改代码后的缓存有效性。

三种策略的区别如下：

这些是底层显式比较策略，用于研究对照。面向接入者的自动模块另外提供 `off`、`observe`、`auto` 三种开关模式，先判断预测是否有足够依据。`auto` 通过门控后才使用下面的 `predictive` 成本公式，开关默认 `off`。模式和策略的对应关系见[自动门控](command-predictability.md#6-offobserve-和-auto-怎样控制行为)。

| 策略 | 行为 | 共同约束 |
|---|---|---|
| `affinity` | 有可用热副本时保留亲和，没有时才选合格冷卡 | 容量必须合格，未知不能变成零 |
| `always_copy` | 有容量合格且完整成本已知的冷卡时复制，可能比亲和更慢 | 不为展示坏结果绕过容量或未知冷成本 |
| `predictive` | 只有估计的复制优势超过门槛才复制，否则保留亲和 | 当前默认是点估计加 2 秒门槛 |

`point` 模式比较 `cold.point + margin < hot.point`。`interval` 模式要求 `cold.upper + margin < hot.lower`。第二种条件更难满足，但这些上下界仍是经验预测，没有变成数学最坏情况保证。

例如热卡估计 16 秒、经验范围 5 至 24 秒，冷卡估计 8 秒、范围 6 至 10 秒。点模式有 `8 + 2 < 16`，可以复制。区间模式的 `10 + 2 < 5` 不成立，保留亲和。两种模式会有不同动作，但它们保存的原始范围相同。

未知热等待时，predictive 回退到热卡串行队列，输出 `wait` 和热卡身份，同时保持 `completion=None`。这表示“执行安排沿用亲和”，不是“已经知道完成时间”。always_copy 仍可以选择成本已知的合法冷卡，因为该对照的定义就是不要求证明复制收益。未知冷加载或未知冷执行不会被当成可预测的冷副本。

默认所有设备共享成本画像，也可用 `device_costs` 提供每张卡的成本。覆盖项必须保持相同模型身份、模型字节与 workspace 要求。这样可以表达某卡加载更慢或执行更慢，同时禁止通过偷偷缩小该卡工作量让它看起来更好。

## 6. 进度怎样变成剩余时间预测

剩余时间必须建立在命令确实知道要执行多少步的前提上。矩阵实验固定模型、两条公开训练样本和更新步数。它没有推断一个任意新 Python 程序会何时退出，也没有预测 Agent 还会修改几轮代码。

服务每 16 步报告一块。假设最近三块分别用 0.400、0.416、0.432 秒，每块 16 步，则每步速度分别是 0.025、0.026、0.027 秒，中位数为 0.026 秒。还剩 1984 步时，点估计约为 `1984 × 0.026 = 51.584` 秒。

中位数是排序后的中间值，比平均值更不容易被单个很慢的热身块拉偏。预测器最多使用近期八块。在第 64 步首次到达时，四块中丢掉第一块热身，实际通常只有三个块可用，不是已经观察到八块。

如果上次报告已经过去 0.1 秒，还要考虑这段时间已经在继续做下一块。代码只从当前尚未报告块的预计工作中扣去这段时间，后面的完整块仍需执行。不能一看到报告变旧就把整个剩余工作扣成零。

报告过期也按“下一整块”判断。每 16 步才报告时，不能用单步 0.026 秒作为过期阈值，否则正常的块间计算就会被误判为失联。超过带余量的下一块上界而仍没有新报告时，预测状态是 `overrun`，时间保持未知。

块的真实墙钟时间包含 CPU、Python 管理和上一次进度发布成本。墙钟时间指现实世界经过的时间，与只统计某个 GPU 内核的执行时间不同。[服务循环](../env-service-verification/scripts/warm_router_study/server.py#L129)在发布前记录边界，下一块从同一边界开始，因此包含这次发布开销。最后一块之后的最终发布还要单独通过完整计算墙钟与 `progress_write_s` 核对。

原子发布的过程是“写完整临时 JSON → 替换正式路径”。读者要么看到旧完整文件，要么看到新完整文件，避免读到只写了一半的 JSON。时间戳仍可能是旧的，所以原子性不等于新鲜度。它也不是断电持久化保证，不是跨机器共同快照。[Python 文件替换说明](https://docs.python.org/3.12/library/os.html#os.replace)给出同一文件系统上的原子替换语义。

每个服务是该进度文件的单一写者，临时文件与正式文件位于同一目录。真实服务使用节点本地短路径 `/run` 挂载，不能把共享目录上每步发布的大开销当成 GPU 核心执行。

经验范围使用分位数加余量。分位数描述已观察样本的分布位置，不会自动提供未来覆盖概率。`statistical_guarantee=False` 明确记录这一点。系统性地把等待预测高十倍，即使同时把上下范围也放大十倍，仍可能错误复制，不能靠“有一个区间”自动解决。

因此可复用模块在这层估计之前增加了[当前稳定性检查与独立完成校准](command-predictability.md#3-怎样从进度估计剩余时间)。至少三条同类命令真正完成，近期误差可接受，且当前进度稳定、新鲜，才获得自动资格。历史按版本、工作量和执行条件分组，设近期窗口与有效期。这里的局部分块范围继续承担表达不确定性的职责，完成校准负责发现“局部平稳但最终仍猜错”的命令。GPU 矩阵冻结时使用原研究预测器，新增门控没有回填到历史结果中。

## 7. 失败取消与资源生命周期

需要分别管理两类生命周期。第一类是 Python 请求任务：创建、等待、返回或取消。第二类是外部资源：容器进程、GPU 模型、workspace、服务和租约。取消第一类，不会自动完成第二类的清理。

`task.cancel()` 是向任务请求取消，通常在下一个可以响应取消的等待点抛出 `CancelledError`。它不是向远程 GPU 发出“撤销计算”的原子指令。客户端停止等 socket，服务仍可能正在训练。

因此真实提供者在 `finally` 中执行外层清理：取消并有限等待客户端，停止本次环境，释放本次租约，再核实所属容器为空、GPU 计算进程消失和显存释放。账本变成零只是记账事实，不能单独证明确实释放了设备。

`finally` 表示无论正常返回还是异常都进入这段收尾代码。但它不是不可打断的承诺。进程被强制终止、清理再次收到取消或外部命令不返回时，仍可能收尾失败，所以要记录物理检查结果，不能只看 Python 已经跳出了 `try`。

[runtime_guard.await_cleanup](../env-service-verification/scripts/warm_router_study/runtime_guard.py#L13)用清理截止时间限制等待，`cancel_and_join` 汇总还没完成的客户端。等待结束时若还有任务，不会宣称 GPU 已释放。把阻塞调用放到线程能避免堵住事件循环，但取消等待该线程的协程，不会强制停止线程里的系统调用。

工作预算和清理预算也要分开。工作预算防止再启动放不下的新案例，清理预算给已经持有的资源留出回收窗口。父进程还应有整个实验的总截止时间。达到预算就保留不完整状态，不为了补齐表格而自动增加时间。

当前 [matrix_suite.run](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L161)通过 [supervise_backend](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L138)监督一个后端。后端以新进程会话启动，便于用进程组身份管理它及其后代进程。父进程等待上限由后端工作预算、清理预留和总截止时间共同决定。超时、父级 Ctrl+C 产生的 `KeyboardInterrupt`、命令行收到 SIGTERM 后转成的 `SystemExit`、其他等待异常和后端非零退出，都进入 [stop_backend](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L103)。

回收先向本次进程组发送 SIGINT，最多等待 10 秒。SIGINT 是允许程序处理的中断信号，给子进程执行 finally 的机会。若进程组仍存在，即使组长已经退出，也会向组发送 SIGKILL 并最多等待 5 秒。SIGKILL 不能被捕获，强杀后的进程无法执行 finally，所以父进程还按本次 actor 标签查找并回收所属容器。这里的标签是实验归属标记，值为本次唯一名字，不能扩大成删除节点所有容器。回收错误另行保留，不覆盖最初的失败。

这个监督分三层：单条请求超时，子进程的工作与清理超时，父进程看门狗。它们限制不同的等待，不能互相替代。父进程的兜底回收记录 `physical_release_confirmed=False`，不会因为尝试删除容器就自行认证 GPU 已释放。没有成功的物理检查就停止后续后端，最终保留未完成或清理未确认状态。终止、定向回收和证据归档也有耗时，报告应保留实际总墙钟，不能只展示名义工作预算。

父级 Ctrl+C 的独立子进程回收缺口已在主目录修复，命令行还在 [main](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L296)安装 SIGTERM 处理函数。SIGTERM 是通常用于请求进程终止的信号，这里把它转成退出码 143 的 `SystemExit`，先沿同一条监督收尾路径退出，再恢复先前的信号处理方式。26 项 CPU 回归验证了监督与清理分支，包含真实父进程 SIGTERM，Ruff 与源码文字检查通过。这项修复发生在 GPU 矩阵运行结束后，未重跑 GPU，原实验冻结源码仍保留修复前版本。不可捕获的 SIGKILL、宿主机故障等情形仍没有 Python 清理保证。

模型缓存与 workspace 的归还时刻不同。命令完成后，workspace 可以释放，但常驻模型仍占显存。使用 episode 租约的服务直到环境退出才整体归还模型预算。不能把一个发送请求的短命令返回写成“GPU 已归还，可以交给别的任务”。

## 8. 矩阵实验究竟验证什么

矩阵把目标长度、阻塞长度和策略分开，避免只在一个人为有利的场景判断策略。下表是预先固定的核心工作量，不是性能结果表。四种工作量的旧 runner 和三种当前策略共 48 个核心区块已全部完成，每组比较各有三次时间块配对。故障检查另完成高估两次、过期两次、容量不足一次，第二轮容量与第三轮三个故障区块因预注册预算门槛未启动，没有补跑。完成证据不等于整个 57 区块计划通过，详见[正式报告](../experiments/gpu-benchmarks/warm-placement-matrix.md)。

| 核心案例 | 目标 A 更新步数 | 阻塞 B 更新步数 | 要观察什么 |
|---|---:|---:|---|
| default | 2 | 128 | 原本很短的目标通常不值得重新加载 |
| short | 1024 | 128 | A 虽长，但热卡很快可用时仍可能该等待 |
| boundary | 1024 | 512 | 等待与重建接近时，2 秒门槛与画像误差怎样影响选择 |
| long | 1024 | 2048 | 较长等待是否足以摊薄冷卡重建 |

本工作量仅更新模型归一化层的参数，其余主要权重冻结。矩阵通过固定更新步数控制任务内计算长度，研究的是等待、加载与放置成本；完整官方题库和自然多轮 rollout 仍按各自协议评价。

沿 `long` 的真实代码路径看一遍。步数 1024/2048、64 步到达和每 16 步报告来自 `matrix.py`。下面的速度与秒数是教学假设，不是矩阵实测：

1. B 在热卡开始，执行满 64 步后报告，A 才到来。假设 B 每步 0.025 秒，此刻已用约 1.6 秒，剩余 1984 步约 49.6 秒。
2. 假设 A 每步 0.035 秒，1024 步约 35.84 秒。另一卡加载 A 需要 12 秒，容量检查也允许放下模型与 workspace。
3. 热卡 A 完成成本约 `49.6 + 35.84 = 85.44` 秒。冷卡约 `12 + 35.84 = 47.84` 秒，另加 2 秒门槛仍更早，所以 predictive 建议冷卡。
4. 驱动创建 A 的请求 Task，B 原来的 Task 继续等待。热服务继续做 B，冷服务先加载 A 再训练，父协程等两者完成。
5. 若忽略所有通信管理开销，从 B 开始到整块结束，亲和约为 `51.2 + 35.84 = 87.04` 秒。复制时 A 约在 `1.6 + 47.84 = 49.44` 秒完成，B 在 51.2 秒完成，因此整块约 51.2 秒。目标延迟和两条命令整体完成时间是不同指标。
6. 两条返回后还要清空冷卡新副本。若它额外花 1 秒，就应同时报告 52.2 秒的闭环时间，不能只保留 51.2 秒。热卡 A/B 模型仍受 episode 预算管理，到环境结束再整体释放。

这个例子解释收益从哪里来，也显示什么会破坏它。若真实 B 只剩 2 秒，错误预测为 49.6 秒，复制就可能比保留亲和更慢。若冷卡容量不够，即使时间计算漂亮也不能执行复制。

三个故障条件分别注入等待高估、报告过期和冷卡预算不足。高估检查预测错误会不会导致不划算的复制。过期检查未知是否保留。预算不足检查容量约束是否能阻止一次时间上看似有利的放置。

容量用服务本身的 `set_budget`、`ping` 和 `admit` 读回检查，而不只修改调度器中的假数字。`set_budget` 不能超过原始租约预算，模型加载也执行同一服务预算检查。它仍是逻辑预算实验，没有把物理 GPU 显存真的切成该容量，也没有通过制造真实 OOM，也就是显存不足时的分配失败，来验证硬隔离。

三种策略使用同一程序、样本、更新次数、模型参数恢复和计时边界。主工作块时间从提交 B 之前，到两个请求都返回。冷卡随后清空的耗时另外计入闭环时间。容器准备和画像不能混入某一组而从另一组删除。

这里还有一个容易遗漏的计时前提：[target_latency_s](../env-service-verification/scripts/warm_router_study/matrix.py#L261)用服务的 `finished_monotonic` 减驱动的 `arrival`，报告年龄也跨进程相减。单调时钟是用于测量经过多久、不会随系统日期调整而倒退的时钟，[Python 说明](https://docs.python.org/3.12/library/time.html#time.monotonic)强调其参考起点本身没有业务含义。本次驱动和容器位于同一主机、共享内核单调时钟，所以可以这样比较。跨机器的单调时间戳不能直接相减。推广到多机时，应把端到端计时放在同一进程的请求发出与返回边界，或设计经过校时并记录误差的共同时间基准，不能照抄这个字段公式。服务内部 `compute_s` 这种同一时钟上的差值与此不同。

固定旧提交对照与三种当前策略也不能混为同一问题。旧提交对照回答“这套执行安排相对固定历史系统如何”。当前 affinity 与 predictive 的对照回答“预测选择相对简单亲和有没有增量”。always_copy 则帮助观察复制是否总划算，不能取代强简单基线。

策略和画像共享进程不等于共享缓存初态。驱动按案例清空冷卡，并轮换逻辑热卡与冷卡的物理设备。纯策略模块冻结后通过私有包加载，使用相同源文件内容。包是一组可导入的 Python 模块，这里的私有包是实验进程中的临时模块命名空间，只装载三个纯算法文件。这样不会为了导入当前调度器而把旧 runner 所依赖的整个 `psrl` 包替换掉。

本文不从计划表推断真实性能。具体已完成单元、失败、预算退出、输出一致性和物理清理，以对应证据审计为准。没有运行时 GPU 利用率数据时，也不能只凭耗时改善声称已经证明所有设备空泡最少。

## 9. 运行 CPU 协程教学实验

配套脚本是[asyncio_warm_demo.py](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py)，只依赖 Python 3.11 或以上的标准库。它通过私有包加载真实的三个纯算法模块，不启动 Ray、Docker、PyTorch、GPU 服务或真实 GPU 分配。

从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py
```

脚本包含两个部分。正常部分先运行热卡阻塞命令，等它报告进度，再调用真实预测器与调度器，提交目标命令并等待两个结果。故障部分让右边设备的命令主动抛出异常，观察 TaskGroup 取消左边任务，并执行两层 finally。

| 教学实现 | 真实实现 | 对应与差异 |
|---|---|---|
| `TeachingDevice.request` 把 Command 放入 `asyncio.Queue`，等待 Future | 容器客户端发送 socket 请求并等服务返回 | 都表示一个已经提交的请求。真实服务没有使用这个 Python Queue |
| 每设备只有一个 `serve` 消费者 | `UnixStreamServer` 串行处理连接 | 都让同设备命令串行，但实现方式不同 |
| `asyncio.sleep` 模拟工作 | `Service.execute` 提交并同步真实 GPU 训练 | 只有真实服务能证明 GPU 工作，sleep 只演示等待重叠 |
| `workspace_bytes=0` 和清空模型字典 | 真实显存释放、进程退出及容器/设备检查 | 教学断言只检查模拟状态，不能取代物理检查 |
| 临时目录原子 JSON 报告 | 本机 `/run` 进度文件 | 文件替换机制相同，记录的数据与频率不同 |

Future 是“以后会产生一个结果”的对象。请求把它交给服务端消费者，自己暂停等它完成。服务端计算成功后调用 `set_result`，请求协程才继续。故障时取消 Future 不会伪造一个成功结果。

教学 Queue 是先进先出队列，`get` 取出一条，处理结束后 `task_done` 记录该条已经处理。它服务于同一事件循环，不是线程安全或跨机器工作队列。可对照 [Python asyncio.Queue 文档](https://docs.python.org/3.12/library/asyncio-queue.html)。本脚本没有额外的 Lock，因为每个设备已经只有一个消费者，不能把并不存在的锁说成真实系统互斥机制。

本次实际运行观察到：

- 正常案例选择 `rebuild` 和逻辑冷卡，两个模拟命令最大重叠数为 2。
- 两条命令都返回预定更新步数，正常结束时模型字典为空、workspace 为零、两个设备都标记关闭。
- 故障案例观察到预期异常，左侧长命令收到取消，最终仍完成相同的模拟清理检查。

秒数会随操作系统调度略变，不把 sleep 的精确时长作为性能断言。如果当次报告已过期，真实控制器也可能保守回退到热卡，脚本保留该决策。输出的 `scope` 明确标记 CPU 教学模拟，不是 GPU 对照。

## 10. 跟着图读源码

按实际输入到收尾的路径读，不必从整个项目的所有 Actor 开始。

`study` 为什么能直接调用真实容器，却没有自己创建容器？入口 [matrix.main](../env-service-verification/scripts/warm_router_study/matrix.py#L289)先选择当前或旧版提供者，把一个 `callback` 函数交给它。提供者准备好环境后，才调用该函数。`callback` 再进入 `study(context, args)`。这里的 callback 是“交给另一段代码、等条件准备好后由它调用的函数”，没有自动创建线程。

提供者通过普通字典 `context` 交付现成能力。[真实 context 构造](../env-service-verification/scripts/warm_router_study/experiment.py#L346)较长，下面只摘录与此处相关的原字段，省略其他字段：

```python
context = dict(
    request=request,
    record=record,
    save=save,
    remaining=lambda: max(0.0, deadline - time.monotonic()),
)
```

`request=request` 左边是字典里的名字，右边是已经定义好的真实请求函数。没有括号，所以这里交的是函数本身，还没有发请求。`study` 通过 [context["request"]](../env-service-verification/scripts/warm_router_study/matrix.py#L112)取出它，再用 `await request(index, payload)`向指定服务提交操作。`record` 是共享的结果字典，`save` 是把当前记录保存到文件的函数。`lambda` 在这里定义一个很短的无名函数，`remaining()` 每次调用都重新计算截止时间还剩多少秒，而非传入创建 context 时已经过期的固定数值。旧 runner 也交付[同样的函数入口](../env-service-verification/scripts/warm_router_study/baseline.py#L418)，所以同一 `study` 可以使用两条真实执行链。容器创建和收尾仍归各自提供者负责。

| 步骤 | 图中编号 | 文件函数 | 输入输出或状态变化 |
|---|---|---|---|
| 0 | W0/T | [matrix_suite.freeze](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L37)、[run](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L161) | 固定源文件、镜像和单元计划，启动监控线程与后端子进程 |
| 1 | W1 | [matrix.main](../env-service-verification/scripts/warm_router_study/matrix.py#L275) | 选择真实提供者，交出 callback。此时还没进入 study |
| 2 | W1 | [experiment 的 context 与调用](../env-service-verification/scripts/warm_router_study/experiment.py#L346)、[baseline 对应位置](../env-service-verification/scripts/warm_router_study/baseline.py#L418) | 环境已就绪，把真实 request、记录保存和剩余预算函数交给 study |
| 3 | W1 | [matrix.study](../env-service-verification/scripts/warm_router_study/matrix.py#L108)、[load_policy](../env-service-verification/scripts/warm_router_study/matrix.py#L70) | 加载冻结纯模块，按 plan_cells 的案例读取服务预算并做固定画像，避免污染旧 runner 包身份 |
| 4，并发开始 | W4 | [matrix TaskGroup](../env-service-verification/scripts/warm_router_study/matrix.py#L185) | TaskGroup 启动 B 请求。父协程继续等待进度，B 请求仍在等待服务执行 |
| 5，进度回调方向 | W5 → W2 | [server.progress](../env-service-verification/scripts/warm_router_study/server.py#L23)、[matrix.observe](../env-service-verification/scripts/warm_router_study/matrix.py#L91) | 服务报告已完成步数，驱动确认命令身份和到达门槛 |
| 6 | W2 | [predict_remaining](../../../psrl/workers/env_service/step_predictor.py#L33)、[estimate_from_forecast](../../../psrl/workers/env_service/warm_dispatch.py#L65) | 已报告块和报告年龄变成点估计及经验范围，未知保持 None |
| 7 | W3 | [WarmPlacementController.decide](../../../psrl/workers/env_service/warm_dispatch.py#L237) | 先检查容量，再按策略比较完整完成成本，返回建议，不改账本 |
| 8，并发分支 | W4 → W5 | [experiment.request](../env-service-verification/scripts/warm_router_study/experiment.py#L129)、[client](../env-service-verification/scripts/warm_router_study/client.py) | 向选定环境发 A。若热卡则 socket 服务串行等待，若冷卡则另一进程重建执行 |
| 9 | W6 | [matrix 正常汇合与清空](../env-service-verification/scripts/warm_router_study/matrix.py#L253) | 正常离开 TaskGroup 后取两条结果，记录工作块与冷卡清空时间 |
| 10，任何退出路径 | W7 | [experiment.finally](../env-service-verification/scripts/warm_router_study/experiment.py#L366)、[cancel_and_join](../env-service-verification/scripts/warm_router_study/runtime_guard.py#L34) | 请求取消、停止所属外部资源、核实释放，清理失败保留错误 |
| 11，父级兜底 | W7 → W0 | [supervise_backend](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L138)、[stop_backend](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L103)、[SIGTERM 处理](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L311)、[cleanup_passed](../env-service-verification/scripts/warm_router_study/matrix_suite.py#L62) | 处理超时、Ctrl+C、协作 SIGTERM、其他等待异常与非零退出，定向停止所属进程组及容器，检查释放后才允许继续。失败保留 incomplete |

CPU 脚本的阅读路径更短：[placement_case](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py#L170) → [request](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py#L84) → [serve](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py#L137) → [execute](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py#L90) → [atomic_progress](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py#L41) → 调度器 → `finally`。再读 [cancellation_case](../env-service-verification/scripts/warm_router_study/asyncio_warm_demo.py#L218)，比较出错后的分支。

其中 serve、request 和进度观察会交错执行，表格的编号表示数据流，不表示全系统是一条顺序函数调用链。看完后应能区分：谁在等待、谁在计算、谁持有模型、谁只是提议，以及谁必须最后证明确实释放了资源。
