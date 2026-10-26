# 真实 GPU 工作负载的同任务多协程对照

> **用户意图纠正（2026-10-26）**：用户明确说明，“一个任务多开几个协程”指同一道题多次独立 rollout 尝试，而非一次尝试的容器里提交多条 GPU 命令。本报告保留的单容器四／十六命令实验是此前对需求的误解所形成的组件对照，不代表原始多 rollout 验证已经完成。OFF 允许同容器多命令重叠的事实仍有效，但不能将其与同题多容器轨迹的结果混用。同容器并发提案是可选扩展，不是完成原始目标的前提。

> **实验层次与归因澄清**：本文使用了 PSRL 的 Science runner、EnvRouter 和容器执行实现，绕过的是外层大模型强化学习训练器。这里“十步”指容器内 CIFAR 分类模型的十次更新，不是 Qwen 的十个强化学习步骤。低压力下四条 GPU 命令串行，直接来自现有同容器锁及同租约协议；实验选择一个容器内四条命令，使这一限制显现。该结果不能代表多容器调度或完整 PSRL 吞吐，详见[第 13–15 节](#13-实验层次与十步的含义)。

> **最新容量压力对照（四命令，各十步）**：固定 OFF 与修复后 ON 在低压力下都完成 4/4，命令阶段分别为 55.00 秒和 156.93 秒，ON 耗时是 OFF 的 2.85 倍。在明确注入占位、只剩约 1.50 GiB 可用显存时，ON 完成 4/4，OFF 完成 0/4，后者出现三条明确显存不足错误及一条 cuBLAS 分配失败。本次证明了这个压力点的完成率收益，也量化了低压力代价，详见第 10–12 节。高压力不是原生任务自身造成的，不能当作完整 MLGym 成绩或多卡调度结论。

> **核心结论（2026-10-22）**：本次不注入显存占位，使用 MLGym 的真实 CIFAR-10 训练代码和真实图片，在一个 Harbor 任务容器内提交 16 个训练命令。OFF 用 143.36 秒全部完成，观测到同卡 16 个 CUDA 进程，峰值显存约 10.78 GiB，没有 OOM。ON 到 212.72 秒只完成 2 个，因此按“结论明确即停止”主动结束，完整 ON 十六命令结果不存在，也不计算完整速度比。本次没有测出收益。
>
> 原始小模型没有形成高显存压力。当前 ON 的同容器锁还会把 Python 导入和数据准备串行化，导致实际 GPU 计算前的大量等待。实验资源已核验释放。

> **后续修复**：已定位依赖导入时的重复文件扫描，并增加显式 CPU 准备 / GPU 执行接口。原始实验保留，修复机制与回归见[第 7 节](#7-启动耗时定位与修复机制)及后续章节。修复没有删除同容器 GPU 回收保护。四个命令各十步的 ON 内部回归全部完成，命令阶段从 190.48 秒降至 158.50 秒，缩短 16.79%。这不是与历史 OFF 的速度对比。
>
> 本文承接[此前的真实训练报告](science-single-gpu-training-control.md)。此前额外申请 72 GiB 的压力实验、原生 CPU LAPS 任务的 PSRL 训练、本次真实 GPU 命令实验，三者分别统计。本次不运行语言模型代理或 PSRL 强化学习训练器，不宣称完整 Science 题目得分。

## 目录

- [1. 执行链路与源码对应](#1-执行链路与源码对应)
- [2. 真实工作负载与公平对照](#2-真实工作负载与公平对照)
- [3. Results：实测结果](#3-results实测结果)
- [4. 为什么同卡并发不一定发生 OOM](#4-为什么同卡并发不一定发生-oom)
- [5. 跟着图读源码与复现](#5-跟着图读源码与复现)
- [6. 验证边界与执行经验](#6-验证边界与执行经验)
- [7. 启动耗时定位与修复机制](#7-启动耗时定位与修复机制)
- [8. Results：修复回归](#8-results修复回归)
- [9. 使用分阶段接口与隔离依赖](#9-使用分阶段接口与隔离依赖)
- [10. 固定 OFF 与修复后 ON 的容量压力对照](#10-固定-off-与修复后-on-的容量压力对照)
- [11. Results：高低显存压力实测](#11-results高低显存压力实测)
- [12. 容量压力实验复现与边界](#12-容量压力实验复现与边界)
- [13. 实验层次与十步的含义](#13-实验层次与十步的含义)
- [14. 同容器限制与问题归因](#14-同容器限制与问题归因)
- [15. CPU 教学复现与后续验收边界](#15-cpu-教学复现与后续验收边界)

## 1. 执行链路与源码对应

```text
[1] 原始 MLGym Net + train_model、640 张真实 CIFAR-10 图片
                         |
[2] Science runner 启动一个 Harbor 任务容器，只暴露同一张 GPU
                         |
[3] 固定脚本代理通过 asyncio.gather 提交 16 个协程
                  /                        \
[4a] OFF：普通 exec                [4b] ON：exec_gpu
静态 GPU compose 映射              同容器异步锁 → Router 授权
16 个独立 Python 进程               当前实现一次执行一个命令
                  \                        /
[5] Python 导入依赖 → 原始 Trainer → 前向、反向、十步更新
                         |
[6] 保存命令 stdout、梯度、显存、GPU UUID 与宿主机遥测
                         |
[7] 验证十步完成 → 清理本次容器、租约、Ray → 核验物理释放
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责、实现与真实源码 |
|---|---|
| 1 | 冻结的 [Net](env-service-verification/results/science-natural-gpu-20261022/workload/baseline.py#L44) 是原始两层卷积分类模型，[train_model](env-service-verification/results/science-natural-gpu-20261022/workload/baseline.py#L77) 创建 Hugging Face Trainer 并执行训练 |
| 2 | [run](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_ab.py#L28) 生成任务、挂载只读工作负载，再调用真实 Science runner |
| 3 | [NaturalGpuAgent.run](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_agent.py#L11) 同时提交协程，分别保存输出 |
| 4a | [冻结 OFF runner](env-service-verification/results/science-natural-gpu-20261022/source/off-runner.py#L187) 使用静态 compose override，没有整卡锁 |
| 4b | [冻结 ON exec_gpu](env-service-verification/results/science-natural-gpu-20261022/source/on-docker.py#L67) 先取得同容器锁，再申请 Router 授权，命令完成后归还 |
| 5 | [训练包装器](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_train.py#L23) 导入原始文件，[第 51 行](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_train.py#L51) 调用原始训练函数 |
| 6 | [代理保存输出](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_agent.py#L26) 与 [NVML 采样](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_ab.py#L15) 相互核验。NVML 是 NVIDIA 提供的设备监控接口 |
| 7 | [有界执行和收尾](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_ab.py#L153) 保留 runner 返回结果并释放隔离 Ray |

这些代码如果有理解上的问题，可以随时追问。图中的 16 个协程是并发调用方，不是 16 张 GPU。每条命令启动独立 Python 进程，GPU 是否真的重叠，必须结合设备进程遥测确认。

## 2. 真实工作负载与公平对照

任务库中的原始 LAPS 修复题主要运行 CPU。`laps-accel-cuda` 虽声明需要 GPU，但题目要求代理先完成 CUDA 移植，现有 `solve.sh` 仍是 CPU oracle，不能拿它冒充已有 GPU 求解器。另一个 Llama 剪枝任务需要当前节点未部署的专用镜像、模型和校准资产。本次因此采用已有 MLGym 图像分类基线，保留其实际计算，不添加占位张量或等待持有显存的进程。

MLGym 不是此前 LAPS 训练数据中的题目。本次将其真实训练入口封装为 Science/Harbor 诊断任务，自定义 verifier 只检查训练命令是否完成。不能把这个封装写成“原始 LAPS GPU 题已经完整跑通”。

| 项目 | 本次配置 |
|---|---|
| OFF | `0eeead861ffafed6bf31c962d259fd847775dcf9`，保留历史静态设备映射与 episode 信号量 |
| ON | `37a06ea8d32b05bbbf385ad40cb404b30ba01680`，不混用当前工作区未提交的性能修改 |
| 工作负载来源 | MLGym `9d40c1b5035202018cd7091fb4e83a9c68b377c0`，原始 baseline 文件没有本地修改 |
| 数据 | `uoft-cs/cifar10`，固定 revision `0b2714987fa478483af9968de7c934580d0bb9a2` 的前 640 张真实训练图片 |
| 模型与 batch | 原始 Net，RGB 32×32 输入，batch size 64，原始 seed、warmup 和优化器配置 |
| 有界改动 | 每命令十步更新，停用评估、checkpoint 保存和外部报告，将输出目录按命令编号隔离 |
| 并发 | 一个容器，16 个协程，各自运行独立进程，所有进程仅能访问同一物理 GPU |
| 主容器资源 | 16 CPU、64 GiB 主存，ON 命令申报 4 GiB，GPU 预算 80 GiB |
| 预算 | 单命令执行 180 秒，全部命令 600 秒，runner 外层 780 秒，零自动重试 |
| 审计 | 代理在宿主机保存 stdout，原始 baseline、数据和训练包装器只读挂载 |

原始 500 个 warmup 步保留，十步仍在学习率预热阶段，不据此讨论收敛或分类准确率。原始训练函数确实执行前向计算、损失计算、反向传播和优化器步骤，完全不同于只申请一块显存后休眠。

OFF 的[运行补丁](env-service-verification/results/science-natural-gpu-20261022/off.patch) 继续保留历史实验中必要的设备配置传递、禁用全局 prune、Ray 和 HTTP 修复。其中本次实际使用 runner 的两处改动，未加入整卡锁，也未更改 GPU 调度语义。ON 源码无补丁。实际训练使用共享环境中的 Trainer，因此本文只声称原始任务模型和训练入口保持，不能声称重建了 MLGym 原始发布环境。

## 3. Results：实测结果

| 指标 | OFF | ON |
|---|---|---|
| 提交命令数 | 16 | 16 |
| 完成十步训练 | 16 | 2，另 14 个未完成 |
| 命令批次耗时 | 143.36 秒，完整完成 | 决定停止时已运行 212.72 秒，未完成同等工作量 |
| 观测 CUDA 进程峰值 | 16 | 1 |
| 观测整卡显存峰值 | 11,038.50 MiB，约 10.78 GiB | 693.06 MiB，约 0.68 GiB，仅覆盖停止前区间 |
| 观测 GPU 利用率峰值 | 14% | 1% |
| 已完成命令 OOM | 0 | 0，不把未完成命令算成成功 |
| 单命令导入与数据准备 | 131.65 至 131.81 秒，16 个进程重叠 | 两个完成命令分别为 99.54、49.82 秒，串行 |
| 单命令 Trainer 初始化至训练结束 | 7.20 至 7.33 秒 | 3.88、3.80 秒 |
| 遥测缺失 | 0 | 4 次，保留异常，不补零 |

OFF 的工作区间从固定代理提交任务前开始，到全部命令输出收集后结束。ON 使用相同起点，在运行超过 OFF 完整耗时且仍有工作未完成时停止。ON 的最终代理记录为 212.75 秒，与发出停止信号时的 212.72 秒相差约 0.03 秒。容器启动、Ray 初始化和最终清理不包含在这个命令批次口径内。

这是有意提前停止的负面结果，不是把 ON 的 2 个成功任务与 OFF 的 16 个任务当作完整等量比较。只要完成同等工作还需要非负时间，ON 在停止时已经不可能比 OFF 更早完成，因此足以否定“这组配置提速”。它不能提供完整 ON 吞吐、完整速度比或长期稳定性结论。

停止前记录见[early-stop.json](env-service-verification/results/science-natural-gpu-20261022/early-stop.json)。本次向已核对 PID 和创建状态的 ON driver 发送 SIGINT，最终 `CancelledError()` 是主动收尾结果，不是新的训练崩溃。第三个命令被中断，其余等待者被取消，缺失的最终训练记录没有补造。

单进程预检已通过十步真实更新，损失和梯度有限。PyTorch 峰值实际分配约 142.68 MiB，缓存约 166 MiB。OFF 的 16 个命令与 ON 完成的 2 个命令，平均训练损失均为 `2.303740954399109`，参数和校验值均为 `-4.757167816162109`。参数和只是粗粒度一致性检查，不等于逐字节验证了整个模型。

本次真实计算的并发使单命令训练阶段变慢，但同时推进 16 个任务。ON 缩短了已完成单命令的训练阶段，却把启动与训练全部串行化。这解释了为什么“单命令更快、显存更低”没有变成“整批任务更快”。导入和数据准备计时、线程等待状态支持存在共享存储启动开销，但本次没有做文件系统归因实验，不能把全部导入时间都归因于某一个存储组件。

原始证据位于[本次结果目录](env-service-verification/results/science-natural-gpu-20261022)，包含固定设计、源码快照、两组提交及补丁、数据摘要、预检、逐命令 stdout、容器配置和 GPU 遥测。单次采样只能给出观测峰值，不能保证捕获比采样间隔更短的瞬时峰值。

汇总见[summary.json](env-service-verification/results/science-natural-gpu-20261022/summary.json)，物理释放核验见[cleanup.json](env-service-verification/results/science-natural-gpu-20261022/cleanup.json)：本次容器、网络、driver 和隔离 Ray 进程均未残留，目标 GPU 无计算进程，NVML v2 used 回到 0.19 MiB。没有清理节点上的其他任务容器。

## 4. 为什么同卡并发不一定发生 OOM

OOM 是 Out Of Memory，即程序申请的显存超过可用容量。协程只是组织异步请求的方式，本身不决定显存占用。真正消耗显存的是命令进程建立的 CUDA 上下文、模型参数、优化器状态、前向激活和计算库工作区。

本次的具体例子是：16 个进程确实重叠，但整卡采样峰值约 11,038.5 MiB，相对于 97,871 MiB 设备总量，只有约 11.3%。因此“全部落在一张卡”已经验证，“因同卡并发而 OOM”没有发生。单进程的 143 MiB 是 PyTorch 张量峰值，不能简单乘 16 后当作整卡消耗，因为驱动上下文和计算库也占显存。

教学假设：若一个真实任务原本需要 30 GiB 工作显存，在约 95 GiB 空闲设备上四个进程同时进入峰值，总需求约 120 GiB，才可能触发 OOM。这里的 30 GiB 和 120 GiB 只是解释容量关系，不是本次数据，也不是建议人为扩大张量来制造结果。真实峰值是否重叠、框架缓存和外部进程占用，仍会影响结果。

ON 的同容器锁一次只放行一个 `exec_gpu` 调用，因此能够减少同时存在的 GPU 进程，但也会串行化命令里的 Python 导入和数据准备。只有当并发造成的 OOM、失败重做或争抢损失大于等待代价时，才能得到吞吐收益。不能从“显存更低”直接推导“完成得更快”。

本次同容器锁之前等待的协程，也不等于 Router 队列中的等待者。即使 15 个协程尚未执行，Router 的 `gpu_commands_waiting` 仍可能为零。这不能证明跨容器或跨 worker 的超额排队行为。

## 5. 跟着图读源码与复现

| 步骤 | 图中编号 | 输入、函数和状态变化 |
|---|---|---|
| 1 | 1 | [准备脚本 main](env-service-verification/scripts/prepare_science_natural_gpu.py#L15) 接收原始 baseline 和固定 Parquet，输出 640 张真实图片与哈希 |
| 2 | 2 | [run](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_ab.py#L28) 核验设备空闲，建立一个 Harbor 任务。ON 同时建立独立 Ray 和 Router |
| 3 | 3 | [asyncio.gather](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_agent.py#L35) 并发提交 16 个请求。OFF 直接 exec，ON 先进入同容器锁 |
| 4 | 4b | [exec_gpu](env-service-verification/results/science-natural-gpu-20261022/source/on-docker.py#L67) 持锁申请、执行、归还。未拿到锁的协程尚未向 Router 发起 GPU 申请 |
| 5 | 5 | [main](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_train.py#L41) 用相同 seed、真实图片构造数据集，调用原始 Trainer 完成十步 |
| 6 | 6 | [代理 finally](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_agent.py#L29) 保存每条命令的返回码与 stdout，遥测协程独立采样，不把缺失补零 |
| 7 | 7 | [run finally](env-service-verification/results/science-natural-gpu-20261022/source/science_natural_gpu_ab.py#L170) 记录账本和设备状态，关闭本次隔离 Ray，检查容器物理释放 |

以下命令用于有 GPU 和 Docker 的已预检节点。`BASELINE` 指原始 MLGym baseline 文件，`PARQUET` 指固定 revision 的训练分片，`OUT` 指全新的结果目录，`SCRIPTS` 指本文脚本目录。准备使用真实数据，不生成随机图片：

```bash
SCRIPTS=docs/env-worker-deep-dive/env-service-verification/scripts
python "$SCRIPTS/prepare_science_natural_gpu.py" \
  --baseline "$BASELINE" --parquet "$PARQUET" --output "$OUT/workload"
```

运行时还需要 `PYTHON` 为容器可只读挂载的环境解释器，`IMAGE` 为已确认可用的本地镜像，`NODE_IP` 为实际执行节点地址，`GPU` 为现场核验的 GPU index，`OVERLAY` 为根据 UUID、minor number 和驱动库生成的单卡 compose override。两组分别将 `PYTHONPATH` 指向对应冻结源码与脚本目录：

```bash
python "$SCRIPTS/science_natural_gpu_ab.py" \
  --mode off --output "$OUT/off" --workload "$OUT/workload" \
  --overlay "$OVERLAY" --python "$PYTHON" --image "$IMAGE" \
  --node-ip "$NODE_IP" --gpu "$GPU" --count 16
```

核验 OFF 资源物理释放后，再切换 ON 的源码路径，以 `--mode on --output "$OUT/on"` 执行同一命令。不要只在当前代码里关闭 env_service 充当历史 OFF。两组收尾后执行：

```bash
python "$SCRIPTS/analyze_natural_gpu.py" "$OUT"
```

分析器只在双方全部完成时计算总时长比值。预期观察是 OFF 的 GPU 进程允许重叠，而当前 ON 同容器命令串行。OOM 是否发生由真实工作负载决定，不作为脚本必须制造出来的结果。

## 6. 验证边界与执行经验

- 本次是真实分类训练内核的调度实验，采用固定脚本代理，没有运行语言模型生成命令，没有 PSRL 强化学习更新，没有完整 MLGym test-set 准确率评测。自定义 verifier 的 reward=1 只代表所有命令完成十步。
- 普通 Science 终端命令不会自动经过 `exec_gpu`。本次固定代理明确调用了该接口，因此不能把结果推广为所有现有 Science 题目已自动受控。
- 同一容器的串行命令没有测试多容器装箱、跨 worker 公平性、与 vLLM 借还设备或多卡放置策略。
- 启动前核实 GPU index、UUID 和 minor number。本次 index 6 实际对应设备 minor 5，不能把 index 直接拼成设备路径。
- NVML v1 的 used 在本节点包含约 586 MiB 驱动保留量。v2 将 `used` 和 `reserved` 分开，设备空闲时实际 used 约 0.19 MiB。先区分接口语义，再判断他人占用，不删除未知进程。
- 分开记录总命令时间、导入和数据准备时间、CUDA 训练区间。`cuda_started_at` 位于数据准备之后、Trainer 构造之前，仍包含 Trainer 初始化开销，不等于纯 GPU 内核时间。
- 实验期间节点存在其他任务容器，本次只控制指定 GPU，并保留目标卡进程遥测。不能声称独占节点 CPU、文件系统或 Docker 服务。

本次五个实验与分析脚本通过定向 Ruff 检查，真实数据准备脚本在 CPU 上复跑，产出的 640 张图片和标签与实验输入逐项相同。修改文档的 195 个本地链接路径检查通过，新增目录锚点和源码行号已核对。完整 `pre-commit run --all-files` 在隔离副本运行，仍因仓库既有 Ruff 与注释、日志风格违规失败，没有将自动修改写回共享工作区。检查记录见[validation.json](env-service-verification/results/science-natural-gpu-20261022/validation.json)与[precommit.log](env-service-verification/results/science-natural-gpu-20261022/precommit.log)。

## 7. 启动耗时定位与修复机制

修复分为依赖扫描和调度边界两部分。前者减少每个 Python 进程反复访问共享存储的次数，后者让多个 CPU 准备阶段能够同时推进。完整链路如下：

```text
[A] 宿主机提交多个 exec_gpu_prepared 请求
           |
[B] 每个请求创建独立监督进程和工作进程，暂不申请 GPU
           |
[C] 工作进程 prepare(payload)：导入已核查的基础依赖、准备 CPU 数据
           |
[D] 工作进程标记 ready，保留已加载的模块和数据，等待启动信号
           |
[E] 宿主机取得同容器 GPU 锁，经原 Router 授权后发送启动信号
           |
[F] 同一个工作进程执行 prepare 返回的 GPU 函数
    此时才导入 Trainer 等会探测 CUDA 的组件
    Transformers 包名映射：读取清单 → 按顶层名分组 → 找到存在文件即停止
           |
[G] 进程退出并被监督进程回收 → GPU 归还确认 → 收集输出
    取消分支：核对 PID 创建身份 → 终止所属进程组 → 再通知 Router 取消
```

图例：图里每个部分对应的源码位置。

| 编号 | 实现与职责 |
|---|---|
| A、B | [execute_prepared](../../psrl/workers/env_service/prepared_gpu.py#L24) 创建唯一工作目录，并启动监督执行与准备就绪观察两个异步操作 |
| F | [packages_distributions](../../patch/transformers/fast_package_metadata.py#L10) 保留声明名称，对需要推断的名称只检查到第一个存在文件 |
| C、D | [worker](../../psrl/workers/env_service/prepared_gpu_process.py#L43) 加载程序，调用 prepare，保存 ready 后等待授权信号 |
| E | [execute_prepared 的 GPU 调用](../../psrl/workers/env_service/prepared_gpu.py#L81) 在 CPU 准备结束后调用原 exec_gpu，不修改 GPU 准入预算或容器回收语义 |
| F | [训练 prepare](env-service-verification/scripts/science_natural_gpu_train.py#L11) 返回捕获了真实数据集和基础模块的执行函数，在授权后导入 baseline，训练仍调用原始 train_model |
| G | [stop](../../psrl/workers/env_service/prepared_gpu_process.py#L30) 核对进程身份并等待终止，监督进程负责 wait 回收，避免留下无人回收的后台工作进程 |

这些代码如果有理解上的问题，可以随时追问。这里各个 CPU 准备阶段是并发的，GPU 执行仍串行，监督进程不是另一份训练进程，不持有 CUDA 张量。

**启动时间为什么长。** `import` 不只是把一行 Python 名称放进内存，它可能继续加载很多依赖，并查询安装包的元数据。元数据是描述软件包的名称、版本、文件清单等信息。例如导入时写 `PIL`，安装包名称却是 `pillow`，Transformers 需要知道两者的对应关系，才能检查依赖版本。

本节点实际 Transformers 在[模块导入时](env-service-verification/results/science-natural-gpu-fix-20261022/source/transformers-import-utils.py#L47) 构造整个名称映射。实际 Python 版本的 [Distribution.files](env-service-verification/results/science-natural-gpu-fix-20261022/source/stdlib-metadata.py#L474) 会对清单中的每个文件检查存在性，然后 [_top_level_inferred](env-service-verification/results/science-natural-gpu-fix-20261022/source/stdlib-metadata.py#L956) 才提取顶层导入名。大型机器学习环境有大量包文件，这些小查询在共享文件系统上逐个等待，每个新进程又独立做一遍。

本次没有根据目录名称猜测原因。限时 CPU 容器使用 Python 的导入计时与周期调用栈抓取，反复观察到 `packages_distributions → Distribution.files → Path.exists/stat`，另有读取模块文件的等待。容器关闭网络，没有加载模型权重或执行 GPU 训练。实际分段如下：

| 阶段 | 原始依赖预检 | 隔离修复后预检 |
|---|---:|---:|
| NumPy 导入 | 0.61 秒 | 0.32 秒 |
| PyTorch 导入 | 10.21 秒 | 4.60 秒 |
| datasets 导入 | 7.87 秒 | 3.27 秒 |
| 原始 baseline 导入，含 Transformers/Trainer | 87.26 秒 | 27.15 秒 |
| 真实图片读取与 Dataset 构造 | 0.06 秒 | 0.05 秒 |
| 合计 | 106.01 秒 | 35.39 秒 |

两个完整导入预检有缓存和共享设备状态差异，不能把整个差值都归因于单个补丁。更直接的机制验证是同一个容器依次调用标准实现与修复实现：两者得到完全相同的 **442 个名称映射**，存在性检查由 **48,400 次降为 169 次**，耗时由 **11.13 秒降为 1.20 秒**。补丁没有跳过依赖版本检查，没有修改 Python 全局元数据接口，也没有修改共享 Transformers。

具体例子：教学假设一个 wheel 安装包在清单中列出 `torch/a.py` 到 `torch/file9999.py` 共一万个文件。为了得到顶层名称 `torch`，原路径先检查一万个文件是否存在。修复后仍读取清单，但把同一顶层名称归为一组，找到首个真实存在文件就知道这个名称有效。如果第一项已被删除，继续检查后面的项，如果全都不存在则不登记。CPU 回归覆盖了一万文件只发生一次存在性查询、已删除文件、导入别名和多个发行包共享名称的情况。这里的一万文件是测试数据，不冒充真实环境文件数量。

**ON 的具体问题是什么。** 原接口收到的是完整 shell 命令，如 `python train.py`。它无法自动知道“导入结束、现在开始用 GPU”的位置，所以把整个命令当作 GPU 工作。本次增加显式分阶段 Python 接口，利用同一个工作进程保留已准备数据，避免释放锁后再开一个 Python 进程重新导入。

不能简单删除同容器锁：设备访问权限以容器为单位，同一租约只允许一个待执行 GPU 请求，一条命令归还时还要确认容器没有存活 CUDA 上下文。直接让同容器多条 GPU 命令同时进入，会使归还和仍在运行的命令互相冲突。修复保留这条不变量，只将明确不使用 CUDA 的准备阶段移到锁外。

**CPU 准备为什么也会碰到 CUDA。** “没有创建 GPU 张量”不足以证明代码不访问 GPU。本次第一版分阶段回归把整个 baseline 导入放在 prepare 中，四个进程均在执行阶段报 `CUDA is unavailable`，这个失败结果完整保存在 [on-prepared](env-service-verification/results/science-natural-gpu-fix-20261022/on-prepared/result.json)。导入追踪确认 `Trainer → Flash Attention 集成 → is_torch_cuda_available → torch.cuda.is_available`，另有 Transformer Engine 导入时检查设备能力。CUDA 是让程序调用 NVIDIA GPU 的运行接口，默认的可用性检查会触发底层驱动初始化，此时容器 GPU 设备访问尚被拒绝，进程内的失败状态与依赖探测缓存不能当成授权后的新状态。我们未尝试绕过设备权限或伪造 CUDA 可用。

修复后的 CPU 阶段只包含 NumPy、PyTorch、datasets 的基础导入和图片准备，将 baseline、Trainer 及其 CUDA 探测留在授权后。另在无 GPU 容器内把 CUDA 可用性、设备数量和底层初始化入口替换为“调用即报错”的检查函数，实际运行 prepare 成功，见 [check-cpu-prepare.log](env-service-verification/results/science-natural-gpu-fix-20261022/check-cpu-prepare.log)。这只是当前依赖版本的回归，升级依赖后需要重新检查。其余导入开销仍存在，例如 Pydantic 加载插件时还会读取安装元数据，所以不能承诺每个进程瞬间启动。

**跟着修复图读源码。** A/B 从 [execute_prepared](../../psrl/workers/env_service/prepared_gpu.py#L24) 接收程序路径、JSON 参数和三个时间预算，创建独立目录。C/D 在 [worker](../../psrl/workers/env_service/prepared_gpu_process.py#L43) 中得到可调用对象并写就绪标记，多个 worker 的这段执行并发。E 由 [GPU 调用](../../psrl/workers/env_service/prepared_gpu.py#L81) 经过原同容器锁与 Router 授权。F 的 [execute](env-service-verification/scripts/science_natural_gpu_train.py#L36) 导入 Trainer、执行十步、保存结果。G 的监督进程等待子进程退出，Router 确认归还，宿主机再清理临时目录。发生取消时先停止所属进程，再通知 Router，不允许留下后台 CUDA 工作后就把卡交给下一条命令。每次调用的输出和预约记录一起返回，避免异步收尾期间下一条命令覆盖全局 last_gpu_command，导致指标归错任务。

## 8. Results：修复回归

CPU 回归已通过 29 项，覆盖真实子进程准备并发、准备失败不请求 GPU、取消时停止工作进程、实际 ManagedDockerEnvironment 在通知 Router 取消前完成进程清理，以及原有命令准入和 GPU job 回收逻辑。完整包名映射的一致性验证也已通过。

实机回归使用同一修复后 ON 源码、同一隔离依赖、同一组真实图片和模型，每组四个命令、每命令十步。旧调用方式仍使用 exec_gpu 执行完整 Python 命令，新调用方式使用 exec_gpu_prepared。这里比较的是 ON 内部修复前后的调用方式，不是更换用户指定的历史 OFF baseline。

| 指标 | ON 完整命令调用 | ON 安全分阶段调用 |
|---|---:|---:|
| 完成命令 / 每命令步数 | 4 / 10 | 4 / 10 |
| 命令阶段，代理开始到全部命令收尾 | 190.48 秒 | 158.50 秒 |
| episode，包含容器、任务、评分与回收 | 221.67 秒 | 187.75 秒 |
| CPU 准备最大重叠进程数 | 1 | 4 |
| 四个 Python 工作进程启动时间跨度 | 142.26 秒 | 0.0032 秒 |
| 观测到的 GPU 进程峰值 | 1 | 1 |
| GPU 显存峰值 | 693.06 MiB | 693.06 MiB |
| 最终 train_loss，所有命令相同 | 2.303740954399109 | 2.303740954399109 |
| 最终参数和，所有命令相同 | -4.757167816162109 | -4.757167816162109 |

命令阶段缩短 **16.79%**，episode 缩短 **15.30%**。这是一次有界配对观察，未重复采样建立统计置信区间。两组均使用相同的元数据补丁，因此上述差值主要用于评价准备与 GPU 授权的边界调整，不是“原始依赖到全部修复”的总收益。CPU 与文件系统缓存仍会影响单次结果，不能把所有时间差都归因于并行准备。

用本次数据理解调度变化：安全分阶段组的四个工作进程几乎同时启动，各自用约 9.53 秒导入基础依赖并准备图片。随后 GPU 授权仍逐个发放，每个命令继续用约 27.05 至 32.06 秒导入 baseline 和 Trainer，再用约 3.32 至 3.77 秒完成 Trainer 初始化与十步训练区间。节约的是重复串行等待的基础准备时间，仍有大量会触发 CUDA 探测的依赖导入留在 GPU 区间。这解释了为什么修复后有改善，但还不是理想的高吞吐方案。

第一次分阶段尝试因错误地提前导入 Trainer 而四个命令全部失败，虽然只耗时 102.42 秒，**不能作为加速结果**。失败源码与输入在 [on-prepared-failed-source](env-service-verification/results/science-natural-gpu-fix-20261022/on-prepared-failed-source) 保留，修正后只续跑失败组，没有重跑已完成的完整命令组。最终两组均观察到单 GPU 进程上限，没有 OOM 或显存占位。它没有验证多卡分配、高显存压力收益，也不能与前面 16 命令 OFF 的总时间直接相除。

GPU 与所属容器、网络、隔离 Ray 进程已做物理清理核验，见 [physical-cleanup.json](env-service-verification/results/science-natural-gpu-fix-20261022/physical-cleanup.json)。原始材料保存在[修复结果目录](env-service-verification/results/science-natural-gpu-fix-20261022)，汇总见 [analysis.json](env-service-verification/results/science-natural-gpu-fix-20261022/analysis.json)。分析脚本只在成功工作量一致时计算改进比例：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_natural_gpu_fix.py \
  docs/env-worker-deep-dive/env-service-verification/results/science-natural-gpu-fix-20261022
```

定向 CPU 回归 29 项通过，实际无 GPU 容器的准备阶段禁探测检查通过。修改文件的定向风格检查通过。完整 pre-commit 在隔离副本执行，仍被本次改动之外的既有 Ruff 和文字风格违规阻塞，见 [precommit-final.log](env-service-verification/results/science-natural-gpu-fix-20261022/precommit-final.log)。未把隔离副本中的自动格式修改写回其他人的文件。

## 9. 使用分阶段接口与隔离依赖

分阶段接口适用于能够明确拆出 CPU 准备阶段的 Python 程序。`prepare(payload)` 必须只做 CPU 工作，并返回无参数的执行函数，CUDA 初始化、模型搬到 GPU 和训练放在返回函数中。会间接探测 CUDA 的模块导入也必须放在返回函数中，不能只检查显式的 `.cuda()` 调用。接口不会自动拆分任意终端命令，也不是同容器内不同进程之间的硬隔离承诺。

训练适配的现有例子见 [science_natural_gpu_train.py](env-service-verification/scripts/science_natural_gpu_train.py)。代理通过 [science_natural_gpu_agent.py](env-service-verification/scripts/science_natural_gpu_agent.py) 调用，多个准备进程并发，GPU 授权与输出收集仍由宿主机负责。返回后进程已退出并回收，不留下持有 CUDA 上下文的常驻 worker。

在仓库根目录运行以下 CPU 回归，不需要 GPU 或 Docker：

```bash
pytest -q tests/env_service/test_prepared_gpu.py tests/env_service/test_package_metadata.py
```

其中并发测试让两个准备函数互相等待对方的就绪标记。如果错误地在 GPU 锁内执行准备，第一个准备函数会因等不到第二个而失败。取消测试启动真实 CPU 子进程，模拟独占 GPU 授权，核验进程退出和调用顺序。它验证生命周期，不代替实际 CUDA 回归。

依赖修复使用明确的源包目录 `TRANSFORMERS_SOURCE` 和全新的输出目录 `DEPS`，不能指向共享安装目录作为输出：

```bash
python patch/transformers/prepare_metadata_fix.py \
  --source "$TRANSFORMERS_SOURCE" --output "$DEPS"
```

源代码中必须存在本次识别出的元数据扫描语句，脚本不匹配时拒绝修改。生成目录采用只读源文件链接及独立修改文件，保留原始和修改后哈希。将 `DEPS` 放到本次进程的 `PYTHONPATH` 首位，并在实际容器中只读挂载该目录及它指向的原始环境。源依赖改变时重新准备和核验，不能把这个链接目录当成完全脱离源环境的不可变镜像。

第 5 节的真实训练脚本现在额外支持 `--prepared --dependencies "$DEPS"`。只有 ON 加 `--prepared` 才启用新的分阶段调用，两种调用方式都加相同的 `--dependencies`，才能单独比较调度边界。原始历史实验快照保持不变。


## 10. 固定 OFF 与修复后 ON 的容量压力对照

本节回答两个问题：低压力时修复后的 ON 相比固定 OFF 慢多少，可用显存明显不足时 ON 能否减少真实训练失败。用户在本次请求中明确要求预先占用显存，因此高压力组有意注入外部显存占位。这与前面的无占位实验分别统计。

```text
[H] 控制脚本固定四个训练命令、每命令十步、相同模型和真实图片
                  |
        低压力：不注入占位
                  |
        OFF → 核验释放 → ON → 核验释放
                  |
[I] 独立占位进程：申请显存，只留下约 1.5 GiB 可用空间
    保留 PID、创建时间、GPU UUID 与实际可用显存
                  |
[J] ON：CPU 准备并发 → 同容器逐条 GPU 授权与训练
    OFF：固定历史 runner → 静态 GPU 映射 → 四条训练命令并发
                  |
[K] 分别记录完成数、显存不足错误、时间和排除占位进程后的 GPU 并发
                  |
[L] 回收所属容器与网络 → 核验 GPU 只剩占位 → 停止占位 → 核验空闲
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与实现 |
|---|---|
| H | [run](env-service-verification/scripts/run_natural_gpu_capacity.py#L26) 按低 OFF、低 ON、高 ON、高 OFF 顺序启动四组，固定工作量，设置独立日志和运行预算 |
| I | [hold](env-service-verification/scripts/hold_gpu_capacity.py#L14) 根据实际 CUDA 空闲量分配显存，保存身份信息，只做一次微小读写校验，随后等待释放 |
| J | [run 的压力预检](env-service-verification/scripts/science_natural_gpu_ab.py#L39) 核验占位进程身份与 GPU 占用，训练执行沿用第 1 节和第 7 节的 OFF / ON 路径 |
| K | [analyze](env-service-verification/scripts/analyze_gpu_capacity.py#L8) 从每条命令的实际日志统计成功和 OOM，成功工作量不相同时不计算速度比 |
| L | [released](env-service-verification/scripts/run_natural_gpu_capacity.py#L15) 有界等待 GPU 回到预期占位状态，控制脚本随后检查 Compose 项目与网络并释放占位 |

这些代码如果有理解上的问题，可以随时追问。按图读源码时，H 启动的四条命令是并发提交，J 的 ON 只有 CPU 准备并发，GPU 执行仍串行。I 的占位进程独立于训练容器，在两组高压力训练之间保持相同分配，不随某条训练命令退出而释放。正常退出、控制进程终止和最长持有时间到期都触发占位释放。

本次仍是 **MLGym 的 CIFAR-10 原始 Net / train_model 加固定脚本代理**，使用 640 张真实图片、batch size 64、seed 42、每命令 10 步，不运行语言模型生成命令，也不做完整 MLGym 测试集评测或 PSRL 强化学习更新。上次 ON 很慢的 16 命令实验同样采用这个工作负载，不应称作完整 MLGym benchmark 成绩。

| 固定条件 | 本次设置 |
|---|---|
| OFF | 用户指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，静态 GPU override，无整卡锁 |
| ON | `37a06ea8d32b05bbbf385ad40cb404b30ba01680` 加已验证的工作区修复，使用 exec_gpu_prepared |
| 共同依赖 | 同一 Python 环境、同一隔离 Transformers 元数据补丁、同一 veRL 补丁来源 |
| GPU | 同一张现场核验空闲的物理 GPU，四组顺序执行 |
| 低压力 | 不占位，不人为限制可用显存 |
| 高压力 | 占位后留下约 1536 MiB 可用显存，原始模型和输入大小保持不变 |
| ON 单命令申报 | 高低压力均为 1024 MiB，基于此前约 693 MiB 实测峰值留出余量 |
| ON unit 预算 | 高低压力均为设备物理总 MiB，外部占位通过实测 used 纳入准入检查，不把物理预算误设成剩余容量再重复扣减 |
| 预算与停止条件 | 四组各四命令、每命令十步、零训练重试，每组工作限时 780 秒，driver 硬限时 900 秒，占位最多 1800 秒，四组结果明确后结束 |

OFF 保留必要运行补丁：传递 GPU override、关闭节点级 dangling-image prune，以及已有 HTTP 连接池 / Ray 地址运行修复。这些补丁没有加入 GPU 锁或改写静态放置语义，实际补丁见 [off-runtime.patch](env-service-verification/results/science-gpu-capacity-20261022/off-runtime.patch)。ON 的实际 Python 源码补丁见 [on-runtime.patch](env-service-verification/results/science-gpu-capacity-20261022/on-runtime.patch)。两组源码清单和哈希、实验设计与相同输入的哈希均保存在[结果目录](env-service-verification/results/science-gpu-capacity-20261022)。

## 11. Results：高低显存压力实测

| 组别 | 完成十步的命令 | 错误 | 命令阶段 | episode | driver 启动至物理释放核验 | 训练 GPU 进程峰值 | 训练增量显存峰值 |
|---|---:|---|---:|---:|---:|---:|---:|
| 低压力 OFF | 4/4 | 无 | 55.00 秒 | 74.31 秒 | 90.46 秒 | 4 | 2761.56 MiB |
| 低压力 ON | 4/4 | 无 | 156.93 秒 | 187.07 秒 | 213.09 秒 | 1 | 692.88 MiB |
| 高压力 OFF | 0/4 | 3 OOM + 1 cuBLAS 分配失败 | 47.48 秒 | 66.16 秒 | 81.87 秒 | 4 | 1522.19 MiB |
| 高压力 ON | 4/4 | 无 | 165.35 秒 | 198.14 秒 | 227.41 秒 | 1 | 689.56 MiB |

命令阶段从代理开始提交命令计到全部命令收尾，包含导入、CPU 准备、等待与 GPU 训练。episode 还包含容器生命周期和评分。第三个时间口径从独立 driver 进程启动计到资源释放核验，包含 driver 导入和 ON 的隔离 Ray 初始化。不同口径必须分别比较，不能用 OFF 的命令时间与 ON 的全程时间相除。

**低压力的代价。** 两组均完成四个命令，因此可以直接比较：ON 命令阶段多用 **101.92 秒**，耗时增加 **185.30%**，即 **2.85 倍**。episode 为 OFF 的 **2.52 倍**，含 driver 启动及释放核验为 **2.36 倍**。修复让 CPU 基础准备并发，但同容器 GPU 区间仍串行，且会探测 CUDA 的 Trainer 导入仍在该区间内。OFF 的四个任务能同时放下时，这个保守限制明显拖慢完成速度。

**高压力的收益。** 占位进程实际分配 95,422 MiB，CUDA 查询剩余 **1537.875 MiB**，两组共用同一个占位进程，所有成功采样都确认它始终存在。ON 四个命令全部完成十步，OFF 没有任何命令完成十步。OFF 的 47.48 秒只是四条失败返回的时间，不能与 ON 完成全部训练的 165.35 秒计算速度比。该压力点显示的是从 0/4 到 4/4 的完成率收益，没有测试自动重试后“完成相同四条成功训练”的总时长。

**具体失败证据。** 两条 OFF 命令报 `OutOfMemoryError`，第三条报 CUDA `out of memory`。日志显示剩余约 **15.69 MiB** 时仍需分配 **32 MiB**，因此申请无法满足。另一条报 `CUBLAS_STATUS_ALLOC_FAILED`，cuBLAS 是 GPU 矩阵运算库，该错误发生在创建库句柄时。它与同时刻显存紧张相符，但这里单独记为库分配失败，不冒充第四条明确 OOM。四条命令的退出码都是 1，原始错误保存于 [analysis.json](env-service-verification/results/science-gpu-capacity-20261022/analysis.json) 的 commands 字段和代理原始日志。

用实测数字理解：没有占位时，OFF 四个训练进程同时运行，增量峰值约 **2761.56 MiB**，这张卡完全放得下。高压力只留下约 **1537.88 MiB**，就无法容纳同样的同时需求。ON 同时只运行一个训练进程，该组增量峰值约 **689.56 MiB**，所以本次四个命令逐个完成。这个例子说明并发限制的取舍，不代表任何模型只要串行就一定不会 OOM，单个任务自身超过剩余显存仍然会失败。

所有成功命令的最终 `train_loss` 都是 **2.303740954399109**，参数和都是 **-4.757167816162109**。这是相同工作负载结果一致性的检查，不是模型效果提升或收敛证明。四组遥测缺失数均为零，高压力训练进程计数已排除占位进程。

![相同四命令的完成数量与返回时间，高压力 OFF 的短时间对应失败而非加速](env-service-verification/results/science-gpu-capacity-20261022/capacity-comparison.svg)

图由 [plot_gpu_capacity.py](env-service-verification/scripts/plot_gpu_capacity.py) 从 analysis.json 生成，柱形左侧分别表示成功与失败数，右侧明确标注高压力 OFF 未完成相同工作量。重新生成图只需要 CPU 和 Matplotlib：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_gpu_capacity.py \
  docs/env-worker-deep-dive/env-service-verification/results/science-gpu-capacity-20261022
```

本次是四组各一次的有界验证，没有反复重跑寻找更漂亮的结果，也没有建立统计置信区间。低压力 OFF→ON、高压力 ON→OFF 的顺序只能部分平衡缓存顺序影响，不能消除共享 CPU、文件系统与启动波动。

四组任务、主容器、sidecar、网络、隔离 Ray 和占位进程已按归属释放，目标 GPU 没有剩余计算进程，实际 used 回到约 0.19 MiB，见 [physical-cleanup.json](env-service-verification/results/science-gpu-capacity-20261022/physical-cleanup.json)。容器预检的首次脚本挂载路径错误已单独保留在 preflight.log，修正后 CUDA 读写预检通过，未重跑已完成训练组。相关 CPU 回归六项通过，修改脚本定向 pre-commit 通过，全仓检查仍被既有违规阻塞，详见结果目录检查日志。


## 12. 容量压力实验复现与边界

已有结果的分析只需要 CPU 和 Python 标准库，在仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_gpu_capacity.py \
  docs/env-worker-deep-dive/env-service-verification/results/science-gpu-capacity-20261022
```

它输出各组完成数、OOM 数、命令阶段、episode 和包含 driver 启动的时间。只有相同数量训练全部完成时才计算 ON / OFF 耗时变化。占位进程被排除出训练 GPU 进程计数，增量显存峰值从该组训练前实际占用量扣除，采样缺失保留为缺失。

重新运行实机实验前，按第 5 节准备真实数据和与目标 GPU 对应的 compose override。`RUN_ROOT` 为全新结果目录，包含 workload 和 gpu-overlay.yaml，`OFF_SOURCE` 与 `ON_SOURCE` 分别为核对过提交和补丁的冻结源码目录，`DEPS` 为隔离 Transformers 包的父目录，`VERL_PARENT` 为实际 veRL 补丁父目录，`NODE_IP` 为执行节点地址，`GPU_INDEX` 为现场核验的设备 index。将本节脚本及所引用的训练代理脚本放到冻结 source 目录，再执行：

```bash
python "$RUN_ROOT/source/run_natural_gpu_capacity.py" \
  --root "$RUN_ROOT" --off-source "$OFF_SOURCE" --on-source "$ON_SOURCE" \
  --dependencies "$DEPS" --verl-parent "$VERL_PARENT" \
  --node-ip "$NODE_IP" --gpu "$GPU_INDEX" --image "$IMAGE"
```

`IMAGE` 必须是预检成功且能运行上述环境解释器的本地镜像。执行节点应先激活项目环境，恢复本次工作区路径，并让驱动脚本可导入冻结的源码和修复依赖。不得在其他用户占用的 GPU 上启动占位脚本。

“留下 1.5 GiB”不等于把硬件改造成一张 1.5 GiB 显卡。设备总容量、计算能力和带宽仍不变，只是当前剩余显存变少。该实验验证的是共卡显存紧张时的行为，不能直接推广为小显存型号的完整性能预测。占位进程不持续执行计算，因此本次也不是高计算利用率或高功耗压力实验。

单容器 ON 串行保护如果减少 OOM，说明在这个场景中限制并发有可靠性收益，不能单独归功于多 GPU 放置、跨 worker 公平性、显存碎片治理或与 vLLM 的借还机制。这里没有测试这些能力。相反，低压力时多个小训练本来能同时放下，串行保护与额外调度会有时间代价，应按实际数字报告。

## 13. 实验层次与十步的含义

先把两种实验放在同一张图中。A 是此前完整训练的外层，B/C 是本次仍实际使用的环境执行层。下面是源码调用关系，不是新一轮实机结果。

```text
[A1] PSRL 训练器：采样任务，收集轨迹，组织大模型更新
       ↓ 调用采样执行链路
[A2] Science AgentLoop → Science runner → Harbor 容器
       ↓ 真实语言模型代理生成命令，执行任务并评分
[A3] 轨迹与奖励返回训练器 → 更新 Qwen 参数 → 下一训练步

最近的组件实验从这里进入，跳过 A1 / A3 和语言模型决策：
[B1] 固定实验脚本 → 同一个 Science runner → 一个 Harbor 容器
       ↓
[B2] 固定脚本代理同时提交四条 CIFAR 训练命令
       ├─ OFF → 普通 exec → 四个进程可重叠运行
       └─ ON  → 并行 CPU 准备 → 同容器锁 → Router 授权
                                              ↓
[B3] 加载分类模型 → 十步分类训练 → 返回输出 → 核验 GPU 归还
       ↓
[B4] 保存结果 → 停止本次容器、租约和隔离 Ray → 物理清理核验
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责、函数与源码 |
|---|---|
| A1、A3 | [训练器 fit](../../psrl/trainer/ppo/ray_trainer.py#L3343) 组织外层循环；[_update_actor](../../psrl/trainer/ppo/ray_trainer.py#L3204) 将轨迹批次交给训练 worker 更新大模型 |
| A2 | [SciAccelAgentLoop._run_harbor_in_thread](../../examples/sciaccel_rl/agent_loop.py#L358) 调用 [run_harbor_episode](../../examples/sciaccel_rl/runner.py#L259)；[PressureTerminus2._query_llm](env-service-verification/scripts/science_pressure_agent.py#L21) 先调用真实模型，再执行明确注入的压力阶段 |
| B1 | [组件实验 run](env-service-verification/scripts/science_natural_gpu_ab.py#L28)，[第 180 行](env-service-verification/scripts/science_natural_gpu_ab.py#L180) 调用真实 runner；[第 127 行](env-service-verification/scripts/science_natural_gpu_ab.py#L127) 启动真实 EnvRouter |
| B2 | [NaturalGpuAgent.run](env-service-verification/scripts/science_natural_gpu_agent.py#L11)，[第 47 行](env-service-verification/scripts/science_natural_gpu_agent.py#L47) 使用 asyncio.gather 同时提交；ON 调用 exec_gpu_prepared，OFF 调用普通 exec |
| B3 | [prepare / execute](env-service-verification/scripts/science_natural_gpu_train.py#L11) 分离准备与执行；[第 50 行](env-service-verification/scripts/science_natural_gpu_train.py#L50) 限制 CIFAR 更新数；[revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) 检查物理归还 |
| B4 | [实验 finally](env-service-verification/scripts/science_natural_gpu_ab.py#L195) 回收所属服务；四组实机物理核验保存在 [physical-cleanup.json](env-service-verification/results/science-gpu-capacity-20261022/physical-cleanup.json) |

这些代码如果有理解上的问题，可以随时追问。这里引用当前工作区实现帮助阅读；实验版本以第 10 节冻结提交及补丁为准，当前 runner 的默认 OFF 不能替代固定 baseline。

**跟着图读源码。** 完整训练沿 A1→A2→A3 循环，A2 中的多条轨迹可以并发，不能理解成所有容器串行调用。组件实验则由 B1 直接创建任务并进入 runner；B2 分出四条协程，ON 的 CPU 准备并行，GPU 执行按同容器锁轮流进入 B3。每条命令完成后先确认归还，再允许下一条进入；四条汇合后 B4 清理。外层大模型的采样、奖励计算与参数更新没有被这个组件脚本调用。

**Results：几轮实验的口径对照。** 下表引用既有结果，本节没有重启 GPU 实验。

| 实验 | 谁在被训练 | 容器里的真实工作 | 压力与并发单位 | 能回答什么 |
|---|---|---|---|---|
| [完整 Science 压力训练](science-single-gpu-training-control.md#7-results十步扩展验证) | 外层 Qwen 大模型，使用 GRPO，即根据同题多条回答的相对奖励组织强化学习更新 | LAPS 的 Fortran 修复，附加 CUDA 压力阶段 | 多个 episode 容器，每次压力申请 72 GiB；ON 跨租约排队 | 这个人工压力条件下，减少失败重试能否改善完整训练步骤耗时 |
| [完整 Science 无压力训练](science-single-gpu-training-control.md#10-results无压力续跑与共同完成的两步) | 同样是外层 Qwen | 原生 CPU Fortran 修复，没有原生 GPU 任务计算；模型推理和更新仍用 GPU | 多个 CPU 环境容器，无额外显存占位 | 环境无 GPU 争用时，完整训练的观测开销 |
| 本文原始十六命令实验 | 容器内 CIFAR 小型图像分类模型 | 真实图片、真实前向反向计算和参数更新，固定脚本提交命令 | 一个容器十六命令，无占位 | 同容器并发及启动开销；不是完整 MLGym 评测 |
| 本文四命令高低容量实验 | 同一个 CIFAR 分类模型，每命令十步 | 修复后分阶段执行，四条相同工作量命令 | 一个容器四命令；高压力额外占位，低压力不占位 | 同容器的完成率与耗时取舍；不是外层 Qwen 的训练吞吐 |

此前完整压力训练 OFF 完成十步、ON 完成五步后第六步中止，双方共同前五步为 48.63→35.17 分钟，OOM（显存不足）57→0。ON 中止是模型删除了实验脚本放在 `/logs` 中的唯一证据，属于采集脚本缺陷，不能解释成本次同容器串行限制。证据保存路径已修复并通过 CPU 回归，但未因此补出完整十步 ON 结果。

完整无压力续跑共同前两步为 521.17→535.42 秒，ON 慢 2.73%；ON 第三步因评分器镜像请求 EOF（连接提前结束）退出。这个比例和本次 CIFAR 低压力慢 185.30% 的分母、工作量、模型与计时范围都不同，不能相互替代。

选择固定代理是为了让 OFF/ON 执行相同四条命令，避免语言模型生成不同命令干扰组件对照。它继承 NopAgent，但覆盖 run 并实际启动训练，不是空跑。脚本传入占位模型地址和 model_name=unused，明确没有调用语言模型。**组件隔离有助于定位原因；若将它称为用户最初要求的完整 PSRL 训练收益验证，就是实验范围和汇报不准确。**

## 14. 同容器限制与问题归因

这里“协程”是可以交错推进的异步调用；同时提交四个协程，只表示四个调用都开始等待或准备，不保证四个 GPU 进程同时运行。“租约”是 Router 给某个环境分配资源、记录归属并负责回收的一份凭据。本轮一个环境容器对应一个租约，四个命令共享它。

```text
[C1] 四条协程已提交，共用一个环境容器和租约
                   ↓
[C2] CPU 准备可以并行
                   ↓
[C3] 同容器 Lock：A 进入；B、C、D 在客户端等锁
                   ↓
[C4] Router：同租约最多一个 pending / active 请求，检查预算和遥测
                   ↓
[C5] 开放整个容器的 GPU 设备访问 → 执行 A
                   ↓
[C6] 撤回容器设备权限，检查该容器所有 GPU 句柄和计算进程
                   ↓
[C7] 确认归还并释放锁 → B 才进入 C4
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责、函数与源码 |
|---|---|
| C1 | [代理 gather](env-service-verification/scripts/science_natural_gpu_agent.py#L47) 同时提交，没有写成逐条 await |
| C2 | [execute_prepared](../../psrl/workers/env_service/prepared_gpu.py#L24) 在 GPU 授权前准备；训练脚本明确将可能探测 CUDA 的 Trainer 导入留在授权之后 |
| C3、C7 | [exec_gpu 的同容器锁](../../psrl/workers/env_service/docker.py#L104) 包住授权、执行和归还；分阶段接口最终仍经过它 |
| C4 | [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L49)，[第 63 行](../../psrl/workers/env_service/command_gpu.py#L63) 拒绝同租约的第二个等待或活动请求，随后检查申报预算及实测显存 |
| C5 | [grant_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L139) 调用 [DeviceCgroupFence.set_access](../../psrl/workers/env_service/command_gpu_node.py#L61)，按整个容器控制设备权限 |
| C6 | [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) 禁止新设备打开，并等待该容器已有句柄和 CUDA 进程消失 |

**跟着图读源码。** C1 传入四份命令参数，C2 的四个 worker 分别完成准备，C3 只让一条进入 C4。因此本轮的其余三条主要在客户端等锁，不应把“四条已提交”写成“Router 有四条等待请求”。C4 授权后，C5 改变容器权限，命令得到 GPU 访问能力。执行结束进入 C6；只有实际释放得到确认，C7 才允许下一条。失败或取消走 exec_gpu 的所属进程停止及 Router cancel 分支，不能提前宣称资源可复用。

为什么有这样的锁？Linux cgroup 是把一组进程放在一起做资源或权限控制的机制。这里使用设备控制规则给**整个容器**开放或关闭 GPU，而不是分别给 A、B 两条命令独立发卡。设 A、B 同时持有 GPU：A 完成时撤回整个容器权限，会影响 B 后续打开设备；而 B 仍在计算或持有设备句柄，容器级“已归还”检查又不能通过。关闭权限也不是立即杀死全部已有 CUDA 内核，所以不能用一次 deny 代替真正的回收检查。现有实现通过一次只运行一个 GPU 命令，避免这个归属冲突。

**因此，不能只删掉 asyncio.Lock。** 服务端同租约单请求检查仍会拒绝第二条；连检查一起删除则破坏现有容量记账、授权代次和容器级归还语义。这是当前协议的限制，不是“显存充足时自动放开四条”的实现。

用已测数字理解它：低压力下四个训练进程峰值显存合计约 2761.56 MiB，OFF 能全部启动；ON 观测峰值约 692.88 MiB，但每次只运行一个。ON 的 1024 MiB 申报预算即使四份相加也只有 4096 MiB，低压力设备容量足够；然而第二条在申请预算前就被同容器锁拦住了，所以增加 GPU 总预算不能解除限制。`max_containers=1` 只限制容器数，把它改成 4 不会自动将一个容器里的四条命令拆成四个租约。

高压力只剩约 1538 MiB 时，容纳一条约 693 MiB 的实测工作较容易，四条重叠则可能分配失败。这解释了 ON 4/4、OFF 0/4。不过这个结果也与简单地把并发上限固定为 1 相容，**不能凭此证明 Router 已经实现“闲时并行四条、忙时自动收缩为一条”，也不能证明高级多卡放置收益。** 单条申报 1024 MiB 是准入估计，不是 CUDA 分配硬上限。

责任应按层分清楚：

| 现象 | 直接归因 | 实验与框架的责任边界 |
|---|---|---|
| 低压力同容器 GPU 进程峰值仍为 1 | 框架现有同容器锁、同租约单请求和容器级设备回收协议 | 对“同任务多个小 GPU 命令自适应并发”的目标，当前实现粒度过粗；不是代理顺序提交造成 |
| 一容器四命令、固定代理、不启动大模型训练器 | 实验入口和工作负载选择 | 合理覆盖用户提出的单任务多协程问题；不覆盖完整 PSRL 吞吐、多 worker 或多容器调度 |
| 十步短任务中启动占很大比例 | 共享存储依赖导入、仍须授权后执行的 Trainer 导入，以及串行临界区 | 依赖扫描与 CPU 准备已修；GPU 阶段仍串行。156.93 秒是命令阶段耗时，不是纯 Router RPC 耗时 |
| 以前 ON 第六个外层训练步证据丢失 | 采集脚本把唯一证据放进模型可删目录 | 是实验脚本缺陷，已修复采集并做 CPU 回归，不是本次串行限制的症状 |
| 将不同层次都简称“Science 十步训练” | 汇报口径 | 必须注明训练对象、入口、容器拓扑、压力来源和计时边界；不能让组件实验替代端到端验收 |

不同租约的命令不共用这个环境对象的锁；从 acquire 的源码条件看，跨租约请求会按申报预算、实测显存、排队和清理状态决定是否准入，而不是全局固定只能一条。这是源码能力边界说明，本轮没有实测“四个独立容器的 CIFAR 并行”，不能把它当作已验证修复。

对于原目标有两类方案，以下均不表示本轮已经实现或验证：独立工作放进各自容器和租约，可以使用现有跨租约准入机制，但需要考虑额外启动开销及共享文件语义；若必须共享同一个容器，就需要重新设计命令级预算、进程归属与取消回收，并让容器设备授权持续到所有活动命令完成，避免 A 归还影响 B。后者是框架协议设计工作，不是调大 max_containers 或实验脚本并发数就能完成。

具体改造方案见 [GPU 作业文档第 11 至 17 节](gpu-jobs-and-task-telemetry.md#11-同容器并发的目标架构)：推荐每作业独立预算和 cgroup 子树，覆盖外部显存与未兑现预约、授权竞态、取消及物理回收，并附已运行的 CPU 契约模型。方案和模型不代表生产框架已经支持同容器 GPU 并行。

## 15. CPU 教学复现与后续验收边界

[explain_same_container_lock.py](env-service-verification/scripts/explain_same_container_lock.py) 是标准库教学模拟，注释对应 C2/C3。在仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_same_container_lock.py
```

它先让四条协程都完成准备，再用短暂异步等待模拟执行区间。无锁时四条区间重叠，有锁时逐条进入。没有启动 Docker、Ray、CUDA 或任何训练，也没有模拟 Router 的遥测、容量和物理回收，因此只解释“并发提交为什么不等于并发执行”。

本次实际 CPU 运行中，两种模型的 submitted、prepared、completed 都为 4，最终 active 都为 0；off_model 的 peak 为 4，same_container_on_model 的 peak 为 1。这里的 peak 是模拟执行区间的重叠数，不能当成新测 GPU 并行能力或性能数字。

本次问题的验收是分清机制与实验层次，已有源码及日志足够，因此没有追加训练。后续若要验证框架改造，组件验收应同时要求低压力下四条真实工作能重叠、高压力时安全限制并发、失败取消后物理资源释放；通过后再接入真实 Science 代理与外层 PSRL，单独检验端到端步骤耗时和完成率。这个建议不构成本轮已完成的验证，也不要求为解释问题现在启动新实验。

本次新增教学脚本的 Ruff 检查和格式检查通过，本文与总索引的 139 个本地链接路径、源码行号范围及章节锚点检查通过，新增源码引用已逐项阅读。全仓 pre-commit 在隔离副本执行，仍因既有格式及注释风格违规失败，自动改动未写回工作区；记录见 [scope-explanation-precommit.log](env-service-verification/results/science-gpu-capacity-20261022/scope-explanation-precommit.log)。
