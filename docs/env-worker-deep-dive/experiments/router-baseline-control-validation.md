# 修复后的 EnvRouter 与固定 OFF 基线：单卡等完成量验证

> **核心结论**：修复后的 ON 在单卡高压力下，16 条成功任务耗时从固定 OFF 基线的 266.52 秒降至 207.84 秒，减少 22.02%，OOM 从 15 次降至 0。计入驱动记录的初始化和最终清理后，总时长减少 15.15%。低压力四任务则从 40.37 秒增加到 55.83 秒，说明安全调度并非所有负载下都更快。基线是指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，失败重试不计作完成。
>
> **低负载优化补充**：第 7–8 节记录后续诊断和代码优化。节点服务复用 NVML 连接、保持每次读取实时状态后，两批各四任务的总任务时间从 98.48 秒降到 90.36 秒；服务已就绪的第二批从 44.83 秒降到 41.87 秒。本轮比较优化前后 ON，不是重新测量 OFF。第 9 节解释预热原理，第 10 节记录已接入生产初始化流程后的私有节点小实验，第 11 节进一步拆分低负载关键路径，区分控制开销与授权后的执行长尾。第 12 节记录真实小任务复现、逐 CUDA 调用与线程栈定位。第 13 节记录两条同步准备路径移出事件循环的实现与实机对照。
>
> 这是实际 science runner、Harbor、Docker 和 CUDA 的控制实验。任务是合成验证任务，CPU 间隔固定，不是原生 LAPS 修复、真实模型 rollout 或训练。本文补充[命令阶段配额与慢查询修复](../knowledge/command-gpu-reservations.md#13-results慢查询定位与直接接口修复)，不替代该文第 7 节的历史训练结果。

> **真实训练补充**：[单张环境 GPU 的真实训练报告](science-single-gpu-training-control.md#7-results十步扩展验证)记录了十步目标扩展：OFF 完成十步，ON 完成五步后因实验审计文件被模型删除而退出。双方完整前五步的训练耗时减少 27.67%，OOM 从 57 次降到 0。完整十步收益尚未验证，节点干扰与测量脚本修复见报告第 7、8 节。 无压力原生任务修正配置后续跑，双方共同完成两步，ON 耗时增加 2.73%，未测出提速。ON 第三步因评分器镜像请求 EOF 退出，完整三步对照未完成，见[报告第 10 节](science-single-gpu-training-control.md#10-results无压力续跑与共同完成的两步)。这组真实训练与本文合成容器数据分别统计。

> 最新同期低负载配对与原设计 Gate A/B 审视见[独立报告](router-paired-latency-and-design-review.md)。该实验与本文早期顺序对照分别统计，并同时报告工作阶段和初始化成本。

## 目录

- [1. 比较链路与源码](#1-比较链路与源码)
- [2. 相同条件与停止边界](#2-相同条件与停止边界)
- [3. Results：等完成量与开销](#3-results等完成量与开销)
- [4. Results：账本与安全回填](#4-results账本与安全回填)
- [5. 复现与证据](#5-复现与证据)
- [6. 能说明什么与不能说明什么](#6-能说明什么与不能说明什么)
- [7. 低负载为何变慢与连接复用](#7-低负载为何变慢与连接复用)
- [8. Results：优化前后与复现](#8-results优化前后与复现)
- [9. 节点服务预热：方案、原理与教学模拟](#9-节点服务预热方案原理与教学模拟)
- [10. Results：私有节点预热与固定 OFF 复验](#10-results私有节点预热与固定-off-复验)
- [11. Results：低负载慢在哪里与优化优先级](#11-results低负载慢在哪里与优化优先级)
- [12. Results：长尾复现与阻塞栈定位](#12-results长尾复现与阻塞栈定位)
- [13. Results：同步准备路径移出事件循环](#13-results同步准备路径移出事件循环)

## 1. 比较链路与源码

```text
[1] 共同实验驱动：四个任务编号，最多四个活动尝试
      ├─ OFF：固定旧提交的 science runner
      │     → 静态 Compose 映射 → Harbor 容器 → 直接执行 GPU 命令
      └─ ON：当前 science runner
            → Router 分配持久会话 → Harbor 容器
            → [2] exec_gpu → [3] FIFO 队列和显存预算
            → [4] 执行节点检查容器身份并开启 GPU 权限
            → GPU 命令完成 → 后台撤销权限和确认释放
                                     │
                    CPU 阶段与归还并行 │ 安全确认后释放预算
                                     ▼
                 另一条任务可以申请 GPU，原任务随后重新申请
      两组均执行第二条 GPU 命令 → [5] verifier 检查结果
      OOM → 清理本次容器 → 相同退避规则 → 从头重试整个任务
      完成固定数量或达到预算 → [6] 清理及物理释放核验
```

图例：图里每个部分对应的源码位置：

| 编号 | 职责与源码 |
|---|---|
| 1 | [实验驱动](../env-service-verification/scripts/science_router_control_ab.py) 的 `run` 选择实际导入的 science runner，保持相同并发及补采规则。旧 runner 的实验快照见[基线源码](../env-service-verification/results/router-baseline-control-20261020/baseline-runner.py#L132) |
| 2 | [固定代理](../env-service-verification/scripts/science_router_control_agent.py#L12) 的 `run` 执行两条相同 CUDA 命令，中间执行 CPU 等待。ON 调用 [exec_gpu](../../../psrl/workers/env_service/docker.py#L73) |
| 3 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L47) 检查先入先出队列、预算及实时显存 |
| 4 | [grant_gpu_command / revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L120) 检查容器归属并授予、撤销设备权限。使用[本机 Docker 接口](../../../psrl/workers/env_service/docker_api.py#L9)与 [NVML 查询](../../../psrl/workers/env_service/probe.py#L74) |
| 5 | 驱动的 `prepare_tasks` 写入 verifier，检查两条命令均成功且 CUDA 写入、读回值均为 42 |
| 6 | 驱动的 `run` 保存租约、命令、失败、清理和 GPU 状态，外层启动脚本限制整个进程时间 |

这些源码如果有不容易理解的地方，可以随时追问。

跟着图读源码：先看驱动的 `episode` 如何选择任务及重试，然后看代理 `run` 的两阶段命令。ON 分支从 `exec_gpu` 进入调度器，由节点服务管理权限。第一条命令返回后，CPU 命令与节点归还检查是并行分支，第二条 GPU 命令必须等待前次归还和新授权。最后 verifier 评分，runner 清理容器，实验驱动核对物理资源。OFF 不创建 Router，不执行这套设备授权协议。

## 2. 相同条件与停止边界

| 条件 | 两组设置 |
|---|---|
| 物理资源 | 同一张 H20，约 95.58 GiB 可见显存。模型不占卡 |
| 低压力 | 四个任务编号各执行一次，每条 GPU 命令分配 1 GiB |
| 高压力 | 四个任务编号各成功执行四次，共 16 条，每条 GPU 命令分配 50 GiB |
| 每条任务 | GPU 分配及读写校验、持有 2 秒，CPU 等待 8 秒，再进行同样的 GPU 命令 |
| 成功标准 | 两条 CUDA 命令均成功且 verifier 奖励为 1，无异常 |
| 并发与重试 | 最多四个活动尝试，每个完成位置最多八次尝试。仅 CUDA OOM 重试，按位置固定等待 2 至 4 秒，重试整个任务 |
| ON 预算 | 一个 unit，显存预算 80 GiB，四个容器位置。每条命令申报实际需求加 1 GiB 余量 |
| 计时 | 任务提交到所有任务、评分及 runner 清理完成，包含排队和全部重试。额外列出初始化与最终清理 |
| 运行预算 | 低压力每组任务 180 秒，高压力 480 秒。外层另留初始化及清理时间，超时不扩大规模 |
| 顺序 | 低压力 OFF → ON，高压力 ON → OFF。每种条件一组，没有重复测量置信区间 |

例如高压力下，两个同时分配 50 GiB 的任务需要 100 GiB，超过卡上可用显存，因此静态映射允许竞争时可能 OOM。ON 每条申报 51 GiB，`51 + 51 > 80`，所以同一卡只放行一个 GPU 阶段。容器仍可以同时活着，其他任务的 CPU 阶段不占 GPU 预算。低压力下四条命令只申报 8 GiB，ON 应允许安全的 GPU 并发，不应退化为固定整卡互斥锁。

固定基线使用指定提交加已有运行修复。实验保存了完整 [baseline.patch](../env-service-verification/results/router-baseline-control-20261020/baseline.patch)：runner 传递 GPU 覆盖配置、取消节点级 dangling-image prune，另外有 Ray 地址和 HTTP 事件循环归属修复。补丁没有增加 GPU 锁、动态分配或 Router。当前版本为 `daff7664d284e338d2ad70eb748ad73d46ee3d86` 加保存的工作树改动，不能把两个版本的对照写成只切换一个布尔配置。

## 3. Results：等完成量与开销

四组均正常完成，没有预算终止，没有追加实验或扩大规模。压力组两边都是四个任务编号各成功四次，32 条有效 CUDA 命令，verifier 奖励均为 1。

| 指标 | 低压力 OFF | 低压力 ON | 高压力 OFF | 高压力 ON |
|---|---:|---:|---:|---:|
| 成功任务 | 4 | 4 | 16 | 16 |
| 总尝试数 | 4 | 4 | 31 | 16 |
| CUDA OOM | 0 | 0 | 15 | 0 |
| 任务阶段，含全部重试与清理 | 40.37 秒 | 55.83 秒 | 266.52 秒 | 207.84 秒 |
| 任务提交前的驱动初始化 | 0.21 秒 | 17.98 秒 | 0.56 秒 | 17.26 秒 |
| 驱动记录的总时长 | 41.45 秒 | 76.92 秒 | 269.70 秒 | 228.84 秒 |

驱动总时长从进入实验函数到最终清理核验结束，包含 ON 建立独立 Ray 的成本，但不包含 Python 模块导入和外层环境激活。任务阶段的起止边界两组相同，均从提交固定任务集合到全部评分及 runner 清理完成。它还包含首次节点服务准备，不是已经预热的稳态微基准。

![单卡等完成量、累计完成数量与 OOM 对照](../env-service-verification/results/router-baseline-control-20261020/completion.png)

高压力任务时间减少 `1 - 207.84 / 266.52 ≈ 22.02%`，同等完成量吞吐约为 OFF 的 1.28 倍。加上上表的初始化和最终清理，减少约 15.15%。ON 没有多使用一张环境卡，也没有少做成功任务。

OFF 的 15 次失败都发生在第一条 GPU 命令的显存分配，返回实际 CUDA OOM。两组成功执行的 CUDA 命令都是 32 条，所以本轮收益主要来自避免额外的失败尝试、容器准备、清理和重试等待，而不是减少了已成功 GPU 阶段的重复计算。总尝试耗时包含并发，不能将 15 次失败的时间简单相加当作墙钟节省。

低压力 ON 反而增加约 38.32% 的任务时间。该场景四条 1 GiB 请求本来就放得下，OFF 没有失败成本可供消除，ON 仍需准入、权限检查和归还。任务开始到代理开始执行的平均准备时间为 OFF 6.83 秒、ON 19.05 秒，说明本次差异很大一部分出现在命令之前的准备路径。这一指标涵盖会话准入、节点服务和容器准备，未再拆分各自开销，不能全部归因于 Docker 接口或某个具体子步骤。

## 4. Results：账本与安全回填

高压力 ON 的账本观测如下：

| 指标 | 本次实测 |
|---|---:|
| 同时存活的会话峰值 | 4 |
| 同时活动的 GPU 命令峰值 | 1 |
| 等待 GPU 命令峰值 | 3 |
| 声明显存超预算样本 | 0 |
| 声明显存与活动命令数不一致样本 | 0 |
| 容器仍活着、GPU 配额为零的样本 | 76 |
| 首条命令归还与 CPU 阶段重叠 | 16／16 条成功任务 |
| 命令请求到获准，均值／最大值 | 3.778／14.258 秒 |
| 命令完成到确认归还，均值／最大值 | 0.456／1.195 秒 |

低压力 ON 的活动 GPU 命令峰值为 4，说明此机制没有变成固定整卡互斥锁。高压力只允许一个活动阶段，原因是两份 51 GiB 申报放不进 80 GiB 预算。等待的是当前 GPU 阶段及安全授权，不要求上一条完整任务先结束。

按真实 CUDA 分配时间落入其他任务 CPU 间隔来统计，高压力 ON 观测到 27 对重叠关系，OFF 也有 10 对。这里计数单位是“GPU 阶段与其他 CPU 区间的配对”，不是 27 个独立 GPU 命令。**OFF 确实也能碰巧使用物理空档，区别是没有安全准入，重叠错位时会 OOM。ON 则在确认归还后允许回填。**

![统一时间轴下的显存、预算和会话状态](../env-service-verification/results/router-baseline-control-20261020/resources.png)

上图显存占用不等于计算利用率，命令间复用不等于 CUDA 分配器内部碎片减少。实际使用的显存与申报预算也不必逐毫秒重合，预算覆盖了授权、实际命令和安全归还阶段。

修复后的高压力 ON 共记录 64 次容器检查，平均 1.012 毫秒，最大 2.157 毫秒。GPU 身份查询平均 0.336 秒，实时占用查询平均 0.157 秒。直接 Docker 接口没有再出现前次探针的数秒级检查，但整条控制路径仍有 GPU 查询、锁、远程调用和设备权限管理开销，不能把“容器检查 1 毫秒”说成“获得 GPU 总共只要 1 毫秒”。这次没有额外加入修复前 ON 组，不能把相对 OFF 的全部收益单独归因于直接接口修复。

四组最终均无本次容器、Compose 网络及私有 Ray 进程，目标 GPU 无计算进程、应用显存为 3.44 MiB。物理清理核验通过后结束实验。

## 5. 复现与证据

先按[集群运行约束](../../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)激活环境、核对执行节点和 Docker，确认目标 GPU 空闲。下列变量由操作者显式指定：

| 变量 | 含义 |
|---|---|
| `REPO` | 当前仓库根目录 |
| `BASELINE_REPO` | 指定历史提交及披露运行修复的独立 checkout |
| `VERIFICATION_DEPS` | 项目隔离依赖的包父目录 |
| `RESULT` | 新建、独立的结果目录 |
| `NODE_IP` | 经路由与 Ray 核对的执行节点地址 |
| `GPU` | 现场确认的 GPU index，脚本同时查询 UUID 和设备 minor |
| `IMAGE` | 执行节点已缓存、包含 Python 3 的测试镜像 |

准备静态映射，不启动容器：

```bash
export PYTHONPATH="$REPO:$VERIFICATION_DEPS:$REPO/docs/env-worker-deep-dive/env-service-verification/scripts"
python "$REPO/docs/env-worker-deep-dive/env-service-verification/scripts/science_router_control_ab.py" \
  --prepare --output "$RESULT" --node-ip "$NODE_IP" --gpu "$GPU" --image "$IMAGE"
```

在两次运行之间确认前一组容器和 GPU 已释放。以下为压力组，低压力组使用 `--gib 1 --count 4 --budget 180`：

```bash
for mode in on off; do
  selected="$REPO"
  if [ "$mode" = off ]; then selected="$BASELINE_REPO"; fi
  export PYTHONPATH="$selected:$VERIFICATION_DEPS:$REPO/docs/env-worker-deep-dive/env-service-verification/scripts"
  (
    cd "$selected"
    timeout --signal=INT --kill-after=60 600 python \
      "$REPO/docs/env-worker-deep-dive/env-service-verification/scripts/science_router_control_ab.py" \
      --mode "$mode" --output "$RESULT/pressure-$mode" --overlay "$RESULT/gpu-overlay.yaml" \
      --node-ip "$NODE_IP" --gpu "$GPU" --image "$IMAGE" --gib 50 --count 16 --budget 480
  ) || break
done
```

`--count 16` 是固定成功任务数，不是总尝试数。原始 `result.json` 同时保存失败尝试，失败日志没有删除。任务中的 `sleep 8` 仅代表固定 CPU／规划间隔，不会发起模型请求。图表由下面命令重建，不会启动 GPU 实验：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_router_control_ab.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-baseline-control-20261020
```

证据入口：

- [comparison.json](../env-service-verification/results/router-baseline-control-20261020/comparison.json)：各组统计和实际回填配对。
- [高压力 OFF 全部尝试](../env-service-verification/results/router-baseline-control-20261020/pressure-off/result.json)、[高压力 ON 全部尝试](../env-service-verification/results/router-baseline-control-20261020/pressure-on/result.json)：成功及 OOM 原始记录、计时和实际导入的 runner 路径。
- [低压力 OFF](../env-service-verification/results/router-baseline-control-20261020/low-off/result.json)、[低压力 ON](../env-service-verification/results/router-baseline-control-20261020/low-on/result.json)：保留 ON 更慢的结果。
- [ON 控制查询分段计时](../env-service-verification/results/router-baseline-control-20261020/pressure-on/control-trace.json)、[ON 账本与显存采样](../env-service-verification/results/router-baseline-control-20261020/pressure-on/telemetry.jsonl)。
- [等工作量与账本审计](../env-service-verification/results/router-baseline-control-20261020/audit.json)、[最终物理清理](../env-service-verification/results/router-baseline-control-20261020/physical-cleanup-final.json)、[源码快照 SHA-256](../env-service-verification/results/router-baseline-control-20261020/source-sha256.json)。

新增三个实验与绘图脚本的 Ruff 检查通过，实验结果审计通过。全仓 `pre-commit run --all-files` 在包含当前文件的隔离副本中执行，仍有其他文件的既有格式和风格错误，未改写工作区文件，详见[检查日志](../env-service-verification/results/router-baseline-control-20261020/precommit-all.log)。不能称为全仓检查通过。

## 6. 能说明什么与不能说明什么

本轮能够检验固定工作量下的 OOM、重试、命令阶段排队、物理空档回填及修复后控制查询耗时。低压力组用于暴露无需排队时的额外开销。只有这次实测更快的场景才报告加速，不把安全性直接等同于性能收益。

本轮没有模型推理、训练 loss、entropy、梯度或学习效果，也没有 Docker 与 vLLM 同卡借还。GPU 工作以显存分配和读写校验为主，不能用它宣称真实计算吞吐或 GPU 利用率提高。合成工作负载的两秒 GPU 阶段与八秒 CPU 间隔不是生产 science 任务的时间分布。每组只运行一次，机器负载和 Docker 状态仍可能影响结果。


## 7. 低负载为何变慢与连接复用

先区分两个概念：**GPU 容量竞争**是多个任务的总需求放不下；**调度控制路径的等待**是检查和记账需要协调，暂时轮到其他请求执行。前者不存在，不代表后者自动消失。低压力四条命令各申报 2 GiB，总共 8 GiB，远低于 unit 的 80 GiB 预算，依然需要经过下面的控制路径。

```text
[1] science 容器发起 exec_gpu
         ↓
[2] Router：进入 unit 锁，检查队列及预算
         ↓ 远程调用实际执行节点
[3] 节点服务持有一个 NvmlProbe
         ├─ 首次：初始化 NVML 驱动连接
         └─ 每次：查询当前显存、计算进程、设备身份
         ↓
[4] 检查本机 Docker 容器归属 → 开启设备权限 → 返回授权
         ↓ Router 退出 unit 锁，下一个请求可以完成检查
     容器执行 GPU 命令，多个已获准的命令可以并行
         ↓
[5] GPU 命令结束 → 节点后台撤权并检查物理归还 → Router 释放预算
         └──────── 容器 CPU／规划阶段同时继续
[6] 连接关闭：等待正在进行的查询结束，再释放 NVML 引用
```

图例：图里每个部分对应的源码位置：

| 编号 | 对应函数与职责 |
|---|---|
| 1 | [ManagedDockerEnvironment.exec_gpu](../../../psrl/workers/env_service/docker.py#L73)：请求授权、执行命令并发起归还 |
| 2 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L47)：unit 锁覆盖预算检查、实时查询和授权确认，避免同一个预算被并发超额承诺 |
| 3 | [GpuLendingNodeMixin._gpu_probe](../../../psrl/workers/env_service/lending_node.py#L41) 持有对象；[gpu_lending_snapshot](../../../psrl/workers/env_service/lending_node.py#L52) 将查询交给线程；[NvmlProbe._session](../../../psrl/workers/env_service/probe.py#L52) 只在首次初始化连接；[read_gpu_snapshot](../../../psrl/workers/env_service/probe.py#L85) 每次读取当前占用 |
| 4 | [_command_container](../../../psrl/workers/env_service/command_gpu_node.py#L109)、[grant_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L120)：本机 Docker 身份校验、实时设备映射及设备权限管理 |
| 5 | [revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L131)、[Scheduler._return](../../../psrl/workers/env_service/command_gpu.py#L124)：锁外检查归还，确认后在锁内释放预算 |
| 6 | [NvmlProbe.close](../../../psrl/workers/env_service/probe.py#L59)：与查询共用线程锁，防止查询途中关闭；短期调用可使用上下文管理器，节点对象注册正常进程退出清理 |

这些源码如果有理解上的问题，可以随时追问。

跟着图读源码：

1. `exec_gpu`（图 1）输入命令和申请标识，调用 Router；`acquire`（图 2）输入租约、容器和请求标识，输出 `wait` 或 `granted`。
2. `acquire` 在 unit 锁内调用节点 `gpu_lending_snapshot`（图 3），获得当前 UUID、显存、进程号。NVML 是 NVIDIA Management Library，即 NVIDIA 的设备管理库，用于从驱动读取设备状态；此处不执行模型计算。
3. 预算允许后，`grant_gpu_command`（图 4）检查本机容器，查当前 GPU index 对应的设备 minor，授予访问权限。Router 确认授权后结束本次锁内检查。**锁不覆盖整个 CUDA 命令，因此不等于整卡串行执行。**
4. GPU 命令结束后，`exec_gpu` 可以让后台归还与 CPU 阶段并行（图 5）。`revoke_gpu_command` 先撤权，确认所属进程已无设备句柄／CUDA 使用，再允许 `_return` 把预算释放。
5. 后续请求沿同一路径重新读取实时状态；节点服务继续复用图 3 的连接。图 6 是独立的生命周期清理，正常退出会关闭引用；强制终止进程不能保证运行 Python 退出回调，操作系统仍回收该进程资源。NVML 连接本身不持有任务的 CUDA 显存。

**低负载变慢的三部分原因。** 第一是首次服务启动。历史低压力实验任务准备均值为 OFF 6.83 秒、ON 19.05 秒，相差约 12.22 秒，但其中混合了准入、节点服务、容器准备，不能把差值全部称为 Router 启动成本。本轮增加分段计时，节点服务首次启动分别为优化前 5.76 秒、优化后 5.31 秒，各只发生一次。

独立实验的 ON 还自行建立私有 Ray 集群。Ray 是管理远程进程和调用的分布式运行框架，这一初始化计在任务提交前，历史低压力 ON 为 17.98 秒。真实 PSRL 训练的 OFF 本来也使用 Ray，所以不能把这 17.98 秒再算成生产训练每批新增的 Router 成本。当前 [create_env_router](../../../psrl/workers/env_service/router.py#L407) 可以复用同名服务，[节点服务缓存](../../../psrl/workers/env_service/router.py#L363) 也避免每条命令重新创建服务。

第二是每条命令重复的安全管理。OFF 直接执行；ON 要检查额度、实时占用、容器身份、设备权限，结束后还要确认归还。即使没有容量竞争，这些工作也不会自动变成零。在不需要补采、没有额外并行收益的理想对照里，ON 的控制路径确实有额外成本。只是这些成本可以被复用、摊薄或与其他工作重叠，不能推导为“无竞争就一定显著更慢”。

第三是慢检查放大了其他请求的等待。以本轮优化前第一批第一条 GPU 命令为例，四个并发申请的 Router 调用依次耗时约 **0.420、0.920、1.878、2.586 秒**。最后一个请求并不是等待前一个任务把 GPU 用完，而是前面三个请求正在完成锁内检查。容量明明够，但检查本身太慢，仍会排队。优化后对应四个调用约 **0.254、0.258、0.262、0.265 秒**：首个请求付出连接／设备首次准备成本，后续检查很短。数值来自本轮 trace，包含调用内锁等待，不等于容器从最外层发起命令的全路径耗时。

**已实现的优化。** 原来每次 GPU 身份或占用查询都 `nvmlInit → 查询 → nvmlShutdown`。现在每个执行节点服务持有 `NvmlProbe`，首次连接后持续复用，关闭与查询由同一把线程锁协调。复用的是连接，不缓存显存空闲结论、进程列表、设备句柄或 index/minor 映射；每次仍向驱动查询，失败仍传播，设备撤权与物理归还检查也保留。

只读计时先验证了这个机制：反复建连的显存／进程查询为 141–242 毫秒，保留连接后的三次同类查询为 1.54–1.92 毫秒。首次连接仍耗时约 116 毫秒，首次设备清单查询约 245 毫秒，没有把首次成本从报告中删掉。随后才进行第 8 节的真实容器验证。

**节点预热与进一步优化。** 第 10 节已验证在其他初始化期间预热节点服务，将约 5 秒的首次启动移出第一批任务关键路径；这只有与其他必要工作重叠才减少端到端时间，否则只是移动计时边界。也可继续测量客户端获取容器 ID 和远程调用开销，再决定是否复用已验证的容器信息。目前没有跳过归属校验或放开锁：如果多个请求各自读取同一份剩余额度后同时放行，会重新引入超额承诺，必须先设计正确的预留和失败回滚协议。

## 8. Results：优化前后与复现

本轮只检验“复用 NVML 连接能否降低安全控制开销，并改善低负载任务时间”，不新增训练、模型指标或高压力目标。同一物理 H20、相同镜像和任务配置，优化前后 ON 各运行一次，每次两批、每批四任务，两个批次复用同一个 Router 和节点服务。每条任务仍是 1 GiB CUDA 分配与读写、持有 2 秒、CPU 等待 8 秒、再执行相同 CUDA 命令。每组任务预算 180 秒、整个进程预算 300 秒，完成两批后停止。

| 指标 | 优化前 ON | 优化后 ON |
|---|---:|---:|
| 成功任务／成功 GPU 命令 | 8／16 | 8／16 |
| OOM | 0 | 0 |
| 第一批，含首次节点服务启动 | 53.65 秒 | 48.48 秒 |
| 第二批，复用已启动服务 | 44.83 秒 | 41.87 秒 |
| 两批总任务时间 | 98.48 秒 | 90.36 秒 |
| GPU 身份查询均值，32 次 | 405.3 毫秒 | 3.5 毫秒 |
| 显存／进程查询均值，40 次 | 185.3 毫秒 | 5.3 毫秒 |
| 客户端请求到获准均值，16 次 | 1.216 秒 | 0.077 秒 |
| 命令结束到确认归还均值，16 次 | 0.521 秒 | 0.010 秒 |
| 活动 GPU 命令峰值 | 4 | 4 |

第一批减少 9.63%，第二批减少 6.59%，两批合计减少 8.25%。查询均值包含首次调用和查询内部锁等待，未人为删除最慢的初始化样本。下图右侧使用对数坐标，以便同时显示毫秒和数百毫秒；归还柱采用服务端 `_return` 分段计时，和上表客户端边界稍有不同。

![低负载任务与控制查询优化前后](../env-service-verification/results/router-low-overhead-20261020/low-overhead.png)

任务时间没有按查询时间的倍数缩短，是因为容器创建、CUDA 上下文准备、固定的 8 秒 CPU 间隔、评分和清理仍然存在，且部分查询与其他工作重叠。本轮优化不能消除它们。两组启动到任务提交分别为 22.87 秒、17.07 秒，尚未执行这些 GPU 命令，不能把这个初始化差值归因于命令查询优化，因此性能百分比采用相同任务阶段边界。

**与历史 OFF 的关系。** 历史低压力 OFF 四任务为 40.37 秒；本轮优化后 ON 第二批为 41.87 秒，数值已经接近，但不是同轮成对重测，也没有重复样本和误差区间，不能据此断言精确开销只剩 1.50 秒，更不能声称 ON 已普遍快于 OFF。当前能验证的是查询开销明显下降，以及这一组低负载任务确实完成得更快。没有测量真实训练吞吐、loss 或 entropy。

安全与清理：两组 16 条 CUDA 命令的读回值均为 42，每条成功任务评分为 1；采样预算没有超限，申报显存与活动命令数一致，峰值为四条并发 GPU 命令；结束账本、容器、Compose 网络、私有 Ray 进程和四个记录的服务 PID 均已清零，GPU 无计算进程、应用显存 3.44 MiB。CPU 回归 62 项通过；强化后的连接生命周期测试所在文件再次通过 7 项，覆盖实时状态更新、查询失败传播、身份映射以及查询途中不能关闭连接。全仓 pre-commit 在隔离副本运行，仍因其他文件的既有格式／风格问题失败，不声称全仓通过。

复现入口（GPU 实验需先完成第 5 节的节点预检；先后两组必须等待物理清理，每组只运行一次）：

```bash
# 只读查询实验，不分配 CUDA 显存。
python "$REPO/docs/env-worker-deep-dive/env-service-verification/scripts/profile_nvml_sessions.py" \
  --gpu "$GPU" --output "$RESULT/nvml-session-profile.json"

# 在对应版本及依赖已经固定的进程中执行，before/after 使用不同输出目录。
timeout --signal=INT --kill-after=45 300 python \
  "$REPO/docs/env-worker-deep-dive/env-service-verification/scripts/science_router_control_ab.py" \
  --mode on --gib 1 --count 8 --wave-size 4 --budget 180 \
  --node-ip "$NODE_IP" --gpu "$GPU" --image "$IMAGE" \
  --output "$RESULT/after" --overlay "$RESULT/gpu-overlay.yaml"

# 只读分析与画图，使用已保存的两组结果，无需 GPU。
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_router_low_overhead.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-low-overhead-20261020
```

最后一个脚本会断言相同工作量、CUDA 校验、账本与最终清理数据，并生成表格 JSON 和 PNG／SVG 图。命令中的 `--wave-size 4` 明确建立两批，第二批复用同一服务。若只运行 after 不能构成优化前后对照。

证据：[统计与审计](../env-service-verification/results/router-low-overhead-20261020/comparison.json)、[优化前原始结果](../env-service-verification/results/router-low-overhead-20261020/before/result.json)、[优化后原始结果](../env-service-verification/results/router-low-overhead-20261020/after/result.json)、[连接只读计时](../env-service-verification/results/router-low-overhead-20261020/nvml-session-profile.json)、[最终物理清理](../env-service-verification/results/router-low-overhead-20261020/physical-cleanup-final.json)、[CPU 回归](../env-service-verification/results/router-low-overhead-20261020/cpu-tests.log)、[最终生命周期回归](../env-service-verification/results/router-low-overhead-20261020/cpu-lifetime-final.log)、[全仓检查日志](../env-service-verification/results/router-low-overhead-20261020/precommit-all.log)。当前基础提交仍为 `daff7664d284e338d2ad70eb748ad73d46ee3d86`，本轮变更由 [before-source](../env-service-verification/results/router-low-overhead-20261020/before-source/env_service) 和 [after-source](../env-service-verification/results/router-low-overhead-20261020/after-source/env_service) 的生产源码快照记录；固定 OFF 提交保持不变，本轮未修改或重新运行它。


## 9. 节点服务预热：方案、原理与教学模拟

**状态：生产预热调用已实现，实机结果见第 10 节。** 第 7 节的 NVML 连接复用已经实现，它减少反复初始化的成本；本节解释提前完成第一次初始化，并让它与已有模型准备过程重叠。教学脚本只模拟时间依赖，没有启动 Ray、Docker、模型或 CUDA。

```text
[1] Trainer：worker 放置确定，创建 Router
       ├─ [2] 继续模型初始化、权重加载 ─────────────────┐
       │                                               │
       └─ [3] 异步请求预热指定环境节点        │
                → Router 按节点复用同一个启动任务        │
                → 实际节点启动 EnvNodeService           │
                → 开启服务端口和清理循环                │
                → Docker 探活，首次初始化并查询 NVML    │
                                                        ▼
[4] 首次使用前确认该节点准备成功，检查超时／异常
       → 借用模型 GPU 的 unit 仍要完成原有模型绑定及借还条件
       ↓
[5] 实际任务：创建租约和容器 → 当次实时查询 → 配额准入 → GPU 命令
       → 后台确认归还并释放配额，容器 CPU／规划继续
       ↓
[6] 结束或启动失败：回收本次资源、结束服务和未完成的预热任务
```

图例：图里每个部分对应的源码位置：

| 编号 | 已有实现与拟新增部分 |
|---|---|
| 1 | [init_workers 调用点](../../../psrl/trainer/ppo/ray_trainer.py#L1906)、[_init_env_service](../../../psrl/trainer/ppo/ray_trainer.py#L549)、[create_env_router](../../../psrl/workers/env_service/router.py#L407)：创建或复用 Router，目前只等待 Router 自己响应，不代表全部节点已就绪 |
| 2 | [模型初始化阶段](../../../psrl/trainer/ppo/ray_trainer.py#L2112)：在 Router 创建之后继续进行，因此存在可与节点准备重叠的窗口 |
| 3 | [_start_node](../../../psrl/workers/env_service/router.py#L296)、[_gpu_command_node](../../../psrl/workers/env_service/router.py#L363)、[EnvNodeService.ready](../../../psrl/workers/env_service/node.py#L31)、[NvmlProbe._session](../../../psrl/workers/env_service/probe.py#L52)：已有节点创建、启动任务复用、端口和连接初始化能力；统一入口为 `begin_prewarm`，由 `create_env_router` 提交 |
| 4 | [模型完成后的绑定入口](../../../psrl/trainer/ppo/ray_trainer.py#L2275)、[_init_gpu_lending](../../../psrl/trainer/ppo/ray_trainer.py#L608)、[bind_rollout_gpu](../../../psrl/workers/env_service/router.py#L367)：保留真实绑定和 GPU 身份验证。节点就绪等待不取代这些检查 |
| 5 | [try_admit](../../../psrl/workers/env_service/router.py#L380)、[acquire](../../../psrl/workers/env_service/command_gpu.py#L47)、[_return](../../../psrl/workers/env_service/command_gpu.py#L124)：首次请求复用已就绪节点，实时预算、授权与归还协议照常执行 |
| 6 | [EnvNodeService.cleanup](../../../psrl/workers/env_service/node.py#L68)、[NvmlProbe.close](../../../psrl/workers/env_service/probe.py#L59)、[训练器结束 Router](../../../psrl/trainer/ppo/ray_trainer.py#L596)：已有清理入口；预热任务通过 `shutdown_nodes` 纳入结束和启动失败的回收路径 |

这些源码如果有理解上的问题，可以随时追问。

**跟着图读源码。** 入口 `init_workers` → 图 1 → `_init_env_service`：输入配置与已放置的 worker 组，解析环境允许节点，创建 Router，保存句柄。当前开启预热时会提前提交节点启动，再继续图 2 的模型准备；关闭预热时，普通节点在图 5 首次需要服务时启动。`ready` 返回 URL 后仍不等于 NVML 已初始化，因为该连接是第一次查询时才建立。

图 3 是并行分支：`create_env_router` 调用 `begin_prewarm`，确认 Router 已提交启动任务后返回；主流程继续图 2，不等待节点完成。Router 让同一节点的预热、准入以及 GPU 借用绑定共享同一个 pending 启动任务，避免首请求与预热同时启动两份服务。服务启动后，还需要在**该节点服务对象**持有的 `NvmlProbe` 上做一次查询；在 driver 或临时进程查询再退出，不能给另一个进程留下已经初始化的连接。

图 4 在真正使用前收敛两个并行分支：需要的节点准备完成才继续任务。首批若在预热完成前到达，只等待剩余时间。图 5 的实际命令仍查询当时状态并走完整协议；完成后按图 6 清理。若模型初始化或预热失败，也要收回本次后台任务及已启动服务，不能因为没有创建容器就忽略服务进程。

**准备内容和范围。** `begin_prewarm` 对本轮配置中的环境节点按 IP 去重并提交准备，`wait_prewarm` 提供显式等待；并行启动数由配置限制。每个节点依次完成：服务进程创建和依赖导入、`ready` 的端口及清理循环、命令模式需要的设备控制能力检查、本机 Docker 快速探活、GPU 节点的首次设备身份和状态读取。CPU 环境节点不做 GPU 查询。它不通过创建假任务租约来“碰巧启动”服务，也不提前预留 unit 容量。

这里的“热”表示后续请求需要的进程和驱动接口已经存在。Ray actor 是 Ray 管理的常驻进程对象；异步结果句柄相当于“启动工作已经提交，可以稍后取结果”的凭据。必须先提交再做其他独立工作，若提交后立刻 `ray.get` 等到结束，仍然是串行执行。

节点 `ready` 与 GPU 空闲必须分开。预热时查到的显存／进程只用于检查查询能力，不能当作之后准入时的空闲证明。模型加载期间显存还在变化，借用 vLLM 卡时更不能把预热读数当成模型 sleep 完成后的归还基线。vLLM 是这里负责模型推理的服务；只有它的实际借还协议确认条件满足，Docker 才能获准用卡。现有 `bind_rollout_gpu` 本身已经会创建节点并查询一次 GPU，这类借用配置的节点可能在首批任务前就已经准备过，所以不应假定所有模式都有同样大的预热收益。

**原理：缩短关键路径。** 关键路径是决定最早何时能完成的一串依赖操作。模型准备和环境节点准备在轻量服务预热范围内可以独立进行，任务却依赖两者都完成，因此可以把两个串行等待改成并行等待。

用教学假设说明，所有时间均非本次实机测量：模型准备 `M=20 秒`，节点准备 `W=6 秒`，任务 `E=10 秒`。

| 做法 | 从共同启动时刻到完成 | 第一条任务需要等待节点多久 |
|---|---:|---:|
| 懒启动：模型准备完，任务到了才启动节点 | `M + W + E = 36 秒` | 6 秒 |
| 串行预热：先等节点准备完，再加载模型 | `W + M + E = 36 秒` | 0 秒，但成本只是移到前面 |
| 并行预热：模型与节点同时准备 | `max(M, W) + E = 30 秒` | 0 秒，节点在模型完成前已就绪 |

理想情况下，端到端节省 `min(M, W)`，首请求仍需等待 `max(0, W-M)`。例如模型只需 2 秒而节点需 6 秒，预热后首请求仍等 4 秒；只能隐藏 2 秒。实际同时启动可能争用 CPU、磁盘或进程启动资源，使 `M`、`W` 变化，因此公式是无额外争用时的教学模型，真实收益要测量。

第 8 节已测到首次节点启动约 5.31 秒，说明这段可能被其他准备工作覆盖，这部分可重叠性由第 10 节的实机实验进一步检验。第一批 48.48 秒与第二批 41.87 秒的差也包含容器、系统缓存和运行波动，不能把两者相减后全称为可被预热消除的时间。

**效果与不变项。** 预热主要改善首批等待、新节点首次使用的延迟，以及对坏节点更早发现问题。连接复用已经解决后续请求反复初始化的问题，预热不会再次把每一批都加速 5 秒。它也不会消除每条命令的实时检查、任务容器创建、CUDA 上下文准备、模型推理或计算耗时。镜像提前拉取和创建容器池属于另一项缓存／资源管理设计，不能混进这次“轻量节点服务预热”收益。

**失败与验证。** 实现使用有界启动预算，包含等待并行启动名额和真正准备时间，失败结果保留给后续请求。失败节点不得当成已就绪，也不绕过检查继续放行。预热与首请求共用任务，服务重启后原就绪状态失效；结束时取消或等待本次启动任务并核对资源释放。第 10 节使用小型实验比较同工作量下的启动总时间、首请求等待、实际节点创建次数，以及空账本、GPU 占用和清理结果。三组分别是懒启动、串行提前准备、与独立工作并行准备，避免只改变计时边界就宣称加速。

**可运行教学模拟。** [node_prewarm_demo.py](../env-service-verification/scripts/node_prewarm_demo.py) 使用 Python 标准库 `asyncio` 模拟上图，模型准备 0.20 秒、节点准备 0.06 秒、任务 0.04 秒，执行三个场景约一秒。注释对应图中步骤；`TeachingNode.ready` 复用同一个启动任务，断言每个场景只启动一次，`finally` 负责收回未完成任务。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/node_prewarm_demo.py
```

预期：懒启动与串行预热总时长都约 0.30 秒；并行预热约 0.24 秒。串行预热虽然把首请求等待降到近零，总时间仍不变；只有重叠后总时间才缩短。该脚本不模拟 Ray 调度、Docker、NVML 的真实成本或硬件争用，不能用于预测训练加速比。本轮实际模拟结果：懒启动总时长 0.3006 秒、首请求等服务 0.0602 秒；串行预热总时长 0.3005 秒、首请求等待接近零；并行预热总时长 0.2405 秒、首请求等待接近零。三个场景都只启动一次节点。实际运行记录保存在 [demo.json](../env-service-verification/results/node-prewarm-explanation-20261020/demo.json)，此脚本只验证教学模型，生产节点预热的实机结果另见第 10 节；本轮均未运行训练。

本节教学脚本的 Ruff 检查和格式检查通过，文档本地路径、源码行号范围及章节锚点校验通过。全仓 pre-commit 在隔离副本执行，仍因其他文件的既有风格问题失败，见[检查日志](../env-service-verification/results/node-prewarm-explanation-20261020/precommit-all.log)；未自动改写工作区其他文件。

## 10. Results：私有节点预热与固定 OFF 复验

**结论**：节点预热已接入 `create_env_router`，会先提交配置节点的准备任务，再让训练器继续模型初始化。真实节点服务的准备与固定 8 秒独立准备阶段重叠后，低负载 ON 的“准备＋任务”从懒启动的 54.78 秒降到 51.02 秒，减少 6.85%；串行预热为 54.86 秒，没有总时间收益。低负载 OFF 仍更快，42.56 秒。高压力下四条任务均须完成，OFF 的 6 次 OOM 和重试使其耗时为 111.08 秒，ON 零 OOM、58.03 秒，减少 47.76%。包含各自 Ray／Router 初始化和最终清理的驱动总时间减少 34.21%。

当前完整架构、各层职责、相对基线的优化和源码阅读路径集中维护在 [EnvRouter 当前流程](../knowledge/env-router-current-flow.md)。下面只记录本次实验及其验证边界。

**实现内容。** 配置 `prewarm.enable` 在环境服务启用时默认打开，`timeout_s: 30`、`max_concurrency: 4`。Router 为每个配置节点只创建一个启动任务，准入、预热和借用绑定共享它。首请求取消不取消公共启动；启动超时或失败会销毁刚创建的服务并保留错误。节点准备包括本机 Docker 探活、首次 GPU 身份查询及其自身 NVML 连接初始化，不占用任务租约或显存配额。正常结束和模型初始化失败路径均接入环境服务停止入口。`shutdown_nodes` 不允许直接丢弃活动租约，空闲时取消未完成准备，逐个关闭并回收服务，某个关闭失败也继续回收其余服务。

**实验条件与停止规则。** 使用经现场检查的私有节点，同一张 H20、同一测试镜像、GPU index／UUID／实际设备 minor 已现场映射，普通容器和 1 GiB CUDA 读写预检通过。低负载四组、高压力两组，每组四个任务各成功一次；每条任务执行两次相同 CUDA 操作，均持有 2 秒，中间 CPU 等待 8 秒。低负载每次分配 1 GiB、申报 2 GiB；高压力每次分配 50 GiB、申报 51 GiB；ON 的预算为 80 GiB、最多四个存活容器。OFF 使用固定提交加披露运行修复，仍是静态映射、进程内并发四，无整卡锁。

所有组都创建相同规格的独立 Ray，且在任务前执行相同的 8 秒独立准备等待。这个等待代表可与节点服务准备并行的模型加载窗口，**不是实际加载模型**。本次真实执行的是 science runner、Harbor、Docker 和 CUDA，未运行 LLM 或优化器。每组任务预算 240 秒、外层进程预算 360 秒，每个任务最多八次尝试，仅 CUDA OOM 重试；整组启动脚本另有 1200 秒总上限。顺序为低负载 OFF、ON 懒启动、ON 并行预热、ON 串行预热、高压力 ON、高压力 OFF。六组达到固定成功数量后停止，没有重跑追求更漂亮的数字。

| 组别 | 准备＋任务，含重试 | 仅任务阶段 | 驱动总时间 | 尝试／成功／OOM |
|---|---:|---:|---:|---:|
| 低负载 OFF | 42.56 秒 | 34.56 秒 | 56.67 秒 | 4／4／0 |
| 低负载 ON 懒启动 | 54.78 秒 | 46.77 秒 | 72.28 秒 | 4／4／0 |
| 低负载 ON 串行预热 | 54.86 秒 | 38.38 秒 | 70.79 秒 | 4／4／0 |
| 低负载 ON 并行预热 | 51.02 秒 | 43.00 秒 | 66.63 秒 | 4／4／0 |
| 高压力 OFF | 111.08 秒 | 103.08 秒 | 127.34 秒 | 10／4／6 |
| 高压力 ON 并行预热 | 58.03 秒 | 50.00 秒 | 83.78 秒 | 4／4／0 |

“准备＋任务”从独立 Ray 和 Router（ON）就绪后，开始共同准备阶段计时，到四条任务评分及 runner 清理完成；其中包含实际节点服务启动。驱动总时间进一步包含 Ray／Router 初始化和最终清理，不含外层环境激活和进入驱动函数前的 Python 模块导入。两种边界同时报告，不能将被移到任务计时之前的串行预热成本隐藏掉。

![预热方式、低负载与高压力等完成量对照](../env-service-verification/results/router-prewarm-private-20261020/prewarm-comparison.png)

上图右上角使用真实 `_start_node` trace，蓝色为节点准备，灰色为固定独立准备阶段。ON 懒启动在灰色阶段结束、任务开始后才准备节点，耗时 5.48 秒；ON 并行预热的 5.42 秒节点准备落在 8 秒独立阶段内。串行预热先准备节点 8.48 秒，再等待共同的 8 秒，任务阶段虽然更短，总时间没有改善。不同组启动耗时有抖动，因此 6.85% 是本轮观察值，没有多次重复测量的置信区间，不能把全部时间差视为稳定的纯预热收益。

高压力 OFF 的六次 OOM 均发生在第一条 GPU 命令分配显存时。两边成功的 CUDA 命令都是八条，因此优势来自避免失败尝试的准备、失败处理、清理和重试等待，没有通过少完成任务或不计失败来缩短时间。也不能把并发失败的耗时逐条相加当作墙钟节省。

**账本与归还证据。** 高压力 ON 同时活动 GPU 命令峰值为 1、等待峰值为 3；三种低负载 ON 的活动峰值均为 4，说明预算足够时没有退化成整卡串行锁。各组采样申报均等于“活动命令数 × 单命令需求”，无超预算样本。高压力 ON 请求到授权均值 2.418 秒、最大 9.220 秒，主要包含真实的额度排队；命令结束到确认归还均值 7.28 毫秒。四条任务的首次归还都与自己的 CPU 阶段重叠，另有 12 对 GPU 分配与其他任务 CPU 区间的重叠关系。配对数不是独立命令数。

![统一时间轴下的显存与命令预算](../env-service-verification/results/router-prewarm-private-20261020/resource-timeline.png)

预算在上图 ON 的 GPU 工作密集阶段看起来持续为 51 GiB，是总量采样中相邻命令快速交接的结果，不代表同一条任务一直保留预算。归还及下一个申请之间的短暂归零可能小于约 0.5 秒采样间隔；应结合命令归还记录和请求身份判断。图中显存占用不是计算利用率，也不能证明 CUDA 内部碎片减少。

**低负载的剩余开销。** 并行预热后 ON 比本轮 OFF 的“准备＋任务”仍高约 19.87%。预热只覆盖节点首次准备，任务会话准入、容器配置、授权与归还等路径仍比固定 OFF 多。机器和任务准备波动也存在，不能把剩余 8.46 秒全部归因于某个未计时子步骤。它说明当前方案有适用条件，不能由高压力优势推导为所有负载都更快。

**回归与清理。** 针对节点、命令协议、借用和 science runner 的 67 项 CPU 回归通过；补充关闭异常及工厂提交行为后，最终预热测试文件 9 项通过。后者覆盖共享启动、取消隔离、超时、失败、关闭继续回收、活动租约保护及“只等待提交确认、不等待节点完成”的生产入口行为。未执行完整训练初始化集成验证。修改模块的针对性 Ruff 检查通过，全仓 pre-commit 在隔离副本仍因既有风格问题失败。最终按六组 actor／Compose project 核验无容器、无网络、无私有 Ray 进程；八个记录的 Router／节点服务 PID 均已退出，GPU 无计算进程，应用显存为 0.19 MiB。

复现选项见 [science_router_control_ab.py](../env-service-verification/scripts/science_router_control_ab.py)。沿用第 5 节的环境与节点预检，先准备新节点自己的 GPU overlay，再对各组固定导入路径运行：

```bash
# ON 三种模式分别使用 lazy、serial、overlap，并使用独立输出目录。
timeout --signal=INT --kill-after=45 360 python \
  "$REPO/docs/env-worker-deep-dive/env-service-verification/scripts/science_router_control_ab.py" \
  --mode on --gib 1 --count 4 --budget 240 --prewarm overlap \
  --preparation-seconds 8 --ray-for-off --node-ip "$NODE_IP" --gpu "$GPU" \
  --image "$IMAGE" --overlay "$RESULT/gpu-overlay.yaml" --output "$RESULT/low-on-overlap"
```

OFF 必须在指定历史 checkout 中执行同一驱动，并将 `PYTHONPATH` 的仓库入口切到该 checkout；使用 `--mode off --prewarm lazy`。压力组两边都改为 `--gib 50`，保持四条成功任务、同样的重试策略和预算。在下一组之前核验前一组物理资源释放，不使用当前代码关闭 Router 来代替历史基线。画图及只读审计命令为：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_router_prewarm_ab.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020
```

证据入口：[六组统计](../env-service-verification/results/router-prewarm-private-20261020/comparison.json)、[完整启动记录](../env-service-verification/results/router-prewarm-private-20261020/suite.log)、[OFF 全部压力尝试](../env-service-verification/results/router-prewarm-private-20261020/pressure-off/result.json)、[ON 压力尝试](../env-service-verification/results/router-prewarm-private-20261020/pressure-on/result.json)、[并行预热 trace](../env-service-verification/results/router-prewarm-private-20261020/low-on-overlap/control-trace.json)、[CUDA 预检](../env-service-verification/results/router-prewarm-private-20261020/cuda-preflight.json)、[最终物理清理](../env-service-verification/results/router-prewarm-private-20261020/physical-cleanup-final.json)、[67 项回归](../env-service-verification/results/router-prewarm-private-20261020/cpu-tests.log)、[最终 9 项预热测试](../env-service-verification/results/router-prewarm-private-20261020/startup-tests-final.log)、[全仓检查](../env-service-verification/results/router-prewarm-private-20261020/precommit-all.log)、[代码摘要](../env-service-verification/results/router-prewarm-private-20261020/source-sha256.json)、[基线运行补丁](../env-service-verification/results/router-prewarm-private-20261020/baseline.patch)。

## 11. Results：低负载慢在哪里与优化优先级

**核心结论**：重新分析第 10 节保存的时间线后，可以把成本分成首次准备、会话准备、逐命令控制和实际执行长尾。ON 并行预热比 OFF 多出的 8.46 秒不能全部解释为 Router 计算或排队。该组最后完成的任务，第一条 GPU 调用花了 9.762 秒，其中申请到获准只有 0.039 秒，获准后到容器脚本开始为 3.922 秒，脚本内部为 5.532 秒。已有数据定位到了延迟区间，但没有定位 Docker 或驱动内部的具体根因。本节只分析已有真实运行，没有重新启动 GPU 实验。

**先把两条路径连起来。** 图中的编号用于下方源码表和分析脚本注释。OFF 指固定提交，不是当前关闭服务的分支。

```text
OFF：
[1] science runner → 静态 GPU 配置 → Harbor 创建容器
  → [3] 直接执行容器命令 → CUDA → CPU 间隔 → 再执行 CUDA
  → [5] 评分、容器和任务清理 → 完成

ON：
[0] Router 与节点服务准备，节点准备可与独立初始化重叠
  → [1] 会话申请 → 准备租约配置 → Harbor 创建容器
  → [2] 等前次归还、取 Router 句柄、Compose 查询容器 ID
  → [3] Router 检查额度 → 节点校验与授权 → 容器命令 → CUDA
  → [4] 原任务继续 CPU 间隔，同时后台确认归还并释放额度
  → [2] 再次需要 GPU 时重复查询与准入
  → [5] 等待必要确认 → 评分、容器和会话清理 → 完成
```

图例：图里每个部分对应的源码位置：

| 编号 | 真实调用、输入和状态 |
|---|---|
| 0 | [create_env_router](../../../psrl/workers/env_service/router.py#L407) 提交预热，[Node.prepare](../../../psrl/workers/env_service/node.py#L49) 探活并初始化查询能力，不占任务配额 |
| 1 | [固定基线 runner 快照](../env-service-verification/results/router-prewarm-private-20261020/baseline-runner.py#L191) 静态映射；ON 经 [Client.acquire](../../../psrl/workers/env_service/client.py#L159) 申请会话，再由 [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L98) 准备设备、驱动库和 Compose 覆盖配置 |
| 2 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73) 在记录 `reservation.requested_at` 前执行归还等待、获取句柄及 `compose ps -q main`；[_docker_compose_paths](../../../psrl/workers/env_service/docker.py#L157) 每次访问会读配置、生成归属文件 |
| 3 | [acquire](../../../psrl/workers/env_service/command_gpu.py#L47) 校验预算并等待节点授权；[exec_gpu 调用父类执行](../../../psrl/workers/env_service/docker.py#L108) 后运行 [CUDA 脚本](../env-service-verification/scripts/cuda_pressure.py#L10) |
| 4 | [begin_return](../../../psrl/workers/env_service/command_gpu.py#L109) 交接后台任务，[_return](../../../psrl/workers/env_service/command_gpu.py#L124) 确认撤权及物理状态后提交配额释放 |
| 5 | [stop](../../../psrl/workers/env_service/docker.py#L126) 汇合未完成的归还，随后清理容器；[runner](../../../examples/sciaccel_rl/runner.py#L507) 收敛归还错误并处理评分结果 |

这些源码如果有不容易理解的地方，可以随时追问。Harbor 内部 Docker 子进程、daemon 请求和 CUDA 每个驱动调用没有独立计时，图中不能将这些未观测步骤的时间强行拆分。

**为什么没有竞争时需要付出成本。** 低负载四条任务各申报 2 GiB，累计 8 GiB，小于 unit 的 80 GiB 预算，实际允许四条 GPU 命令并行。OFF 本轮没有 OOM，因此 ON 没有可节省的失败重试时间，但仍要确认容器归属、额度和归还。这些额外工作有实际用途，只是不保证在每种负载下都带来净时间收益。这个例子也排除了“因为 unit 只让一条任务运行”作为本轮低负载主因。

**测到的成本与不能声称的结论。** 下表的 ON 是并行预热组，每条命令的统计覆盖八次调用。均值不代表关键路径，各行也不能随意相加。

| 环节 | 本轮证据 | 判断与优化空间 |
|---|---|---|
| 首次节点启动 | 懒启动 5.481 秒；并行预热 5.415 秒，落在共同的 8 秒准备窗口内 | 已覆盖本轮并行 ON 的这部分等待，不应继续把它算作剩余 8.46 秒的原因。Router 自身创建仍在驱动初始化边界内 |
| 任务开始到代理开始 | OFF 平均 6.694 秒，ON 8.002 秒 | ON 多 1.308 秒，但此区间混有遥测、会话准入、租约配置、Harbor 准备及文件操作。没有逐步 trace，不能全部归因于账本 |
| Harbor 容器环境准备 | OFF 平均 5.491 秒，ON 5.205 秒 | 本轮没有证据表明 ON 创建容器更慢。此行嵌套在上一行，不能再相加 |
| GPU 申请前准备 | ON 平均 0.202 秒，最大 0.232 秒 | 包含查询容器 ID 等步骤，是源码可确认的重复工作。该区间整体有计时，但没有独立证明 202 毫秒全部属于 `compose ps` |
| 申请到获准 | ON 平均 0.0144 秒，最大 0.0390 秒 | 包含 Ray 跨进程调用、短锁等待、预算检查和节点授权，不等于纯显存排队。它解释不了数秒长尾 |
| 节点 Docker inspect | 平均 0.915 毫秒 | 直接访问执行节点本机接口已经很快，不是本轮优先瓶颈 |
| 命令结束到确认归还 | 平均 9.884 毫秒 | 第一段与 CPU 工作重叠，不能把所有归还耗时都加到墙钟时间上 |
| 代理结束到任务结束 | OFF 平均 13.727 秒，ON 13.903 秒 | 两边都有明显收尾成本，本轮差值仅 0.176 秒。该区间包含评分、日志、清理等，不能全叫“Docker 删除耗时” |
| 获准后的执行阶段 | ON 的任务 1 第一条命令出现显著长尾，细分见下表 | 是本轮批次尾部的主要延迟位置。Docker 交付、进程调度、文件系统、CUDA 同卡执行等只能列为候选，不能当作已确认原因 |

NVML 是 NVIDIA 的 GPU 管理查询接口。当前节点复用连接，但每次仍读取实时占用。它和直接 Docker inspect 的优化已经落地，不能再用早期慢查询的旧数字解释当前结果。

**跟着一个真实长尾任务看。** 并行预热 ON 的任务 1 第一条 GPU 调用，以下区间互不重叠：

| 时间段 | 实测耗时 | 包含什么 |
|---|---:|---|
| 调用开始到发起额度请求 | 0.219 秒 | 获取句柄、查容器等客户端准备 |
| 发起请求到获准 | 0.039 秒 | Router／节点控制协议 |
| 获准到 CUDA 脚本记录开始 | 3.922 秒 | 父类容器执行、进程启动到脚本计时点，内部没有更细计时 |
| 脚本开始到分配成功 | 0.585 秒 | 加载驱动、建立上下文、设备查询及分配 |
| 分配成功到脚本结束 | 4.946 秒 | CUDA 操作、同步、读回、配置的 2 秒等待、显存释放及上下文销毁 |
| 脚本结束到调用返回 | 0.050 秒 | 写日志／输出、命令退出及返回交接 |
| 合计 | **9.762 秒** | 与外层调用计时一致 |

特别注意，4.946 秒不能直接叫“GPU 计算耗时”，也不能断定是等待睡过头。脚本没有把 CUDA 同步、2 秒等待、上下文销毁分别计时。归还确认发生在命令结束后，无法解释获准后、脚本开始前的 3.922 秒延迟。当前证据也不足以排除 ON 的配置或执行方式间接影响长尾。

整个批次必须等最后一个任务完成，因此不能把四条并行任务的均值或耗时总和当作批次墙钟时间。挑出各组最后完成的任务，连续阶段如下，另有不足 10 毫秒的提交偏移和最终记录间隔：

| 最后完成的任务 | 代理前准备 | 代理执行，含两条 GPU 命令及 CPU 间隔 | 代理后收尾 | 从该任务提交到结束 |
|---|---:|---:|---:|---:|
| OFF 任务 0 | 7.497 秒 | 13.450 秒 | 13.601 秒 | 34.548 秒 |
| ON 任务 1 | 8.238 秒 | 21.270 秒 | 13.478 秒 | 42.986 秒 |

这是描述性关键路径拆分，不是让两条不同任务组成严格的因果实验。ON 其余三条任务约在批次开始后 37.81、38.09、38.21 秒结束，任务 1 在 42.99 秒结束。最后约 4.78 秒是这条任务独自拖住批次的尾部，不能据此宣称某项优化必定节省 4.78 秒。

![低负载关键路径与长尾命令拆分](../env-service-verification/results/router-prewarm-private-20261020/low-load-breakdown.png)

上图只画任务阶段，没有重复画两边共同的 8 秒准备。图中 `pre_agent` 是代理前准备，`agent` 是代理执行，`post_agent` 是代理后收尾。第二幅图的 `grant_to_script` 在 OFF 对应直接调用到脚本开始，因为 OFF 没有授权步骤。

**建议优化顺序，以下均为待实现或待验证方案。**

1. **先定位获准后的长尾。** 给 `super().exec` 内的命令构造、子进程启动／退出，以及脚本的 CUDA 初始化、分配、同步、等待、销毁增加少量分段计时。只有再出现该问题时，才能区分控制线程阻塞、Docker 服务延迟、进程启动和驱动等待。已经确认毫秒级的账本路径不应先被大幅重写。
2. **减少逐命令容器发现。** 容器启动后保存不可变的容器 ID，在同一容器生命周期内复用，重建或消失时失效；节点每次授权仍检查标签、状态和归属。这样有机会消除大部分 0.202 秒申请前准备，但不能预先承诺全部消除。两条 GPU 命令对应每任务约 0.40 秒的该区间，四条任务并发时不能直接乘四当成墙钟收益。
3. **避免重复解析和写 Compose 配置。** 当前 `_docker_compose_paths` 每次访问都读取 YAML 并重写归属文件。可按容器配置生命周期生成，服务集合或租约变化时明确失效。源码能确认重复工作存在，现有 trace 没有测出它独立占多少时间。
4. **复用轻量控制对象与合并固定配置查询。** 可按生命周期复用 Router 句柄，缓存与设备／驱动版本绑定的静态解析结果。实时显存、容器存活和归还结果必须继续查询。先拆分会话准备的 1.308 秒平均差值再确定收益，不把未计时步骤都当成缓存优化收益。
5. **高频短命令再考虑有界批量授权。** 若一段确定的 GPU 工作由多个极短命令组成，可以在同一次已批准的预算内完成后统一归还。教学例子：十条 20 毫秒命令，如果每条额外需要 200 毫秒准备，200 毫秒实际工作可能伴随 2 秒控制成本。批量执行可以摊薄这些成本，但额度持有会延长，必须有明确时间／命令数上限，长 CPU 间隔仍需归还。不能重新退化为整条任务长期占额，也不能因为当前空闲就跳过准入。

先入先出等待目前采用轮询，请求被拒绝时等 0.1 秒再问，可以考虑改为额度归还后通知等待者。但本轮低负载的申请到获准最大不足 0.1 秒，没有跨过一次完整的 100 毫秒退避，轮询并非这组数秒差距的解释。

**跟着图读源码。** 从 [代理 run](../env-service-verification/scripts/science_router_control_agent.py#L12) 观察 OFF 的 `exec` 与 ON 的 `exec_gpu` 分支；沿 [1] 先看 `Client.acquire` 和 `_apply_env_lease`，再沿 [2] 看 `reservation.requested_at` 为什么不含容器发现；沿 [3] 看 `acquire` 返回后才执行父类容器命令；沿 [4] 看第一次归还与 CPU 命令并发、下一次 GPU 命令等待前次错误收敛；最后沿 [5] 看 `stop` 和 runner 收尾。源码跳转位置见图例，不能把这条并发路径当作所有阶段简单串行相加。

**可运行的只读复核。** [analyze_low_load_path.py](../env-service-verification/scripts/analyze_low_load_path.py) 使用保存的真实运行时间戳，不是教学模拟，也不启动 Ray、Docker、GPU 或训练。默认仅依赖 Python 标准库，`--plot` 额外使用 matplotlib 生成上图。运行前按项目约定激活环境，在仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_low_load_path.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020

# 可选：同时重建 PNG／SVG。
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_low_load_path.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020 --plot
```

本次已运行，检查每组四条成功任务、四份 Harbor 记录、命令区间非负且相加等于外层调用时间，输出 [low-load-breakdown.json](../env-service-verification/results/router-prewarm-private-20261020/low-load-breakdown.json)。预期可看到 OFF 任务阶段约 34.557 秒、ON 并行组约 42.999 秒，以及 ON 最后任务第一条命令 0.039 秒授权、3.922 秒授权后启动、5.532 秒脚本执行的分解。单次观察不提供稳定平均性能或明确的系统级根因，新增的代码仅用于离线分析，未修改生产调度行为。

## 12. Results：长尾复现与阻塞栈定位

**核心结论**：原记录的“获准后 3.922 秒才进入脚本，同时脚本执行 5.532 秒”组合没有完整重现。新增三组共 24 条成功任务、48 条 CUDA 命令定位了两类真实成本：宿主事件循环会被同步 NVML／Docker 探测以及子进程创建阻塞；CUDA 脚本内部的较长耗时主要落在驱动初始化、上下文创建和销毁。本轮 OFF 也出现了 4.878 秒脚本时长，因此不能把这类耗时都归因于 Router。新增证据不能替代旧任务缺失的调用栈，旧 9.762 秒调用的完整根因仍未证实。

**先看计时和阻塞如何对应。** 事件循环是一条线程轮流推进多个异步任务的调度机制。某一步使用 `await` 不代表它内部所有工作都不阻塞，普通同步驱动调用、同步子进程等待，以及异步进程接口里的实际创建步骤，都可能暂时占住这条线程。

```text
宿主事件循环线程
  ├─ [1] A 准备会话：_apply_env_lease → build_lease_overlay
  │        → 同步 NVML 初始化、设备枚举、关闭
  ├─ [2] Harbor 准备环境：同步 Docker 模式／内核探测
  ├─ [3] B 已获 GPU 额度：准备并启动 Docker 子进程
  │        → asyncio 接口内部仍经过同步 _fork_exec
  └─ [4] 处理其他任务的就绪回调、命令结果、心跳
       上面任一步同步阻塞时，同线程的其他协程暂时不能推进

Docker 容器内独立进程
  [5] 加载 libcuda → cuInit → cuCtxCreate → 分配／读写
      → 等待 2 秒 → 释放显存 → cuCtxDestroy → 返回
  这些驱动操作有自己的耗时，不能与宿主事件循环延迟混为一项
```

图例：图里每个部分对应的源码位置：

| 编号 | 实现、观测和边界 |
|---|---|
| 1 | 本节实验快照中的 `_apply_env_lease` 在异步函数中直接调用 [build_lease_overlay](../../../psrl/workers/env_service/overlay.py#L180)，后者进入 [read_gpu_devices](../../../psrl/workers/env_service/probe.py#L104)。实际采样栈覆盖 NVML 初始化、设备句柄查询及关闭 |
| 2 | 实际使用的 [Harbor Docker 源码快照](../env-service-verification/results/router-tail-profile-20261020/source/harbor-docker.py#L125) 的 `_detect_daemon_os` 和 [_egress_control_kernel_support](../env-service-verification/results/router-tail-profile-20261020/source/harbor-docker.py#L834) 使用同步 `subprocess.run`；内核探测已有函数级缓存，不能说每条命令都重复执行它 |
| 3 | [Harbor 命令启动](../env-service-verification/results/router-tail-profile-20261020/source/harbor-docker.py#L606) 和 [_is_rootless_docker](../env-service-verification/results/router-tail-profile-20261020/source/harbor-docker.py#L985) 调用 `asyncio.create_subprocess_exec`，实际线程栈停在 Python 的 `_execute_child → _fork_exec`。后者是外部 Python 运行时实现，不是 Router 锁 |
| 4 | [profile_science_tail](../env-service-verification/scripts/profile_science_tail.py#L92) 每 50 毫秒检查事件循环推进，独立线程对超过 300 毫秒的停顿采集调用栈。栈只含文件、行号和函数，不读取局部变量 |
| 5 | [cuda_timing_probe](../env-service-verification/scripts/cuda_timing_probe.py#L26) 保留原来的显存分配和读回工作，逐个驱动调用记录单调时钟、进程 CPU 时间及容器限流计数 |

这些源码如果有不容易理解的地方，可以随时追问。图里的 A／B 同时活动是机制示意，不是旧任务的逐帧重放。例如 A 的同步准备若堵住事件循环两秒，即使 B 的 GPU 额度已够，B 也可能迟迟不能执行下一个 Python 步骤。实测堵塞主要出现在本轮任务准备阶段，不能据此直接认定旧任务的 3.922 秒完全来自 A。

**实验条件与预算。** 使用第 10 节同一私有节点、同一 GPU UUID、同一镜像。现场重新检查 Docker、GPU index／minor 映射、导入来源及 1 GiB CUDA 读回，其他 GPU 上的任务未改动。ON 与固定 baseline OFF 各两批，每批四条任务，每条仍为两次 1 GiB 分配、各持有 2 秒，中间 8 秒 CPU 等待。每组任务预算 240 秒、外层 360 秒。ON 第一组出现准备阶段的 6.639 秒事件循环停顿后，追加且仅追加一组相同规模的 ON 栈采样，预算相同。随后只做了八对 Docker 客户端 `--version` 的 CPU 路径检查，未再启动 GPU 实验。

所有有效组均一次成功，没有 OOM。最初一轮计时包装误把 Harbor 静态方法当作实例方法，导致容器准备失败，未产生有效代理／CUDA 数据；已保留在 [instrumentation-failed](../env-service-verification/results/router-tail-profile-20261020/instrumentation-failed/result.json)，修正的是诊断包装，不是生产逻辑。生产环境服务源码与第 10 节快照逐文件一致，本次提交标识及工作树补丁另行保存，未修改共享 Harbor 安装或生产调度策略。

| 观察值 | ON，16 条命令 | 固定 OFF，16 条命令 | ON 栈采样，16 条命令 |
|---|---:|---:|---:|
| 两批任务墙钟时间 | 39.53／44.22 秒 | 36.70／39.53 秒 | 40.88／38.00 秒 |
| 获准后到脚本开始，最大 | 0.561 秒 | 无授权，直接调用到脚本开始最大 0.641 秒 | 0.293 秒 |
| 脚本执行，最大 | 3.198 秒 | 4.878 秒 | 3.769 秒 |
| 请求到 GPU 获准，最大 | 39.6 毫秒 | 无此协议 | 49.6 毫秒 |
| 任务期间事件循环停顿，最大 | 6.639 秒 | 2.694 秒 | 1.705 秒 |
| 记录的容器 CPU 限流计数增量 | 0 | 0 | 0 |

这不是新的稳定吞吐结论。三组包含不同诊断开销、顺序和运行时波动，没有重复试验置信区间。主机 Ray 初始化和最终退出也会阻塞事件循环，上表已排除这两个阶段，不能把初始化时十几秒的停顿拿来解释任务执行。

**CUDA 侧复现到了什么。** OFF 第二批任务 4 的第一段脚本用时 4.878 秒，其中主要组成如下：

| 实际操作 | 耗时 | 含义 |
|---|---:|---|
| `cuInit` | 0.732 秒 | 初始化进程使用 CUDA 驱动的能力 |
| `cuCtxCreate_v2` | 1.071 秒 | 建立该进程的 GPU 执行上下文，即驱动管理的执行状态和资源环境 |
| 配置的等待 | 2.001 秒 | 与设定的 2 秒一致，没有出现秒级睡眠超时 |
| `cuMemFree_v2` | 0.037 秒 | 释放本次分配 |
| `cuCtxDestroy_v2` | 1.031 秒 | 销毁上下文并回收关联状态 |
| 其他记录操作及微小间隔 | 约 0.006 秒 | 库加载、设备查询、分配、读写校验和计时开销 |

显存分配本身约 0.8 毫秒、同步约 0.13 毫秒。慢的是上下文生命周期操作，不是本次一小段显存读写，也不是 OFF 不存在的 GPU 配额队列。`cuCtxCreate` 的约 1.071 秒内，进程 CPU 时间约 0.342 秒，说明墙钟时间并非全是进程消耗 CPU。驱动内部的锁等待、跨进程干扰等没有进一步计时，不能把某一种猜测当成已经证明的内部根因。

全部 48 条命令同时记录墙钟和单调时钟，两者的脚本时长差小于 7 微秒；CPU 限流计数增量均为零。它们排除了本轮已记录脚本时段的秒级系统时钟跳变和可见 CPU 配额限流，不能反推旧任务当时也必然没有这些问题。原始较长 CUDA 操作已在脚本返回前完成，后续归还协议无法解释这些调用内部的耗时。

**宿主侧捕获到了什么。** ON 栈采样组在任务窗口采到 37 份停顿栈：11 份位于 NVML，9 份位于 Harbor 同步 Docker 探测，15 份位于创建子进程，2 份位于文件操作／路径解析。采样份数不是独立故障次数，也不能乘采样周期当作精确耗时分摊。原始栈见 [execution-trace.json](../env-service-verification/results/router-tail-profile-20261020/on-stack/execution-trace.json)。

其中一条完整链路是：

```text
run_harbor_episode
  → _apply_env_lease
  → build_lease_overlay
  → 短生命周期 read_gpu_devices / NvmlProbe
  → nvmlInitWithFlags / nvmlDeviceGetHandleByIndex / nvmlShutdown
```

这说明节点服务复用 NVML 连接的优化没有覆盖所有调用者：runner 构造租约 overlay 时仍走短生命周期查询，而且在事件循环线程同步执行。这是已经有现场栈证据的阻塞路径。节点服务本身的快速查询不能代表客户端所有 NVML 调用都已优化。

另一条链路为 `_chown_to_host_user → _is_rootless_docker → asyncio.create_subprocess_exec → _execute_child → _fork_exec`。因此一个标成“启动 Docker 子进程”的长区间，可能发生在 Docker daemon 接到请求之前，不能仅凭它就判定 daemon 慢。本轮 CPU 路径小检查中，`docker --version` 的进程启动平均 1.92 毫秒、最大 2.54 毫秒，使用解析后的 `/usr/bin/docker` 平均 0.64 毫秒、最大 0.70 毫秒。这个小差异没有重现秒级停顿，不能认定 PATH 查找就是长尾根因。

**可确定的修复方向与尚未确定的部分。** 应优先将 runner 的设备枚举与 overlay 准备从事件循环线程移走，保持身份新鲜度与异常传播；再处理 Harbor 同步探测，减少反复创建命令行进程。不能仅用“函数已经是 async”作为没有阻塞的证明。若复用节点服务的静态设备身份，仍需明确设备变化和服务重启时的失效规则，不能缓存实时显存空闲结论。CUDA 上下文创建和销毁是另一条成本链，进程常驻或批量执行可能摊薄它，但必须兼容“没有遗留设备句柄才允许归还”的协议，不能为了提速让容器长期私自持有上下文。

第 12 节这轮只完成诊断与复现，后续实现见第 13 节，当时没有宣布上述方向已经实现，也没有把新增准备阶段的阻塞等同于旧 GPU 调用的完整根因。后续修复应分别验收事件循环能否继续推进、原任务是否正常完成和实际资源能否归还，而不只看总时间是否碰巧下降。

**跟着图读源码与只读复核。** 先沿 [1] 阅读 runner 到 overlay、短生命周期 probe 的同步调用，再沿 [2] 阅读实际 Harbor 快照中的 `subprocess.run`。沿 [3] 看宿主命令启动，结合 [4] 的独立线程采样，避免把单线程阻塞误当作远端等待。最后沿 [5] 按每个 CUDA 调用的单调时钟区间阅读，返回到 `exec_gpu` 后才进入归还协议。批次结束先检查评分和命令成功，再检查容器、网络、actor 和物理 GPU 状态。

在仓库根目录、项目环境中执行下列只读分析，不重新占用 GPU：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_science_tail.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-tail-profile-20261020 \
  --arms on off on-stack
```

本次已运行，预期得到三组批次时间、逐 CUDA 调用统计、事件循环停顿和原始栈分类。[profile_science_tail.py](../env-service-verification/scripts/profile_science_tail.py) 是真实任务埋点入口，沿第 10 节命令的节点、镜像与 overlay 参数使用，`--stack-sampling` 开启独立线程采样。[probe_subprocess_path.py](../env-service-verification/scripts/probe_subprocess_path.py) 是不访问 Docker daemon 的 CPU 检查。它们均不是训练脚本。

最终按真实 Compose project 参数核验容器与网络，按记录 PID 核验 Router／节点服务，检查独立 Ray 进程。四组含失败诊断轮的所属资源均已释放，GPU 无计算进程，应用显存约 0.19 MiB，未删除镜像或改动其他用户服务。证据入口：[完整统计](../env-service-verification/results/router-tail-profile-20261020/tail-analysis.json)、[本轮源码摘要](../env-service-verification/results/router-tail-profile-20261020/source-hashes-final.json)、[实验预算](../env-service-verification/results/router-tail-profile-20261020/experiment-plan.json)、[清理核验](../env-service-verification/results/router-tail-profile-20261020/physical-cleanup-final.json)、[CPU 路径检查](../env-service-verification/results/router-tail-profile-20261020/spawn-path-probe.json)。

本次四个诊断脚本的 Ruff 检查通过，文档本地链接及源码行号范围检查通过。全仓 pre-commit 在隔离副本仍被既有格式／风格问题阻塞，未对工作区执行全仓自动修正；记录见 [precommit-all.log](../env-service-verification/results/router-tail-profile-20261020/precommit-all.log)。


## 13. Results：同步准备路径移出事件循环

**结论与范围。** 同一张空闲 GPU、同一组 science runner 合成任务，ON 优化前后分别完成八条任务、十六条 CUDA 命令。任务总时间由 93.00 秒降为 77.56 秒，减少 16.61%，事件循环最大停顿由 2.143 秒降为 0.583 秒。本节比较 ON 修改前后，前者固定在 `37a06ea8d32b05bbbf385ad40cb404b30ba01680` 的隔离 checkout，后者为该提交加本节补丁。没有把它称为用户指定历史 OFF 的新结果，也没有启动模型或训练。

```text
[1] 执行节点上的 science runner 获得会话租约
       │
       ├─ [2] await 后台线程：读驱动库、NVML 设备身份、构造 overlay 内容
       │         主事件循环继续处理其他任务和完成回调
       │         查询返回 → 同步写文件并登记清理路径
       │
       ├─ [3] await prepare_runtime：后台线程准备 Harbor 内核能力缓存
       │         同进程并发调用串行访问原缓存，取消时汇合探针
       │         准备完成 → Job.create → Harbor 环境构造
       │
       └─ [4] ManagedDockerEnvironment.start
                 后台线程新查 Docker daemon 类型 → 原校验逻辑 → 容器启动
                       ↓
             [5] exec_gpu：请求预算 → 授权 → CUDA 命令 → 确认归还
                       ↓
             CPU 间隔／下一条命令 → 评分 → [6] 清理容器、租约和物理资源
```

图例：图里每个部分对应的源码位置：

| 编号 | 职责、函数与源码 |
|---|---|
| 1 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L258) 调用会话准入，再进入容器准备 |
| 2 | [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L129) 在线程执行只读准备，`write_overlay` 和清理列表登记仍在调用协程里连续完成 |
| 3 | [runner 调用入口](../../../examples/sciaccel_rl/runner.py#L500)、[prepare_runtime / _warm_kernel_probe](../../../psrl/workers/env_service/docker.py#L18) 使用 Harbor 原有缓存，探测失败结果也保持原值 |
| 4 | [start / _detect_daemon_os](../../../psrl/workers/env_service/docker.py#L133) 每次启动新查 daemon 类型，原 Harbor 校验随后读取这次结果 |
| 5 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73)、[GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L47) 继续执行原准入和归还协议 |
| 6 | [stop](../../../psrl/workers/env_service/docker.py#L126) 汇合延迟归还，[runner finally](../../../examples/sciaccel_rl/runner.py#L550) 负责本次容器与租约清理 |

**为什么有用。** 事件循环是让多个任务轮流处理请求、完成通知和定时器的执行线程。函数声明为 `async` 并不能让其中的同步调用自动变成后台操作。NVML 是 NVIDIA 的设备查询库，初始化、枚举或关闭连接如果耗时，直接在事件循环调用它，就会同时推迟其他任务收到完成通知。Docker 的同步 `subprocess.run` 也有同样问题。

`asyncio.to_thread` 把这类同步工作交给线程池，`await` 只暂停当前任务对查询结果的使用，事件循环可继续处理其他协程。查询仍发生在原来的实际执行节点，并没有改成让管理节点查询错误的本机 Docker。

举一个**教学假设**：任务 A 在 0 秒发起需要 0.8 秒的设备查询，任务 B 的命令在 0.1 秒完成。如果 A 同步堵住事件循环，B 的完成处理只能等到约 0.8 秒，额外等约 0.7 秒。把 A 的查询放入后台线程后，B 可以在约 0.1 秒开始处理结果，A 仍要等自己的 0.8 秒查询。这不允许 B 跳过设备撤权或物理归还确认，只是完成回调不必被无关查询压住。

**具体改动与安全边界。**

- 设备身份和驱动库每次重新查询，没有缓存“GPU 空闲”的结论。后台只构造 overlay 内容，不创建临时文件。取消等待时线程可能继续完成只读查询，但不能随后留下无人登记的 overlay 文件。
- Harbor 内核能力探针原本就有进程内缓存。本次提前在线程里填充原缓存，用线程锁避免多个冷请求同时运行探针。并非把原来每次运行的探针改成缓存，也没有把探测失败伪装成支持。托管 runner 会在创建任务前准备这一能力，纯公网任务也会支付进程首次准备成本，这是本轮没有单独评估的场景。
- daemon 类型在每次 `start` 时重新查询，再运行 Harbor 原校验，类型不匹配仍报错，查询失败仍沿用 Harbor 的后续报错行为。没有长期保存 daemon 类型来跳过验证。
- 内核探针会运行临时容器，取消准备时等待探针退出后再进入调用方清理。线程不能被 Python 强制中止，因此取消不是立即完成，沿用原探针的有限超时。共享安装的 Harbor 源码没有修改。
- Python 创建子进程时的 `_fork_exec`、共享文件系统读取和 CUDA 上下文创建／销毁仍存在。没有为了提速保留容器 CUDA 上下文，也没有改 GPU 显存配额、设备权限、队列顺序或确认归还条件。

**Results：同等工作量。** 两组均四并发、两批各四条任务，每条分配 1 GiB、申报 2 GiB，两次 GPU 命令各等待 2 秒，中间 CPU 等待 8 秒，使用相同镜像及 GPU UUID。节点预热与 8 秒独立准备重叠，设置相同 240 秒工作预算。按固定的“修改前 → 修改后”顺序各跑一次，任务总时间包含容器准备、命令、评分及每条 runner 清理，不包含整个 Ray 初始化和最终检查。

| 观测项 | 修改前 ON | 修改后 ON | 解读 |
|---|---:|---:|---|
| 成功任务／CUDA 命令 | 8 / 16 | 8 / 16 | 相同工作量，全部校验通过 |
| 第一批四任务 | 48.059 秒 | 39.943 秒 | 含首次托管 Docker 探测 |
| 第二批四任务 | 44.936 秒 | 37.604 秒 | 同进程后续任务 |
| 全部任务时间 | 93.003 秒 | 77.556 秒 | 本次减少 16.61% |
| 事件循环最大停顿 | 2.143 秒 | 0.583 秒 | 本次减少约 72.82% |
| 授权后到 CUDA 脚本开始，最大 | 1.096 秒 | 0.352 秒 | 宿主发送／处理阶段的长尾缩短 |
| 记账请求到收到授权，最大 | 0.721 秒 | 0.070 秒 | 包含调用者恢复执行延迟，不全是账本排队 |
| CUDA 脚本平均时间 | 3.587 秒 | 3.606 秒 | 未见内核脚本本身加速 |
| CUDA 脚本最大时间 | 5.971 秒 | 5.914 秒 | 仍有约六秒的设备侧脚本长尾 |
| 主线程停顿采样中的 NVML / 同步 Docker 栈份数 | 10 / 11 | 0 / 0 | 两条目标阻塞路径未再被采到 |

![同步准备优化的同量任务、事件循环停顿和 CUDA 脚本耗时](../env-service-verification/results/router-sync-offload-20261020/startup-offload.png)

这些结果支持“目标查询已移出事件循环，且本次同量任务更快”。它们不证明长期固定加速 16.61%。只有一组顺序对照，没有重复样本和置信区间，文件系统缓存、进程启动和节点噪声仍会影响整体耗时。修改后也没有采到进程创建停顿，但该路径本次没有修改，不能声称已经修复它。采样份数不是阻塞调用次数或精确耗时分摊。两项优化一起启用，本次没有单独估算各自贡献。

**跟着图读源码。** 从 [1] `run_harbor_episode` 的会话准入进入 [2]，输入是租约，后台输出是 overlay 字典，主协程登记文件路径。随后 [3] 准备原 Harbor 缓存，多个任务可并发等待但冷探针串行执行，取消分支必须先汇合。进入 [4] 每个环境的异步启动，查询结果回到原验证分支。沿 [5] 阅读 `exec_gpu` 到命令归还，注意后台归还与 CPU 间隔可重叠。最后 [6] 先收敛返回错误、清理所属容器，再释放租约。代码中这些调用或并发边界有理解上的问题，可以随时追问。

**可运行回归和只读复核。** 在仓库根目录的项目环境运行：

```bash
python -m pytest tests/env_service/test_startup_offload.py -q
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_science_tail.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-sync-offload-20261020 \
  --arms before after
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_startup_offload.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-sync-offload-20261020
```

第一条是 CPU 故障注入回归，使用线程事件阻塞假的设备／Docker 查询，由事件循环解除阻塞，验证循环持续推进、并发探针去重、取消汇合、错误保持和无临时文件泄漏。预期五项通过，它不需要 Docker daemon 或 GPU。后两条分析本次保存的真实结果并绘图，不再启动任务，结果目录是本地实验材料。需要重跑 GPU 时使用 [profile_science_tail.py](../env-service-verification/scripts/profile_science_tail.py) 的有界入口，设备、镜像和 overlay 按现场重新准备，不沿用过期占用判断。

本次五项新回归加 runner、命令配额和 GPU 借还回归共 **59 项通过**。修改文件的 Ruff 检查通过。全仓 pre-commit 已在隔离副本执行，仍被既有格式和 prose-style 问题阻塞，没有对工作区做全仓自动修正。相关原始证据：[完整统计](../env-service-verification/results/router-sync-offload-20261020/tail-analysis.json)、[修改补丁](../env-service-verification/results/router-sync-offload-20261020/after.patch)、[测试日志](../env-service-verification/results/router-sync-offload-20261020/tests.log)、[全仓检查](../env-service-verification/results/router-sync-offload-20261020/precommit-all.log)、[实验预算](../env-service-verification/results/router-sync-offload-20261020/experiment-plan.json)。

两组结束后核验所属容器、Compose 网络、Router／节点服务和独立 Ray 进程均已退出，目标 GPU 无计算进程、应用显存约 0.19 MiB。证据见[物理清理检查](../env-service-verification/results/router-sync-offload-20261020/physical-cleanup-final.json)。达到本轮结论即停止，没有为提高优势重复运行或追加训练。


**最新 ON 与已有低负载 OFF 的延迟参考。** 以下是对已有原始结果的只读重算，本次没有新启动 GPU 实验。OFF 取第 12 节指定历史提交的八任务结果，ON 取第 13 节修改后的八任务结果。工作量、节点、设备和镜像一致，但运行时间不同，而且 ON 开启了独立线程栈采样、OFF 未开启，不能把这张表视为最新版本严格配对的统计结论。

| 延迟口径 | 已有 OFF | 最新 ON | ON 相对差值 |
|---|---:|---:|---:|
| 单任务端到端平均 | 37.621 秒 | 38.181 秒 | +0.560 秒，+1.49% |
| 单任务端到端最大 | 39.526 秒 | 39.928 秒 | +0.402 秒，+1.02% |
| 单次 GPU 命令调用平均 | 4.136 秒 | 4.168 秒 | +0.032 秒，+0.76% |
| 单次 GPU 命令调用最大 | 6.198 秒 | 6.414 秒 | +0.217 秒，+3.50% |
| 八条任务全部完成 | 76.243 秒 | 77.556 秒 | +1.313 秒，+1.72% |

当前已有证据显示最新 ON 与这批 OFF 的低负载延迟数值接近，尚未证明 ON 在低负载更快，也不能把约 1% 至 2% 的差值认定为稳定协议成本。上一表的 93.003 秒是修改前 ON，不能拿它充当 OFF。

单任务端到端延迟由[驱动的 episode](../env-service-verification/scripts/science_router_control_ab.py#L216)记录的 `finished_at - started_at` 得出，覆盖任务容器准备、执行、评分和 runner 清理，还包含驱动读取任务结果的小开销。八任务总时间是两批四并发全部完成的墙钟，不是单任务 latency，也不能除以八当作单任务平均。例如 OFF 的 76.243 / 8 约为 9.53 秒，而真实单任务平均是 37.621 秒，因为四条任务在时间轴上重叠。

命令调用延迟采用[代理的计时边界](../env-service-verification/scripts/science_router_control_agent.py#L23)，包含 ON 的准入与返回确认路径，第一条允许归还与后续 CPU 阶段重叠。它不是单纯 CUDA 内核执行时间。两组 CUDA 脚本耗时本身也不同，因此不能把平均多出的约 32 毫秒全部归因于 Router。

低负载下四条命令合计申报 8 GiB，低于 unit 的 80 GiB 预算，两边都不发生 OOM，ON 的安全机制没有失败重试可以节省。现有优化主要减轻主循环阻塞，容器身份确认、预算检查、设备授权及归还仍有开销。本表不代表单请求零并发延迟、模型响应延迟或完整训练速度。

原始结果与本次重算：[历史 OFF](../env-service-verification/results/router-tail-profile-20261020/off/result.json)、[最新 ON](../env-service-verification/results/router-sync-offload-20261020/after/result.json)、[描述性对比 JSON](../env-service-verification/results/router-sync-offload-20261020/historical-off-reference.json)。继续使用第 13 节已有只读分析入口可复核每组命令统计。若要判断最新 ON 是否稳定追平 OFF，需要另做同期交错对照，目前不据这一次跨轮结果宣称已经证明。
