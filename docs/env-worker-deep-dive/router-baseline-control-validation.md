# 修复后的 EnvRouter 与固定 OFF 基线：单卡等完成量验证

> **核心结论**：修复后的 ON 在单卡高压力下，16 条成功任务耗时从固定 OFF 基线的 266.52 秒降至 207.84 秒，减少 22.02%，OOM 从 15 次降至 0。计入驱动记录的初始化和最终清理后，总时长减少 15.15%。低压力四任务则从 40.37 秒增加到 55.83 秒，说明安全调度并非所有负载下都更快。基线是指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，失败重试不计作完成。
>
> **低负载优化补充**：第 7–8 节记录后续诊断和代码优化。节点服务复用 NVML 连接、保持每次读取实时状态后，两批各四任务的总任务时间从 98.48 秒降到 90.36 秒；服务已就绪的第二批从 44.83 秒降到 41.87 秒。本轮比较优化前后 ON，不是重新测量 OFF。第 9 节解释预热原理，第 10 节记录已接入生产初始化流程后的私有节点小实验。
>
> 这是实际 science runner、Harbor、Docker 和 CUDA 的控制实验。任务是合成验证任务，CPU 间隔固定，不是原生 LAPS 修复、真实模型 rollout 或训练。本文补充[命令阶段配额与慢查询修复](command-gpu-reservations.md#13-results慢查询定位与直接接口修复)，不替代该文第 7 节的历史训练结果。

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
| 1 | [实验驱动](env-service-verification/scripts/science_router_control_ab.py) 的 `run` 选择实际导入的 science runner，保持相同并发及补采规则。旧 runner 的实验快照见[基线源码](env-service-verification/results/router-baseline-control-20261020/baseline-runner.py#L132) |
| 2 | [固定代理](env-service-verification/scripts/science_router_control_agent.py#L12) 的 `run` 执行两条相同 CUDA 命令，中间执行 CPU 等待。ON 调用 [exec_gpu](../../psrl/workers/env_service/docker.py#L52) |
| 3 | [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L47) 检查先入先出队列、预算及实时显存 |
| 4 | [grant_gpu_command / revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L120) 检查容器归属并授予、撤销设备权限。使用[本机 Docker 接口](../../psrl/workers/env_service/docker_api.py#L9)与 [NVML 查询](../../psrl/workers/env_service/probe.py#L74) |
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

固定基线使用指定提交加已有运行修复。实验保存了完整 [baseline.patch](env-service-verification/results/router-baseline-control-20261020/baseline.patch)：runner 传递 GPU 覆盖配置、取消节点级 dangling-image prune，另外有 Ray 地址和 HTTP 事件循环归属修复。补丁没有增加 GPU 锁、动态分配或 Router。当前版本为 `daff7664d284e338d2ad70eb748ad73d46ee3d86` 加保存的工作树改动，不能把两个版本的对照写成只切换一个布尔配置。

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

![单卡等完成量、累计完成数量与 OOM 对照](env-service-verification/results/router-baseline-control-20261020/completion.png)

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

![统一时间轴下的显存、预算和会话状态](env-service-verification/results/router-baseline-control-20261020/resources.png)

上图显存占用不等于计算利用率，命令间复用不等于 CUDA 分配器内部碎片减少。实际使用的显存与申报预算也不必逐毫秒重合，预算覆盖了授权、实际命令和安全归还阶段。

修复后的高压力 ON 共记录 64 次容器检查，平均 1.012 毫秒，最大 2.157 毫秒。GPU 身份查询平均 0.336 秒，实时占用查询平均 0.157 秒。直接 Docker 接口没有再出现前次探针的数秒级检查，但整条控制路径仍有 GPU 查询、锁、远程调用和设备权限管理开销，不能把“容器检查 1 毫秒”说成“获得 GPU 总共只要 1 毫秒”。这次没有额外加入修复前 ON 组，不能把相对 OFF 的全部收益单独归因于直接接口修复。

四组最终均无本次容器、Compose 网络及私有 Ray 进程，目标 GPU 无计算进程、应用显存为 3.44 MiB。物理清理核验通过后结束实验。

## 5. 复现与证据

先按[集群运行约束](../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)激活环境、核对执行节点和 Docker，确认目标 GPU 空闲。下列变量由操作者显式指定：

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

- [comparison.json](env-service-verification/results/router-baseline-control-20261020/comparison.json)：各组统计和实际回填配对。
- [高压力 OFF 全部尝试](env-service-verification/results/router-baseline-control-20261020/pressure-off/result.json)、[高压力 ON 全部尝试](env-service-verification/results/router-baseline-control-20261020/pressure-on/result.json)：成功及 OOM 原始记录、计时和实际导入的 runner 路径。
- [低压力 OFF](env-service-verification/results/router-baseline-control-20261020/low-off/result.json)、[低压力 ON](env-service-verification/results/router-baseline-control-20261020/low-on/result.json)：保留 ON 更慢的结果。
- [ON 控制查询分段计时](env-service-verification/results/router-baseline-control-20261020/pressure-on/control-trace.json)、[ON 账本与显存采样](env-service-verification/results/router-baseline-control-20261020/pressure-on/telemetry.jsonl)。
- [等工作量与账本审计](env-service-verification/results/router-baseline-control-20261020/audit.json)、[最终物理清理](env-service-verification/results/router-baseline-control-20261020/physical-cleanup-final.json)、[源码快照 SHA-256](env-service-verification/results/router-baseline-control-20261020/source-sha256.json)。

新增三个实验与绘图脚本的 Ruff 检查通过，实验结果审计通过。全仓 `pre-commit run --all-files` 在包含当前文件的隔离副本中执行，仍有其他文件的既有格式和风格错误，未改写工作区文件，详见[检查日志](env-service-verification/results/router-baseline-control-20261020/precommit-all.log)。不能称为全仓检查通过。

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
| 1 | [ManagedDockerEnvironment.exec_gpu](../../psrl/workers/env_service/docker.py#L52)：请求授权、执行命令并发起归还 |
| 2 | [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L47)：unit 锁覆盖预算检查、实时查询和授权确认，避免同一个预算被并发超额承诺 |
| 3 | [GpuLendingNodeMixin._gpu_probe](../../psrl/workers/env_service/lending_node.py#L41) 持有对象；[gpu_lending_snapshot](../../psrl/workers/env_service/lending_node.py#L52) 将查询交给线程；[NvmlProbe._session](../../psrl/workers/env_service/probe.py#L52) 只在首次初始化连接；[read_gpu_snapshot](../../psrl/workers/env_service/probe.py#L85) 每次读取当前占用 |
| 4 | [_command_container](../../psrl/workers/env_service/command_gpu_node.py#L109)、[grant_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L120)：本机 Docker 身份校验、实时设备映射及设备权限管理 |
| 5 | [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L131)、[Scheduler._return](../../psrl/workers/env_service/command_gpu.py#L124)：锁外检查归还，确认后在锁内释放预算 |
| 6 | [NvmlProbe.close](../../psrl/workers/env_service/probe.py#L59)：与查询共用线程锁，防止查询途中关闭；短期调用可使用上下文管理器，节点对象注册正常进程退出清理 |

这些源码如果有理解上的问题，可以随时追问。

跟着图读源码：

1. `exec_gpu`（图 1）输入命令和申请标识，调用 Router；`acquire`（图 2）输入租约、容器和请求标识，输出 `wait` 或 `granted`。
2. `acquire` 在 unit 锁内调用节点 `gpu_lending_snapshot`（图 3），获得当前 UUID、显存、进程号。NVML 是 NVIDIA Management Library，即 NVIDIA 的设备管理库，用于从驱动读取设备状态；此处不执行模型计算。
3. 预算允许后，`grant_gpu_command`（图 4）检查本机容器，查当前 GPU index 对应的设备 minor，授予访问权限。Router 确认授权后结束本次锁内检查。**锁不覆盖整个 CUDA 命令，因此不等于整卡串行执行。**
4. GPU 命令结束后，`exec_gpu` 可以让后台归还与 CPU 阶段并行（图 5）。`revoke_gpu_command` 先撤权，确认所属进程已无设备句柄／CUDA 使用，再允许 `_return` 把预算释放。
5. 后续请求沿同一路径重新读取实时状态；节点服务继续复用图 3 的连接。图 6 是独立的生命周期清理，正常退出会关闭引用；强制终止进程不能保证运行 Python 退出回调，操作系统仍回收该进程资源。NVML 连接本身不持有任务的 CUDA 显存。

**低负载变慢的三部分原因。** 第一是首次服务启动。历史低压力实验任务准备均值为 OFF 6.83 秒、ON 19.05 秒，相差约 12.22 秒，但其中混合了准入、节点服务、容器准备，不能把差值全部称为 Router 启动成本。本轮增加分段计时，节点服务首次启动分别为优化前 5.76 秒、优化后 5.31 秒，各只发生一次。

独立实验的 ON 还自行建立私有 Ray 集群。Ray 是管理远程进程和调用的分布式运行框架，这一初始化计在任务提交前，历史低压力 ON 为 17.98 秒。真实 PSRL 训练的 OFF 本来也使用 Ray，所以不能把这 17.98 秒再算成生产训练每批新增的 Router 成本。当前 [create_env_router](../../psrl/workers/env_service/router.py#L407) 可以复用同名服务，[节点服务缓存](../../psrl/workers/env_service/router.py#L363) 也避免每条命令重新创建服务。

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

![低负载任务与控制查询优化前后](env-service-verification/results/router-low-overhead-20261020/low-overhead.png)

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

证据：[统计与审计](env-service-verification/results/router-low-overhead-20261020/comparison.json)、[优化前原始结果](env-service-verification/results/router-low-overhead-20261020/before/result.json)、[优化后原始结果](env-service-verification/results/router-low-overhead-20261020/after/result.json)、[连接只读计时](env-service-verification/results/router-low-overhead-20261020/nvml-session-profile.json)、[最终物理清理](env-service-verification/results/router-low-overhead-20261020/physical-cleanup-final.json)、[CPU 回归](env-service-verification/results/router-low-overhead-20261020/cpu-tests.log)、[最终生命周期回归](env-service-verification/results/router-low-overhead-20261020/cpu-lifetime-final.log)、[全仓检查日志](env-service-verification/results/router-low-overhead-20261020/precommit-all.log)。当前基础提交仍为 `daff7664d284e338d2ad70eb748ad73d46ee3d86`，本轮变更由 [before-source](env-service-verification/results/router-low-overhead-20261020/before-source/env_service/) 和 [after-source](env-service-verification/results/router-low-overhead-20261020/after-source/env_service/) 的生产源码快照记录；固定 OFF 提交保持不变，本轮未修改或重新运行它。


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
| 1 | [init_workers 调用点](../../psrl/trainer/ppo/ray_trainer.py#L1906)、[_init_env_service](../../psrl/trainer/ppo/ray_trainer.py#L549)、[create_env_router](../../psrl/workers/env_service/router.py#L407)：创建或复用 Router，目前只等待 Router 自己响应，不代表全部节点已就绪 |
| 2 | [模型初始化阶段](../../psrl/trainer/ppo/ray_trainer.py#L2112)：在 Router 创建之后继续进行，因此存在可与节点准备重叠的窗口 |
| 3 | [_start_node](../../psrl/workers/env_service/router.py#L296)、[_gpu_command_node](../../psrl/workers/env_service/router.py#L363)、[EnvNodeService.ready](../../psrl/workers/env_service/node.py#L31)、[NvmlProbe._session](../../psrl/workers/env_service/probe.py#L52)：已有节点创建、启动任务复用、端口和连接初始化能力；统一入口为 `begin_prewarm`，由 `create_env_router` 提交 |
| 4 | [模型完成后的绑定入口](../../psrl/trainer/ppo/ray_trainer.py#L2275)、[_init_gpu_lending](../../psrl/trainer/ppo/ray_trainer.py#L608)、[bind_rollout_gpu](../../psrl/workers/env_service/router.py#L367)：保留真实绑定和 GPU 身份验证。节点就绪等待不取代这些检查 |
| 5 | [try_admit](../../psrl/workers/env_service/router.py#L380)、[acquire](../../psrl/workers/env_service/command_gpu.py#L47)、[_return](../../psrl/workers/env_service/command_gpu.py#L124)：首次请求复用已就绪节点，实时预算、授权与归还协议照常执行 |
| 6 | [EnvNodeService.cleanup](../../psrl/workers/env_service/node.py#L68)、[NvmlProbe.close](../../psrl/workers/env_service/probe.py#L59)、[训练器结束 Router](../../psrl/trainer/ppo/ray_trainer.py#L596)：已有清理入口；预热任务通过 `shutdown_nodes` 纳入结束和启动失败的回收路径 |

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

**可运行教学模拟。** [node_prewarm_demo.py](env-service-verification/scripts/node_prewarm_demo.py) 使用 Python 标准库 `asyncio` 模拟上图，模型准备 0.20 秒、节点准备 0.06 秒、任务 0.04 秒，执行三个场景约一秒。注释对应图中步骤；`TeachingNode.ready` 复用同一个启动任务，断言每个场景只启动一次，`finally` 负责收回未完成任务。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/node_prewarm_demo.py
```

预期：懒启动与串行预热总时长都约 0.30 秒；并行预热约 0.24 秒。串行预热虽然把首请求等待降到近零，总时间仍不变；只有重叠后总时间才缩短。该脚本不模拟 Ray 调度、Docker、NVML 的真实成本或硬件争用，不能用于预测训练加速比。本轮实际模拟结果：懒启动总时长 0.3006 秒、首请求等服务 0.0602 秒；串行预热总时长 0.3005 秒、首请求等待接近零；并行预热总时长 0.2405 秒、首请求等待接近零。三个场景都只启动一次节点。实际运行记录保存在 [demo.json](env-service-verification/results/node-prewarm-explanation-20261020/demo.json)，此脚本只验证教学模型，生产节点预热的实机结果另见第 10 节；本轮均未运行训练。

本节教学脚本的 Ruff 检查和格式检查通过，文档本地路径、源码行号范围及章节锚点校验通过。全仓 pre-commit 在隔离副本执行，仍因其他文件的既有风格问题失败，见[检查日志](env-service-verification/results/node-prewarm-explanation-20261020/precommit-all.log)；未自动改写工作区其他文件。

## 10. Results：私有节点预热与固定 OFF 复验

**结论**：节点预热已接入 `create_env_router`，会先提交配置节点的准备任务，再让训练器继续模型初始化。真实节点服务的准备与固定 8 秒独立准备阶段重叠后，低负载 ON 的“准备＋任务”从懒启动的 54.78 秒降到 51.02 秒，减少 6.85%；串行预热为 54.86 秒，没有总时间收益。低负载 OFF 仍更快，42.56 秒。高压力下四条任务均须完成，OFF 的 6 次 OOM 和重试使其耗时为 111.08 秒，ON 零 OOM、58.03 秒，减少 47.76%。包含各自 Ray／Router 初始化和最终清理的驱动总时间减少 34.21%。

当前完整架构、各层职责、相对基线的优化和源码阅读路径集中维护在 [EnvRouter 当前流程](env-router-current-flow.md)。下面只记录本次实验及其验证边界。

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

![预热方式、低负载与高压力等完成量对照](env-service-verification/results/router-prewarm-private-20261020/prewarm-comparison.png)

上图右上角使用真实 `_start_node` trace，蓝色为节点准备，灰色为固定独立准备阶段。ON 懒启动在灰色阶段结束、任务开始后才准备节点，耗时 5.48 秒；ON 并行预热的 5.42 秒节点准备落在 8 秒独立阶段内。串行预热先准备节点 8.48 秒，再等待共同的 8 秒，任务阶段虽然更短，总时间没有改善。不同组启动耗时有抖动，因此 6.85% 是本轮观察值，没有多次重复测量的置信区间，不能把全部时间差视为稳定的纯预热收益。

高压力 OFF 的六次 OOM 均发生在第一条 GPU 命令分配显存时。两边成功的 CUDA 命令都是八条，因此优势来自避免失败尝试的准备、失败处理、清理和重试等待，没有通过少完成任务或不计失败来缩短时间。也不能把并发失败的耗时逐条相加当作墙钟节省。

**账本与归还证据。** 高压力 ON 同时活动 GPU 命令峰值为 1、等待峰值为 3；三种低负载 ON 的活动峰值均为 4，说明预算足够时没有退化成整卡串行锁。各组采样申报均等于“活动命令数 × 单命令需求”，无超预算样本。高压力 ON 请求到授权均值 2.418 秒、最大 9.220 秒，主要包含真实的额度排队；命令结束到确认归还均值 7.28 毫秒。四条任务的首次归还都与自己的 CPU 阶段重叠，另有 12 对 GPU 分配与其他任务 CPU 区间的重叠关系。配对数不是独立命令数。

![统一时间轴下的显存与命令预算](env-service-verification/results/router-prewarm-private-20261020/resource-timeline.png)

预算在上图 ON 的 GPU 工作密集阶段看起来持续为 51 GiB，是总量采样中相邻命令快速交接的结果，不代表同一条任务一直保留预算。归还及下一个申请之间的短暂归零可能小于约 0.5 秒采样间隔；应结合命令归还记录和请求身份判断。图中显存占用不是计算利用率，也不能证明 CUDA 内部碎片减少。

**低负载的剩余开销。** 并行预热后 ON 比本轮 OFF 的“准备＋任务”仍高约 19.87%。预热只覆盖节点首次准备，任务会话准入、容器配置、授权与归还等路径仍比固定 OFF 多。机器和任务准备波动也存在，不能把剩余 8.46 秒全部归因于某个未计时子步骤。它说明当前方案有适用条件，不能由高压力优势推导为所有负载都更快。

**回归与清理。** 针对节点、命令协议、借用和 science runner 的 67 项 CPU 回归通过；补充关闭异常及工厂提交行为后，最终预热测试文件 9 项通过。后者覆盖共享启动、取消隔离、超时、失败、关闭继续回收、活动租约保护及“只等待提交确认、不等待节点完成”的生产入口行为。未执行完整训练初始化集成验证。修改模块的针对性 Ruff 检查通过，全仓 pre-commit 在隔离副本仍因既有风格问题失败。最终按六组 actor／Compose project 核验无容器、无网络、无私有 Ray 进程；八个记录的 Router／节点服务 PID 均已退出，GPU 无计算进程，应用显存为 0.19 MiB。

复现选项见 [science_router_control_ab.py](env-service-verification/scripts/science_router_control_ab.py)。沿用第 5 节的环境与节点预检，先准备新节点自己的 GPU overlay，再对各组固定导入路径运行：

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

证据入口：[六组统计](env-service-verification/results/router-prewarm-private-20261020/comparison.json)、[完整启动记录](env-service-verification/results/router-prewarm-private-20261020/suite.log)、[OFF 全部压力尝试](env-service-verification/results/router-prewarm-private-20261020/pressure-off/result.json)、[ON 压力尝试](env-service-verification/results/router-prewarm-private-20261020/pressure-on/result.json)、[并行预热 trace](env-service-verification/results/router-prewarm-private-20261020/low-on-overlap/control-trace.json)、[CUDA 预检](env-service-verification/results/router-prewarm-private-20261020/cuda-preflight.json)、[最终物理清理](env-service-verification/results/router-prewarm-private-20261020/physical-cleanup-final.json)、[67 项回归](env-service-verification/results/router-prewarm-private-20261020/cpu-tests.log)、[最终 9 项预热测试](env-service-verification/results/router-prewarm-private-20261020/startup-tests-final.log)、[全仓检查](env-service-verification/results/router-prewarm-private-20261020/precommit-all.log)、[代码摘要](env-service-verification/results/router-prewarm-private-20261020/source-sha256.json)、[基线运行补丁](env-service-verification/results/router-prewarm-private-20261020/baseline.patch)。
