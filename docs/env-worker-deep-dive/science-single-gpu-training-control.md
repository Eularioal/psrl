# Science 真实训练 OFF / ON 对照：显存压力与原生任务

> **核心结论（2026-10-20，十步扩展）**：预定 OFF / ON 各十步，实际 OFF 完成十步，ON 完成五步后在第六步被实验采集逻辑中止。模型执行 `rm -rf /logs` 删除了容器可写的压力证据文件，脚本随后判定证据缺失。这是本次测量脚本的缺陷，完整十步对照未完成。
>
> 双方完整完成的前五步均有 40 条四轮真实轨迹，训练步骤总耗时 **48.63 → 35.17 分钟，减少 27.67%**，OOM **57 → 0 次**。节省集中在等待采样阶段。OFF 第五步与同节点另卡实验有重叠，因此比例是本次观测值。压力来自额外注入的 72 GiB 显存阶段，两组奖励均为零，不证明模型效果提升。
>
> 证据已改为在模型执行命令前保存到宿主机独立目录，CPU 回归五项通过，没有追加长训练。前六节保留原单步实验及其边界，十步目标与结果见[第 7 节](#7-results十步扩展验证)，失败机制和修复见[第 8 节](#8-证据文件删除故障与修复)。本文承接[单卡容器对照](router-baseline-control-validation.md)。

> **无压力续跑（2026-10-21）**：修正配置后，OFF 完成三步，ON 完成两步，在第三步遇到评分器镜像仓库请求 EOF 而退出。共同完成的前两步各 16 条轨迹，耗时 **521.17 → 535.42 秒，ON 慢 2.73%**，均无 OOM，本次没有测出无压力收益。完整三步对照未完成，资源已释放。结果与故障边界见[第 10 节](#10-results无压力续跑与共同完成的两步)，首次配置失败保留在[第 9 节](#9-results不注入压力的原生任务对照)。

## 目录

后续[真实 GPU 命令多协程实验](science-natural-gpu-concurrency.md)使用 MLGym 原始分类训练，不添加显存占位，与本文的 PSRL 强化学习训练结果分别统计。

- [1. 真实执行链路](#1-真实执行链路)
- [2. 对照配置与停止条件](#2-对照配置与停止条件)
- [3. 压力怎样触发收益](#3-压力怎样触发收益)
- [4. Results：训练与调度结果](#4-results训练与调度结果)
- [5. 跟着图读源码与复现](#5-跟着图读源码与复现)
- [6. 验证边界与资源清理](#6-验证边界与资源清理)
- [7. Results：十步扩展验证](#7-results十步扩展验证)
- [8. 证据文件删除故障与修复](#8-证据文件删除故障与修复)
- [为什么曲线从第六步开始没有数据](#为什么曲线从第六步开始没有数据)
- [9. Results：不注入压力的原生任务对照](#9-results不注入压力的原生任务对照)
- [10. Results：无压力续跑与共同完成的两步](#10-results无压力续跑与共同完成的两步)

## 1. 真实执行链路

```text
[1] 相同 Qwen3.5-4B、四个 LAPS 修复任务、每题两条轨迹
                         |
[2] PSRL 训练器启动实际 vLLM 推理、AgentLoop 和训练 worker
                         |
[3] 真实 Terminus2 代理首次请求模型，得到真实回复
                         |
       两组在执行回复中的命令前注入相同 CUDA 显存阶段
                         |
       +-----------------+----------------------+
       |                                        |
[4a] 固定 OFF 提交                        [4b] 当前 ON
进程内 episode 数量限制                   Router 会话准入
静态映射同一张环境 GPU                    exec_gpu 命令申请
直接申请 72 GiB                          73 GiB 申报 / 80 GiB 预算
       |                                  排队、授权、执行、归还
       +-----------------+----------------------+
                         |
[5] 代理继续真实模型交互、修改 Fortran 代码、执行命令
    独立 verifier 比较科学计算结果，返回 reward_repair
                         |
[6] OOM 尝试丢弃并有界重试，凑齐四题各两条可训练轨迹
                         |
[7] GRPO 组内优势、概率重算、前反向与优化器更新
                         |
[8] 保存时间、显存、等待队列与训练指标，清理所属资源
```

图例：图里每个部分对应的源码位置。

| 编号 | 实际实现及职责 |
|---|---|
| 1 | [prepare](env-service-verification/scripts/science_fixed_work_training.py#L40) 从原始 Parquet 选取四题，只在隔离任务副本加入 CUDA 脚本与资源声明 |
| 2、8 | [run_arm](env-service-verification/scripts/science_fixed_work_training.py#L173) 建立隔离 Ray、指定实际源码目录、启动训练并采集遥测 |
| 3 | [PressureTerminus2._query_llm](env-service-verification/scripts/science_pressure_agent.py#L21) 先调用真实模型，再执行一次额外显存阶段 |
| 4a | [固定基线运行补丁](env-service-verification/results/science-single-gpu-training-20261020/off.patch) 只包含静态设备配置传递、共享资源清理限制、Ray 地址与 HTTP 运行修复，没有加入整卡锁 |
| 4b | [CommandPressureAgent._run_pressure_command](env-service-verification/scripts/science_fixed_work_agent.py#L30) 调用 [ManagedDockerEnvironment.exec_gpu](../../psrl/workers/env_service/docker.py#L73)，经 [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L47) 排队，在 [_return](../../psrl/workers/env_service/command_gpu.py#L124) 确认释放 |
| 5 | [SciAccelAgentLoop._run_harbor_in_thread](../../examples/sciaccel_rl/agent_loop.py#L358) 运行 Harbor，协调真实代理和评分器 |
| 6 | [FixedWorkAgentLoop.run](env-service-verification/scripts/science_fixed_work_agent.py#L57) 保留每个采样位置，只有 OOM 允许补采，不把失败算成完成 |
| 7 | [训练器 fit](../../psrl/trainer/ppo/ray_trainer.py#L3343) 和 [_update_actor](../../psrl/trainer/ppo/ray_trainer.py#L3204) 执行实际训练更新 |
| 8 | [分析脚本](env-service-verification/scripts/analyze_single_gpu_training.py#L11) 从尝试记录、训练日志和遥测计算结果 |

这里的 vLLM 负责根据模型权重生成回复，Harbor 负责启动任务容器、运行代理和评分器，Ray 负责将训练组件调度到实际进程。CUDA 是调用 NVIDIA GPU 计算与显存接口的平台，本实验通过它申请并校验真实设备显存。

这些源码如果有理解上的问题，可以随时追问。图中多个轨迹并发推进，模型请求、环境命令、评分和清理会交错执行。ON 的 GPU 归还可在后台完成，后续 CPU 阶段继续推进，但下一条 GPU 命令仍需等归还确认。

## 2. 对照配置与停止条件

| 项目 | 两组共同条件 |
|---|---|
| OFF 源码 | 用户指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9` 加已记录运行修复 |
| ON 源码 | `37a06ea`，实际指纹与补丁见证据 |
| 模型 | Qwen3.5-4B，训练两张 GPU，推理一张 GPU |
| 环境设备 | 两组同一张 H20，实测总显存 97871 MiB，约 95.58 GiB |
| 原始任务 | LAPS 的 coef / subtle 四个 Fortran 修复任务，保留原提示和评分规则 |
| 工作量 | 四题各两条非 OOM 且有评分的轨迹，合计八条，一次 GRPO 更新 |
| 活动尝试 | 两组同一实验信号量，上限四个。OFF 原有进程内信号量仍保留 |
| 压力 | 每条轨迹首次模型回复后申请 72 GiB，校验写入读取结果 42，持有 30 秒 |
| ON 预算 | 80 GiB GPU 预算，每条命令申报 73 GiB，最多四个持久会话 |
| 交互边界 | 最多四轮，prompt 2048 token，response 8192 token，关闭思考模式 |
| 截断处理 | 两组均保留已产生的有效 token，`overlong_filtering=False` |
| 采样与优化 | 数据不打乱，数据 seed 17，相同生成与优化参数，关闭 checkpoint 和额外评估 |
| 重试 | 每个采样位置最多十二次，只重试 CUDA OOM，相同确定性 12–20 秒退避 |
| 时间预算 | 每组训练子进程最多 1200 秒，代理任务 300 秒，评分器 900 秒，整场启动器另有 3000 秒外部上限 |
| 停止 | 达到一次更新即停止。失败保留证据，不为了奖励变好或获得漂亮速度反复补跑 |

token 是模型分词后的基本文本单位，不能直接当成汉字或单词数。GRPO 是“组相对策略优化”。这里同一道题的两条模型轨迹构成一组，通过两条轨迹的奖励差异计算更新方向。八条轨迹来自真实模型，不是固定答案代理。实验额外 CUDA 脚本不由模型编写，它只负责制造相同的资源竞争。

准备阶段在实际执行节点验证了 Docker 探活、GPU index / UUID / minor 映射、实际驱动库加载和 1 GiB CUDA 操作。依赖测试覆盖单条、等长和变长 position_ids，三项通过。八个任务相关镜像的公共预热在首批任务执行前完成，不按 OFF 的采样耗时计入第一次镜像构建。

## 3. 压力怎样触发收益

显存是 GPU 保存模型、张量和中间结果的空间。OOM 是 Out Of Memory，即分配时可用空间不足。本轮环境卡约 95.58 GiB，一条命令申请 72 GiB，约占 75.3%。两个命令同时申请就需要 144 GiB，已经超过物理容量，还没有计入 CUDA 上下文。因此四个 episode 的“数量上限”本身不足以保证显存安全。

以本轮配置作一个计算示例，具体时间先后仍以日志为准：

1. A、B、C、D 都获得活动尝试名额，并完成各自首次模型请求。
2. OFF 的四个容器静态映射同一 GPU。如果 A 已占 72 GiB，B 再申请 72 GiB 就可能 OOM。B 的失败尝试需要清理，再等待退避，重新启动任务和模型交互。
3. ON 在真正执行 CUDA 命令前计算 `73 + 73 > 80`，因而最多批准一条命令，另外的命令排队。原有 CPU 容器可以继续存在。
4. A 完成显存阶段后，节点确认进程结束和设备权限归还，再释放申报额度。B 获得额度时，A 可以继续真实模型交互或 CPU 科学计算。
5. 两组都必须最终收集八条符合条件的轨迹。少发生 OOM 有机会减少重复启动、重复模型请求和清理成本，但 ON 自己也有排队与控制开销，最终谁更快要实测。

OFF 在 GPU 阶段结束后也可以与其他任务的 CPU 阶段交叠。本轮要验证的是资源竞争时的准入和重试成本，不能将所有阶段重叠都说成 ON 独有能力。

这主要是**显存容量压力**。脚本分配大块显存并做短暂 CUDA 校验，其余持有时间是等待，不代表 30 秒持续进行高强度矩阵计算。模型训练和推理有真实 GPU 计算，但分别位于另外三张卡。因此不能将本轮称为 Docker 与 vLLM 同卡借还验证，也不能将显存占用率写成计算利用率。

## 4. Results：训练与调度结果

本轮确实运行了真实模型、原始科学任务、独立评分器和训练器。两组成功轨迹都逐条核实为四轮模型交互，没有代理或评分基础设施异常，没有用更少轮数或任务超时换取 ON 的时间优势。

| 指标 | 固定 OFF | EnvRouter ON |
|---|---:|---:|
| 目标成功轨迹 | 8，四题各 2 条 | 8，四题各 2 条 |
| 总尝试数 | 17 | 8 |
| CUDA OOM 尝试 | 9 | 0 |
| 成功轨迹每条交互轮数 | 全部 4 轮 | 全部 4 轮 |
| 含失败与补采的采样时间 | 483.90 秒 | 402.00 秒 |
| 成功轨迹的模型 completion token | 8993 | 10509 |
| 驱动启动至首个采样请求 | 763.25 秒 | 661.87 秒 |
| 完成训练 step | 0，概率重算时预算截断 | 1，正常退出 |
| 完成更新的有效训练 token | 无最终 step 指标 | 10509 |
| 驱动启动至清理结束的实际时长 | 1262.24 秒，截断记录 | 1141.34 秒，正常完成 |
| 环境 GPU 显存峰值 | 74326 MiB，约 72.58 GiB | 74053 MiB，约 72.32 GiB |
| GPU 遥测失败次数 | 0 | 0 |

**本轮观测到等量采样减少 81.91 秒，即 16.93%，采样吞吐约为 OFF 的 1.20 倍。** 计时从本批第一个采样位置开始，到第八条有效轨迹完成，包含排队、OOM 失败、清理和重试。ON 成功轨迹的模型输出反而多了约 16.9%，因此这次不是通过少生成 token 得到更短采样时间。

模型 completion token 表示模型实际生成的 token 数。它不是“所有多轮拼接文本的长度”，也不能在没有训练记录时冒充已训练 token。ON 的最终 `termination/trained_tokens=10509`、`termination/masked_sample_fraction=0` 和 `group/count=4` 另行确认了八条轨迹都进入了这次更新。OFF 没有最终 step 记录，不填造它的 loss、entropy 或更新耗时。

ON 账本的同一份现场快照观察到 **4 个会话、1 条活动 GPU 命令、3 条等待命令**，显存申报为 74752 MiB，也就是 73 GiB。整个采集区间未观察到 CPU、主机内存、会话数或申报显存超过账本预算。两组 CUDA 记录指向同一 GPU UUID，成功 CUDA 阶段的重叠峰值都是 1。OFF 也只能让一份 72 GiB 分配实际成功，但它通过失败暴露容量竞争，ON 则在执行前排队。

![实际显存、等量采样、OOM 与截断边界](env-service-verification/results/science-single-gpu-training-20261020/comparison.png)

[SVG 图](env-service-verification/results/science-single-gpu-training-20261020/comparison.svg)和[结构化统计](env-service-verification/results/science-single-gpu-training-20261020/comparison.json)可供复核。右下角展示的是实际运行到停止的时间，已标注 OFF 的预算截断，**不是两组都完成一个 step 的时间对比**。

ON 的 `timing_s/step=460.05` 秒，其中等待生成 407.32 秒、旧策略概率重算 8.18 秒、actor 更新调用 44.26 秒。它随后完成权重推送，输出 `Training completed successfully!`，子进程退出码为 0。OFF 的退出码为 -15，是实验看门狗按原定预算发送终止信号，不能描述成模型 OOM 或 Docker 服务崩溃。

训练指标为 entropy（输出分布的不确定程度，亦称熵）0.37109、策略 loss（策略优化损失）0、grad norm（梯度范数，表示更新方向的整体大小）0。两组四题的奖励都是 `[0, 0]`，实际训练读取 `reward_repair`。GRPO 在一组 `[0, 0]` 中减去组均值后仍得到 `[0, 0]`，所以没有奖励差异形成的策略梯度。这证明训练执行链路完成，不证明模型修复能力提高，也不把零梯度自动判作训练崩溃。

**节点并非全程独占。** [全部设备遥测复核](env-service-verification/results/science-single-gpu-training-20261020/shared-node-activity.json)发现，未分配给本实验的 GPU 7 出现最高 5385 MiB 的短时活动。OFF 的 15:15:30 至 15:16:17 采样末段有 6 个此类采样点，ON 初始化期间有 16 个，ON 采样窗口内为 0。这不等于本实验环境卡发生外部占用，但同节点 CPU、Docker 和存储竞争可能影响尾部时间。因此不将 16.93% 当作排除其他因素后的纯 Router 加速比例，也不删除这次结果或追加重跑来挑选更漂亮的数据。

本轮只做一次配对。OFF 和 ON 的初始化时间相差约 101.38 秒，两个版本和异步启动时序并不完全相同，模型生成量也不同，日志中两组都出现了首次推理内核编译。因此不能把全程时间差或全部 16.93% 的观测差额都精确归因于单个调度子模块，也不能据此宣称任意负载都会加速。最直接的机制证据是同卡同压力下出现等待队列、9 次 OOM 消失，以及同等成功轨迹数下减少了重试。

## 5. 跟着图读源码与复现

| 步骤 | 图中编号 | 输入、输出和状态变化 |
|---|---|---|
| 准备 | 1 | 原始四行数据复制成独立任务目录，Parquet 保持原生 struct/list |
| 初始化 | 2 | OFF 和 ON 分别创建自己的 Ray，显式传递 GCS 地址和 Python 包路径，模型 worker 在对应设备初始化 |
| 并发采样 | 3–5 | 最多四个尝试同时运行，真实首次模型回复触发一次 CUDA 阶段，之后恢复原代理流程 |
| OOM 分支 | 6 | 丢弃该尝试的训练数据，记录失败、清理并退避，只重试同一采样位置 |
| 正常分支 | 6–7 | 返回有评分的有效轨迹，八条收齐后参与同一个训练 step |
| 收尾 | 8 | 更新结束或预算到达后回收所属进程和容器，验证环境卡与模型卡释放 |

下列脚本启动真实训练，需要先按[集群预检约束](../psrl/psrl-run-guide/cluster-preflight.md)检查节点。它不是 CPU 教学模拟。`PSRL_REPO` 为当前源码目录，`BASELINE_REPO` 为固定提交及已披露运行修复的隔离 checkout，`MODEL_PATH` 和 `DATASET_PATH` 指已有模型和原始 LAPS Parquet，`RESULT_DIR` 必须为新的结果目录，`NODE_IP` 为现场确认的执行节点地址。

```bash
source "$PSRL_ENV_SCRIPT"
export PSRL_WORKSPACE="$PSRL_REPO"
export PYTHONPATH="$PSRL_REPO:$VERIFICATION_DEPS:$PSRL_REPO/docs/env-worker-deep-dive/env-service-verification/scripts"
export CUDA_VISIBLE_DEVICES=0,1,2 TMPDIR=/tmp RAY_TMPDIR=/tmp
export LOCAL_IP="$NODE_IP" VLLM_HOST_IP="$NODE_IP"
cd "$PSRL_REPO"
python docs/env-worker-deep-dive/env-service-verification/scripts/science_fixed_work_training.py \
  --repo "$PSRL_REPO" --baseline-repo "$BASELINE_REPO" \
  --model "$MODEL_PATH" --dataset "$DATASET_PATH" \
  --node-ip "$NODE_IP" --output "$RESULT_DIR" \
  --steps 1 --timeout 1200 --task-timeout 300 --turns 4 \
  --environment-gpus 6 --rollout-n 2 --pressure-gib 72 --pressure-hold 30 \
  --response-length 8192 --modes off command
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_single_gpu_training.py "$RESULT_DIR"
```

命令中的 `command` 是 ON 的命令阶段配额模式，不能误读为第三组实验。模型设备 0、1、2 和环境设备 6 是本次选择，复现前必须现场确认实际可用性与 minor 映射。分析命令只读取已有结果并生成图表，不重新运行训练。

原始证据目录：[science-single-gpu-training-20261020](env-service-verification/results/science-single-gpu-training-20261020/)。[设计](env-service-verification/results/science-single-gpu-training-20261020/design.json)、[预检](env-service-verification/results/science-single-gpu-training-20261020/preflight.json)、[源码指纹](env-service-verification/results/science-single-gpu-training-20261020/sources.json)和[脚本快照](env-service-verification/results/science-single-gpu-training-20261020/scripts-snapshot/)保留运行条件。

## 6. 验证边界与资源清理

- 一次短对照用于验证资源竞争时的执行收益，不证明长期稳定性、模型收敛或科学任务能力提高。
- 固定历史 OFF 与当前 ON 是两个完整版本的对照，并非只切换一个布尔配置。运行补丁和代码版本必须与结果一起阅读。
- 模型生成的 token 数、执行命令和任务时长可能不同，不能将全部耗时差归因于调度器。分别列出有效 token、采样时间与完整驱动时间。
- 零奖励与零有效 token 不同。如果组内两条奖励都是零，GRPO 的相对优势为零，优化器调用可以完成但没有奖励差异产生的策略梯度。
- 资源预算是准入规则，不是物理显存硬隔离。实验只操作自己的进程、容器、网络和预热镜像标签，不执行全局 prune 或停止其他 Ray 集群。


最终按本轮 Compose project、进程树及 GPU UUID 核验清理：无所属容器、网络或独立 Ray 进程残留，四张目标卡都没有 CUDA 计算进程，NVML 应用显存回到约 0.19 MiB 驱动基线。本轮创建的九个预热镜像标签已移除，没有执行全局 prune。详见[物理清理证据](env-service-verification/results/science-single-gpu-training-20261020/physical-cleanup-final.json)。

[容器集合观测](env-service-verification/results/science-single-gpu-training-20261020/containers.jsonl)包含主容器、独立 verifier 和网络 sidecar。主服务配置为 4 CPU / 8 GiB，sidecar 没有独立 CPU / 内存硬限额。因此“账本无申报越界”不等于整个容器集合具备完整的物理资源硬隔离，本轮没有证明这一点。

验证记录包括[张量回归 3 项通过](env-service-verification/results/science-single-gpu-training-20261020/position-ids-preflight.log)、[修改脚本定向 Ruff 检查](env-service-verification/results/science-single-gpu-training-20261020/targeted-ruff.log)和[隔离副本中的全仓检查](env-service-verification/results/science-single-gpu-training-20261020/precommit-all.log)。全仓检查被已有长行、导入位置和格式问题阻塞，没有批量修改无关文件。准备阶段有两次小型采集脚本错误，分别是把 Compose environment 列表当成字典、使用系统 Python 3.6 启动需要新版本的观测器，均在采集工具层修正并保留初始日志，没有把它们算成训练或 EnvRouter 的失败。

## 7. Results：十步扩展验证

**预定十步没有全部完成。** 两组实际终态如下，ON 的较短退出时间不能当作完整十步加速。

| 全部运行记录 | 固定 OFF | EnvRouter ON |
|---|---:|---:|
| 完成的训练步骤 | 10 | 5 |
| 通过压力与评分核验的轨迹 | 80 | 42，其中第六批两条未进入完成的训练步骤 |
| 已落盘的尝试结果 | 192 | 44，另四个已启动采样位置在清理时没有最终结果 |
| 已记录 OOM | 112 | 0 |
| 非 OOM 拒绝 | 0 | 2，第一条压力证据缺失，第二条为清理时的 ActorDiedError |
| 退出码 | 0 | 1 |
| 包含初始化和清理的驱动耗时 | 109.58 分钟 | 50.98 分钟，因失败提前结束 |

**相同任务数比较只使用双方都完整完成的前五步。** 这是失败后选取的共同连续前缀，不替代预定的十步验收。没有挑选其中最快的步骤，也没有删除受干扰的第五步。

| 前五步的对照指标 | OFF | ON | 解释 |
|---|---:|---:|---|
| 有效轨迹 | 40 | 40 | 每条四轮真实模型交互 |
| 总尝试 | 97 | 40 | ON 不需要 OOM 补采 |
| OOM | 57 | 0 | 相同 72 GiB / 30 秒压力 |
| 训练步骤总耗时 | 2917.93 秒 / 48.63 分钟 | 2110.49 秒 / 35.17 分钟 | 减少 27.67%，约节省 13.46 分钟 |
| 等待采样数据 | 2723.96 秒 / 45.40 分钟 | 1919.02 秒 / 31.98 分钟 | 减少 29.55% |
| 参数更新阶段 | 163.86 秒 | 162.00 秒 | 差约 1.86 秒，基本相同 |
| 有效训练 response token | 48890 | 53665 | ON 多约 9.77%，没有靠更少 token 获得本轮优势 |

具体以第一步为例：OFF 完成相同八条轨迹需要 20 次尝试，12 次 OOM，完整一步 575.19 秒。ON 八次尝试全部成功，一步 437.04 秒。二者都执行了真实评分和参数更新，差值不能解释成 ON 跳过了训练。五步合计节省约 807.44 秒，其中等待采样减少约 804.95 秒，说明本轮时间差主要出现在采样、任务与评分链路，参数更新内核几乎没有变化。模型输出内容、版本差异及节点干扰仍可能影响具体比例。

![双方已完成的前五步对比](env-service-verification/results/science-ten-step-training-20261020/matched-prefix.png)

逐步数据如下。表中“未观测到”仅指全卡遥测未发现其他 GPU 超过 64 MiB 的显存占用，不构成 CPU、存储或节点独占保证。

| 步骤 | OFF 完整步骤秒数 | ON 完整步骤秒数 | OFF OOM | ON OOM | OFF 该批观测边界 |
|---|---:|---:|---:|---|---|
| 1 | 575.19 | 437.04 | 12 | 0 | 未观测到另卡显存活动 |
| 2 | 573.85 | 384.93 | 9 | 0 | 未观测到另卡显存活动 |
| 3 | 567.93 | 405.02 | 12 | 0 | 未观测到另卡显存活动 |
| 4 | 583.88 | 456.81 | 12 | 0 | 未观测到另卡显存活动 |
| 5 | 617.08 | 426.69 | 12 | 0 | 另卡活动重叠 |
| 6 | 606.52 | 未完成 | 11 | 不作完成对比 | 另卡活动重叠 |
| 7 | 697.49 | 未完成 | 13 | 不作完成对比 | 未观测到另卡显存活动 |
| 8 | 575.12 | 未完成 | 12 | 不作完成对比 | 未观测到另卡显存活动 |
| 9 | 528.75 | 未完成 | 8 | 不作完成对比 | 未观测到另卡显存活动 |
| 10 | 618.80 | 未完成 | 11 | 不作完成对比 | 未观测到另卡显存活动 |

![实际完成的步骤与 token 数](env-service-verification/results/science-ten-step-training-20261020/per-step.png)

ON 确实出现过同一快照四个会话、一条 GPU 命令活动、三条命令等待，未发现申报预算越界。两组通过核验的 CUDA 阶段都在同一 UUID 的环境卡运行，峰值显存 OFF 74565 MiB、ON 74053 MiB。这里仍是显存容量竞争，不能推导出持续计算满载、显存碎片改善或 Docker 与 vLLM 同卡借还。

OFF 第五、六批分别有 16、15 个另卡显存占用观测点，最大 5385 MiB。已确认有另一项 profiling 进程使用同节点资源。ON 未观测到其他 GPU 超过 64 MiB 的占用。共同前五步包含受干扰的 OFF 第五步，因而 27.67% 是这一对照的观测收益，不是严格隔离后的纯调度器因果比例。两组已完成步骤的奖励、策略 loss 和梯度范数均为零，不能把优化器执行成功写成科学任务能力改善。

结果与图均由 [analyze_single_gpu_training.py](env-service-verification/scripts/analyze_single_gpu_training.py) 从本轮文件生成。[完整统计](env-service-verification/results/science-ten-step-training-20261020/comparison.json)保留全部运行、共同前缀与干扰记录，[失败分析](env-service-verification/results/science-ten-step-training-20261020/failure-analysis.json)保留首个拒绝及源码指纹核验。两组运行时的提交、关键源码哈希和补丁均未改变。

[物理清理记录](env-service-verification/results/science-ten-step-training-20261020/physical-cleanup-final.json)确认所属容器、网络、独立 Ray 进程及 CUDA 计算进程已释放，四张目标 GPU 的应用显存回到约 0.19 MiB。只删除本轮八个预热镜像标签，没有执行全局 prune。监控中的 Docker inspect 偶发遇到正在退出的容器，记录为遥测缺失，没有用于触发训练失败。


本节对应用户授权的十步扩展，前六节的一步实验仍保留原始条件和结论。扩展问题是：重复采样和训练之后，显存准入减少 OOM 补采的作用是否持续，完整十步工作是否更快。执行链路沿用第 1 节，四个原始任务重复十次，每次各采样两条轨迹，合计 80 条。OFF 仍为固定历史提交加已披露运行修复，ON 固定在 `37a06ea` 的独立 checkout，运行时不吸收其他会话正在编辑的代码。

初始化预算从训练子进程启动起计 1200 秒。代理首次收到采样位置时写入原子开始标记，正式工作预算从该时刻起计 7200 秒。预算分开是为了避免初始化时间较长导致尚未完成第一个更新就被截断，不代表允许运行中自动延长时间。任务超时、最多十二次 OOM 尝试、四个活动尝试、72 GiB 持有 30 秒、四轮模型交互和 token 上限均与一步实验相同。每组达到十步即停止。

逐步比较训练日志的 `timing_s/step`、`timing_s/wait_for_gen` 和 `timing_s/update_actor`。它们分别表示该步整体耗时、训练器等待可用采样数据的时间和参数更新阶段时间。注意，等待采样还可能包含评分、任务清理及流水线协调，不能直接当成纯模型生成耗时。多步运行中“第一条采样开始到最后一条采样完成”还跨过中间九次优化更新，不能继续将整个跨度称为纯采样时间。

可复现命令沿用第 5 节，只将步数与预算改为：

```bash
--steps 10 --timeout 8400 --initialization-timeout 1200 --work-timeout 7200
```

此代码块是参数替换片段，不是独立 shell 命令。完整实际启动命令保存在本次结果目录的 `launch.sh`。可先给完整命令追加 `--prepare-only`，只生成数据和配置而不启动模型，用于公共镜像预热。分析仍使用同一个 `analyze_single_gpu_training.py`，按本次设计中的目标步数核验完成数，并按输入批次汇总尝试和 OOM。请求中的 `version_tag=-1` 是占位标签，不能直接当成实际模型版本。批次由连续的 parent_id 和四题批量计算，生成规则见 [get_train_sample_ids](../../psrl/utils/dataset/data_processor.py#L491) 与 [request_ids 生成](../../psrl/utils/dataset/data_processor.py#L568)。

本次证据：[预先冻结的协议](env-service-verification/results/science-ten-step-training-20261020/protocol.md)、[设计](env-service-verification/results/science-ten-step-training-20261020/design.json)、[预检](env-service-verification/results/science-ten-step-training-20261020/preflight.json)、[源码和运行补丁](env-service-verification/results/science-ten-step-training-20261020/sources.json)、[张量回归](env-service-verification/results/science-ten-step-training-20261020/position-ids-test.log)、[镜像预热](env-service-verification/results/science-ten-step-training-20261020/image-prewarm.json)。本次运行已结束，ON 在第六步停止，未完成十步验收。运行时脚本快照保留了故障版，当前工作区脚本包含第 8 节的事后修复，不能把修复后的 CPU 回归写成本次十步训练通过。

## 8. 证据文件删除故障与修复

先把实际失败和修复后的路径串起来。图中的旧路径来自本轮运行快照，新路径是本轮结束后修改并做 CPU 回归的实验代码，没有用它追加真实训练。

```text
[1] 真实模型生成第一条回复
      |
[2] 在执行模型命令之前，注入 CUDA 压力并得到 stdout JSON
      |
      +-- 运行时旧路径：证据仅位于容器可写的 /logs
      |      |
      |    [3] 模型执行 rm -rf /logs，删除证据内容
      |      |
      |    [4] 实验循环读不到压力 JSON，拒绝 UID 41
      |      |
      |    [5] 补采熔断器令训练退出，清理代码 ray.kill Router
      |           后续请求出现 ActorDiedError
      |
      +-- 修复后的路径：[6] 宿主机先保存原始 stdout 到独立审计目录
                         |
                       [7] 模型执行自己的命令，即使删除 /logs
                         |
                       [8] 实验循环仍从宿主机审计目录读取压力状态
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与已核对的源码或证据 |
|---|---|
| 1、2 | [PressureTerminus2._query_llm](env-service-verification/scripts/science_pressure_agent.py#L21) 在真实模型回复后插入 CUDA 阶段，再把回复交还给代理执行 |
| 3 | [UID 41 模型轨迹](env-service-verification/results/science-ten-step-training-20261020/command/pool-0/harbor-jobs/01a158e1-c461-7940-be27-e930da2d73da/2026-10-20__18-35-55/laps-repair-coef-2d-rktmod-l17k1__iuX4FdH/agent/trajectory.json#L70) 记录实际清理命令和终端输出 |
| 4 | [本轮运行快照的读取逻辑](env-service-verification/results/science-ten-step-training-20261020/scripts-snapshot/science_fixed_work_agent.py#L36) 只搜索容器导出的压力文件，找不到就标为 missing |
| 5 | [失败计数与熔断](../../psrl/workers/agent_loop/manager.py#L558)、[TaskRunner finally](../../psrl/trainer/main_ppo.py#L429)、[stop_env_service](../../psrl/trainer/ppo/ray_trainer.py#L596) 串起拒绝、训练退出和 Router 清理 |
| 6 | [FixedWorkPressureAgent._exec_pressure](env-service-verification/scripts/science_fixed_work_agent.py#L21) 拦截真实执行结果，调用 [save_pressure_evidence](env-service-verification/scripts/science_pressure_audit.py#L19) 原子落盘 |
| 7 | 继续原始代理流程，没有限制模型清理任务产物的命令，也没有更改科学任务评分 |
| 8 | [FixedWorkAgentLoop._run_harbor_in_thread](env-service-verification/scripts/science_fixed_work_agent.py#L45) 调用 [load_pressure_evidence](env-service-verification/scripts/science_pressure_audit.py#L36)，不再依赖容器导出文件 |

这些源码如果有理解上的问题，可以随时追问。第 5 节的正常执行阅读路径仍适用，故障路径按 1 → 2 → 3 → 4 → 5 阅读，修复后的证据路径按 1 → 2 → 6 → 7 → 8 阅读。多个任务仍并发执行，图中只跟踪 UID 41 这一条轨迹，不能把全系统误读为串行。

本次模型执行的命令是 `rm -rf /logs && mkdir -p /logs/artifacts`。终端虽然提示挂载点本身 `Device or resource busy`，但挂载点里的文件已经被递归删除。`/logs/agent` 与 `/logs/artifacts` 分别映射到宿主机本次 trial 的目录，所以把同一证据另存到 `/logs/agent` 也无法避免这次删除。最终 artifact manifest 明确记录目录为空。

UID 41 实际完成四轮模型交互和原评分，`exception_type=null`，但采集脚本得到 `pressure.status=missing`，因此按事先确定的严格规则拒绝该样本。此时只重试 CUDA OOM 的规则不允许将缺失证据当成功，组失败阈值为一，训练停止。随后 `TaskRunner` 的清理代码调用 `stop_env_service` 并杀掉 Router，另一个尚在启动的 UID 46 才报告 ActorDiedError。根因在实验审计数据的存放位置，不能只截取最后的 ActorDiedError 就判定 Router 自发崩溃。

修复让宿主机在压力命令返回后、模型命令执行前保存原始 stdout 和返回码。`SCIENCE_PRESSURE_AUDIT_DIR` 由启动器显式传给 worker，指向输出根目录下独立的 `pressure-audit/`，不挂载到任务容器。文件名依据 trial 日志路径生成，OFF 和 ON 共用同一捕获逻辑。宿主机审计目录不能位于 trial 目录内，避免再次放进 agent、verifier 或 artifacts 的可写挂载。原容器证据仍可用于观察，但不再是通过判定的唯一来源。

可运行的 CPU 回归如下，先按第 5 节激活项目环境并设置源码、依赖路径：

```bash
python -m pytest tests/env_service/test_science_pressure_audit.py -q
```

[测试代码](../../tests/env_service/test_science_pressure_audit.py#L26)给 OFF 与 ON 的实际代理方法提供 CPU 执行结果替身，随后删除临时 trial 的整个 agent 日志目录，确认宿主机证据仍可读取。另三项覆盖 OOM 保留与缺失拒绝、损坏输出保留、拒绝把审计目录放进 trial。[实测结果为五项通过](env-service-verification/results/science-ten-step-training-20261020/pressure-audit-regression.log)。这复现并验证的是证据生命周期，未启动 CUDA、容器或新的训练，不能据此声称修复版已跑完十步。

下次实验需在启动前确认审计目录没有被任何任务容器挂载，保持严格失败判定，保留本轮故障结果。不要将代理能改写或删除的任务产物作为唯一可信运行证据，也不要为了得到完整曲线自动提高重试次数或重跑已完成的 OFF。

本轮检查结果：position_ids 张量回归三项通过，证据生命周期 CPU 回归五项通过，本次五个新增或修改 Python 文件的 Ruff 检查与格式检查通过。全仓 `pre-commit run --all-files` 在隔离快照执行，仍有仓库既有的风格问题，包括 `examples/sciaccel_rl/diagnostics/gate_a_after.py` 的注释与文档字符串规则，未修改其他会话文件。运行日志保存在 [precommit-final.log](env-service-verification/results/science-ten-step-training-20261020/precommit-final.log)，其中本次分析脚本的一处超长行已在最终工作区格式化并通过定向检查。运行时脚本保留在 `scripts-snapshot/`，事后修复另存 `post-run-fix/`，不会覆盖失败现场。

### 为什么曲线从第六步开始没有数据

这里需要区分三层数据：模型每次生成回复的数据、一条完整任务轨迹的数据，以及训练器完成一个批次更新后打印的 step 指标。每步配置是四题、每题两条轨迹，即八条符合验收条件的轨迹，再执行一次训练更新。轨迹内部最多四轮模型交互，“第六个训练 step”不是“第六轮模型对话”。

本轮前五步已经消费 40 条合格轨迹，第六步对应 UID 40 至 47。原始记录里 [UID 40](env-service-verification/results/science-ten-step-training-20261020/command/pool-0/attempts/40-01.json) 和 [UID 42](env-service-verification/results/science-ten-step-training-20261020/command/pool-1/attempts/42-01.json) 已被接受，而 [UID 41](env-service-verification/results/science-ten-step-training-20261020/command/pool-0/attempts/41-01.json) 有 `has_output=true`、评分字典及 `exception_type=null`，却因 `pressure.status=missing` 得到 `accepted=false`。这说明第六步已产生部分采样数据，但没有凑齐实验要求的批次并完成更新。训练日志只存在步骤 1 至 5 的完整指标，第六步应标“未完成”，第七至十步未推进，不能补零、插值或把它解释成绘图漏读。

“missing”为何必须拒绝：本实验要比较同样的 72 GiB 压力，验收不仅要求任务有输出，还要求保留压力状态为 ok 的证据。文件消失后，不能只凭模型正常返回就将这条轨迹计为满足同等压力的成功样本。事先规则只允许对 CUDA OOM 补采，证据缺失属于另一类异常，达到组失败阈值后退出。问题在于实验脚本把唯一证据放进模型能删除的目录，而非严格验收本身。

第六步没有特殊的调度阈值，也没有证据支持“运行五步后 Router 必然失效”。本次恰好在第六批的一条真实模型轨迹里出现清理 `/logs` 的命令，暴露了旧采集路径的问题。后续 ActorDiedError 是退出时关闭 Router 的连带结果。它也不同于第 10 节无压力续跑第三步的镜像请求 EOF，两次失败不能混作一个原因。

当前修复已把压力返回结果先存到模型不可写的宿主机独立目录，证据删除 CPU 回归五项通过，运行命令及边界见本节上文。修复后没有重跑这组十步压力训练，故仍只能报告共同前五步的观测收益，不能称完整十步已验证。

## 9. Results：不注入压力的原生任务对照

**本轮未得到无压力性能结果。** OFF 在首批任务代理构造时因实验脚本配置错误退出，未进入采样，完成零个训练步骤。ON 已取消启动，不能将“没有结果”解读为“没有收益”。本节保留用户追问、实验设计、失败证据及修复边界。

| 实际状态 | OFF | ON |
|---|---|---|
| 计划训练步骤 | 3 | 3 |
| 完成训练步骤 | 0 | 未启动 |
| 原生任务采样位置 / 结果 | 0 / 0 | 未启动 |
| 终止原因 | `harbor.gpu_compose_override=null` 不符合字符串字段要求 | OFF 失败后取消 |
| 可用于性能比较的数据 | 无 | 无 |

GPU 模型初始化已经消耗时间，但这不是训练性能样本，不计算零秒训练或加速比。[分析脚本输出](env-service-verification/results/science-native-training-20261020/comparison.json)将缺失训练指标记为 `null`，不绘制伪造的零耗时对比。

以前一轮约 95.58 GiB 的环境卡为例，两个任务各申请 72 GiB，合计 144 GiB，无法同时容纳。OFF 允许并发尝试，出现 OOM 后补采。ON 将每条 GPU 命令按 73 GiB 记入 80 GiB 的预算，一条执行时，其余命令排队。前五步的 57 次 OOM 降到零，支持这个受控显存竞争条件下的收益。它不等于任意外部程序占满 GPU 都能被安全调度，也不证明持续计算满载时一定提速。环境卡与模型卡分开，也不是 Docker 与 vLLM 同卡借还实验。

本轮移除人为负载。四个原始 LAPS 任务是在 CPU 上修复 Fortran 程序，每任务原始声明为 4 CPU 和 8192 MiB，未要求 GPU。模型推理与训练仍使用真实 GPU，任务环境恢复为原生 CPU 工作量，不能称为原生 GPU science 实验。

下面是预定的执行链路。实际只完成模型等组件初始化，在进入代理运行前的配置校验处失败，未执行图中的修复、评分和训练更新。

```text
[1] 相同 Qwen3.5-4B 与四个原始 LAPS 修复任务
                       |
[2] 真实模型生成回复，原始 TruncatingTerminus2 执行命令
                       |
       +---------------+---------------+
       |                               |
[3a] 固定历史 OFF                 [3b] 冻结版本 ON
进程内并发准入                   Router 的 CPU / 内存会话准入
       |                         16 CPU / 32 GiB / 四个会话
       +---------------+---------------+
                       |
[4] CPU 容器修改 Fortran、独立 verifier 评分
                       |
[5] 四题各两条轨迹组成一个训练 batch，执行参数更新
                       |
[6] 达到三步或预算/故障条件即退出，核验物理清理
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与源码 |
|---|---|
| 1 | 任务按原文件内容复制，生成训练数据与哈希，[science_native_workload.py](env-service-verification/scripts/science_native_workload.py#L20) |
| 2、3、6 | 选择原始代理、建立独立 Ray、启用 CPU unit、设置预算并清理，[science_fixed_work_training.py](env-service-verification/scripts/science_fixed_work_training.py#L173) |
| 4 | 通过实际 science runner 执行任务与评分，[science_native_workload.py](env-service-verification/scripts/science_native_workload.py#L93) |
| 5 | 限制四并发，保留原生任务结果并交还训练器，[science_native_workload.py](env-service-verification/scripts/science_native_workload.py#L97) |
| 5 | [训练器 fit](../../psrl/trainer/ppo/ray_trainer.py#L3343) 消费采样批次并执行真实优化步骤 |

跟着图读源码：1 先生成相同任务数据 → 2、3 建立对应组的训练器与任务环境 → 4 在各并发轨迹中调用真实代理和 verifier → 5 汇总有效 token、评分并更新参数 → 6 在停止条件到达后清理所属资源。图中多个轨迹会交错完成，ON 的会话心跳与账本回收也并发运行。这些代码如果有理解上的问题，可以随时追问。

第 2 步没有插入压力代理、CUDA 分配、额外等待或 GPU 命令预约。两组使用相同原始任务文件副本，生成时逐文件比较内容并保存哈希。图表示依赖关系，真实任务并发运行。

每组预设 3 个更新、24 条轨迹、4 并发、每轨迹最多 4 轮模型交互，response 上限 8192。初始化限时 1200 秒，正式工作从首次采样标记起限时 2400 秒，每条采样位置只尝试一次。两组沿用上一轮相同的 OFF / ON 源码指纹和运行补丁，镜像在计时前共同预热。比较步骤耗时、等待采样、参数更新与包含初始化和清理的总墙钟，另外记录 token 数、评分和同节点活动。

本轮证据：[协议](env-service-verification/results/science-native-training-20261020/protocol.md)、[设计](env-service-verification/results/science-native-training-20261020/design.json)、[任务一致性](env-service-verification/results/science-native-training-20261020/task-integrity.json)、[源码指纹](env-service-verification/results/science-native-training-20261020/sources.json)。运行入口沿用第 5 节启动方式，参数设置为 `--workload native --steps 3 --modes off on`，完整实际命令保存在结果目录的 `launch.sh`。这里的参数是配置片段，不是独立 shell 命令。

**失败机制与修复。** 宿主机将任务代理配置写成 YAML，worker 构造代理时调用 [build_runtime_config](../../examples/sciaccel_rl/config.py#L52)，它按 dataclass 的字段类型合并配置。[HarborConfig](../../examples/sciaccel_rl/config.py#L25) 将 `gpu_compose_override` 声明为 `str`，默认值是空字符串。空字符串表示不使用 GPU Compose 文件，Python 的 `None` 与 YAML 的 `null` 则不是字符串，不能互换。

```text
运行时错误路径：启动模型 → worker 读取 agent.yaml → null 合并到 str 字段 → 校验异常
修复后的路径：生成代理配置 → 用同一个 schema 校验 → 通过后才创建 Ray 和加载模型
```

| 环节 | 已核对源码或证据 |
|---|---|
| 错误配置来源 | [运行时快照第 199 行](env-service-verification/results/science-native-training-20261020/scripts-snapshot/science_fixed_work_training.py#L199) 将字段写成 None |
| 实际传给 worker 的 YAML | [off/agent.yaml](env-service-verification/results/science-native-training-20261020/off/agent.yaml#L4) 保留 null，未覆盖失败现场 |
| 提前校验和空字符串修复 | [build_agent_configs](env-service-verification/scripts/science_fixed_work_training.py#L139) 调用真实 `build_runtime_config`，在 [run_arm](env-service-verification/scripts/science_fixed_work_training.py#L173) 创建 Ray 前执行 |
| 防止失败后继续启动另一组 | [组间退出检查](env-service-verification/scripts/science_fixed_work_training.py#L589) 检查失败返回码 |
| CPU 回归 | [回归测试](../../tests/env_service/test_science_native_config.py#L12) 验证两种模式配置，并复现 null 的拒绝行为 |

主训练配置日志中的 `All configuration checks passed` 不覆盖随后加载的代理 YAML，因此不能把前者当作所有 worker 配置已通过的证明。这里新增的提前校验使用与真实代理相同的构造函数。

先按第 5 节激活环境并固定源码路径，再运行以下 CPU 回归，不需要加载模型或启动容器：

```bash
python -m pytest tests/env_service/test_science_native_config.py -q
```

[本轮回归三项通过](env-service-verification/results/science-native-training-20261020/native-config-regression.log)。两项将修复后的 OFF / ON 配置送入真实 schema，另一项故意传入 None 并确认抛出同样错误。这证明配置问题被修复，不证明修复版已完成真实训练。本轮未追加 GPU 重试。

**准备与清理边界。** 原始任务副本内容核验、八个镜像预热、三张模型 GPU 的小计算、CPU 容器探针以及 position_ids 三项回归均已通过。首次预热误用了 host 构建网络，与 Harbor 默认构建网络不同，缓存未命中并在 120 秒超时。保留失败日志后，使用实际默认构建选项完成一次有界纠正，未计入训练比较。

发现代理配置错误后，对本轮训练子进程组发送 SIGTERM，驱动完成所属资源清理，并阻止故障版驱动自动启动 ON。最终驱动日志中的 `FileExistsError` 是为阻止继续启动而设置的目录保护，不是原始故障原因。修复版驱动现在会在任一组失败后直接停止，不再需要该保护手段。

[失败分析](env-service-verification/results/science-native-training-20261020/failure-analysis.json)和[物理清理证据](env-service-verification/results/science-native-training-20261020/physical-cleanup-final.json)确认本轮所属容器、网络、独立 Ray 和 GPU 计算进程已释放，只移除本轮八个预热镜像标签。运行时源码指纹和补丁未改变，故障脚本保留在 `scripts-snapshot/`，事后修复另存 `post-run-fix/`。

本轮代码的定向 Ruff 与格式检查通过。全仓 pre-commit 在隔离快照执行，仍有既有风格问题，日志保存在结果目录，没有修改其他会话代码。无压力条件下是否更快仍然未验证，前一轮压力收益不能填补这个空白。

## 10. Results：无压力续跑与共同完成的两步

**2026-10-21 续跑结论：没有观察到无压力提速。** 修正第 9 节的代理配置后，OFF 完成三步，ON 完成两步，在第三步评分器构建时遇到镜像仓库请求 EOF 并退出。双方共同完成的前两步分别为 **521.17 秒和 535.42 秒，ON 慢 2.73%**。这是一次顺序对照的观测结果，完整三步对照未完成，不能将失败后的总墙钟作为等完成量性能比较。

本次沿用第 9 节的完整执行链路，实际已走通真实模型生成、CPU 容器执行、独立评分、有效 token 汇总和优化器调用。结果保存在新的 [science-native-training-20261021](env-service-verification/results/science-native-training-20261021/) 目录，上一轮初始化失败现场没有覆盖。

**条件与真实工作量。** OFF 仍是 `0eeead861ffafed6bf31c962d259fd847775dcf9`，ON 仍是 `37a06ea8d32b05bbbf385ad40cb404b30ba01680`，两组运行补丁与上一轮压力实验一致。它不是对工作区其他尚未纳入该提交的优化重新测量。Qwen3.5-4B 用两张 GPU 训练、一张 GPU 推理，四个原始 LAPS 修复任务的文件内容未改变。每步四题、每题两条轨迹，共八条，每条最多四轮，两组活动尝试上限均为四。

启动前在两份实际冻结源码中校验了全部代理配置，position_ids 和 native 配置回归共六项通过，八个镜像公共预热成功。准备限时 900 秒，每组初始化限时 1200 秒，首次采样后工作限时 2400 秒。实际工作未超时，本次停止原因是第三步的明确任务异常。

### 共同完成部分的数字

下表只统计双方都完成的步骤 1、2。第 3 步的未完成工作、故障等待与清理不混入该表。

| 指标 | OFF | ON | 解释 |
|---|---:|---:|---|
| 完成更新步骤 | 2 | 2 | 各自完整的前两步 |
| 有评分且无异常的轨迹 | 16 | 16 | 每题每步各两条，均四轮 |
| 第一步耗时 | 274.08 秒 | 277.42 秒 | 同一计时字段 |
| 第二步耗时 | 247.09 秒 | 258.00 秒 | 同一计时字段 |
| 步骤总耗时 | 521.17 秒 | 535.42 秒 | ON 增加 14.25 秒，慢 2.73% |
| 等待采样数据 | 434.95 秒 | 453.46 秒 | ON 增加 4.25% |
| 参数更新阶段 | 72.74 秒 | 68.60 秒 | ON 减少 5.69%，不能单独解释为调度收益 |
| 有效训练 token | 20,281 | 21,855 | ON 多 7.76%，实际生成文本不完全相同 |
| OOM / 失败补采 | 0 / 0 | 0 / 0 | 仅指这两步，无显存失败可供消除 |
| 实际 reward_repair / 策略梯度范数 | 均为 0 | 均为 0 | 不证明模型学会修复任务 |

![无压力两步对照](env-service-verification/results/science-native-training-20261021/comparison.png)

图由 [analyze_native_training.py](env-service-verification/scripts/analyze_native_training.py) 生成。分析器保留完整运行状态，同时单列 `common_prefix`，即双方从第一步开始连续完成的共同部分，并核验对应组的轨迹数。它不会把 OFF 的三步和 ON 的两步画成等工作量比较，也不会因为两步完成就将整场标记为成功。

**这与之前的压力收益为什么不矛盾。** 之前在约 95.58 GiB 的环境 GPU 上，让多个容器分别申请 72 GiB，两个同时申请就需要 144 GiB。ON 的准入把这种冲突变成排队，减少失败重试。本次四个环境任务本来就只用 CPU。现场四个会话各申报 4 CPU、8 GiB，合计正好是 16 CPU、32 GiB，且共同活动上限就是四，没有第五个活动请求进入 Router 等待。因而既没有 GPU 显存争用，也没有超额会话排队可供优化。这个实际例子说明，调度器解决特定资源冲突，不能据此推导所有任务都会变快。

ON 的 835 份有效账本快照中，最高四会话、16 CPU、32 GiB，等待会话和 GPU 命令峰值均为零。Docker 审计记录的两组主容器与独立评分器均为 4 CPU、8 GiB，未发现 GPU DeviceRequests 或显式设备映射。网络辅助容器没有单独 CPU、内存硬上限，因此账本预算仍不能解释为整个容器集合的物理消耗上限。遥测未发现其他 GPU 超过 64 MiB 的占用，但这不等于排除了所有 CPU、网络或存储干扰。

这里的“有效轨迹”是有模型输出、评分记录且没有运行异常，不表示题目修复成功。两组共同部分均保留有效 token，实际执行优化器调用，但同组奖励全零，策略梯度范数也为零。不能把它写成模型效果提高。两步、单次顺序执行以及不同生成长度，也不足以断言 Router 稳定慢 2.73%，能够给出的结论是本次没有测出无压力收益。

### 第三步为何失败及为什么预热仍不够

下面是本次查证的异常传播过程，训练与清理回调并发执行，不能将日志显示顺序误读成所有工作串行发生。

```text
[1] 任务完成模型交互，独立 verifier 准备评分镜像
    ↓ Harbor 选择源码构建路径，执行 docker compose build
[2] 构建器解析 python:3.13-slim 的远端镜像元数据，请求返回 EOF
    ↓ 该轨迹带 RuntimeError，实验包装器拒绝将它算作成功
[3] 组失败阈值达到 1，通知等待批次的训练器退出
    ├─ 训练器清理 Router，活动租约尚未完全退出
    └─ 并发失败回调仍派发两个替补请求，因 Router 已关闭再次失败
[4] 外层驱动按所属资源清理，核验 Docker、Ray 和 GPU 物理释放
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与已核对源码 |
|---|---|
| 1、2 | 本次实际 Harbor 依赖快照的 [DockerEnvironment.start:868](env-service-verification/results/science-native-training-20261021/failure-sources/harbor-docker.py#L868)，非预构建模式在第 896 行执行 compose build |
| 2 | 本次评分器 [Dockerfile](env-service-verification/results/science-native-training-20261021/tasks/laps-repair-subtle-2d-rktmod-l18k3/tests/Dockerfile) 使用 python:3.13-slim，[原始故障片段](env-service-verification/results/science-native-training-20261021/failure-excerpt.log) 保存构建返回码 17 和 EOF |
| 2、3 | [NativeWorkAgentLoop.run:97](env-service-verification/scripts/science_native_workload.py#L97) 对非空 exception_type 拒绝计入成功 |
| 3 | ON 实际源码快照的 [_record_group_failure:558](env-service-verification/results/science-native-training-20261021/failure-sources/manager.py#L558) 唤醒等待者，随后 [notify_group_failed:842](env-service-verification/results/science-native-training-20261021/failure-sources/manager.py#L842) 仍继续调用补采 |
| 3 | [stop_env_service:596](env-service-verification/results/science-native-training-20261021/failure-sources/ray_trainer.py#L596) 尝试关闭节点服务，finally 关闭 Router |
| 4 | [run_arm:173](env-service-verification/scripts/science_fixed_work_training.py#L173) 的 finally 负责所属进程、容器清理和显存释放核验 |

跟着图读源码：1 看 `_use_prebuilt` 的分支 → 2 看本次 Dockerfile 与 EOF 原始记录 → 3 看实验成功判定、失败阈值和并发清理回调 → 4 看驱动 finally 和物理释放结果。这些代码如果有理解上的问题，可以随时追问。

**预热缓存与离线运行不是一回事。** 镜像由基础镜像和多个构建层组成。预热可以使安装依赖、编译 Fortran 等层命中缓存，但当前 Harbor 的源码任务路径仍会发起 compose build。此次构建器仍尝试查询 `python:3.13-slim` 标签对应的远端元数据，这一步在复用后续层之前就失败了。EOF 表示连接未返回完整预期响应，现有证据只支持“该次镜像请求失败”，不能断言镜像仓库整体长期不可用，也不是本机 Docker daemon 挂掉。

例如本次第 22 号采样位置已经有模型输出，评分记录也有字段，但同时带有 RuntimeError。只检查 reward 字典存在就可能把基础设施故障当作正常零分。本次包装器检查 exception_type，正确拒绝了这条轨迹，因此第三步没有形成完整训练批次。未来若要避免这类网络依赖，应准备固定版本的预构建任务与评分镜像，并确认实际 Harbor 选择预构建分支。这是下次运行准备建议，本次没有修改共享 Docker、Harbor 或任务文件来重新训练。

另一个发现是 `retry_limit=1` 与 `refill_failure_threshold=1` 不等于绝对禁止创建替补请求。前者约束同一请求内尝试次数，后者唤醒训练器退出，但本版本失败回调仍可能继续调用组级补采。因此 ON 实际出现 26 个开始标记、24 份最终记录，其中 21 条接受、1 条原始构建异常、2 条退出期间 ActorDiedError，另 2 条在退出时没有完整尝试记录。不能把这 2 条未完成记录算成成功或零耗时，也不能宣称整场严格只有 24 次开始。本次没有无限补采或额外手工重跑。

### 完成边界、复现与资源释放

| 整场状态，不用于等完成量提速比较 | OFF | ON |
|---|---:|---:|
| 完成步骤 / 目标 | 3 / 3 | 2 / 3 |
| 退出码 | 0 | 1 |
| 首次采样前准备 | 612.65 秒 | 649.98 秒 |
| 含初始化、失败与清理的总墙钟 | 1435.93 秒 | 1429.88 秒 |
| 最终物理释放核验 | 通过 | 通过 |

ON 的总墙钟略小是因为没有完成第三次更新，不能写成端到端加速。OFF 交接清理时有一次 GPU 查询超时，后续有效查询确认释放，未将缺失数据补零或误判训练失败。最终所属容器、网络和独立 Ray 无残留，三张模型 GPU 无计算进程，只删除本轮八个预热镜像标签。

复现分析不需要重新训练。在仓库根目录激活项目环境后执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_native_training.py \
  docs/env-worker-deep-dive/env-service-verification/results/science-native-training-20261021
```

预期输出中的 `step_seconds_reduction_percent` 为约 `-2.7341`，负值表示 ON 耗时增加，并生成两步共同部分的 PNG、SVG 和 JSON。实际已执行该分析并检查图表，不是教学模拟。重新启动真实训练需要重新预检并使用新的输出目录，不能直接覆盖此目录。

证据入口：[实验协议](env-service-verification/results/science-native-training-20261021/protocol.md)、[配置设计](env-service-verification/results/science-native-training-20261021/design.json)、[源码指纹](env-service-verification/results/science-native-training-20261021/sources.json)、[分析 JSON](env-service-verification/results/science-native-training-20261021/comparison.json)、[任务与容器审计](env-service-verification/results/science-native-training-20261021/evidence-audit.json)、[六项回归](env-service-verification/results/science-native-training-20261021/position-ids-test.log)、[最终物理清理](env-service-verification/results/science-native-training-20261021/physical-cleanup-final.json)。定向 Ruff 和格式检查通过，全仓 pre-commit 在隔离副本执行，仍有其他文件的既有风格问题，见[检查日志](env-service-verification/results/science-native-training-20261021/precommit-all.log)。
