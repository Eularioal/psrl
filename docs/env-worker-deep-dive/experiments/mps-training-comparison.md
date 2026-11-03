# MPS 真实 GPU 训练并发对照

> **核心结论**：本报告验证相同训练工作量下，串行、普通 CUDA 多进程和 MPS 的完成时间。使用 autoresearch 的真实模型、数据、前后向与优化器，经 EnvRouter 启动 Docker GPU 作业。这不是原生 LAPS science 收益，也不是固定历史 OFF/ON 对照。第一轮 MPS 完成时间比普通多进程短约 10.7%，比串行短约 2.7%，初次第二轮发现收尾控制错误，后续已定位并修复管理连接竞争，原失败组定向复测通过。性能数字仍按原实验条件解释，不推导为长期稳定性证明。修复后新配对结果见第 7 节：MPS 训练段耗时减少 10.4%，但包含创建、准备和清理的短流程耗时增加 2.1%。

## 目录

- [1. 验证问题和执行路径](#1-验证问题和执行路径)
- [2. 三组对照和计时方法](#2-三组对照和计时方法)
- [3. Results](#3-results)
- [4. 对 Router 的含义与验证边界](#4-对-router-的含义与验证边界)
- [5. 复现与跟着图读源码](#5-复现与跟着图读源码)
- [6. 控制通道故障定位与修复](#6-控制通道故障定位与修复)
- [7. 修复后的 MPS 开关速度对照](#7-修复后的-mps-开关速度对照)

## 1. 验证问题和执行路径

已经能启动多个 MPS 客户端、独立取消与回收，不代表它们完成相同训练一定更快。本次只问两个问题：这套真实训练是否获得吞吐收益，以及是否能观察到不同进程的 CUDA 内核在时间上重叠。CUDA 内核是实际交给 GPU 执行的一段计算程序，进程生命周期重叠只是两份程序都没退出，不能代替内核重叠证据。

```text
[1] 固定工作量对照脚本
    │ 每组两个环境，各申请 40000 MiB，episode 持有预算
    ▼
[2] EnvRouter → 原生 EnvWorker → 两个独立 Docker 容器
    │ 相同模型、数据、批大小、3 步预热
    ▼
[3] 等待两份训练准备好，再发送开始标记
    ├─ serial：A 训练 10 步完成后，B 才训练 10 步
    ├─ plain：A、B 同时开始，普通 CUDA 多进程
    └─ mps：A、B 同时开始，连接同一个 MPS 服务
    ▼
[4] 保存完成时间、训练步数、loss、显存和 MPS 客户端证据
    │ 第二轮 plain / mps 在计时之后额外采集 2 步时间线
    ▼
[5] 回收本组容器和服务 → 检查 GPU 物理释放 → 下一组
```

图例：图里每个部分对应的源码位置。

| 图中步骤 | 实现与职责 |
|---|---|
| 1、2 | [mps_training_compare.py](../env-service-verification/scripts/mps_training_compare.py#L30) 的 `arm` 创建 Router、租约、原生环境和受管作业 |
| 3 | [mps_train_workload.py](../env-service-verification/scripts/mps_train_workload.py#L30) 的 `barrier` 用文件标记协调准备、训练和采样阶段 |
| 4 | [mps_training_analyze.py](../env-service-verification/scripts/mps_training_analyze.py#L42) 的 `summarize` 汇总固定工作量耗时并单独分析时间线 |
| 5 | [mps_training_compare.py](../env-service-verification/scripts/mps_training_compare.py#L30) 的 `arm` 在 `finally` 中清理，并检查本组容器和 GPU |

这些源码如果有理解上的问题，可以随时追问。

## 2. 三组对照和计时方法

选择 autoresearch，是因为现场可用的原生 LAPS CUDA 任务 oracle 实际运行 CPU MPI/FFTW，不能拿它证明 MPS 加速。此处直接运行真实 GPU 训练负载，经过 PSRL 的环境调度和作业管理链路，但没有执行 LLM agent rollout 或 PSRL 强化学习训练。

| 条件 | 本次设置 |
|---|---|
| GPU | 一张 NVIDIA H20，NVML 报告总显存 97871 MiB，开始时空闲 |
| 模型 | autoresearch，8 层、约 5033 万参数，序列长度 2048 |
| 数据 | 镜像中的真实 autoresearch tokenizer 和训练数据 |
| 每份工作量 | 3 个预热更新 + 10 个计时更新，每步 524288 tokens，计时段每份 5242880 tokens |
| 批大小适配 | 单设备批大小从上游 128 改为 64，梯度累积从 2 次变成 4 次，总 token batch 不变 |
| 学习率适配 | 从按运行时间计算进度改为按 `step / 13`，避免更慢的组执行不同的调度进度 |
| GPU 预算 | 每个环境申报 40000 MiB，两个环境共 80000 MiB，episode 模式 |
| CPU / 主存 | 每个训练容器 4 CPU、16384 MiB，MPS 服务另留 1 CPU、512 MiB |
| 背景压力 | 没有人工显存占位，也没有 vLLM 与环境同卡运行 |
| 组别顺序 | 原定第一轮 serial → plain → mps，第二轮 mps → plain → serial。实际在第二轮 mps 收尾失败后停止 |
| 采样 | 原定第二轮 plain / mps 在计时后额外执行 2 步。实际取得 mps 时间线，plain 第二轮未运行 |

所有组使用同一份冻结的 PSRL 工作区源码，快照基于 `8d1fc65230d7bd4c78305e140c41ac8b5ea14fb0`，包含当时工作区的 episode 记账修复，完整文件哈希保存在原始结果的 `source/provenance.json`，紧凑 JSON 另附 Env Service 文件哈希与镜像内容 ID。本次没有修改 Router 或 MPS 后端实现。autoresearch revision 为 `228791fb499afffb54b46200aca536f79142f117`，上游 `train.py` SHA256 为 `2954175f4ac42ad65164aef40910ef953789abcd05a5cc886ac9ba5a00814414`。适配脚本启动时核对哈希，保留上游模型、数据加载、前后向和优化器实现。省略上游最终完整评估，因此没有官方分数或收敛结论。

**计时口径**：先依次准备两个进程，让它们都完成 3 步预热，然后放行计时段。总完成时间从第一份开始到最后一份结束，两端都做 CUDA 同步，包含串行组切换的少量控制间隙。初始化、编译、清理单列。第二轮 profiler 的额外两步不计入这段时间。

串行组中第二份模型已经驻留显存，只有训练计算串行。这是为了固定准备条件，并不是“第一份进程退出后才创建第二份”的完整冷启动队列对照。

## 3. Results

以下是 2026-11-02 实测。每组都完成两份各 10 步的相同计时工作量。第一轮三组完整通过，第二轮仅完成 MPS 的计算和采样，框架收尾未通过，因此停止后续组。

| 组别 | 第一轮总完成时间 | 每份 10 步耗时 | 第二轮 | 完整通过情况 |
|---|---:|---|---|---|
| 串行 direct | 20.719 秒 | 10.320 / 10.328 秒 | 未运行 | 第一轮通过 |
| 普通 CUDA 多进程 | 22.578 秒 | 22.576 / 22.578 秒 | 未运行 | 第一轮通过 |
| MPS 多进程 | 20.167 秒 | 20.165 / 20.166 秒 | 计算段 20.208 秒，随后收尾出错 | 第一轮通过，第二轮失败 |

第一轮的等工作量完成时间降低比例为：

- MPS 相比普通并发：`1 - 20.167 / 22.578 ≈ 10.7%`。
- MPS 相比串行：`1 - 20.167 / 20.719 ≈ 2.7%`。
- 普通并发相比串行增加约 9.0%。不能把同时启动两份训练理解为吞吐翻倍。

这里的 10.7% 是时间缩短比例，换算等工作量吞吐约为 1.12 倍。只有一轮完整三组对照，没有统计显著性结论。第二轮 MPS 的计算段接近第一轮，但因完整作业失败，不能混入成功组平均数证明可靠性。

**单任务延迟的具体例子**：串行时 A 约 10.32 秒完成，B 从共同放行基准算约 20.72 秒完成。MPS 时 A、B 都在约 20.17 秒完成。整体只略早完成，却让 A 的完成延迟接近翻倍。吞吐和单任务延迟是不同目标，不能只报其中有利的一项。

**启动与准备另计**：第一轮 serial / plain / mps 的容器及节点准备分别为 18.28 / 15.39 / 21.83 秒，两个进程依次初始化、编译和预热分别为 89.16 / 89.22 / 88.29 秒。包含创建、准备、训练和清理的本组墙钟时间分别为 133.95 / 129.63 / 138.18 秒。在本次短任务流程中，MPS 的训练段小幅收益没有转换为完整流程更快。这些墙钟数不包含 Ray 集群启动，并受启动波动影响，不能解释为 MPS 的固定额外开销。

**真实 GPU 与显存证据**：两次 MPS 组都在预热后查询到两个真实 MPS 客户端，且连接同一服务。第一轮每份 PyTorch 峰值分配约 22853 MiB，整卡峰值约 50311～50364 MiB，包含缓存和上下文。没有人工显存占位，没有 OOM 记录。训练 loss 和步数已保存，第一轮各进程第 10 个计时更新的原始 loss 约 5.84，但没有进行模型质量或收敛评估。

**内核重叠证据**：节点上的 `nsys` 只是缺少实际程序的包装入口，改用 PyTorch profiler。第二轮 MPS 在计时结束后额外运行的 2 步中，两个客户端各导出 4354 个 CUDA kernel 事件。依据 trace 的共同 `baseTimeNanoseconds` 对齐，分别合并每个进程内部的内核区间后，两者交集约 **348.758 毫秒**，至少一方运行的区间并集约 **4039.895 毫秒**，交集占并集约 **8.63%**。

这个比例证明采样窗口内确实存在跨进程内核重叠。它不是 SM 利用率、不是 10 步计时段的重叠率，也不是吞吐提升比例。profiler 本身会影响执行，且没有取得普通并发的配对时间线，不能用该数字断言普通 CUDA 的本次重叠率为零。

**第二轮收尾故障**：两份工作负载都写入 `passed: true` 的结果并导出完整 trace，但框架只把一条作业标为 completed，另一条标为 failed，错误为 `MPS control failed: Cannot send command to MPS control daemon process`。错误发生在 [MpsDomain.cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L342) 调用控制命令的路径，当前入口已改为 [MpsControlChannel](../../../psrl/workers/env_service/mps.py#L11)。异常处理再次清理后确认归还，但仍保留 failed 状态，环境停止也传播该错误。

初次实验所用源码让各作业清理并发调用同一服务的控制命令。第 6 节的后续最小复现已确认这一集成缺口：无需训练或 profiler，仅并发管理连接就能触发相同断连。当前实现已增加按服务串行的控制通道，初次失败记录仍保留。

本节记录的是修复前的性能对照，未为补齐第二轮而继续重跑。后续修复及验证见第 6 节，仍保留“确认物理回收后才归还预算”的约束。

**收尾与证据留存**：最后核对本次 Docker 容器及 MPS 服务均已移除，GPU 无计算进程，NVML 占用回到约 0.19 MiB。物理资源已释放不等于作业协议成功，两者在结果中分别记录。紧凑数据见 [mps-training-results.json](mps-training-results.json)，完整失败日志、训练输出、trace 和源码哈希保存在 [本轮原始结果](../env-service-verification/results/mps-training-20261102/)。

新增脚本通过语法编译和 Ruff 检查，修改文档通过本地链接检查。全仓 pre-commit 在独立快照运行，仍因仓库既有格式及 prose 检查项失败，没有自动修改无关文件。

**基础设施失败单独记录**：第一轮 plain 首次在申请环境时超时，GPU 空闲且没有启动训练。补充显式节点预热与诊断后，只续跑未完成的组。第二轮 mps 首次在节点预热 65 秒内未完成，也没有启动训练。Ray 日志显示等待新的 actor 进程就绪，尚未确认更深层根因。最后一次续跑改为每组独立创建并关闭自己的 Ray 实例，保持 GPU 工作负载、准备屏障和计时边界不变。两份失败记录均保留，不能算作 OOM 或 MPS 训练错误，也不能据此宣称 Ray 问题已修复。

## 4. 对 Router 的含义与验证边界

MPS 提供跨进程 GPU 计算重叠的条件，Router 决定哪些工作可以进来和何时归还预算。两份工作能放进显存，不代表计算单元或显存带宽还有足够余量。即使实际发生内核重叠，共享硬件后的单份执行时间也可能变长，最终总完成时间仍需要实测。

本次使用两个环境容器、每个容器一条 GPU 作业。普通 direct 后端也能在这种条件下跨容器并发，因此没有通过其同容器串行限制人为削弱对照。三组都经过当前 Router 和相同 episode 预算管理，**serial / plain 都不是用户指定的历史 OFF 提交**。

本次最多证明所测训练负载、批大小、并发度和硬件上的短程效果。没有验证原生 LAPS GPU solver、高显存不足下排队、自动选择 MPS、自动 command/episode、offload 或 vLLM 共存。本次只有一轮完整三组对照，第二轮未完整通过，不能给出长期稳定性或统计显著性结论。

## 5. 复现与跟着图读源码

先按集群运行指南检查实际执行节点、Docker、CUDA、GPU 空闲状态和源码依赖来源。下面命令需要当前项目环境，以及含 autoresearch 依赖、真实训练数据和 tokenizer 的 GPU 镜像。`NODE_IP` 是实际执行节点地址，`GPU_INDEX` 是现场核验的 GPU 索引，`IMAGE` 是已准备好的镜像，`AUTORESEARCH_SOURCE` 是上述固定版本源码目录，`RESULT_DIR` 必须是新的结果目录。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/mps_training_compare.py \
  --node-ip "$NODE_IP" --gpu "$GPU_INDEX" --image "$IMAGE" \
  --source "$AUTORESEARCH_SOURCE" --output "$RESULT_DIR"
python docs/env-worker-deep-dive/env-service-verification/scripts/mps_training_analyze.py \
  "$RESULT_DIR" --output "$RESULT_DIR/analysis.json"
```

比较脚本总期限 1500 秒，每条作业执行期限 480 秒。正式运行应通过后台会话或 `nohup` 保存日志，断开 SSH 后任务仍可完成。`--resume` 只跳过已完成且清理确认的组，失败组保留原始结果后才重新执行，不能无限重试。

跟着图读源码：

1. **步骤 1 → 图 1**：[main](../env-service-verification/scripts/mps_training_compare.py#L211) 创建独立 Ray 实例，核对节点地址和依赖来源，然后按预定顺序调用 `arm`。
2. **步骤 2 → 图 2**：[arm](../env-service-verification/scripts/mps_training_compare.py#L30) 准备节点服务、申请两个租约并启动原生 Docker 环境，再通过 `start_gpu_job` 提交训练。这一步后续工作是并发执行的，不能理解为提交调用一直阻塞到训练结束。
3. **步骤 3 → 图 3**：[barrier](../env-service-verification/scripts/mps_train_workload.py#L30) 所在工作负载在预热后写入 ready 标记，协调器看到两份 ready 才放行。串行组等待 A 的 measured 文件后再放行 B，其他组同时放行。
4. **步骤 4 → 图 4**：[训练脚本](../env-service-verification/scripts/mps_train_workload.py#L50)同步 GPU 后记录结束时间，协调器再确认作业退出码和最终回收状态。profile 阶段发生在计时段之后。
5. **步骤 5 → 图 5**：[arm 的清理段](../env-service-verification/scripts/mps_training_compare.py#L181) 只停止本组环境、释放租约、关闭自己的节点服务，检查本组容器为空且 GPU 无计算进程、显存回到空闲阈值内。

本报告补充 [MPS 并行执行后端](../knowledge/mps-parallel-execution.md) 的性能证据，不替代其中独立取消、超时和故障回收的功能验收。


## 6. 控制通道故障定位与修复

本节记录对上述收尾故障的后续定位。初次性能数据、失败记录和停止范围保留原样，修复后的回归不混入性能对照。

```text
作业 A、B 的 CUDA 计算继续并行
              │ 两条作业相近时间结束，各自进入 cleanup
              ▼
[1] MpsDomain.cleanup：查询服务、核对客户端、必要时定向终止
              ▼
[2] 每个 domain 共用一个 MpsControlChannel
    A 的管理请求先获得通道，B 的管理请求有界等待
              ▼
[3] docker exec → nvidia-cuda-mps-control → 同一个服务的控制 socket
    每次仅保持一条受管管理连接，完成后交给下一请求
              ▼
[4] 查询或终止返回 → 继续各作业的物理清理核验 → 确认后归还预算
```

图例：图里每个部分对应的源码位置。

| 步骤 | 入口与状态变化 |
|---|---|
| 1 | [MpsDomain.cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L342) 禁止新设备访问，检查进程与 CUDA 客户端归属 |
| 2 | [MpsControlChannel](../../../psrl/workers/env_service/mps.py#L11) 每个服务有独立线程锁，排队和执行共用一个 10 秒期限 |
| 3 | [MpsControlChannel.__call__](../../../psrl/workers/env_service/mps.py#L21) 完成一次管理连接的发送、接收和退出，保留具体命令、退出码、标准输出和错误 |
| 4 | [MpsClientController.terminate](../../../psrl/workers/env_service/mps.py#L123) 仍核对身份和终止确认，[cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L342) 仍核验进程与显存归属消失 |

**根因与证据。** 框架让不同作业的清理协程通过后台线程同时启动 `nvidia-cuda-mps-control`。当前节点这套 MPS 控制服务在并发管理连接下会关闭部分连接。socket 是进程间传递消息的通信端点，查询没有发完而对端关闭时，发送端可能收到 `EPIPE`，即向已断开的通信通道写入。

最小复现只创建本次自己的 Docker MPS 服务，发送 `get_server_list` 查询，不启动 CUDA 客户端或训练，不用 profiler。修复前串行 8 次全部成功，8 路并发共 32 次有 15 次失败。随后查询仍成功，服务没有退出。由此确认，并发管理连接本身就足以触发同类错误，不能将原错误归因于显存不足或 profiler 才会发生。

修复后同样通过框架发起的 32 次并发请求全部成功。额外绕过框架通道、直接启动控制客户端，16 次中仍有 8 次失败。`strace -f` 记录到控制连接建立成功后，接收线程先读到 0 字节，发送线程再得到 `EPIPE`，并打印原来的 `Cannot send command to MPS control daemon process`。这里“根因”指已复现的框架集成缺口，不声称已获得 NVIDIA 闭源控制服务内部的完整实现原因。

**为什么没有把训练也串行化。** 锁只覆盖查询和定向取消等管理请求。A、B 的训练命令、GPU 内核执行和各自的输出收集没有持有这把锁。它相当于两位工作人员可以同时干活，但向同一个管理窗口提交手续时依次办理。不同 MPS domain 的管理窗口也各用各的锁，互不排队。

修复没有增加失败后的盲目重试，特别是不能在结果未知时随意重复 `terminate_client`。排队或执行超时仍返回明确错误，无法确认物理回收仍保留预算并隔离该 domain。外部人员绕过框架直接并发操作同一控制 socket，不在这把进程内锁的保护范围内。

NVIDIA 的 [MPS 控制接口说明](https://docs.nvidia.com/deploy/mps/mpsv2-interface.html)解释了 `get_server_list`、`ps`、`terminate_client` 的用途。此处并发断连与修复效果来自本节点实测，并非从接口说明推导出来。

**Results。** 控制通道修复通过 25 项相关 CPU 回归，覆盖同服务串行、不同服务独立、排队超时不发送命令、失败后解锁、终止错误不吞掉，以及原有取消与预算保留约束。原先失败的真实训练组已单独复测通过：两份任务各完成 3 步预热、10 步计时训练和 2 步 profiler 采样，均得到 `completed / exit_code=0`，没有清理错误。计时段为 20.159 秒，仅用来核对同负载执行，不加入原性能统计。容器和 MPS 服务均已移除，GPU 无计算进程，显存回到约 0.19 MiB。没有重跑其他性能对照。

修复基于 `affe7c2`，生产代码改动是 `mps.py` 的控制通道和 `parallel_gpu.py` 的通道接线、关闭异常处理。源码哈希、请求数量和训练状态见 [修复结果 JSON](mps-control-fix-results.json)，原始控制查询、系统调用跟踪、服务日志和训练结果在 [修复验证目录](../env-service-verification/results/mps-control-fix-20261102/)。代码与文档仍是主开发目录未提交改动，未推送、未合并，也没有全局部署。

可运行的最小诊断脚本是 [mps_control_check.py](../env-service-verification/scripts/mps_control_check.py)。它需要已核验的空闲 GPU、Docker 和所选镜像，只操作自己创建的服务。`--trace` 会刻意绕过通道复现原始连接竞争，用于诊断，不能把这部分预期失败算作修复后受管请求失败。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/mps_control_check.py \
  --gpu "$GPU_INDEX" --image "$IMAGE" --output "$CONTROL_RESULT_DIR"
python -m pytest tests/env_service/test_mps.py tests/env_service/test_parallel_gpu.py -q
```

`CONTROL_RESULT_DIR` 是新的诊断输出目录，其余参数沿用第 5 节约定。真实训练回归使用第 5 节脚本的 `--arm 1-mps`，只运行带 profiler 的 MPS 组，并使用独立结果目录。所有实验仍需后台保存日志和设置时限。

跟着图读源码：`cleanup` 通过 `asyncio.to_thread` 调用共用通道 → 通道在线程中等待本 domain 的锁 → 完成一次 Docker 控制客户端调用 → 解锁后其他管理请求继续 → 原清理流程检查物理释放。线程锁不会占住 Ray 的异步事件循环，多个 CUDA 作业仍在后台并行。这些代码如果有理解上的问题，可以随时追问。


## 7. 修复后的 MPS 开关速度对照

这是 2026-11-02 在控制通道修复后重新执行的一组配对。两组均使用提交 `96acfcc` 的独立只读源码快照，未修改执行代码或实验参数。先运行普通 CUDA 并发，确认物理释放后再运行 MPS，并为各组创建独立 Ray 实例。

两组都是两个原生 EnvWorker Docker 容器同时训练，同一张空闲 H20，每份 episode 预算 40000 MiB。负载、批大小和数据沿用第 2 节的真实 autoresearch 训练，每份先预热 3 步，再计时 10 步。这次不启用 profiler，不额外占位显存，不运行 vLLM。关闭 MPS 时 Router 仍开启，也不是把两份训练强行串行，因此这仍不是历史 Router OFF/ON 对照。

| 指标 | MPS 关闭，普通 CUDA 并发 | MPS 开启 |
|---|---:|---:|
| 两份各 10 步全部完成 | 22.499 秒 | 20.160 秒 |
| 每份计时训练耗时 | 22.499 / 22.492 秒 | 20.158 / 20.159 秒 |
| 节点与容器准备 | 17.707 秒 | 19.799 秒 |
| 两份模型初始化、编译与预热 | 90.124 秒 | 92.590 秒 |
| 环境创建开始至物理清理结束 | 133.150 秒 | 136.010 秒 |
| 整卡峰值显存 | 50310.6 MiB | 50364.0 MiB |
| 作业和清理状态 | 两条均 completed、退出码 0、清理确认 | 两条均 completed、退出码 0、清理确认 |

训练段耗时减少 `1 - 20.160 / 22.499 ≈ 10.4%`，完成相同工作量的吞吐约为 **1.116 倍**，即提高约 **11.6%**。用具体数字理解：两份任务各完成 10 步，关闭 MPS 约需 22.50 秒，开启后约需 20.16 秒，总共节省约 2.34 秒，不是并发后缩短一半。

但若关心从创建环境到彻底清理的整个短流程，开启 MPS 这组反而多花约 2.86 秒，即增加约 **2.1%**。初始化和编译占用了约 90 秒，计算阶段节省的两秒多不足以抵消这次其他阶段的差异。这里的总时间不包含 shell 环境激活和 Ray 集群启动，且仅有一次配对，不能把全部差异归因于 MPS，也不能把 2.86 秒当成固定启动成本。

本次确认 MPS 组有两个真实客户端接入服务，两组全部完成，无 OOM、无控制通信错误、无清理错误。采样没有缺失，最后容器和 MPS 服务为空，GPU 无计算进程，NVML 回到约 0.19 MiB。结论明确后结束，没有为了放大收益继续增加训练步数。

**适用边界**：这说明该负载在预热后的持续训练段有约一成收益，但不足以说明短任务完整流程更快。仅有一组新配对，没有统计显著性或长期稳定性结论。本次没有重新采集内核时间线，第 3 节的重叠数据仍属于之前实验。它也不是原生 science/LAPS solver、完整 PSRL 强化学习训练、显存不足时的排队收益或训推同卡收益。

复现使用第 5 节的同一脚本，分别指定 `--arm 0-plain`、`--arm 0-mps`，各用新的输出目录。两组之间必须确认物理释放。紧凑结果与镜像内容 ID 见 [修复后速度结果 JSON](mps-speed-fixed-results.json)，完整日志和源码快照保存在本地 `env-service-verification/results/mps-speed-fixed-20261102/`，原始目录未纳入仓库。

本次只补充性能证据，没有改变架构、接口或资源生命周期，现状入口和专题索引已有的能力边界仍适用，无需改写设计。文档本地链接通过检查，全仓 pre-commit 在隔离副本运行，仍有既有格式与 prose 问题，未为本次实验改动无关文件。
