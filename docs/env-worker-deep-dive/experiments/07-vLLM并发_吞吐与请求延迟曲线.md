# vLLM并发：吞吐与请求延迟曲线

> **核心结论**：2026-11-18 在用户私有节点的一张 H20 上，Qwen2.5-3B 的短输入从 1 增至 64 并发，平均请求延迟增加约 12%；128 并发增加约 23%，256 并发增加约 71%。输入从 128 增至 2,048 token 后，32 并发已经增加约 52%。吞吐在测量范围内仍增长，没有测到明确平台。因此可以建立依赖输入长度的服务曲线，暂不能把某个请求数认定为 H20 的 SM 饱和上限。

本次只测单个 vLLM 引擎，没有加入环境命令、MPS、SMG、Ray 或 PPO 训练。原理、同模型合批与当前 PSRL 的 PD 状态见 [vLLM 推理机制](../../psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md)，如何将曲线用于 Env 决策见[共享准入与 PD 取舍](../knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#怎样把单卡曲线用于准入)。

## 目录

- [研究问题与方法](#研究问题与方法)
- [测量结果](#测量结果)
  - [短输入的吞吐与延迟](#短输入的吞吐与延迟)
  - [长输入为何更早变慢](#长输入为何更早变慢)
- [怎样理解上限与等待](#怎样理解上限与等待)
- [与 Roofline 和 PD 分离的关系](#与-roofline-和-pd-分离的关系)
- [复现与证据](#复现与证据)
  - [运行条件和计时](#运行条件和计时)
  - [启动故障与监督边界](#启动故障与监督边界)
  - [命令与验证](#命令与验证)

## 研究问题与方法

上一轮[动态分区实验](09-资源分区_静态卡池与推理余量共享.md#实验二借卡期间突然来了推理请求)的校准扫描到 16 并发，引擎也限制为 16，只说明这一范围内合批有效。这次把软件序列上限提高至 256，直接观察增加并发后的代价。

| 项目 | 本次设置 |
|---|---|
| GPU 与模型 | 一张 H20，Qwen2.5-3B-Instruct，bf16，TP=1 |
| 短输入 | 每请求 128 token，并发 1、2、4、8、16、32、64、128、256 |
| 长输入 | 每请求 2,048 token，并发 1、8、32、64 |
| 输出 | 每请求固定 512 token，忽略 EOS，temperature=0 |
| 引擎预算 | `max_num_seqs=256`，`max_num_batched_tokens=2048`，`max_model_len=2560` |
| 缓存 | 固定 16 GiB KV 池，关闭 prefix caching |
| 执行 | eager 模式，关闭 CUDA Graph 和编译，与此前小规模校准保持这一模式一致 |
| 请求构造 | 确定性 token ID 输入，直接调用 AsyncLLM，关闭输出文本反解码 |
| 重复 | 先预热 16 条短请求，每个长度升序一遍、降序一遍，共 26 个测量波次 |
| 预算 | 单波最多 150 秒，主进程最多 1,200 秒，完成预定扫描即停止 |

“一波 64 并发”表示同一事件循环连续提交 64 个请求，等待它们全部结束后才进入下一波。引擎内部仍会依照每轮 token 预算安排 P 和 D。因此提交并发、瞬时运行请求数和每一轮实际合批 token 数要区分。所有测量波次合计完成 1,232 个请求，另有 16 个预热请求，每个均生成了要求的 512 token。

TTFT 是从本地提交到拿到第一个 token 的时间，包含引擎等待和输入处理。TPOT 是首 token 后的耗时除以其余 511 个 token，表示平均生成间隔。完整请求延迟从本地提交计到最后一个 token。输出吞吐用两次波次的总输出数除以波次总墙钟时间，波次包含所有请求的结束等待。这里没有 HTTP、分词和上层路由延迟，也不代表持续到达流量下的排队稳态。

## 测量结果

[![吞吐、完整请求延迟与每输出 token 时间（1/5）](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3-screen.png)](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.png)

[![吞吐、完整请求延迟与每输出 token 时间（2/5）](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3-screen-02.png)](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.png)

[![吞吐、完整请求延迟与每输出 token 时间（3/5）](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3-screen-03.png)](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.png)

[![吞吐、完整请求延迟与每输出 token 时间（4/5）](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3-screen-04.png)](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.png)

[![吞吐、完整请求延迟与每输出 token 时间（5/5）](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3-screen-05.png)](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.png)

<!-- explained-source:explained-single-gpu-inference-curve-evidence-7df94d3 -->
**读图结论：**并发上限依模型、长度和延迟目标而变。延迟增加的拐点不等于硬件已饱和。 同批请求波次，不是持续到达稳态。未加入环境共运行或 PPO。 [完整解释图（SVG）](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.svg) · [原始机制图或实验图](single-gpu-inference-curve-evidence.svg)。
<!-- /explained-source:explained-single-gpu-inference-curve-evidence-7df94d3 -->

### 短输入的吞吐与延迟

输入 128 token，输出 512 token。下表是两次波次的汇总：

| 并发请求数 | 平均完整延迟 | 相对单请求 | 平均首 token 等待 | 平均 TPOT | 输出吞吐 token/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.062 s | 1.000× | 0.030 s | 11.804 ms | 84 |
| 2 | 6.129 s | 1.011× | 0.031 s | 11.933 ms | 167 |
| 4 | 6.112 s | 1.008× | 0.045 s | 11.872 ms | 335 |
| 8 | 6.167 s | 1.017× | 0.053 s | 11.966 ms | 664 |
| 16 | 6.384 s | 1.053× | 0.096 s | 12.306 ms | 1,283 |
| 32 | 6.479 s | 1.069× | 0.176 s | 12.334 ms | 2,527 |
| 64 | 6.782 s | 1.119× | 0.276 s | 12.730 ms | 4,820 |
| 128 | 7.479 s | 1.234× | 0.451 s | 13.753 ms | 8,693 |
| 256 | 10.337 s | 1.705× | 0.845 s | 18.576 ms | 12,500 |

从 1 到 64，同一模型合批让总吞吐大幅增长，而每条请求的生成速度变化较小。从 128 增至 256，吞吐提高约 44%，平均请求延迟增加约 38%，每 token 时间增加约 35%。因此这里已经进入“继续合批有收益，但响应变慢的代价明显”的区间。

若把平均请求延迟比单请求增加 25% 作为一个研究判据，短输入的越界位置被当前采样点夹在 128 与 256 之间。128 点只有很小的余量，而且只重复两次，不应直接设成生产硬上限。对于更严格的 10% 判据，32 点仍满足而 64 点越界。阈值取决于目标允许的延迟，而非硬件给出唯一答案。

### 长输入为何更早变慢

输入 2,048 token，输出仍为 512 token：

| 并发请求数 | 平均完整延迟 | 相对单请求 | 平均首 token 等待 | 平均 TPOT | 输出吞吐 token/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.067 s | 1.000× | 0.097 s | 11.683 ms | 84 |
| 8 | 6.752 s | 1.113× | 0.497 s | 12.241 ms | 604 |
| 32 | 9.201 s | 1.517× | 1.647 s | 14.783 ms | 1,745 |
| 64 | 12.279 s | 2.024× | 3.184 s | 17.799 ms | 2,585 |

长输入的 25% 越界位置只能夹在 8 与 32 之间，因为本轮没有测 16，不能给出精确阈值。固定 64 并发比较两种输入，引擎记录如下：

| 时间分量 | 128-token 输入 | 2,048-token 输入 |
|---|---:|---:|
| 平均首次调度前等待 | 0.091 s | 2.886 s |
| 平均引擎 prefill 时间 | 0.173 s | 0.285 s |
| 平均引擎 decode 时间 | 6.505 s | 9.096 s |

每轮最多处理 2,048 个 token，长输入请求会消耗更多轮次预算，已经开始 D 的请求还要与后续 P 工作穿插。结果既有首次调度前等待，也有输出生成阶段变慢，不能把 12.279 秒全部解释成“解码计算量变大”。长上下文带来的 KV 读取也同时变化，本次没有用单独消融分解两者贡献。

原始请求还显示：64 条长请求中，最先提交的 8 条平均 TPOT 在两轮分别为 22.38 与 22.21 ms，最后提交的 8 条为 13.38 与 13.01 ms。早进入 D 的请求经历更多后续 P 工作，晚进入 D 的请求则已经承担了前面的等待。这个模式支持进一步检查 P/D 干扰，但仍包含活跃批次随时间变化的影响，不能仅凭这组差值计算 PD 分离收益。

所有波次的引擎 KV 抢占计数均为 0，最大 KV 池使用约 35.0%；短输入最高档实际达到 256 个运行请求。因此本轮明显变慢的解释不需要诉诸 KV 不足后的重算，也没有被原来的 16 序列上限截住。显存尚有空间时，延迟已经可能显著增加。

## 怎样理解上限与等待

并发更大时，请求本身计算更慢，但可能少排很久的队。调度应比较：

```text
现在接纳：当前时刻 + 扩大批次后的预计剩余时间
继续等待：当前时刻 + 预计排队时间 + 较小批次下的服务时间
```

用短输入实测波次作一个估算，假设同时来了 256 条同类请求。每波最多 128 条，第一波约 7.54 秒完成，第二波再运行约 7.54 秒，总计约 15.08 秒。如果 256 条一起进入，本轮整波实测约 10.49 秒完成。即使单条已接纳请求的服务时间由约 7.48 增至 10.34 秒，一起接纳仍可能缩短全部完成时间。

上述两波排队是用独立波次推算，没有另起一次 256 条排队策略对照。它说明为什么“超过某个并发后单请求变慢”不自动推出“超过就必须排队”。实际还需考虑先完成的一部分请求能否提前触发环境命令、请求优先级、未来到达和其他空闲 GPU。正式延迟预算必须从请求到达开始计时，不能把准入前等待移出统计来满足指标。

同模型请求合批还能共享读取权重的成本；额外启动一个 Env GPU 进程通常不具备这种共享。Env 准入仍需要混合负载测量，不能把本表的“还可增加 32 条模型请求”翻译成“可以增加一个环境命令”。

## 与 Roofline 和 PD 分离的关系

经典 Roofline 用算术强度解释算力上界与带宽上界。此次图的横轴是并发请求数，纵轴是 token 吞吐与延迟，属于服务曲线。eager 模式保留了 CPU 发起内核的成本，图中还包括引擎调度和 P/D 穿插。没有采集 Nsight 的算子 FLOP、访存字节或计算管线计数，因此本轮没有确定 SM 饱和点。

这组数据对 PD 分离的启发是：相同 64 并发，长输入显著增加了首次调度等待和生成时间。若真实 agent 的工具反馈经常制造长输入突发，就值得验证把 P 移到专门资源上能否减少 D 延迟。对照应固定 GPU 总数，比较普通 P+D 副本、调优 chunked prefill 和 PD 分离，计入 KV 传输、模型副本和两池不均衡。

当前 PSRL [SMG 适配代码](../../../psrl/workers/gen/smg_adapter.py#L76)关闭 PD 分离。本次没有实现或运行 P/D 专用池。对 Env 更优先的方向是用现有实例的 P/D 状态和实测曲线改进共享决策，然后在真实任务显示长期 P/D 干扰时引入分离，详细理由见[共享专题](../knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#pd-分离对环境调度有什么价值)。

## 复现与证据

### 运行条件和计时

本次基于 `research/env-elastic-partition` 分支、提交 `c5f2d65a70f6923c992a7be867edd0864db2bd0c` 加主目录未提交实验脚本。没有提交、推送或合并本次改动。vLLM 实际版本为 `0.22.1.dev0+g0b3ba88f1.d20260729.cu128`，PyTorch 为 `2.11.0+cu128`，Transformers 为 `5.10.1`。版本与路径以原始证据为准，不据版本字符串推断所有上游特性。

第三次启动完成模型初始化约 67.76 秒，包含初始化、预热、26 个测量波次、引擎退出及回收确认的监督观察跨度约 275.56 秒。表格只统计测量波次，初始化和预热单列。模型加载使用已有共享依赖及进程内元数据发现优化，未修改共享依赖安装。

本次运行结束后，所选 GPU 的计算进程列表为空，NVML v2 的使用量回到约 0.19 MiB；约 586.06 MiB 的驱动保留显存单列。没有创建实验容器或 Ray 集群，也没有停止共享服务。

### 启动故障与监督边界

保留了两个未进入有效测量的启动记录。第一次预检使用旧版 NVML 内存结构，把驱动保留区计入占用，触发空闲检查退出；改用 v2 结构分别记录使用量和保留量。第二次安装版 vLLM 将 `CUDA_VISIBLE_DEVICES` 解析为整数，不能直接接受 UUID 字符串，启动时报错。随后先通过 NVML 核对 UUID 对应的 index，再传数字 index 与 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。第二次退出后的物理释放已核对。

第三次启动的原监督进程在实验中退出，独立会话中的 vLLM worker 继续运行。发现后附加了独立会话 watchdog，继承原进程创建时间计算的 1,200 秒总预算，没有重启或重复测量。保留了两段遥测，中间缺测不补零。附加监督进程无法用 `waitpid` 获取非子进程的退出码，因此该字段为 `null`，没有将其写成 0。有效性来自 worker 完整写出的 `passed`、26 个波次逐请求与引擎结束记录、观察到进程结束，以及 GPU 物理释放；它们不能补充缺失的硬件利用率剖析。

### 命令与验证

先按[运行前检查](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)激活项目环境，设置 `PSRL_WORKSPACE` 和 `PYTHONPATH`，核对 GPU 空闲、模型路径与依赖来源。`GPU_UUID` 是现场核对的目标设备，`RUN_DIR` 必须是尚不存在的本次输出目录，`MODEL_PATH` 指本地 Qwen2.5-3B 模型。

```bash
setsid nohup python -u docs/env-worker-deep-dive/env-service-verification/scripts/roofline_sweep.py \
  --repo "$PSRL_WORKSPACE" --model "$MODEL_PATH" --gpu "$GPU_UUID" \
  --output "$RUN_DIR" --budget-s 1200 > "$RUN_LOG" 2>&1 < /dev/null &
```

`RUN_LOG` 指当前用户有权写入的独立日志文件。完成且物理释放后生成汇总和图：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_roofline_sweep.py \
  --source "$RUN_DIR" \
  --output docs/env-worker-deep-dive/experiments/single-gpu-inference-curve-evidence.json
```

汇总脚本核对每个点两波、每波完成数量、每请求 512 token、引擎结束记录数量、任务状态和物理释放，生成 JSON、SVG、PNG。脚本通过 Ruff 与项目 prose audit，元数据发现辅助函数的两项针对性回归通过。没有为本次测量运行额外训练。

- [汇总 JSON](single-gpu-inference-curve-evidence.json)、[PNG 图](../../_static/img/document-figures/explained-single-gpu-inference-curve-evidence-7df94d3.png)。
- [原始波次与逐请求数据](../env-service-verification/results/roofline-sweep-20261118-03/evidence.json)、[监督与物理释放](../env-service-verification/results/roofline-sweep-20261118-03/completion.json)、[版本与脚本摘要](../env-service-verification/results/roofline-sweep-20261118-03/provenance.json)。
- [扫描脚本](../env-service-verification/scripts/roofline_sweep.py)、[统计与绘图脚本](../env-service-verification/scripts/analyze_roofline_sweep.py)。
- [首次预检失败](../env-service-verification/results/roofline-sweep-20261118-01/preflight.json)、[第二次启动失败](../env-service-verification/results/roofline-sweep-20261118-02/worker.log)。
