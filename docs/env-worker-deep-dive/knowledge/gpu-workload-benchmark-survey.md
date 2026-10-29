# GPU 环境工作负载调研与 benchmark 接入顺序

> **核心结论**：优先补齐 KernelBench 的短时 GPU 评测、MLE-bench 的容器内训练、InferenceBench 的常驻推理服务。autoresearch 可作为有界研究迭代负载，PostTrainBench、RE-Bench 和 InfraDev 作为后续扩展。已有 AIRS recipe 不等于已有 GPU 环境验证，默认脚本仍设置环境 GPU 槽位为零。
>
> 本文是截至 2026-10-29 的公开资料调研与本地源码核对。接入顺序、资源分类和验收预算是设计建议，不是已接入或实测结果。未运行外部 benchmark，未下载模型、数据或启动集群实验。

## 目录

- [目标与全局流程](#目标与全局流程)
- [比较维度](#比较维度)
- [候选对照表](#候选对照表)
- [首批接入建议](#首批接入建议)
- [具体例子与评分干扰](#具体例子与评分干扰)
- [本地实现与源码阅读路径](#本地实现与源码阅读路径)
- [可运行的教学模拟](#可运行的教学模拟)
- [验收与未验证事项](#验收与未验证事项)

## 目标与全局流程

这里的 workload 指实际交给计算资源执行的一段工作。我们关心 agent 在环境中启动的训练、计算核和模型服务。驱动 agent 思考的大语言模型也需要算力，但它属于 PSRL 的策略模型训推资源，需要另记。

下图是接入目标图。实线主流程可映射到本地 science 路径，候选 benchmark 分支尚未实现。

```text
[1] benchmark 任务与资源需求
                 |
                 v
[2] runner 向 EnvRouter 请求资源租约
                 |
                 v
[3] 容器内 agent 编辑代码、执行命令、读取反馈 <---+
                 |                            |
                 +--> 候选：训练模型 / kernel / 服务
                 |                            |
                 +------------ 实验反馈 -------+
                 |
                 v
[4] verifier 独立验证提交，产生评分
                 |
                 v
[5] 回收所属容器和进程，归还租约
```

图例：图里每个部分对应的源码位置。以下链接是当前工作目录中的实现，不代表所有外部 benchmark 已经能复用。

| 编号或箭头 | 职责 | 对应函数及源码 |
|---|---|---|
| 1 → 2 | science 任务转换资源需求并申请租约 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L368) |
| 2 | 向路由器申请资源 | [EnvServiceClient.acquire](../../../psrl/workers/env_service/client.py#L159) |
| 3 → 4 | 创建并运行 Harbor job，由 Harbor 内部执行 agent 与 verifier | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L537) |
| 4 | 提取奖励，按条件处理 GPU 作业报告 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L575) |
| 5 | finally 清理容器，随后归还资源 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L597)、[EnvLeaseHandle.release](../../../psrl/workers/env_service/client.py#L69) |
| 候选分支 | 训练、kernel、服务的专属任务适配 | 本次调研提案，没有新增本地实现 |

租约表示某个任务获准使用一份资源预算，不自动保证显存硬隔离。verifier 是负责重新检查结果的评分程序，不能直接相信 agent 打印出来的分数。一个 episode 是一次从任务开始到评分和清理结束的完整尝试。若这些源码有理解上的问题，可以随时沿图中编号追问。

## 比较维度

图片适合比较研究能力覆盖范围，例如任务来源、研究数据、任务跨度、科学研究环节和计算需求。对本次资源调度目标，还需要以下字段。图片没有给 H/I/E/A 的完整定义，本文不凭缩写补造含义。

| 字段 | 为什么影响接入 |
|---|---|
| 任务来源、版本、固定 commit | 任务数量与评测协议会变化，不能把截图当作最新事实 |
| GPU 使用者与阶段 | agent 模型推理、环境训练、最终评分可能分别需要 GPU |
| 单卡或多卡、硬件绑定 | CUDA 是 NVIDIA 的计算平台，ROCm 是 AMD 的计算平台，不能仅换镜像名就互通 |
| 显存峰值、空闲保留显存 | 利用率低不代表显存已释放，服务进程可能一直保留模型 |
| 占用时间、调用频率、生命周期 | 短命令适合阶段借还，常驻服务需要覆盖整个进程生命周期 |
| 评分与计时边界 | 时延和吞吐分数会受同卡竞争影响，不能把调度干扰算作代码质量 |
| 数据、模型、镜像、许可证 | 提前确认获取条件和可再分发范围，不统一打包传播未知许可数据 |
| 反馈、最终提交、可信评分 | 公开反馈与隐藏测试隔离，避免 agent 改评分器获利 |
| 接入方式与证据等级 | 区分文档宣称、代码审核、容器通过、真实 CUDA 执行和正式评测 |

RSI 在这里按 Recursive Self-Improvement，即递归自我改进理解。研究 agent 自动改训练代码属于相关工作负载，但若它改的是另一个模型，而没有把改进反馈到自身能力中，就不能仅凭这个循环声称实现了严格的递归自我改进。MLE 是机器学习工程，包括处理数据、训练模型、调参和提交预测。AI4Infra 指利用 AI 改进基础设施，可覆盖推理服务、训练框架、通信与底层计算。

## 候选对照表

GPU 形态是根据任务机制作的接入判断，不是本地测得的资源曲线。未列具体显存时表示尚未核定，不能解释为低显存。

| 候选与一手来源 | 任务和来源 | GPU 使用形态 | 结果与接入判断 |
|---|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | PyTorch 程序改写为更快的 GPU kernel，基础三级为 100/100/50 个任务，另有第 4 级 | 编译后反复在 GPU 上检查正确性和计时，通常可按单次评测组织 | **第一批**。覆盖短作业与计时隔离。不能假设所有任务都能放进小显存卡 |
| [MLE-bench](https://github.com/openai/mle-bench) | 75 个 Kaggle 机器学习竞赛，Lite 为 22 个低复杂度任务 | 数据准备、训练、预测交替，选图像任务可形成持续训练负载 | **第一批**。官方参考资源为单张 24GB A10、24 小时，不是所有任务的最低要求。部分表格任务可以只用 CPU |
| [InferenceBench](https://github.com/aisa-group/InferenceBench) | agent 优化模型推理服务，涵盖首 token 时延、后续 token 时延、吞吐及综合场景 | 模型服务常驻显存，外部请求产生 GPU 压力 | **第一批 AI4Infra**。官方仓库公开 Mistral-7B 和每次 2 小时设置。重点是服务生命周期与性能隔离，不是普通命令执行 |
| [autoresearch](https://github.com/karpathy/autoresearch) | agent 修改训练代码，运行实验后保留或放弃修改 | 单张 NVIDIA GPU，每次训练固定 5 分钟，另计启动、编译和评估 | **第一批补充负载**。是研究循环框架，不直接视为标准多任务 benchmark。必须额外设置总轮数与总期限 |
| [MLGym](https://github.com/facebookresearch/MLGym) | 13 个开放研究任务，覆盖监督学习、强化学习等 | 不同任务差异大，训练类可用 GPU | AIRS 已复用其 agent，接入成本可能较低，但新增资源形态有限。上游页面显示仓库已归档，需固定依赖 |
| [RE-Bench](https://github.com/METR/RE-Bench) | 7 类研究工程任务，包括训练脚本优化、模型修复、Triton 前缀和 | 同时覆盖长训练与短 kernel，逐任务审核卡数和评分环境 | **第二批**。可选优化 LLM Foundry 任务补充训练系统优化。上游已归档，不默认依赖会继续维护 |
| [PostTrainBench](https://github.com/aisa-group/PostTrainBench) | 自动对基础语言模型做后训练 | 官方设置为一张 H100、10 小时，训练和评估均可能占 GPU | **第二批 RSI 相关**。权重、优化器状态与评分成本较重。官网已出现版本更新入口，不沿用图片的固定目标集数量 |
| [TritonBench](https://github.com/thunlp/TritonBench) | 生成 Triton 算子，Triton 是便于编写 GPU 并行计算的语言与编译系统 | GPU 正确性与性能评测 | KernelBench 接口稳定后扩充 kernel 题型，初期无需两个近似负载同时接 |
| [InfraDev 论文](https://arxiv.org/abs/2610.36589) | 2026-10-27 发布，36 个源自 Megatron-LM、vLLM、verl 的真实修改任务 | 多 GPU 执行与隐藏验证 | **跟进候选**。论文定位明确，公开任务包、运行入口、每题硬件预算本次尚未核实，不能宣称可立即运行 |
| [sammysun0711/InfraBench](https://github.com/sammysun0711/InfraBench) | 20 个 Harbor 格式 AI 基础设施任务 | 目标为 AMD MI35X，任务表包含 1 卡和 2 卡 | Harbor 格式接近已有路径，但硬件不匹配时不宜首接。与其他同名论文不自动视为同一项目 |
| [MLSysBench](https://github.com/haiyuanwan/MLSysBench) | 配置搜索、调度和跨规模迁移，包含模拟器与原型任务 | 多个任务只运行模拟，真实 CUDA 路径另有边界 | **暂不作为真实 GPU 负载证明**。其仓库明确说明尚非完整正式评测套件 |

这张表关注可运行环境，故没有把图片中所有论文复现或科学问答 benchmark 一并列为 GPU 候选。是否使用研究论文数据，与是否实际启动 GPU 任务是两回事。

## 首批接入建议

建议分三个正式 benchmark 和一个补充 workload，分别覆盖短作业、训练、服务、研究循环。以下是拟议目录，不是本次创建完成的 recipe。

| 顺序 | 拟议目录 | 首次选题 | 适配核心 |
|---|---|---|---|
| 1 | `examples/kernel_bench/` | 从 Level 1 中选一个矩阵运算任务，核定输入尺寸 | 保留官方参考实现、正确性检查与计时协议，外包可信 GPU evaluator |
| 2 | `examples/mle_bench/` | Lite 中的 `aerial-cactus-identification` | 准备数据、容器训练、导出预测，由可信 grader 评分 |
| 3 | `examples/inference_bench/` | 固定一个推理场景 | 启动服务、等待就绪、压测、验证质量、停止服务及其子进程 |
| 4 | `examples/autoresearch/` | 一个固定训练起点，最多两轮修改 | 保留官方单次训练预算，设置有界外层循环和不可修改的评估输入 |

第一项先验证一题的完整执行和回收。第二项不一次下载完整 75 题，Lite 也不等于只需很少磁盘。第三项要把 agent 自己调用的策略模型服务和它正在优化的目标服务分开，端口、权重、GPU 账目也分开。

共同架构建议：benchmark 适配器只负责任务准备、提交协议、评分和终止条件。资源申请、容器生命周期、GPU 归属和清理复用公共后端。science runner 中包含 LAPS 专用评分分支，不能直接把整个 runner 当作无条件通用接口复制四遍。是否需要抽取公共 runner，应在首个适配器实现时结合真实接口确定。

## 具体例子与评分干扰

**教学假设，不是测量结果**：一个 KernelBench episode 花 40 秒读题和生成代码、10 秒编译、2 秒 GPU 运行与计时，三个阶段共 52 秒。若整个 episode 独占一张 GPU，预留为 52 GPU 秒，其中只有 2 秒执行该任务的 GPU 计算。按评测阶段申请理论上可缩短预留，但还要计算初始化、数据传输、切换和等待成本，不能据此直接宣称 26 倍吞吐收益。

输入是一段 PyTorch 矩阵计算和固定输入规范。agent 生成 CUDA 或 Triton 程序，编译器生成可执行代码，evaluator 在 GPU 上运行参考与候选。只有输出符合容差才比较速度，最终再销毁评测进程并核实释放。GPU kernel 是由大量 GPU 线程协作执行的计算函数，其速度来自并行计算与访存组织，不是把 Python 文件改个后缀。

假设参考为 100 微秒，候选为 80 微秒，加速比是 100/80=1.25。若另一个任务竞争显存带宽，使候选测到 120 微秒，则测出的 0.83 会把正确优化误判为退步。因此正式性能计时应使用受控独占窗口，同样约束参考与候选。开发阶段是否允许共享可以另外评估。

对比 MLE：agent 可能先用 CPU 处理数据，再训练 10 分钟，然后用 GPU 预测，最终提交 CSV。GPU 分配应覆盖实际训练和预测进程，不能在 shell 返回但后台训练仍活着时归还。对比 InferenceBench：服务在请求之间利用率可能很低，但模型仍占显存，不能将请求间空隙直接解释为可借出的整卡。

## 本地实现与源码阅读路径

本次核对的是带有未提交修改的主开发目录，而不是一个干净的已发布版本。

- science 的 [runner.py](../../../examples/sciaccel_rl/runner.py#L273) 已有 `needs_gpu`、资源租约和 GPU 作业报告路径。其 [README](../../../examples/sciaccel_rl/README.md) 也提示 ScienceInfra 为后续开发入口。任务是否实际执行 GPU 仍需逐题证明。
- AIRS 的 [run_mlgym_agent](../../../examples/airs_bench/runner.py#L123) 复用 MLGym，其 [启动脚本](../../../examples/airs_bench/run_qwen3-4b.sh#L193) 明确设置 `gpu_slots_per_worker=0`。因此本次只能确认已有 recipe，不能确认默认 AIRS 已提供环境 GPU 训练。
- 这篇文档补充的是候选选择，公共调度机制继续以 [当前流程总览](env-router-current-flow.md) 和 [统一 GPU 执行后端](unified-gpu-execution.md) 为入口，避免重复维护架构说明。

跟着图读源码，按真实 science 路径查看：

| 步骤 → 图中编号 | 文件、函数 | 输入输出或状态 |
|---|---|---|
| 申请 → 1、2 | [runner.py:368](../../../examples/sciaccel_rl/runner.py#L368)，`run_harbor_episode` | 任务路径和需求转换为租约，可能异步等待 |
| 维持租约 → 2 | [client.py:50](../../../psrl/workers/env_service/client.py#L50)，`start_heartbeat` | 后台心跳与任务并发，不是任务执行完才调用 |
| 执行 → 3、4 | [runner.py:537](../../../examples/sciaccel_rl/runner.py#L537)，`Job.create`、`job.run` 调用点 | Harbor 执行任务并返回结果，内部 agent/verifier 顺序由 Harbor 管理 |
| 评分处理 → 4 | [runner.py:575](../../../examples/sciaccel_rl/runner.py#L575)，`run_harbor_episode` | command 模式读取 GPU 报告，LAPS 条件分支检查活动证据 |
| 清理 → 5 | [runner.py:597](../../../examples/sciaccel_rl/runner.py#L597)，`run_harbor_episode` 的 finally | 正常、异常、取消均进入收尾，再调用 lease release |

## 可运行的教学模拟

下面是自包含 CPU 脚本，在仓库根目录执行即可。只展示“episode 预留”和“实际计算阶段”的差异，不调用 PSRL、Docker 或 GPU，也不模拟抢占、显存分配和排队收益。

```bash
python - <<'PY'
# 对应图 1：人为定义三个阶段，单位为秒。
stages = [("edit", 40, False), ("compile", 10, False), ("evaluate", 2, True)]
# 对应图 2、3、4：比较整个任务预留和仅计算阶段预留。
episode_reserved = sum(seconds for _, seconds, _ in stages)
phase_reserved = sum(seconds for _, seconds, gpu in stages if gpu)
for name, seconds, gpu in stages:
    print(f"{name}: {seconds}s, GPU={gpu}")
print(f"episode_reserved={episode_reserved} GPU-s")
print(f"phase_reserved={phase_reserved} GPU-s")
# 对应图 5：模型假设进程退出后释放，真实系统必须额外核验。
print("simulation_end: assumed resources released")
PY
```

预期分别输出 52 和 2 GPU 秒。差值代表本例可避免的预留空窗，不是已经证实的加速。真实容器可能在编译或初始化时建立 GPU 上下文，真实训练和服务也可能跨命令保留状态，必须按实际进程与遥测修正阶段边界。

本次已实际运行此 CPU 模拟，输出为 52 和 2 GPU 秒。本篇及两个更新索引的本地链接检查通过，新增目录锚点和源码行号范围检查通过。全仓 `pre-commit run --all-files` 在临时副本执行，因既有 Python 行长、格式和注释规范问题失败，自动格式化仅发生在副本，没有回写主开发目录。

## 验收与未验证事项

接入验收分开记录：任务包准备、固定动作或已知解通过、真实 CUDA 执行、agent 完整 episode、资源物理释放、正式 benchmark 协议评测。前一项通过不能替代后一项。

建议首批每类只选一题、最多两个尝试。KernelBench 的集成检查可先给 10 分钟总期限，MLE 的小样本路径给 30 分钟，InferenceBench 的服务启动和固定请求检查给 20 分钟，autoresearch 两轮给 20 分钟含启动评估。这些是拟议工程检查预算，需要结合实际任务可行性核定，不是官方评测预算。到期证据不足即记录未验证，不自动扩容重跑。

资源紧张时首先选择满足要求的独立 GPU。若研究共享机制，单列共享实验与正式分数实验。多 GPU 任务需确认同时分配的设备集合、互联要求以及失败后整个进程组的清理，不能用单卡通过代替。

本次尚未逐题审核许可证、下载权限、镜像可构建性、峰值显存、评分隔离漏洞和硬件兼容性。InfraDev 仅核验到论文，其他项目也仅核验公开入口和说明，没有声称其在本工作区可直接运行。执行任何真实实验前按 [集群检查指南](../../psrl/psrl-run-guide/cluster-preflight.md) 做现场预检。
