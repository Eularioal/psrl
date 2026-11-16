# GPU 环境工作负载调研与 benchmark 接入顺序

> **核心结论**：当前有两条并行研究路线。非计时评分的调度优先看 [SWE-Serve 功能开发场景](../research-proposals/non-timing-gpu-scheduling-scenarios.md)，性能反馈可靠性看 [AI4Infra benchmark 选择与 FlashInfer 验证](ai4infra-performance-benchmarks.md)，两者不互相替代。已有 recipe 的实测状态看 [导览](../experiments/gpu-benchmarks/README.md)。[RE-Bench 原生接入](../knowledge/re-bench-recipe.md)已有本地有界偏好评分与策略闭环。PostTrainBench 的 9 个公开本地组合已完成闭环，整库 24 组合仍未完成，范围统一查[路线图](../milestones/science-gpu-bench-status-and-roadmap.md)。高 GPU 计算需求本身不说明必须使用 episode。
>
> 最新核对：2026-11-12，仅同步本地接入能力与已验证范围，基于 `6aa328d` 加未提交修改，未重新做外部文献调研。官方协议与本地协议分开，RE-Bench 的资产和运行边界见 [recipe](../knowledge/re-bench-recipe.md) 与 [验证记录](../experiments/history/gpu-benchmark-onboarding-chronology.md#re-bench-原生接入验证)。


## 目录

- [目标与全局流程](#目标与全局流程)
- [比较维度](#比较维度)
- [候选对照表](#候选对照表)
- [图片中六项 high GPU 的具体选题](#图片中六项-high-gpu-的具体选题)
- [首批接入建议](#首批接入建议)
- [具体例子与评分干扰](#具体例子与评分干扰)
- [本地实现与源码阅读路径](#本地实现与源码阅读路径)
- [可运行的教学模拟](#可运行的教学模拟)
- [验收与未验证事项](#验收与未验证事项)
- [非计时评分场景的当前优先级](#非计时评分场景的当前优先级)

## 目标与全局流程

这里的 workload 指实际交给计算资源执行的一段工作。我们关心 agent 在环境中启动的训练、计算核和模型服务。驱动 agent 思考的大语言模型也需要算力，但它属于 PSRL 的策略模型训推资源，需要另记。

下图描述本地 science 的公共执行结构，具体 benchmark 的完成数量以路线图和对应 recipe 为准。PostTrain 的 9 个本地组合与 RE 的有界协议已有执行及策略闭环证据，不能由此推断官方全量评测完成。

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
| 1 → 2 | science 任务转换资源需求并申请租约 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L372) |
| 2 | 向路由器申请资源 | [EnvServiceClient.acquire](../../../psrl/workers/env_service/client.py#L159) |
| 3 → 4 | 创建并运行 Harbor job，由 Harbor 内部执行 agent 与 verifier | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L550) |
| 4 | 提取奖励，按条件处理 GPU 作业报告 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L574) |
| 5 | finally 清理容器，随后归还资源 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L610)、[EnvLeaseHandle.release](../../../psrl/workers/env_service/client.py#L69) |
| 候选分支 | 训练、kernel、服务的专属任务适配 | [已有 recipe 与边界](../knowledge/gpu-benchmark-recipes.md)，新增候选仍需适配 |

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
| [MLGym](https://github.com/facebookresearch/MLGym) | 13 个开放研究任务，覆盖监督学习、强化学习等 | 不同任务差异大，训练类可用 GPU | 本地 9 个 GPU 任务已有逐题闭环，RL、视觉和 NLP 使用明确的小预算协议，源码与依赖需固定 |
| [RE-Bench](https://github.com/METR/RE-Bench) | 7 类研究工程任务，包括训练脚本优化、模型修复、Triton 前缀和 | 同时覆盖长训练与短 kernel，逐任务审核卡数和评分环境 | 7 类本地协议已有逐题 GPU 快检，聊天训练代表题闭环，官方模型规模和完整裁判协议仍未复现 |
| [PostTrainBench](https://github.com/aisa-group/PostTrainBench) | 自动对基础语言模型做后训练 | 官方设置为一张 H100、10 小时，训练和评估均可能占 GPU | **第二批 RSI 相关**。权重、优化器状态与评分成本较重。官网已出现版本更新入口，不沿用图片的固定目标集数量 |
| [TritonBench](https://github.com/thunlp/TritonBench) | 生成 Triton 算子，Triton 是便于编写 GPU 并行计算的语言与编译系统 | GPU 正确性与性能评测 | KernelBench 接口稳定后扩充 kernel 题型，初期无需两个近似负载同时接 |
| [InfraDev 论文](https://arxiv.org/abs/2610.36589) | 2026-10-27 发布，36 个源自 Megatron-LM、vLLM、verl 的真实修改任务 | 多 GPU 执行与隐藏验证 | **跟进候选**。论文定位明确，公开任务包、运行入口、每题硬件预算本次尚未核实，不能宣称可立即运行 |
| [sammysun0711/InfraBench](https://github.com/sammysun0711/InfraBench) | 20 个 Harbor 格式 AI 基础设施任务 | 目标为 AMD MI35X，任务表包含 1 卡和 2 卡 | Harbor 格式接近已有路径，但硬件不匹配时不宜首接。与其他同名论文不自动视为同一项目 |
| [MLSysBench](https://github.com/haiyuanwan/MLSysBench) | 配置搜索、调度和跨规模迁移，包含模拟器与原型任务 | 多个任务只运行模拟，真实 CUDA 路径另有边界 | **暂不作为真实 GPU 负载证明**。其仓库明确说明尚非完整正式评测套件 |

这张表关注可运行环境，故没有把图片中所有论文复现或科学问答 benchmark 一并列为 GPU 候选。是否使用研究论文数据，与是否实际启动 GPU 任务是两回事。

## 图片中六项 high GPU 的具体选题

图片的 high GPU 是计算需求类别，不是实测显存、GPU 利用率或跨命令驻留保证。下面按官方任务定义选择，不把套件中所有题都当成 GPU 任务。command 判断是对进程与状态边界的分析，不是官方发布的 Env 模式兼容声明。

| 图中项目与官方来源 | 真正要做的任务 | GPU 形态与 command 判断 | 当前建议 |
|---|---|---|---|
| [AIRS-Bench](https://github.com/facebookresearch/airs-bench) | 20 个来自论文的研究任务，自行提出方法、训练、评估 | 按具体题选择 GPU 训练。训练进程保存结果并退出，可作为 command | 19/19 非 Kaggle 任务有 CUDA 快检，SICK 代表题闭环，原生 sandbox 分支尚无 command 借还 |
| [Automated LLM Speedrunning](https://github.com/facebookresearch/llm-speedrunner/blob/main/config/task/nanogpt_speedrun/default_config.yaml) | 修改 GPT-2 训练代码，在验证损失不高于 3.28 的条件下缩短训练时间 | 官方配置固定 8 张 H100，使用 8 进程 `torchrun`。完整训练作业可是一条 command | 多卡资源重，优先级低于单卡后训练。不把缩成单 H20 的结果称作原版成绩 |
| [MLE-bench](https://github.com/openai/mle-bench) | 75 个 Kaggle 竞赛，做特征、训练、预测和提交 | 选择 GPU 图像或语言模型方案。训练和预测可以分别作为 command | 保留现有接入，按用户安排暂缓 Kaggle 权限和数据 |
| [MLGym-Bench](https://github.com/facebookresearch/MLGym/blob/main/configs/tasks/languageModelingFineWeb.yaml) | 例如 `languageModelingFineWeb`，训练 GPT-2 风格模型做下一个 token 预测 | 官方要求保存 `model.pt` 与 `model_config.pt`，天然有落盘边界。单次训练超时 2400 秒，不等于必须跨命令保留模型 | 本地 4 个 RL、3 个视觉、2 个 NLP 任务均有闭环，CPU 博弈题不计入 GPU 范围 |
| [RE-Bench](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/ai_rd_nanogpt_chat_rl.py) | 例如用强化学习把 GPT-2 XL 训练成聊天模型，按外部模型偏好训练 | GPU 生成、外部裁判等待、GPU 更新交替。官方起始代码在等待时保留 CUDA 模型 | 已有原生接入、本地 Qwen 反馈观测及有界评分闭环，官方全量偏好评测未复现 |
| [PostTrainBench](https://github.com/aisa-group/PostTrainBench/blob/main/src/harbor_adapter/README.md) | 对基础语言模型后训练，提高数学、知识或代码等目标能力 | 官方单 H100、默认 10 小时，训练后交付最终模型。完整训练或评估可作为 command | 已接入数学、代码目标配三个公开模型的 9 个本地组合，全部闭环，整库 24 组合未完成 |

**PostTrainBench 提供的工作形态。** 后训练是在已有基础模型上继续优化参数。当前本地协议使用真实基础权重和真实数据，仅训练最终归一化层，提交 safetensors 权重，再由独立 verifier 重载基础模型并评分隐藏答案似然。Qwen3-1.7B × GSM8K 已包含在通过的 9 个组合中。这个协议保留真实训练、模型交付与独立 GPU 评分，不是官方生成答案准确率。

固定上游的范围为 4 个模型、6 个目标，共 24 个组合。现有 9 个组合对应三个公开模型和 AIME2025、GSM8K、HumanEval。其余组合涉及 Gemma-3 许可、GPQA 数据授权及 ArenaHard / HealthBench 裁判适配。单卡 H20 已有已选子集的实际闭环证据，不能外推其余组合或官方完整训练预算。逐组合数据入口、边界与证据见[任务协议](../milestones/science-gpu-bench-status-and-roadmap.md#posttrainbench-与-fastkernels-任务协议)。

**RE-Bench 的驻留证据更具体。** 下面是已查阅的上游起始代码流程，本地接入复用该训练程序。它描述正式裁判链的状态生命周期，不是无裁判 smoke 已测得的等待曲线：

```text
[R1] GPT-2 XL 放到 CUDA，建立 Adam 优化器
    ↓
[R2] GPU 为每道提示生成两份回答
    ↓
[R3] await 外部裁判比较回答，模型仍由当前进程持有
    ↓
[R4] 根据偏好选回答，GPU 反向传播并更新参数
    └→ 下一轮 R2，最终保存模型
```

图例：此图是外部实现，未复制进本地仓库，因此给官方源码位置。

| 编号 | 官方函数与状态 |
|---|---|
| R1 | [rl_finetune.main:49](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L49)，`model.to(device)` 和 Adam 实例在循环外创建 |
| R2 | [main:74](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L74)，模型生成两组候选回答 |
| R3 | [main:90](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L90)，等待 `evaluate` 返回偏好，没有先把模型移出 GPU |
| R4 | [main:119](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L119)，计算 loss、backward、optimizer step，循环后保存模型 |

跟着图读源码：R1 建立跨迭代状态，R2 将提示变为文本，R3 通过异步 API 取得偏好，R4 用这些偏好更新同一模型，随后重复。这是外部异步等待，不是本地 GPU 在等待期间一定释放。Adam 优化器保存参数更新所需的历史统计，第一次更新后这些状态也会占资源。只保存模型权重不能完整恢复同一训练轨迹，还要考虑优化器、缩放器和随机数状态。以上源码若有不清楚的地方，可以随时追问。

**它仍能包成一条长 command。** 但这样 R3 的等待仍占预约。若想在每次 R3 把 GPU 真正借走，就要适配卸载和恢复，并满足 Env 的实际回收要求。仅在 Python 中 `await` 不会搬走模型。此任务为研究这种机制提供了自然切入点，不等于官方禁止 command，也不等于当前 Env 已自动实现状态迁移。无需 GPU 的相近生命周期演示见 [导览脚本](../env-service-verification/scripts/explain_resident_command_lifetime.py)，它只模拟客户端结束而服务仍在，不模拟 RL 算法或优化器恢复。

后续接入应沿本文全局图，把 task、agent、独立 verifier 与最终清理都纳入同一套 Env 资源管理。对照保持相同真实数据和工作量，分别记录常驻准入与状态卸载恢复的成本，不能以换 benchmark 本身作为 ON 必然更快的保证。

## 首批接入建议

已有入口和运行条件见 [GPU benchmark recipe](../knowledge/gpu-benchmark-recipes.md)，完成范围以[当前路线图](../milestones/science-gpu-bench-status-and-roadmap.md)为准。研究选题应从已通过的协议中取代表任务，避免重复处理依赖和数据错误。

| 研究顺序 | 已有目录 | 当前可用选题 | 适合检验的机制 |
|---|---|---|---|
| 1 | `examples/kernel_bench/` | 270 题身份中的矩阵运算代表题 | 短 GPU 工作、编译成本、计时争用与反馈可靠性 |
| 2 | `examples/mlgym_bench/`、`examples/posttrain_bench/` | MLGym 9 题，PostTrain 已闭环的 9 个组合 | 训练与 CPU 阶段交替、权重交付、独立重载的成本 |
| 3 | `examples/inference_bench/`、`examples/swe_serve/` | 4 个本地服务场景与 SWE score 本地闭环任务 | 常驻服务状态、重启、自测和物理回收 |
| 4 | `examples/autoresearch/`、`examples/re_bench/` | 固定训练起点与有界偏好任务 | 多次实验、外部反馈等待和有代价的状态恢复 |

MLE 官方 Kaggle 题等待凭据和逐竞赛准备，不作为当前可直接执行的研究输入。任务自己的目标服务与 agent 使用的策略模型服务继续分别核对端口、权重和 GPU 归属。对照必须保持真实数据、工作量和评分边界一致。

当前适配器负责任务准备、提交和评分，资源申请、容器生命周期及 GPU 清理复用 [GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L14)与公共 science runner。不要为每个 benchmark 复制一套调度器。AIRS 使用另一条原生 Coordinator/sandbox 分支，后端统一混训尚未验证。

## 具体例子与评分干扰

**教学假设，不是测量结果**：一个 KernelBench episode 花 40 秒读题和生成代码、10 秒编译、2 秒 GPU 运行与计时，三个阶段共 52 秒。若整个 episode 独占一张 GPU，预留为 52 GPU 秒，其中只有 2 秒执行该任务的 GPU 计算。按评测阶段申请理论上可缩短预留，但还要计算初始化、数据传输、切换和等待成本，不能据此直接宣称 26 倍吞吐收益。

输入是一段 PyTorch 矩阵计算和固定输入规范。agent 生成 CUDA 或 Triton 程序，编译器生成可执行代码，evaluator 在 GPU 上运行参考与候选。只有输出符合容差才比较速度，最终再销毁评测进程并核实释放。GPU kernel 是由大量 GPU 线程协作执行的计算函数，其速度来自并行计算与访存组织，不是把 Python 文件改个后缀。

假设参考为 100 微秒，候选为 80 微秒，加速比是 100/80=1.25。若另一个任务竞争显存带宽，使候选测到 120 微秒，则测出的 0.83 会把正确优化误判为退步。因此正式性能计时应使用受控独占窗口，同样约束参考与候选。开发阶段是否允许共享可以另外评估。

对比 MLE：agent 可能先用 CPU 处理数据，再训练 10 分钟，然后用 GPU 预测，最终提交 CSV。GPU 分配应覆盖实际训练和预测进程，不能在 shell 返回但后台训练仍活着时归还。对比 InferenceBench：服务在请求之间利用率可能很低，但模型仍占显存，不能将请求间空隙直接解释为可借出的整卡。

## 本地实现与源码阅读路径

本次核对的是带有未提交修改的主开发目录，而不是一个干净的已发布版本。

- science 的 [runner.py](../../../examples/sciaccel_rl/runner.py#L275) 已有 `needs_gpu`、资源租约和 GPU 作业报告路径。其 [README](../../../examples/sciaccel_rl/README.md) 也提示 ScienceInfra 为后续开发入口。任务是否实际执行 GPU 仍需逐题证明。
- AIRS 的 [run_mlgym_agent](../../../examples/airs_bench/runner.py#L137) 复用 MLGym。原 CPU recipe 保持零 GPU，新增 [GPU 配置](../../../examples/airs_bench/config/gpu_agent_config.yaml) 与 [训练入口](../../../examples/airs_bench/gpu_train.sh) 已覆盖 19 个非 Kaggle 任务，并有 SICK 代表题的策略闭环。实际协议与证据见 [AIRS GPU 接入](../knowledge/airs-gpu-recipe.md)。这是 AIRS 分支的证据。MLGym 的 4 个 RL、3 个视觉和 2 个 NLP 任务已有独立 GPU 快检与策略闭环，实际协议及未完成范围见[当前接入状态](../milestones/science-gpu-bench-status-and-roadmap.md#目前有哪些任务及可用边界)，不等于整个 MLGym 全部任务已完成。
- 这篇文档补充的是候选选择，公共调度机制继续以 [当前设计](../knowledge/current-design.md) 和 [统一 GPU 执行后端](../knowledge/unified-gpu-execution.md) 为入口，避免重复维护架构说明。

跟着图读源码，按真实 science 路径查看：

| 步骤 → 图中编号 | 文件、函数 | 输入输出或状态 |
|---|---|---|
| 申请 → 1、2 | [runner.py:372](../../../examples/sciaccel_rl/runner.py#L372)，`run_harbor_episode` | 任务路径和需求转换为租约，可能异步等待 |
| 维持租约 → 2 | [client.py:50](../../../psrl/workers/env_service/client.py#L50)，`start_heartbeat` | 后台心跳与任务并发，不是任务执行完才调用 |
| 执行 → 3、4 | [runner.py:550](../../../examples/sciaccel_rl/runner.py#L550)，`Job.create`、`job.run` 调用点 | Harbor 执行任务并返回结果，内部 agent/verifier 顺序由 Harbor 管理 |
| 评分处理 → 4 | [runner.py:589](../../../examples/sciaccel_rl/runner.py#L589)，`run_harbor_episode` | command 模式读取 GPU 报告，LAPS 条件分支检查活动证据 |
| 清理 → 5 | [runner.py:610](../../../examples/sciaccel_rl/runner.py#L610)，`run_harbor_episode` 的 finally | 正常、异常、取消均进入收尾，再调用 lease release |

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

路线图已标注的通过范围有本地 GPU 和策略证据，不能扩展为逐题完整许可审计、峰值显存上界、漏洞审计或全部硬件兼容保证。未接入的候选仍按公开入口和调研结论阅读，InfraDev 只核验到论文。执行任何真实实验前按 [集群检查指南](../../psrl/psrl-run-guide/cluster-preflight.md) 做现场预检。

## 非计时评分场景的当前优先级

非计时研究按[场景与策略](../research-proposals/non-timing-gpu-scheduling-scenarios.md)进入，测速反馈按[AI4Infra 选择](ai4infra-performance-benchmarks.md)进入。固定 SWE 清单包含 41 个 GPU 题，其中 3 个按 speedup 评分、38 个按正确性评分，适用问题不同。后续 score 服务题已完成原始评分与相同工作量准备前移对照；fused-kernels 的六项原始正确性测试完成 11 组 H20 运行，观察到跨命令同机换卡仍可利用编译缓存。这些分别支持任务功能、准备取舍和缓存机制，完整接入范围统一见[当前路线图](../milestones/science-gpu-bench-status-and-roadmap.md)。
