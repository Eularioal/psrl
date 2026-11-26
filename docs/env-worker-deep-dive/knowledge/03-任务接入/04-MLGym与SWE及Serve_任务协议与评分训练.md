# MLGym与SWE及Serve：任务协议与评分训练

> **核心结论**：MLGym 任务可提交候选模型权重，由独立 verifier 恢复模型、在隐藏数据上计算分数；外层 Agent 则用交互 token、动作概率、掩码和这个分数更新自身策略。SWE/Serve 提交代码补丁并按各自协议重放。任务协议、评分算例、内外两层训练和本地准备入口在本篇连续说明。

## 目录

- [两层训练如何串联](#两层训练如何串联)
- [MLGym 的范围与评分](#mlgym-的范围与评分)
- [SWE-Serve 的完整导入与运行边界](#swe-serve-的完整导入与运行边界)
- [权重如何评分，探索轨迹如何训练 Agent](#权重如何评分探索轨迹如何训练-agent)
  - [先把两层训练连起来](#先把两层训练连起来)
  - [verifier 怎样把权重变成分数](#verifier-怎样把权重变成分数)
  - [外层训练拿到哪些数据](#外层训练拿到哪些数据)
  - [跟着四条尝试走一轮 DAPO](#跟着四条尝试走一轮-dapo)
  - [分数不可导为何仍能训练](#分数不可导为何仍能训练)
  - [当前 MLGym 入口实际启用了什么](#当前-mlgym-入口实际启用了什么)
  - [CPU 教学实验](#cpu-教学实验)
  - [源码阅读路径与适用边界](#源码阅读路径与适用边界)
  - [相关知识回顾与详细阅读导航](#相关知识回顾与详细阅读导航)
- [准备与验证入口](#准备与验证入口)
- [跟着图读源码](#跟着图读源码)
- [CPU 回归与故障边界](#cpu-回归与故障边界)

> 最新核对：2026-11-16。主目录 `psrl/`，接入代码已在共享目录提交 `8755e20`，后续修改仍含未提交内容。本次核对 MLGym 权重评分、轨迹奖励与公共训练配置，并关联新增 SWE GPU 快检及 DSv32、dLLM 策略闭环。历史实验保留原报告条件，本任务未发起提交、推送、合并或共享服务部署，远端状态未核验。


## 两层训练如何串联

MLGym 在任务环境内训练模型并交付权重；SWE-Serve 修改服务代码并交付补丁。下面重点看这些提交物怎样进入独立评分器，以及评分如何接回 PSRL 策略训练。

<!-- figure:two-model-training-loops -->
这张图沿同一个场景解释：Agent 让环境训练了一个模型，PSRL 更新的到底是哪一个模型？

[![两个模型、两层训练：工具里的模型与Agent策略分开更新（1/2）](../../../_static/img/document-figures/two-model-training-loops-screen.png)](../../../_static/img/document-figures/two-model-training-loops.png)

[![两个模型、两层训练：工具里的模型与Agent策略分开更新（2/2）](../../../_static/img/document-figures/two-model-training-loops-screen-02.png)](../../../_static/img/document-figures/two-model-training-loops.png)

策略模型 θ 提出训练方案，环境任务模型 φ 执行更新并保存产物，评分器返回质量。 φ 的内部训练次数与 θ 的策略 step 属于两层循环。任务产物不会自动成为下一轮 Agent。 θ、φ 是教学符号。任务模型接管 Agent 是额外设计，需要能力评估与发布协议。 [查看矢量图](../../../_static/img/document-figures/two-model-training-loops.svg)。
<!-- /figure:two-model-training-loops -->

```text
[1] 题目准备器 → 指令、任务目录、Parquet 数据行
                         ↓
[2] PSRL 策略采样 → agent 生成命令 → GpuBenchmarkAgentLoop
                         ↓
[3] EnvRouter 准入 → Harbor 创建任务容器
                         ↓
[4] 环境 GPU：JAX/PyTorch 训练，或修改 SGLang 后启动服务
                         ↓ 提交权重 / 代码补丁
[5] 独立 verifier：重新加载权重或代码 → GPU 评估 → reward.json
                         ↓
[7] Harbor finally：停止所属容器、归还租约
                         ↓ 分数与 Agent token 轨迹返回
[6] 奖励适配 → 组内优势 → PSRL 更新 Agent → 发布新版本 → 回到 [2]
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与入口 | 源码 |
|---|---|---|
| 1 | `prepare_rl` / `prepare` 生成完整任务封装 | [RL 准备器](../../../../examples/mlgym_bench/prepare.py#L18)、[视觉准备器](../../../../examples/mlgym_bench/prepare_vision.py#L17)、[NLP 准备器](../../../../examples/mlgym_bench/prepare_nlp.py#L18)、[SWE 导入器](../../../../examples/swe_serve/prepare.py#L24) |
| 2–3 | `_run_harbor_in_thread` 创建资源请求并调用 Harbor；未启用 EnvRouter 时拒绝运行 | [GpuBenchmarkAgentLoop](../../../../examples/gpu_benchmarks/agent_loop.py#L32)、[run_harbor_episode](../../../../examples/sciaccel_rl/runner.py#L275) |
| 4–5 | `execute` / `evaluate` 分别处理训练与独立验证，CUDA 不可用则失败 | [JAX RL](../../../../examples/mlgym_bench/rl.py#L19)、[PyTorch 视觉](../../../../examples/mlgym_bench/vision.py#L14)、[NLP](../../../../examples/mlgym_bench/nlp.py#L11)、[SWE 快速协议](../../../../examples/swe_serve/quick_profile.py#L17) |
| 6 | `compute_score` 检查有效性；`_update_actor` 更新外层语言模型 | [奖励适配](../../../../examples/gpu_benchmarks/reward.py#L6)、[策略更新](../../../../psrl/trainer/ppo/ray_trainer.py#L3222) |
| 7 | `run_harbor_episode` 在返回前清理任务容器与租约，独立快检另核对物理快照 | [任务 finally](../../../../examples/sciaccel_rl/runner.py#L610)、[有界验证器](../../../../examples/gpu_benchmarks/run.py#L33) |

这些代码如果有理解上的问题，可以随时追问。图中的多条 rollout 可以并发，资源准入会排队；它不是全局单线程流程。策略更新等待所需的轨迹批次，不能由单个任务里的训练日志推断策略更新已完成。

这里有两层参数：容器内训练程序更新任务模型，PSRL 更新负责写代码与调用工具的 Agent。前者交给 verifier 评分，后者使用“生成了哪些动作、获得多少奖励”的轨迹学习。当前公共入口采用 GRPO 中心化优势、非对称裁剪和 token 平均损失，动态采样默认关闭，完整解释见[逐步算例与配置](#当前-mlgym-入口实际启用了什么)。

例如 Breakout：agent 收到训练指令，在环境容器运行公开基线。基线用 JAX 的 GPU 后端并行推进 16 个游戏环境，每个最多 64 帧，做 2 次小策略更新，再写 `policy.msgpack`。独立 verifier 在另一容器用种子 109、211 各评估 16 条轨迹，得到原始平均回报。固定参考提交的实测回报为 0.46875，映射为 `1-exp(-0.46875/10)≈0.0457933`。这个奖励随后才进入 PSRL 外层语言模型的训练。它证明真实采样、训练、权重交付和独立评分可工作，不说明游戏已经学好。

## MLGym 的范围与评分

本地 GPU 范围为 **4 个 RL 配置、3 个视觉任务、2 个 NLP 任务身份**，共 9 个，不包含仓库中的 CPU 任务，也不等于复现官方完整训练预算。上游 MLGym 源码核对为干净提交 `9d40c1b5035202018cd7091fb4e83a9c68b377c0`，实际网络文件另有内容哈希。

| 类型 | 任务身份 | 本地协议 |
|---|---|---|
| JAX 强化学习 | Breakout-MinAtar、MetaMaze-misc、MountainCarContinuous、MountainCarContinuousReinforce | 真实 Gymnax 环境，上游网络代码，宽度 32、2 层；共享本地 actor-critic 基线，2 次更新；提交 Flax 权重 |
| 图像分类 | CIFAR-10、Fashion-MNIST | 真实图像，64 训练样本、16 隐藏测试标签，32×32 RGB，小型卷积模型，2 个 epoch |
| 图像描述 | COCO Karpathy | 真实图片与描述，卷积编码器加循环解码器，词表只来自训练描述，独立计算 BLEU-4 |
| 语言建模 | FineWeb | 原上游 GPT 风格网络，2 层、宽度 64、字节词表，2 次 CUDA 更新，隐藏文本计算 `exp(-loss)` |
| 句对分类 | MNLI | 冻结真实 BERT-base 编码器，只训练三分类头 2 次，隐藏句对计算准确率 |

JAX 是数组计算与自动求导框架，Gymnax 提供可在 JAX 中并行推进的环境。CUDA 是本轮实际使用的 GPU 计算后端。Flax 的 msgpack 文件保存网络参数，verifier 根据可信网络结构恢复它。PyTorch 视觉分支只加载 `weights_only=True` 的状态字典，隐藏标签不挂入 agent 容器。

两份 MountainCar 配置在本地采用同一种小型 actor-critic 训练程序，因此是 **4 个配置、3 种实际环境**。不能据此宣称已复现上游 PPO 与 REINFORCE 算法比较。图像在准备阶段下载、解码并缩小，任务运行时读取像素数组并做真实 GPU 前后向；这不覆盖持续在线解码、增强或完整大规模视觉数据流水线。

RL 评分采用有限、可比较的本地映射：MountainCar 为 `exp(min(return,0)/100)`，其他环境为 `1-exp(-max(return,0)/10)`；同时保留原始回报。视觉任务返回准确率或 BLEU-4，不采信提交文件中自报的分数。NLP 提交使用 safetensors，可信评估器严格检查参数键、尺寸和有限数值。小样本得分不适合作为模型能力排名。

## SWE-Serve 的完整导入与运行边界

固定上游 `84d8fd92739a4a76ceddba56e5a70605921c4ad2` 有 53 题，其中 41 道声明需要 GPU，12 道 CPU 题不计入本任务。准备器导入全部 41 道 GPU 题，保留指令、参考补丁和独立测试，并记录每个文件的 SHA-256。`runtime_verified_count=0` 是准备器的正确初始状态，之后必须关联真实执行证据，不能在导入时设为通过。

独立测试目录只挂入 verifier，agent 只看到公开任务资料。实际接入时修复了独立容器缺少 `/tests` 挂载及 `test.sh` 不可执行的问题。使用预构建镜像时，部分题还需要原始 `/base` 和私有 `/speed-check`：`--base-map` 提供逐题未打补丁的源码路径，只给 verifier 只读挂载，不能指向 agent 修改后的 `/code`。准备器缺少必要基线就拒绝生成；基线源码重建另经真实 CPU 容器检查通过，不冒充该性能题 GPU 回归通过。模型缓存只读挂载，每道题继续执行上游资产和测试包完整性检查。

`ss-sgl22544-score-multi-item-delimiters` 已有一轮完整原测试的真实 GPU 验证：18 个 fail-to-pass 测试和 8 个 pass-to-pass 测试全部通过。前者检查修复目标，后者检查原有功能未损坏。另建的快速协议只运行同一测试包中 2 个服务适配器测试，并要求 agent 实际调用 CUDA `Engine.score`。快速协议明确公布参考补丁用于功能验收，因此使用独立的 `swe_serve_local` 数据源，不作为原 26 测试成绩或 agent 修复能力成绩。

原始完整导入保留上游交付配置。[bounded_tests](../../../../examples/swe_serve/bounded_tests.py#L107) 和 [indexer_profile](../../../../examples/swe_serve/indexer_profile.py#L42) 生成的本地协议明确启用代码补丁交付，要求 `/logs/artifacts/submission.patch`，独立 verifier 将其应用到干净源码后再运行测试，见 [patch_submission.configure](../../../../examples/swe_serve/patch_submission.py#L18)。不能将全部 41 题的原始导入或 score 历史闭环写成补丁路径全部实测。本地快速协议可选择少量原测试节点，Indexer 另有明确的 CUDA 投影协议，这些成绩与原题完整测试分开。

其他题目存在 SGLang 版本、CUDA 扩展、模型资产或下载权限差异。仅构建出镜像不能证明其测试通过。批量构建镜像首先服务 oracle 测试，交互 agent 所需工具和策略闭环仍需逐题补齐。当前本地镜像构建器可复用已验证的 CUDA 运行时并替换为逐题固定源码，但记录了上游与本地基础镜像的差异；它不是官方镜像等价性声明。

本地 dLLM 快检另由 [dllm_profile](../../../../examples/swe_serve/dllm_profile.py) 准备：从不可变上游测试中明确选取六项生产采样算法回归，用 [cuda_pytest](../../../../examples/swe_serve/cuda_pytest.py) 将默认张量放到 CUDA，并逐项拒绝零 CUDA 操作的测试。它保留补丁交付和独立重放，不启动完整扩散语言模型服务。实际六项全过，原完整服务测试仍保存在 `official-test.sh`。DSv32 cache 和 Embedding LoRA 使用各自源码兼容的镜像，所选原测试分别 4/4 与 3/3 全过，逐题条件见[当前状态审计](../../experiments/gpu-benchmarks/onboarding-status-audit-20261116.json)。


## 权重如何评分，探索轨迹如何训练 Agent

下面沿一个候选模型的提交和四条 Agent 尝试，依次说明评分、轨迹数据和外层策略更新。
>
> 核对日期：2026-11-16。源码为主目录提交 `6aa328d` 加已有未提交内容，重点核对 MLGym 评分器、公共训练脚本、轨迹掩码、奖励适配和优势计算。本次运行标准库 CPU 教学实验，没有启动 GPU、Docker 或实际策略训练。任务接入与历史实测仍以 [recipe](04-MLGym与SWE及Serve_任务协议与评分训练.md) 为入口。

### 先把两层训练连起来

假设题目是训练图像分类器。Agent 是解决题目的语言模型，它生成 Python 代码与工具命令。分类器是任务容器中的另一个模型，它接受图片并预测类别。把 Agent 参数记作 θ，把候选分类器参数记作 φ。

```text
外层 Agent 参数 θ
    │ [1] 读题、写代码、发命令、根据工具结果继续尝试
    ▼
Agent 的一次探索轨迹 τ ─────────────────────────────┐
    │ [2] 容器执行训练代码                         │
    ▼                                             │
内层模型 φ：图片 + 训练标签 → loss → 更新 φ         │
    │ 保存 checkpoint，即模型参数文件              │
    ▼                                             │
[3] verifier：可信网络结构 + φ + 隐藏评估数据        │
    │ 推理、比较答案，输出 reward.json             │
    ▼                                             │
Harbor 收尾清理容器、释放租约                      │
    ▼                                             │
[4] 奖励适配得到标量 R ────────────────────────────┤
    ▼                                             ▼
[5] 同题多条轨迹 τ、各自 R、生成概率、掩码 → 相对优势 A
    │ [6] 对 Agent 的生成概率计算策略损失，反向传播
    ▼
Agent 参数 θ 更新 → 发布给采样引擎 → 新一轮探索
```

这是数据依赖图。同题的多次探索可以并发，各自生成提交物。PSRL 还可以重叠不同轮的采样与训练，实际执行顺序取决于配置。

| 层次 | 更新谁 | 使用什么数据 | 改善什么 |
|---|---|---|---|
| 内层任务训练 | 分类器、游戏策略或任务模型 φ | 图片与标签、文本、游戏交互等 | 这份候选模型的任务表现 |
| 外层 Agent 强化学习 | 写代码、调用工具的语言模型 θ | Agent 轨迹、生成概率、任务评分 | 今后解决这类题目的决策方式 |

所以，“权重本身应该不能拿来训练”需要补上对象：权重是内层训练得到的模型状态，也可作为继续训练的起点。在这里，它交给 verifier 评估；PSRL 外层优化器使用的是 Agent 轨迹的学习信号。

### verifier 怎样把权重变成分数

权重文件保存参数张量，例如卷积核和分类矩阵。执行推理还需要网络结构，告诉程序怎样把输入与这些张量组合。当前本地协议要求提交物匹配可信评估器的固定结构，Agent 可以改变训练过程，最终参数键和尺寸要符合协议。

本地 CIFAR-10 分类分支执行以下步骤：

1. **内层训练**：读取公开图片与训练标签，构建卷积网络，计算交叉熵损失，用 `loss.backward()` 与 `optimizer.step()` 更新 φ，保存参数字典。
2. **独立恢复**：verifier 自己构建相同网络，以 `weights_only=True` 读取参数，`strict=True` 检查键与尺寸。隐藏标签只挂入 verifier 容器。
3. **隐藏评估**：调用 `model.eval()`，在 `torch.no_grad()` 内推理，选择输出分数最大的类别，与隐藏答案比较。
4. **输出奖励**：计算准确率，把分数与有效性标记写入 `reward.json`。奖励适配器读取分数并检查标记。

源码见 [训练与保存](../../../../examples/mlgym_bench/vision.py#L70)、[恢复与评分](../../../../examples/mlgym_bench/vision.py#L103)、[数据挂载](../../../../examples/mlgym_bench/prepare_vision.py#L49)、[奖励检查](../../../../examples/gpu_benchmarks/reward.py#L40)。

**教学数字**：隐藏测试有 16 张图片，提交权重预测对 12 张，准确率就是 `12/16=0.75`，这次探索获得基础奖励 0.75。Agent 在日志中自报的分数不会代替独立计算。

`eval()` 改变特定网络层的运行行为，例如关闭 dropout 的随机丢弃。`no_grad()` 关闭这次推理的梯度记录，节省反向传播所需的中间状态。评分器同时使用二者。

其他任务保留同样的“恢复候选、独立评估”结构：

| 任务 | verifier 用权重做什么 | 基础分数 |
|---|---|---|
| 图像分类 | 对隐藏图片预测类别 | 准确率 |
| 图像描述 | 生成描述，与隐藏参考描述比较 | BLEU-4，即词语片段匹配指标 |
| FineWeb 语言建模 | 预测隐藏文本的下一个字节，计算平均损失 | `exp(-平均损失)` |
| MNLI 句对分类 | 加载固定 BERT 与候选分类头，判断句对关系 | 准确率 |
| 游戏强化学习 | 恢复候选游戏策略，在独立种子的环境中行动 | 平均游戏回报经本地映射得到奖励 |

语言分支见 [nlp.execute](../../../../examples/mlgym_bench/nlp.py#L11)。游戏分支见 [rl.execute](../../../../examples/mlgym_bench/rl.py#L19)：训练种子为 42、43，验证种子为 109、211。游戏内的“状态、动作、回报”轨迹训练游戏策略 φ；外层的“Agent 写代码、运行程序、读结果”轨迹训练语言模型 θ。这里的确有两层强化学习，但内层视觉分类采用监督学习，并非每类任务内层都是 RL。

### 外层训练拿到哪些数据

一次探索可以是：

```text
题目：训练分类器，提交符合协议的权重
Agent：检查数据、写训练脚本
工具：返回数据形状
Agent：运行脚本
工具：返回 loss 日志，提示模型已保存
Agent：确认提交文件并结束
verifier：加载权重，算出奖励 0.75
```

**trajectory（轨迹）**是一次完整尝试的交互。**token** 是文本编码单位，代码、命令与自然语言都会编码成 token。训练记录包括：

| 数据 | 用途 |
|---|---|
| 题目与历史上下文 | 记录生成动作时看到了什么 |
| Agent 输出 token | 学习提高或降低这些动作的生成概率 |
| 工具观察 | 作为后续决策的条件，例如根据报错修复 |
| 生成时的对数概率 | 比较采样策略与更新策略的概率 |
| `response_mask` | 指明模型动作与环境观察的位置 |
| verifier 分数 | 判断完整尝试相对同题其他尝试的好坏 |

mask 即掩码，是逐位置的训练开关。通常 Agent 输出对应 1，工具观察对应 0，预算截断过滤还可能进一步屏蔽位置。工具文本进入上下文，影响后续动作，但自身不作为策略动作计入损失。

例如工具返回 `loss=0.2`，外层模型学习“看到这个结果后应该生成什么命令”。这是环境训练损失的文字观察，与外层反向传播使用的策略损失分属两层。

真实接线中，[SciAccelAgentLoop](../../../../examples/sciaccel_rl/agent_loop.py#L227) 从会话取 token 轨迹、附上 `harbor_rewards`；[build_token_output](../../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L226) 保留掩码与对数概率；[AgentData](../../../../psrl/workers/agent_loop/agent_data/base.py#L311) 区分环境与模型位置；[compute_score](../../../../examples/gpu_benchmarks/reward.py#L6) 读取独立分数。外层样本可概括为 `上下文 + 动作 token + mask + 概率 + 奖励`。

### 跟着四条尝试走一轮 DAPO

DAPO 的名称来自“解耦裁剪与动态采样策略优化”。它保留同题多次尝试的相对奖励思想，并调整裁剪、采样、token 损失聚合及超长反馈。以下按启用动态采样的教学流程说明，本地开关见下一节。[原论文](https://arxiv.org/html/2503.14476v2)提供算法定义。

**第一步：用采样策略 θ_old 对同题生成多条探索。** 每条轨迹可能采用不同代码、超参数和修复过程，各自产生候选权重。共享初始题目与评分规则的多次尝试构成一个 group，即比较组。

**第二步：分别评分，计算 advantage（优势）。** 优势表达“这次比同题的比较基准好多少”。以下都是教学数字：

| 探索 | 隐藏测试正确数 | 奖励 R | R 减去组内均值 |
|---|---:|---:|---:|
| τ1 → φ1 | 12/16 | 0.7500 | +0.15625 |
| τ2 → φ2 | 10/16 | 0.6250 | +0.03125 |
| τ3 → φ3 | 8/16 | 0.5000 | −0.09375 |
| τ4 → φ4 | 8/16 | 0.5000 | −0.09375 |

均值为 `0.59375`。原论文还除以组内标准差：`A_i=(R_i-平均R)/标准差`。当前公共 MLGym 入口关闭标准差归一化，因此上表最后一列就是其基础优势形式，尚未叠加其他过滤或校正。

**第三步：动态采样保留有比较信号的组。** 若四次都得 0.5，中心化优势全为零。动态采样过滤这类组，在有界预算内补足有效组。连续评分任务应检查奖励差异，不能仅按“提交是否都合法”过滤。

**第四步：重算已采动作的概率，构造裁剪损失。** 把旧轨迹再次输入当前 Agent θ，计算当时每个动作 token 的概率。训练前向处理的是记录下来的文本，不需要再次执行整套容器和内层训练。

```text
ρ_i,t = 当前 Agent 对这个 token 的概率 / 采样策略对它的概率
u_i,t = min(ρ_i,t × A_i,
            clip(ρ_i,t, 1-ε_low, 1+ε_high) × A_i)
loss = -Σ(mask_i,t × u_i,t) / Σ(mask_i,t)
```

`clip` 把数值限制在范围内。非对称门槛分别控制概率比下降和上升两侧。以上展示核心项，真实流水线还可包含采样版本校正等处理。

以当前脚本的 low=0.2、high=0.3 为例：正优势 token 的旧概率 0.1，当前为 0.12，概率比 1.2，处于上升范围；若变为 0.15，比值 1.5，裁剪目标不再因继续提高这一比值而增大。这里限制的是该项的优化收益，共享参数和其他样本仍会影响最终概率。

token 平均指先累加全部有效模型 token 的损失，再除以有效位置数。两条轨迹有 10、30 个有效 token，就有 40 个位置参与平均，长轨迹贡献更多位置。

**第五步：反向传播，更新 Agent θ。** 正优势倾向于提高对应动作概率，负优势倾向于降低它。大量 token 共享参数，最终更新是所有样本梯度的合成。终局分数提供整次尝试的反馈，不会自动精确指出哪一行代码起了决定作用。

**第六步：发布新 Agent，继续采样。** θ 更新后，新探索可能生成不同代码并训练出新候选权重。真实 PSRL 支持流水重叠，需要记录采样策略版本并处理其与更新策略的差异。θ_old 表示帮助理解的一轮固定采样假设。

### 分数不可导为何仍能训练

准确率中的离散比较、shell 执行、文件保存和独立评分没有连通的自动求导图。策略梯度换了一条求导路径：把奖励视为观测到的常数，用它给 Agent 动作的对数概率梯度加权。

```text
目标梯度 ≈ 平均[ A_i × ∇θ log Pθ(动作 | 当时上下文) ]
```

`∇θ` 表示对 Agent 参数求导。有梯度的部分是语言模型的动作概率。候选 φ 文件和评分程序提供奖励值，无需参与这次外层反向传播。

用只有两种方案的教学模型手算：Agent 选好方案的概率 p，选差方案的概率 `1-p`；两种方案期望得分分别为 0.8、0.4，则期望奖励为 `0.8p+0.4(1-p)=0.4+0.4p`。提高 p 会改善期望奖励，而产生这两个分数的程序完全可以是黑盒。

因此，“用探索轨迹训练”抓住了外层对象。完整说法还应包括：轨迹关联独立评分，并保留模型动作概率与掩码，候选权重让这次探索的成果可被验证。

### 当前 MLGym 入口实际启用了什么

[公共入口](../../../../examples/gpu_benchmarks/train.sh#L29)调用 [fsdp_qwen35_4b.sh](../../../../examples/sciaccel_rl/fsdp_qwen35_4b.sh#L161)。以下为源码默认值，实际运行以启动参数和配置转储为准：

| 机制 | 当前默认配置 | 意义 |
|---|---|---|
| 同题多次探索 | 每题 8 次，`rollout_N=8` | 收集同题差异 |
| 相对优势 | `adv_estimator=grpo`，关闭标准差归一化 | 组内中心化奖励 |
| 动态采样 | `GROUP_FILTER=False` | 已配置过滤器，默认关闭 |
| 非对称裁剪 | low=0.2，high=0.3 | 分开限制概率比两侧 |
| 损失聚合 | `loss_agg_mode=token-mean` | 按有效 token 汇总 |
| 超长轨迹过滤 | `OVERLONG_FILTERING=True` | 截断轨迹屏蔽策略损失，评分保留在组内基准 |
| 超长软惩罚 | `overlong_buffer_cfg.enable=False` | 不额外叠加这项长度惩罚 |
| KL 约束 | 奖励与 loss 中均关闭 | 不加与冻结参考模型的分布偏移惩罚 |
| 采样版本差异 | 默认 staleness=1，配置 token 级重要性校正 | 处理重叠采样与训练产生的策略差异 |

KL 是衡量两个概率分布差异的量，参考模型约束用它限制 Agent 偏离初始模型。`staleness` 是允许采样版本落后训练版本的配置。重要性校正则按采样概率与目标概率关系调整样本作用，具体计算见[采样修正章节](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#rollout-correction-与版本滞后)。

`dapo` reward manager 是奖励模块名。当前入口更准确的描述是 **GRPO 中心化优势，加上部分 DAPO 风格配置**。完整机制是否启用，要逐项核对开关。

### CPU 教学实验

从仓库根目录运行，无需 PyTorch、Docker 或 GPU：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/mlgym_dapo_layers_demo.py
```

脚本让四个候选分类器在相同隐藏小数据上推理，再计算分数、中心化优势、可选标准差归一化、同分组过滤与掩码，最后用解析梯度更新一个只会在两种方案间选择的外层策略。

这是教学模拟：固定的一维分类阈值代替真实 CNN 训练，一个标量策略代替语言模型，最后一步演示基础策略梯度方向，没有实现完整 DAPO 裁剪优化器或分布式执行。

实际观察：四份候选得分为 0.75、0.625、0.5、0.5；同分组优势全零；策略损失位置排除工具观察；外层 θ 从 0 变为 0.1，选择好方案的概率从 0.5 变为约 0.52498，候选权重保持原值。

### 源码阅读路径与适用边界

| 图中步骤 | 函数与源码 | 输入、输出与状态变化 |
|---|---|---|
| 1，任务入口 | [GpuBenchmarkAgentLoop](../../../../examples/gpu_benchmarks/agent_loop.py#L32) | 题目与资源需求 → Harbor Agent 会话 |
| 2，内层训练 | [vision 训练分支](../../../../examples/mlgym_bench/vision.py#L70) | 训练数据 → loss → 更新 φ → 参数文件 |
| 3，独立评分 | [vision 验证分支](../../../../examples/mlgym_bench/vision.py#L103) | 固定结构、φ、隐藏数据 → 预测 → 准确率 |
| 4，任务收尾 | [结果](../../../../examples/sciaccel_rl/runner.py#L574)、[finally](../../../../examples/sciaccel_rl/runner.py#L610) | 获取 verifier 分数，返回前清理所属容器与租约 |
| 4 → 5 | [AgentLoop](../../../../examples/sciaccel_rl/agent_loop.py#L227)、[奖励适配](../../../../examples/gpu_benchmarks/reward.py#L6) | token、分数与有效性信息 → 外层训练输出 |
| 5，过滤与优势 | [组过滤](../../../../psrl/utils/post_processor/group_post_process/filter.py#L28)、[优势计算](../../../../psrl/trainer/ppo/utils.py#L295) | 检查差异、按同题分组计算优势、多段会话按实现广播 |
| 6，更新与发布 | [_compute_advantage](../../../../psrl/trainer/ppo/ray_trainer.py#L3011)、[_update_actor](../../../../psrl/trainer/ppo/ray_trainer.py#L3222) | 组织批次、调用 Actor 更新并要求发布模型 |

这些代码中的任意一步都可以继续追问。本篇核对本地权重协议和外层机制，历史 GPU 验收、实际外层参数更新、Agent 能力改善分别需要相应证据。

### 相关知识回顾与详细阅读导航

PPO、GRPO、DAPO 的相连原理已合并到[从奖励到模型更新的完整教程](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md)。奖励来源、GAE、同题分组、裁剪和损失都可在同篇连续阅读，通过章节跳转复习：

- [奖励怎样进入策略梯度](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#1-从评分到更新的完整链路)。
- [PPO 与 GAE](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#2-ppo优势估计与裁剪更新)、[GRPO 的组内比较](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#3-grpo用同题回答互相比较)。
- [DAPO 四项机制](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#4-dapo四项机制怎样配合)与[多轮 token 损失](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#5-一条多轮轨迹怎样变成训练损失)。
- [熵与 KL](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#6-熵与-kl探索和参考模型约束)、[裁剪与共享参数手算](../../../psrl/02-强化学习_PPO与GRPO及DAPO.md#7-深入理解裁剪与共享参数)。

独立的系统与排障问题继续从下列文档进入：

| 想回顾的问题 | 文档 | 内容与范围 |
|---|---|---|
| MLGym、SWE、Serve 分别怎样提交与评分 | [任务协议](04-MLGym与SWE及Serve_任务协议与评分训练.md) | 各类产物、隐藏验证器和奖励适配 |
| Agent 交互怎样成为训练数据 | [Agent Loop、Harbor 与 TITO](../../../psrl/01-PSRL原理_AgentLoop与工具调用.md) | 工具调用、token 记录和训练载荷 |
| 采样、训练与权重发布怎样重叠 | [训推协作](../../../psrl/推理与训练/03-训推协作_异步流水线与版本协调.md) | 进程分工、完整数据路径与版本滞后 |
| batch 参数怎样映射轨迹数 | [批量配置](../../../psrl/psrl-run-guide/01-PSRL运行_准备启动与监控.md#train_prompt_bsz-对应哪个配置是什么意思) | 问题数、采样数与计算批量 |
| loss、奖励、长度与梯度怎样联合判断 | [历史退化诊断](../../../psrl/05-训练排障_DAPO退化诊断.md#训练曲线指标奖励长度梯度和效率怎样计算) | 指标计算与历史退化证据 |

历史 demo 与当前 MLGym 配置应分别阅读，原理教程第 8 节与本篇的实际开关表注明各自范围。


## 准备与验证入口

命令从仓库根目录执行。路径变量须显式设置为实际资产和输出目录；输出目录必须是新目录，避免混入旧结果。执行 GPU 命令前遵守[集群预检](../../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)。

```bash
# RL_PYTHON 指向任务专用 JAX 环境，新包安装到该环境。
python -m examples.mlgym_bench.prepare --repo "$MLGYM_REPO" \
  --output "$RL_TASKS" --python "$RL_PYTHON" --image "$MLGYM_IMAGE"

# 下载真实小样本，分别保存公开像素与训练标签、隐藏测试标签。
python -m examples.mlgym_bench.download_vision --output "$VISION_DATA"
python -m examples.mlgym_bench.prepare_vision --data "$VISION_DATA" \
  --output "$VISION_TASKS" --python "$TORCH_PYTHON" --image "$MLGYM_IMAGE"

# 导入全部 GPU 题；镜像映射不意味着该题已验证。
python -m examples.swe_serve.prepare --repo "$SWE_REPO" \
  --output "$SWE_TASKS" --hf-cache "$HF_CACHE" --image-map "$IMAGE_MAP" \
  --base-map "$PRISTINE_BASE_MAP"

# 对一题做真实 PSRL 环境快检，包含独立 verifier 和回收检查。
python -m examples.gpu_benchmarks.run --task "$TASK" --output "$NEW_RESULT" \
  --node-ip "$NODE_IP" --gpu "$IDLE_GPU" --gpu-memory-mb 8192 --timeout 300
```

最后一条对 MLGym 可使用 8192 MiB 准入预算；SWE 服务题需按题目与实测设置，当前评分服务使用 80000 MiB。预算是准入声明，不是硬显存隔离。三种 MLGym 准备器可加 `--guided-baseline`，给出 direct 后端明确的 `psrl-gpu start/wait` 验收步骤。这个标记只用于功能验收：`psrl-gpu` 是框架截获的协议行，不能在前面加 `sleep` 或 `export`，也不能给 direct 后端传逐作业 `gpu_memory_mb`。验收提示包含明确的工具调用指引，不用于声称策略学习效果。公共 `train.sh` 已接受 `mlgym_bench` 和 `swe_serve`，正式策略训练还需显式设置模型、数据、agent 配置、训推布局和 EnvRouter units，参考[公共训练入口](01-任务接入_公共GPU任务与PSRL配方.md#接入训练)。

NLP 准备入口为 `python -m examples.mlgym_bench.prepare_nlp`，需要 `--repo`、`--assets`、`--output`、`--python`、`--image`，也支持 `--guided-baseline`。资产目录须包含真实 BERT 权重，以及两题分开的公开和隐藏数据，具体结构见 [prepare_nlp](../../../../examples/mlgym_bench/prepare_nlp.py#L18)。公共 `train.sh` 同样接受 `posttrain_bench` 和 `fastkernels_bench`，它们的当前范围和数据入口见[路线图](../../milestones/02-Science现状_任务覆盖与接入路线.md#posttrainbench-与-fastkernels-任务协议)。

当前 JAX 环境固定版本见 [requirements-rl.txt](../../../../examples/mlgym_bench/requirements-rl.txt)。准备器读取真实 CUDA 工具链与 NVIDIA 动态库位置，生成 `/opt/task/python`，解决独立虚拟环境内 CUDA 库发现的问题。仅能 import JAX 不代表 GPU 后端可运行，固定实验已有容器内真实 CUDA 执行证据。JAX 虚拟环境启用 `system-site-packages`，其他若干任务只读复用主 Python 前缀。SWE 的安装发生在镜像构建或任务容器内部，下载的 wheel 与模型保存在 bench 专用资产目录。镜像与容器可写层、只读挂载、主环境复用的区别，结合三类真实任务和可运行教学实验，统一见[环境隔离详解](../../milestones/02-Science现状_任务覆盖与接入路线.md#环境是否隔离及为何反复下载)。

## 跟着图读源码

1. **步骤 1 → 图 1**：准备器接收上游源码、资产与解释器，输出任务目录和 `extra_info.task_path`。检查原始清单与私有数据齐备。
2. **步骤 2 → 图 2–3**：[AgentLoop](../../../../examples/gpu_benchmarks/agent_loop.py#L32) 读取任务资源声明，向 EnvRouter 请求准入，再调用 [Harbor runner](../../../../examples/sciaccel_rl/runner.py#L275)。不同 rollout 在此可能并发或等待。
3. **步骤 3 → 图 4**：[RL execute](../../../../examples/mlgym_bench/rl.py#L19) 或 [视觉 evaluate](../../../../examples/mlgym_bench/vision.py#L14) 的训练分支生成权重，NLP 分支生成严格限定的权重文件，SWE 分支提交服务代码补丁。
4. **步骤 4 → 图 5**：同一程序的验证分支运行在独立容器，读取提交物及私有标签或固定种子；输出数值奖励和 `eval_valid/cuda/local_evaluation` 等证据。
5. **步骤 5 → 图 6**：[compute_score](../../../../examples/gpu_benchmarks/reward.py#L6) 拒绝缺少证据的本地奖励；批次进入 [策略更新](../../../../psrl/trainer/ppo/ray_trainer.py#L3222)。发布后的采样须另查轨迹版本，不能用初始版本采样替代。
6. **资源收尾 → 图 7**：[任务 finally](../../../../examples/sciaccel_rl/runner.py#L610) 在任务返回前清理所属容器与租约，然后图 6 用分数与轨迹继续外层处理。[独立快检](../../../../examples/gpu_benchmarks/run.py#L33)另核对物理容器与 GPU 状态，账本归零不能代替物理核查。

## CPU 回归与故障边界

以下是实际准备器的自包含 CPU 回归，临时生成小型题库，不启动 Docker 或 GPU，不是 GPU 执行证据：

```bash
python -m pytest tests/examples/gpu_benchmarks/test_swe_serve.py \
  tests/examples/gpu_benchmarks/test_mlgym_vision.py -q
```

检查覆盖：全量 GPU 题导入和 CPU 排除、训练验证身份不重叠、私有测试只挂给 verifier、缺隐藏标签时拒绝生成任务、缺 CUDA 有效性标记不给奖励，以及性能题缺少独立基线时拒绝生成。该检查保护接线和数据边界，GPU 与策略闭环结果以路线图链接的实测证据为准。

失败实验保留原日志，不覆盖成成功。下载失败、镜像失败、任务失败、评分为零、策略优化器更新和清理失败分别统计。PostTrain 的 15 个组合和 FastKernels 的 77 个目标已有真实 CUDA 验证，SWE 原测试和本地协议去重后 21 题全过，整库仍未完成。逐题证据、合法输入、依赖与硬件缺口见路线图，Kaggle 按用户要求后续处理。
