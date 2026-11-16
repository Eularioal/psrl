# MLGym 与 SWE-Serve 的本地 GPU 接入

> **核心结论**：MLGym 在任务内部训练 RL、视觉或 NLP 模型并交付权重，SWE-Serve 修改推理服务代码并做真实 GPU 功能回归。二者使用已有 `GpuBenchmarkAgentLoop → EnvRouter → Harbor` 接线。verifier 用提交物独立算分，PSRL 用 Agent 轨迹与分数更新外层语言模型。两套参数、评分与 DAPO 的完整例子见[权重评分与 Agent 训练](mlgym-weight-scoring-and-agent-dapo.md)。本页保留运行协议，完成范围与历史实测查[当前路线图](../milestones/science-gpu-bench-status-and-roadmap.md)。
>
> 最新核对：2026-11-16。主目录 `psrl/`，基准 `6aa328d` 加未提交改动。本次核对 MLGym 权重评分、轨迹奖励与公共训练配置，并关联新增 SWE GPU 快检。历史实验保留原报告条件。本页改动在主目录未提交文件中，未推送、未合并，未更改共享服务部署。


## 目录

- [两层训练如何串联](#两层训练如何串联)
- [MLGym 的范围与评分](#mlgym-的范围与评分)
- [SWE-Serve 的完整导入与运行边界](#swe-serve-的完整导入与运行边界)
- [准备与验证入口](#准备与验证入口)
- [跟着图读源码](#跟着图读源码)
- [CPU 回归与故障边界](#cpu-回归与故障边界)

## 两层训练如何串联

MLGym 在任务环境内训练模型并交付权重；SWE-Serve 修改服务代码并交付补丁。下面重点看这些提交物怎样进入独立评分器，以及评分如何接回 PSRL 策略训练。

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
| 1 | `prepare_rl` / `prepare` 生成完整任务封装 | [RL 准备器](../../../examples/mlgym_bench/prepare.py#L18)、[视觉准备器](../../../examples/mlgym_bench/prepare_vision.py#L17)、[NLP 准备器](../../../examples/mlgym_bench/prepare_nlp.py#L18)、[SWE 导入器](../../../examples/swe_serve/prepare.py#L24) |
| 2–3 | `_run_harbor_in_thread` 创建资源请求并调用 Harbor；未启用 EnvRouter 时拒绝运行 | [GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L32)、[run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275) |
| 4–5 | `execute` / `evaluate` 分别处理训练与独立验证，CUDA 不可用则失败 | [JAX RL](../../../examples/mlgym_bench/rl.py#L19)、[PyTorch 视觉](../../../examples/mlgym_bench/vision.py#L14)、[NLP](../../../examples/mlgym_bench/nlp.py#L11)、[SWE 快速协议](../../../examples/swe_serve/quick_profile.py#L17) |
| 6 | `compute_score` 检查有效性；`_update_actor` 更新外层语言模型 | [奖励适配](../../../examples/gpu_benchmarks/reward.py#L6)、[策略更新](../../../psrl/trainer/ppo/ray_trainer.py#L3222) |
| 7 | `run_harbor_episode` 在返回前清理任务容器与租约，独立快检另核对物理快照 | [任务 finally](../../../examples/sciaccel_rl/runner.py#L610)、[有界验证器](../../../examples/gpu_benchmarks/run.py#L33) |

这些代码如果有理解上的问题，可以随时追问。图中的多条 rollout 可以并发，资源准入会排队；它不是全局单线程流程。策略更新等待所需的轨迹批次，不能由单个任务里的训练日志推断策略更新已完成。

这里有两层参数：容器内训练程序更新任务模型，PSRL 更新负责写代码与调用工具的 Agent。前者交给 verifier 评分，后者使用“生成了哪些动作、获得多少奖励”的轨迹学习。当前公共入口采用 GRPO 中心化优势、非对称裁剪和 token 平均损失，动态采样默认关闭，完整解释见[逐步算例与配置](mlgym-weight-scoring-and-agent-dapo.md#当前-mlgym-入口实际启用了什么)。

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

原始完整导入保留上游交付配置。[bounded_tests](../../../examples/swe_serve/bounded_tests.py#L107) 和 [indexer_profile](../../../examples/swe_serve/indexer_profile.py#L42) 生成的本地协议明确启用代码补丁交付，要求 `/logs/artifacts/submission.patch`，独立 verifier 将其应用到干净源码后再运行测试，见 [patch_submission.configure](../../../examples/swe_serve/patch_submission.py#L18)。不能将全部 41 题的原始导入或 score 历史闭环写成补丁路径全部实测。本地快速协议可选择少量原测试节点，Indexer 另有明确的 CUDA 投影协议，这些成绩与原题完整测试分开。

其他题目存在 SGLang 版本、CUDA 扩展、模型资产或下载权限差异。仅构建出镜像不能证明其测试通过。批量构建镜像首先服务 oracle 测试，交互 agent 所需工具和策略闭环仍需逐题补齐。当前本地镜像构建器可复用已验证的 CUDA 运行时并替换为逐题固定源码，但记录了上游与本地基础镜像的差异；它不是官方镜像等价性声明。

本地 dLLM 快检另由 [dllm_profile](../../../examples/swe_serve/dllm_profile.py) 准备：从不可变上游测试中明确选取六项生产采样算法回归，用 [cuda_pytest](../../../examples/swe_serve/cuda_pytest.py) 将默认张量放到 CUDA，并逐项拒绝零 CUDA 操作的测试。它保留补丁交付和独立重放，不启动完整扩散语言模型服务。实际六项全过，原完整服务测试仍保存在 `official-test.sh`。DSv32 cache 和 Embedding LoRA 使用各自源码兼容的镜像，所选原测试分别 4/4 与 3/3 全过，逐题条件见[当前状态审计](../experiments/gpu-benchmarks/onboarding-status-audit-20261116.json)。

## 准备与验证入口

命令从仓库根目录执行。路径变量须显式设置为实际资产和输出目录；输出目录必须是新目录，避免混入旧结果。执行 GPU 命令前遵守[集群预检](../../psrl/psrl-run-guide/cluster-preflight.md)。

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

最后一条对 MLGym 可使用 8192 MiB 准入预算；SWE 服务题需按题目与实测设置，当前评分服务使用 80000 MiB。预算是准入声明，不是硬显存隔离。三种 MLGym 准备器可加 `--guided-baseline`，给出 direct 后端明确的 `psrl-gpu start/wait` 验收步骤。这个标记只用于功能验收：`psrl-gpu` 是框架截获的协议行，不能在前面加 `sleep` 或 `export`，也不能给 direct 后端传逐作业 `gpu_memory_mb`。验收提示包含明确的工具调用指引，不用于声称策略学习效果。公共 `train.sh` 已接受 `mlgym_bench` 和 `swe_serve`，正式策略训练还需显式设置模型、数据、agent 配置、训推布局和 EnvRouter units，参考[公共训练入口](gpu-benchmark-recipes.md#接入训练)。

NLP 准备入口为 `python -m examples.mlgym_bench.prepare_nlp`，需要 `--repo`、`--assets`、`--output`、`--python`、`--image`，也支持 `--guided-baseline`。资产目录须包含真实 BERT 权重，以及两题分开的公开和隐藏数据，具体结构见 [prepare_nlp](../../../examples/mlgym_bench/prepare_nlp.py#L18)。公共 `train.sh` 同样接受 `posttrain_bench` 和 `fastkernels_bench`，它们的当前范围和数据入口见[路线图](../milestones/science-gpu-bench-status-and-roadmap.md#posttrainbench-与-fastkernels-任务协议)。

当前 JAX 环境固定版本见 [requirements-rl.txt](../../../examples/mlgym_bench/requirements-rl.txt)。准备器读取真实 CUDA 工具链与 NVIDIA 动态库位置，生成 `/opt/task/python`，解决独立虚拟环境内 CUDA 库发现的问题。仅能 import JAX 不代表 GPU 后端可运行，固定实验已有容器内真实 CUDA 执行证据。JAX 虚拟环境启用 `system-site-packages`，其他若干任务只读复用主 Python 前缀。SWE 的安装发生在镜像构建或任务容器内部，下载的 wheel 与模型保存在 bench 专用资产目录。镜像与容器可写层、只读挂载、主环境复用的区别，结合三类真实任务和可运行教学实验，统一见[环境隔离详解](../milestones/science-gpu-bench-status-and-roadmap.md#环境是否隔离及为何反复下载)。

## 跟着图读源码

1. **步骤 1 → 图 1**：准备器接收上游源码、资产与解释器，输出任务目录和 `extra_info.task_path`。检查原始清单与私有数据齐备。
2. **步骤 2 → 图 2–3**：[AgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L32) 读取任务资源声明，向 EnvRouter 请求准入，再调用 [Harbor runner](../../../examples/sciaccel_rl/runner.py#L275)。不同 rollout 在此可能并发或等待。
3. **步骤 3 → 图 4**：[RL execute](../../../examples/mlgym_bench/rl.py#L19) 或 [视觉 evaluate](../../../examples/mlgym_bench/vision.py#L14) 的训练分支生成权重，NLP 分支生成严格限定的权重文件，SWE 分支提交服务代码补丁。
4. **步骤 4 → 图 5**：同一程序的验证分支运行在独立容器，读取提交物及私有标签或固定种子；输出数值奖励和 `eval_valid/cuda/local_evaluation` 等证据。
5. **步骤 5 → 图 6**：[compute_score](../../../examples/gpu_benchmarks/reward.py#L6) 拒绝缺少证据的本地奖励；批次进入 [策略更新](../../../psrl/trainer/ppo/ray_trainer.py#L3222)。发布后的采样须另查轨迹版本，不能用初始版本采样替代。
6. **资源收尾 → 图 7**：[任务 finally](../../../examples/sciaccel_rl/runner.py#L610) 在任务返回前清理所属容器与租约，然后图 6 用分数与轨迹继续外层处理。[独立快检](../../../examples/gpu_benchmarks/run.py#L33)另核对物理容器与 GPU 状态，账本归零不能代替物理核查。

## CPU 回归与故障边界

以下是实际准备器的自包含 CPU 回归，临时生成小型题库，不启动 Docker 或 GPU，不是 GPU 执行证据：

```bash
python -m pytest tests/examples/gpu_benchmarks/test_swe_serve.py \
  tests/examples/gpu_benchmarks/test_mlgym_vision.py -q
```

检查覆盖：全量 GPU 题导入和 CPU 排除、训练验证身份不重叠、私有测试只挂给 verifier、缺隐藏标签时拒绝生成任务、缺 CUDA 有效性标记不给奖励，以及性能题缺少独立基线时拒绝生成。该检查保护接线和数据边界，GPU 与策略闭环结果以路线图链接的实测证据为准。

失败实验保留原日志，不覆盖成成功。下载失败、镜像失败、任务失败、评分为零、策略优化器更新和清理失败分别统计。PostTrain 的 15 个组合和 FastKernels 的 74 个目标已有真实 CUDA 验证，SWE 原测试和本地协议去重后 21 题全过，整库仍未完成。逐题证据、合法输入、依赖与硬件缺口见路线图，Kaggle 按用户要求后续处理。
