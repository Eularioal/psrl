# Science验收：接入流程与环境隔离

> **核心结论**：接入一类 GPU 任务，要依次确认“任务可运行并独立评分”“真实 agent 能操作任务”“交互记录能进入 PSRL 训练”“发布后的策略能继续执行任务”。2026-11-10 的固定实验用六类代表题打通了这条链路。本文先解释怎样接入与检查，再保留该次验收的结果、失败和适用范围。

最新核对：**2026-11-16**。流程说明对应主目录已提交的接入代码 `8755e20`、分组修复 `30997ea` 加后续未提交内容，并关联本轮新增 GPU 任务验证；环境安装审计保留 11-12 的原范围。上述提交在本轮实验期间由共享工作区的并行工作产生，本任务未发起提交、推送或合并，远端状态未核验。历史六类验收基于 `bc5665f` 加当时未提交内容，收尾 HEAD 为 `ab74120`，提交间差异仅为会话归档文档。**本页保留历史六类代表题条件，当前全部题库覆盖看[任务现状与路线图](02-Science现状_任务覆盖与接入路线.md)。**

## 目录

- [验收目标与执行链](#验收目标与执行链)
  - [用数字分类题串起两种训练](#用数字分类题串起两种训练)
  - [跟着流程读源码](#跟着流程读源码)
- [环境与依赖怎样隔离](#环境与依赖怎样隔离)
  - [镜像、容器与挂载各负责什么](#镜像容器与挂载各负责什么)
  - [例子一：PostTrain 怎样训练只读模型](#例子一posttrain-怎样训练只读模型)
  - [例子二：SWE 怎样交付代码修改](#例子二swe-怎样交付代码修改)
  - [例子三：MLGym 虚拟环境怎样复用依赖](#例子三mlgym-虚拟环境怎样复用依赖)
  - [安装审计与长期复现](#安装审计与长期复现)
  - [不需要 GPU 的教学实验](#不需要-gpu-的教学实验)
- [本地任务协议](#本地任务协议)
- [运行入口](#运行入口)
  - [先准备可运行的任务](#先准备可运行的任务)
  - [再生成配套数据与训练配置](#再生成配套数据与训练配置)
- [Results 与验证边界](#results-与验证边界)
  - [真实 GPU 与策略闭环结果](#真实-gpu-与策略闭环结果)
  - [怎样解释失败和受控停止](#怎样解释失败和受控停止)
- [验收相关修复与回归](#验收相关修复与回归)
- [复现时最容易遗漏的条件](#复现时最容易遗漏的条件)
- [下一步研究优先级](#下一步研究优先级)

## 验收目标与执行链

接入有三个验收层次：先用固定参考程序（oracle）在真实 GPU 上跑通任务执行与独立 verifier 评分；再让真实 agent 生成工具调用；最后把交互记录交给 PSRL，执行策略训练与更新后采样。oracle 用于先检查环境和评分规则，把接入故障与 agent 解题困难分开。

<!-- figure:gpu-task-asset-boundaries -->
这张图沿同一个场景解释：同一个 GPU 任务里，哪些能复用、哪些随代码变、哪些必须保密？

[![GPU任务：代码、模型、输入与评分资产有不同寿命（1/2）](../../_static/img/document-figures/gpu-task-asset-boundaries-screen.png)](../../_static/img/document-figures/gpu-task-asset-boundaries.png)

[![GPU任务：代码、模型、输入与评分资产有不同寿命（2/2）](../../_static/img/document-figures/gpu-task-asset-boundaries-screen-02.png)](../../_static/img/document-figures/gpu-task-asset-boundaries.png)

Agent 多次修改候选训练代码，每次使用固定数据和依赖，最后由独立评估器评分。 资产的生命周期不同，调度能否复用状态必须连同版本与评分边界一起判断。 公共结构示意。具体 benchmark 是否已接入、常驻到哪一层，查本篇能力表。 [查看矢量图](../../_static/img/document-figures/gpu-task-asset-boundaries.svg)。
<!-- /figure:gpu-task-asset-boundaries -->

```text
[1] 真实数据、源码与模型 → 准备任务目录、数据表和资源声明
                                      ↓
[2] 策略语言模型 v0 生成操作 ←→ [3] 任务容器执行 CPU/GPU 工作
                                      ↓ 提交代码、预测或权重
                              [4] 独立评分器检查提交物
                                      ↓
                      [5] 模型交互记录 + 任务奖励
                                      ↓
                      [6] PSRL 执行一次策略优化器步骤
                                      ↓
                      [7] 发布 v1 → 用 v1 再交互和评分

每次任务退出 → 清理本次容器和进程 → 核实物理释放 → 归还资源
```

图描述数据怎样流动；多个任务、模型生成和监控可以异步执行。Harbor 负责驱动公共 GPU 任务，EnvRouter 负责资源准入，即判断任务当前能否占用所声明的 CPU、内存与 GPU。租约记录这次资源占用，任务结束后释放。AIRS 的独立协调组件与沙箱路径见 [AIRS recipe](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md)。

### 用数字分类题串起两种训练

以 MLE 本地 digits 题为例：1797 张真实的 8×8 数字图像被固定分成 **1347 张训练图像、450 张测试图像**。agent 只能看到训练标签和测试图像。它调用 GPU 训练分类器，然后提交每个测试 ID 的数字预测。

独立评分器先检查提交的 ID 是否齐全、是否重复、类别是否合法，再读取隐藏标签计算准确率。固定参考提交答对 437 张，所以结果是 `437/450≈97.11%`。

这时完成的是**任务内训练**：分类器学会识别数字，对应图中 [3]。PSRL 随后把 agent 生成的命令、工具结果和奖励组合成训练样本，更新负责操作的**策略语言模型**，对应 [5]→[6]。最后还要看到 v1 生成真实操作并取得评分，才完成 [7] 的策略闭环。两份模型的参数与结果各自记录，97.11% 说明该份分类提交的表现。

### 跟着流程读源码

TITO 是 Token-In-Token-Out 的简称，这里指保存真实模型输入、输出 token 及生成概率的会话记录机制。它让多轮工具交互能进入策略训练。

| 图中步骤 | 本地实现 | 输入如何变成结果 |
|---|---|---|
| [1] 准备 | [write_dataset](../../../examples/gpu_benchmarks/prepare.py#L47)、[混合准备器](../../../examples/gpu_benchmarks/mix.py#L12) | 任务路径、奖励身份和显存需求 → Parquet 数据表与配套 agent 配置 |
| [2] 创建会话 | [SciAccelAgentLoop.run](../../../examples/sciaccel_rl/agent_loop.py#L144) | 数据行 → 模型会话、任务交互与 TITO 记录 |
| [3] 申请与执行 | [GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L14)、[run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275) | 任务资源需求 → 获准资源 → 容器、模型与工具交互 |
| [4] 评分 | [MLE grader](../../../examples/mle_bench/local_grade.py#L9)、[公共奖励适配](../../../examples/gpu_benchmarks/reward.py#L6) | 提交物与隐藏标签 → 协议指标 → PSRL 奖励 |
| [5]→[6] 策略训练 | [完整批次策略](../../../psrl/trainer/ppo/strategies/full_batch.py#L110) → [_update_actor](../../../psrl/trainer/ppo/ray_trainer.py#L3222) | 轨迹、生成概率和奖励 → 训练 worker 的优化器步骤 |
| [7] 参数发布 | [模型版本推送](../../../psrl/workers/ps/ps_manager.py#L1194)、[模型拉取](../../../psrl/workers/ps/ps_manager.py#L1269) | 参数服务登记新版本，推理端接收参数并继续采样；实际参数位于 PS worker |
| 退出清理 | [runner 的 finally](../../../examples/sciaccel_rl/runner.py#L610)、[快检收尾](../../../examples/gpu_benchmarks/run.py#L183) | 清理所属容器，再检查容器、GPU 进程和显存状态，保存证据 |

初读只需沿表格理解数据流；要追查实现时，再按编号进入源码。资源管理的完整原理见[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)。

## 环境与依赖怎样隔离

不同题可能要求不同版本的 PyTorch、模型服务或 GPU 扩展。接入时需要让这些依赖各有安装位置，同时把公开数据、隐藏测试和提交物分开管理。**环境隔离回答“文件、包与进程会影响谁”，资源调度回答“CPU、内存和 GPU 当前给谁用”。**

### 镜像、容器与挂载各负责什么

| 对象 | 用通俗语言理解 | 本项目中的作用 |
|---|---|---|
| 镜像 image | 预先准备的软件模板，本身不执行任务 | 保存固定源码、Python 和依赖；同一模板可以反复启动 |
| 容器 container | 从模板启动的一次运行实例 | agent 和独立评分器有各自的进程与文件修改 |
| 容器可写层 | 当前实例新增或修改的文件 | 临时修改源码、安装包；删除容器时随实例移除 |
| 绑定挂载 bind mount | 将宿主目录接到容器中的指定路径 | 复用模型、数据或 Python；`:ro` 表示只读，可写挂载则直接修改宿主文件 |
| 虚拟环境 venv | 一套单独的 Python 包目录与入口 | 把某类任务新增的包放在专用目录；可与容器一起使用 |
| 提交物 artifacts | 明确交给评分器、并保留供复查的文件 | 代码补丁、预测 CSV 或模型权重 |

同一镜像启动 A、B 两个容器时，各自拥有可写层。A 修改镜像中的普通文件，B 仍读取原版本。这个机制称为**写时复制**：共享模板保持只读，修改保存在当前实例的上层。停止容器后可写层仍在，删除容器才移除它。

挂载目录按其读写配置处理。例如 `/model:ro` 代表从宿主只读提供模型文件；可写的输出挂载会把结果保存到宿主。要判断安装影响范围，应看实际使用哪个 Python、安装目标在哪、该路径是否挂载，以及是否只读。[公共 compose 准备器](../../../examples/gpu_benchmarks/prepare.py#L23)在需要复用解释器时，将解释器前缀和其实际基础前缀去重后都以 `:ro` 挂载。

GPU 显存由运行进程占用。删除容器文件、租约计数归零与显存释放是不同观察，实验收尾还要检查实际进程和设备。

### 例子一：PostTrain 怎样训练只读模型

以已验证的 **GSM8K + Qwen3-1.7B-Base** 为例。GSM8K 是数学应用题数据，当前本地任务做两次小范围模型更新，并交付更新后的参数。

1. [准备器](../../../examples/posttrain_bench/prepare.py)向 agent 提供只读 `/model`、公开训练数据和 Python。独立评分器另获得隐藏数据与测试程序。
2. [模型加载函数](../../../examples/posttrain_bench/local_model.py#L9)读取磁盘权重，把模型加载到进程内存和 GPU 显存。只读约束作用在磁盘文件；加载后的内存张量可以计算梯度和更新。
3. 程序固定模型主体，只更新最后一个归一化参数向量 `model.norm.weight`。本地协议选择这一个向量来控制训练成本。
4. 两次更新后，将该向量写为新的 `final_model/adapter.safetensors` 提交文件。原始 `/model` 文件继续保持原样。
5. 评分器启动自己的模型进程，重新读取基础模型，检查提交权重的名字、形状和数值，再复制到自己的模型内存，在隐藏答案上评分。

因此，**磁盘基础权重、GPU 中的模型、提交的参数文件是三个对象**。评分器再次加载模型，是为了独立检查提交物；agent 的模型进程和显存状态不会随提交文件传过去。该任务目录保存在[固定资产](../../../../benchmark-runs/complete-onboarding-20261112/posttrain-tasks/posttrainbench-gsm8k-qwen3-1.7b)。

写作和健康问答采用同一提交路径，但多一个只读 `/judge` 挂载，仅提供给独立评分器。以 **HealthBench + Qwen3-1.7B** 为例，agent 在两条公开记录上更新任务模型的归一化参数并交权重；verifier 加载提交物，在另外两条隐藏记录上生成回答。随后它释放任务模型，加载冻结的 Qwen3.5-4B 裁判，对每条回答的四项原始标准评分。裁判来自已有模型缓存，冻结权重不会进入 agent 的提交文件，也不随外层 PSRL 策略更新。[准备器](../../../examples/posttrain_bench/prepare.py)明确区分 agent 与 verifier 的挂载，[裁判评分](../../../examples/posttrain_bench/judged_model.py)实现生成、模型释放与独立评判。

### 例子二：SWE 怎样交付代码修改

SWE 要求修改真实 SGLang 推理服务。不同题固定的源码和 GPU 扩展版本可能不同，所以各题使用选定运行镜像。下面解释当前本地快速协议的补丁交付路径，历史原题协议仍按自己的记录阅读。

1. [构建器](../../../examples/swe_serve/build_local_images.py#L47)取得固定源码，将其装入镜像中的 `/code`，配齐该题依赖。
2. agent 在自己的容器中编辑 `/code`，改动保存到当前容器可写层。
3. [补丁配置](../../../examples/swe_serve/patch_submission.py#L5)将改动导出为 `submission.patch`。Harbor 收集并交付该文件。
4. 评分器从干净环境启动，应用补丁，读取只给评分器的测试，再独立运行。agent 临时安装的包、进程和整个可写层仍留在原实例；需要的评分依赖由评分环境自己准备。
5. 本次容器删除后，宿主保留的补丁与日志仍可复查，任务基础镜像可继续复用。

部分性能题还给评分器只读提供未修改源码 `/base` 和私有计时程序，用于独立比较。[SWE 准备器](../../../examples/swe_serve/prepare.py#L62)核对这些挂载。本地 score 闭环题公开参考补丁并选择两个原测试，验收目标是这条交互、执行和评分路径。

### 例子三：MLGym 虚拟环境怎样复用依赖

MLGym 强化学习任务使用 JAX 数组计算、Flax 网络和 Gymnax 游戏环境。新增依赖放入 `benchmark-assets/mlgym-rl-env/`，这是 Python 的 venv，包含自己的包安装目录和解释器入口。

实际 [pyvenv.cfg](../../../../benchmark-assets/mlgym-rl-env/pyvenv.cfg)启用 `include-system-site-packages = true`，表示它可以同时读取基础 Python 中已有的包。新包写到该 venv，基础解释器与部分包继续复用。容器因此需要同时只读挂载 venv 和它依赖的基础 Python 前缀。[RL 准备器](../../../examples/mlgym_bench/prepare.py#L18)还生成 `/opt/task/python`，显式设置 GPU 动态库搜索路径。

例如 Breakout 本地题使用该入口运行 16 个并行游戏环境和两次参数更新，提交 `policy.msgpack`。独立评分器用相同软件、可信网络结构和新的随机种子重新评分。软件文件可以只读复用，任务进程和模型状态各自独立。

### 安装审计与长期复现

| 对象 | 实际依赖位置 | 复现时要固定什么 |
|---|---|---|
| SWE、部分 RE/Inference | 专用任务镜像与各自容器 | 源码版本、实际 image ID、评分资产 |
| MLGym RL | 专用 JAX venv，部分包来自基础 Python | venv 与基础包两个来源 |
| 多种公共任务、MLGym 视觉/NLP、PostTrain、当前 FastKernels 子集 | 容器内只读复用已有主 Python | 已有包版本及任务脚本，后续宜固化镜像 |
| AIRS | 预装 agent 镜像或显式只读复用解释器 | 按 GPU 配置选择的实际分支，见[沙箱构建](../../../psrl/environments/mlgym_env.py#L57) |
| PSRL 验证进程的额外依赖 | `.verification-deps/` 与实验专用依赖目录 | 进程 `PYTHONPATH`，即 Python 模块搜索顺序 |

2026-11-12 审计检查主环境 **2540 个包元数据文件**，未观察到 11-09 以来的安装变更，检查的 **90 份 Compose 配置**中，匹配的主 Python/JAX 前缀挂载均为只读。[状态 JSON](../experiments/gpu-benchmarks/onboarding-status-audit-20261112.json)保存结果，[审计脚本](../experiments/gpu-benchmarks/scripts/audit_onboarding_status.py#L115)说明检查方法。审计依据是安装路径、包元数据和挂载配置，接入前的完整文件快照未保存。

只读挂载保护任务到宿主的写入方向。宿主之后升级包，任务仍可能读取到新版本，所以长期复现需把验证过的依赖固化到镜像。各次输出也要使用独立目录。CPU、GPU、磁盘、驱动与 Docker 服务仍由宿主共享，按实际预算调度。

下载反复发生，主要是为了准备真实数据、模型和各版本 GPU 扩展。wheel 是 Python 安装包文件，下载 wheel 后还要决定装入哪个镜像或 venv。**下载、安装、导入和 GPU 运行分别验收**，可以据此判断准备卡在哪一步。

本轮启动前现场检查共享存储约剩余 **140 TiB**，节点本地存储约剩余 **8 TiB**。扩题复用了现有任务模型、冻结裁判与源码兼容镜像，没有下载新模型。SDAR 的一个约 1.82 GB 依赖包在重新检查磁盘后开始下载，达到 180 秒预算时保留约 1.6 GB 的未完成文件，尚未安装。PostTrain 与 FastKernels 继续只读挂载 Python 前缀，SWE 的依赖准备在各自容器内执行。新增 57 项协议的 114 份配置通过[只读挂载与私有数据检查](../../../../benchmark-runs/complete-bench-onboarding-20261116/task-mount-audit.json)。现场磁盘和下载结果分别见[预检](../../../../benchmark-runs/complete-bench-onboarding-20261116/preflight.json)、[下载记录](../../../../benchmark-runs/complete-bench-onboarding-20261116/sdar-wheel-download.json)，后续下载前须重新检查容量。

原生 CUDA 算子还需要编译器：只有 CUDA 运行库的镜像能执行已有 GPU 程序，却没有 `nvcc` 将 `.cu` 源码编译成新扩展。本轮把节点已安装、约 7.3 GB 的 CUDA 12.8 工具链复制到专用镜像的 `/opt/psrl-cuda`，并从只读 Python 前缀调用 Ninja 构建工具。宿主机已有安装保持原样，复制过程没有联网下载。准备过程及编译探活见[镜像构建证据](../../../../benchmark-runs/complete-bench-onboarding-20261116/fast-toolchain-build.json)。

例如 FastKernels 的 SiLU-and-Mul 把 `silu_and_mul.py` 与同目录 `silu_and_mul.cu` 一起交给 agent。agent 可修改两者，提交目录进入独立 verifier，再通过上游编译器构建并比较输出。MoeSum 与 SiLU-and-Mul 使用 `psrl-fast-native-clean:20261116`；RMSNorm 使用 `psrl-fast-native-pinned:20261116`，另将 H20 的编译目标固定为 `sm_90`，让预编译缓存与运行环境一致。镜像和 image ID 见[干净镜像](../../../../benchmark-runs/complete-bench-onboarding-20261116/fast-native-rebuild.json)、[编译目标固定](../../../../benchmark-runs/complete-bench-onboarding-20261116/fast-native-pinned.json)。这些镜像保留工具链与已编译缓存，缓存属于初始模板；agent 与 verifier 各自修改自己的容器可写层，彼此没有共享可写编译缓存。运行时的设备映射由本次 GPU 租约提供，镜像不固定 GPU 编号。首次派生镜像因继承空的 `CUDA_VISIBLE_DEVICES` 而快检失败，失败证据保留，最终镜像从无设备绑定的工具链镜像重建。

### 不需要 GPU 的教学实验

在仓库根目录可运行：

```bash
python docs/env-worker-deep-dive/experiments/gpu-benchmarks/scripts/bench_environment_layers_demo.py
```

脚本用临时目录模拟镜像、两份可写层、只读主 Python、隐藏数据和提交物。先修改 agent 的代码，观察评分器仍读取原版本；再交付一个文件快照，独立读取和评分；最后删除两份可写层，观察模板和提交物仍保留。已有运行的 11 条观察全部通过，三份版本输出为 `image-version`、`agent-version`、`image-version`，见[固定记录](../experiments/gpu-benchmarks/bench-environment-layers-demo.json)。

这是 Python 标准库教学模拟，只读与分层由脚本实现。真实容器权限、Docker 和 GPU 的验收依据仍来自各实验。想理解“客户端结束后服务为何还占资源”，另有[常驻服务生命周期示例](../env-service-verification/scripts/explain_resident_command_lifetime.py)。本文整理期间未重跑这些脚本。

## 本地任务协议

下面是 **2026-11-10 六类代表题验收**实际采用的协议。后续扩展到更多任务的输入和规模见[当前路线图](02-Science现状_任务覆盖与接入路线.md#题库数量位置与逐题覆盖)。

| 家族 | agent 提交与独立评分 | 结果身份与解释 |
|---|---|---|
| KernelBench | 原始矩阵乘法题，提交计算代码，独立 GPU 比较正确性并记录速度 | `kernel_bench`；训练奖励主要为正确性 |
| autoresearch | 真实缓存与固定训练预算，提交训练程序，独立重放并计算验证 BPB | `autoresearch`；BPB 为每字节预测编码代价，奖励为 `1/(1+BPB)` |
| MLE 本地 digits | CUDA 训练后提交 `id,label` CSV，按隐藏标签计算准确率 | `mle_bench_local`；本地公开数据任务，官方 Kaggle 单列 |
| Inference 本地服务 | 提交前台服务脚本；评分器重启服务，发送两个真实 MMLU-Pro 请求 | `inference_bench_local`；输出先与冻结参考一致，再按批请求耗时计分 |
| RE 有界聊天训练 | 真实 GPT-2 XL、Alpaca 与偏好训练，使用原评价函数、本地 Qwen 裁判、两条隐藏提示 | `re_bench_local` 的 `bounded-local`；完整官方协议为 512 条提示 |
| FlashInfer RMSNorm | 提交规定格式 Solution JSON，独立运行官方评估器 | `flashinfer_bench`；固定子集全部正确才给正确性奖励，时延另报 |

本地协议使用显式身份标记，让准备器、评分器和奖励函数能对齐同一规则。RE 的 `feedback-smoke` 是接线探活模式，训练奖励保持零；正式本地偏好评价使用上表的 `bounded-local`。

## 运行入口

以下命令用于理解复现步骤；运行前先完成[集群最小预检](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)。节点与设备用现场确认的值，准备资产和结果放新目录。

### 先准备可运行的任务

真实 Terminus agent 使用终端工具，需要镜像中有 `tmux`。固定参考程序 oracle 的启动方式更简单，所以还需单独检查真实 agent 所需工具。公共 science runner 关闭终端录制，交互训练记录由 TITO 收集。

对于 apt 系统的基础镜像，可使用已有 agent 镜像层：

```bash
docker build --network "${BUILD_NETWORK:-default}" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f examples/gpu_benchmarks/agent.Dockerfile -t "$AGENT_IMAGE" .
docker run --rm --network none --entrypoint bash "$AGENT_IMAGE" \
  -c 'tmux -V && asciinema --version'
```

`BASE_IMAGE` 是已准备任务依赖的模板，`AGENT_IMAGE` 是新的 agent 镜像标签，`BUILD_NETWORK` 按节点构建网络设置。第二条命令在禁网条件下检查工具。独立评分器可使用自己的镜像；RE 评分镜像还必须包含隐藏资产与当前评分脚本。

| 准备器 | 关键输入 |
|---|---|
| `python -m examples.mle_bench.local` | `--output`、`--image`、`--python`；使用 sklearn 自带真实公开数据 |
| `python -m examples.inference_bench.local` | 上述三项及 `--references`，其中保存预先独立生成的请求参考输出 |
| `python -m examples.re_bench.prepare` | 固定源码、资产、镜像，另加 `--local-judge-model`、`--bounded-local`；配好独立裁判与桥接器 |
| `python -m examples.flashinfer_bench.prepare` | `--repo`、`--subset`、`--cuda`、`--output`、`--image`、`--python` |

`--output` 使用新任务目录，`--python` 是容器中实际可执行的解释器路径。详细模型和数据准备分别见[公共 recipe](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)与 [RE recipe](../knowledge/03-任务接入/05-REBench_聊天模型训练接入.md)。先独立冻结服务参考输出，再评候选，可避免候选给自己定义正确答案。

### 再生成配套数据与训练配置

混合准备器读取已准备任务。将清单保存为 `task-manifest.json`，每项声明：

| 字段 | 含义 |
|---|---|
| `task_path` | 节点可读的任务目录，含指令、配置与来源记录 |
| `environment_backend` | `docker` 或 `env_worker`；后者使用预构建、无 Compose 依赖的任务镜像 |
| `data_source` | 选择哪份奖励协议，本地任务保留本地身份 |
| `gpu_memory_mb` | 任务显存准入需求，单位 MiB；按模型与候选实际用量设置 |

从仓库根目录执行：

```bash
python -m examples.gpu_benchmarks.mix \
  --manifest task-manifest.json --output prepared-gpu-suite
```

产物中的 `tasks.parquet` 是训练任务表，`agent.yaml` 决定每行任务由谁执行，二者配套使用。设置 `TRAIN_FILES`、`VAL_FILES`、`HF_MODEL_PATH`、`OUTPUT_DIR` 和实际资源布局后，公共训练入口为：

```bash
bash examples/gpu_benchmarks/train.sh kernel_bench \
  gen_actor_rollout_ref.rollout.agent.agent_loop_config_path="${PWD}/prepared-gpu-suite/agent.yaml"
```

这里 `kernel_bench` 选择公共启动模板，实际任务由数据行中的 `agent_name` 和 `data_source` 决定。资源节点与独立 GPU unit 仍按[公共配置](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)显式设置。正式效果实验使用独立训练/验证划分；下面的历史功能验收关闭独立验证，检查重点是任务与训练链路。

## Results 与验证边界

### 真实 GPU 与策略闭环结果

固定参考程序先得到下列独立任务结果：

| 代表题 | GPU 与评分结果 | 收尾观察 |
|---|---|---|
| MLE digits | 437/450，提交格式有效 | 所属容器与 GPU 进程归零 |
| Inference 服务 | 两条输出与当前运行时冻结参考一致，奖励约 0.0612 | 所属容器与 GPU 进程归零 |
| RE 有界聊天训练 | 本地偏好分 0.25，裁判与评价有效 | 任务资源归零，裁判在策略验收后关闭 |
| FlashInfer RMSNorm | 三项评估结果全部通过，正确性奖励 1 | 所属容器与 GPU 进程归零 |

随后六类代表题进入同一次真实策略训练。首批 12 条轨迹产生 **11525 个有效训练 token**，执行一次优化器步骤，策略 loss 约为 0.003347，梯度范数约为 0.798689。这两个数字用于检查该次训练计算，学习效果还要用任务表现评估。版本 1 发布并由推理端拉取后，六类任务均至少有一条实际 v1 交互通过独立评分。

| 任务 | v0 有效正分 / 已结束 | v1 有效正分 / 已结束 |
|---|---:|---:|
| KernelBench | 2/2 | 2/2 |
| autoresearch | 2/2 | 1/1 |
| MLE digits | 2/2 | 1/1 |
| Inference 服务 | 1/2 | 1/1 |
| FlashInfer RMSNorm | 2/2 | 1/1 |
| RE 有界聊天训练 | 1/2 | 1/2 |

有效正分要求评分器实际返回对应协议的有效标记和正分。该次证据支持**六类任务能经过一次策略训练并在发布后继续评分**。任务提示提供公开起始程序，验收聚焦功能；策略解题能力、泛化与性能收益各有不同实验目标。

### 怎样解释失败和受控停止

v0 的 12 条采样已结束，用于首轮训练；v1 后续又有 8 条采样结束，合计 **20 条，其中 17 条有效正分、3 条失败**。三个失败是：两个 RE 样本没有提交模型，一个 Inference 样本试运行常驻服务却没有结束，最终超时。v1 的 RE 缺失提交在当时被记录为 `verifier_error`，历史记录保留原样。

配置原本最多运行两个训练步。一次更新和六类 v1 评分齐备后，实验达到功能验收的停止条件，主动终止剩余 4 条已派发采样。这些取消样本另行统计；子进程 `-15` 对应主动 SIGTERM 收尾。实际完成的优化器步骤为 **1 次**。

清理覆盖本次主容器、独立评分器、网络 sidecar 与裁判。固定现场检查显示所属容器为 0，八张实验卡无 GPU 计算进程、显存查询均为 0 MiB。这描述当次收尾状态，下一次实验仍现场预检。

完整指标、模型版本、失败、退出原因与源码 SHA256 保存在 [full-onboarding-20261110.json](../experiments/gpu-benchmarks/full-onboarding-20261110.json)。实验只清本次资源。AIRS、MLGym、PostTrain 等后续接入的结果分别见[当前覆盖](02-Science现状_任务覆盖与接入路线.md#results验证覆盖与证据索引)。

<a id="本轮边界修复与回归"></a>

## 验收相关修复与回归

2026-11-10 GPU 验收之后，基于 `ab74120` 加主目录未提交改动，补充了提交错误、来源身份与清理路径的修复。该阶段证据是 **30 项 CPU 回归全部通过**，包含一个真实 CPU 服务进程的停止与回收。GPU 原始结果仍对应此前任务资产与镜像。

| 触发问题 | 修复后的行为 | 实现 |
|---|---|---|
| RE 缺少模型或 `model.py` | 返回明确的提交无效零分，记录原因并跳过模型评价；裁判配置错误仍报告环境异常 | [RE verify](../../../examples/re_bench/verify.py#L12) |
| MLE CSV 多列、超额行、损坏引号或非法编码 | 拒绝无效提交并限制读取行数；隐藏标签损坏报告准备错误 | [MLE grade](../../../examples/mle_bench/local_grade.py#L9) |
| 本地任务被标为官方数据源 | 根据显式来源记录与裁判身份检查一致性，在生成数据前拒绝冲突 | [mix.prepare](../../../examples/gpu_benchmarks/mix.py#L12) |
| Inference 评分 shell 退出但子进程仍在 | 回收整个所属进程组；缺启动脚本时产生明确零分 | [stop_service](../../../examples/inference_bench/local_verify.py#L11) |

例如 RE 的 agent 花完预算却没有交出模型，新脚本会写出“提交物缺失、奖励零分、未调用裁判”。这让训练获得明确失败反馈。裁判自身配置损坏则属于需要修复的运行环境问题。

MLE、Inference 准备器会复制评分脚本，所以应用修复后需重新准备任务；RE 把评分脚本打入镜像，需重建评分镜像再准备。agent 自己试运行常驻服务时，仍要按协议结束所属服务；独立评分器的进程组清理只负责它自身启动的服务。

针对性回归代码见 [test_task_boundaries.py](../../../tests/examples/gpu_benchmarks/test_task_boundaries.py)，复现命令见[路线图验收节](02-Science现状_任务覆盖与接入路线.md#可用性验收与复现)。本次文档整理未重跑这些回归。

## 复现时最容易遗漏的条件

| 条件 | 历史实验中遇到什么 | 下一次怎样检查 |
|---|---|---|
| 容器集合预算 | RE 的主存需求比 unit 多 256 MiB，首次准入被拒 | 主容器、评分器与 sidecar 一并计入预算 |
| agent 工具 | oracle 可运行，真实 agent 镜像却缺 `tmux`，产生零轨迹 | 禁网启动真实 agent 所需终端工具做预检 |
| 完整上下文长度 | 6144 token 窗口遇到 6227–7604 token 请求被拒 | 算上完整提示、工具说明和多轮输出；该次最终窗口与单样本预算为 16384，回复预算 4096，`MAX_PROMPT_LENGTH=12288` |
| 训练与推理预算 | `MAX_TOKENS_PER_GPU` 只控制训练打包 | 推理窗口单独核对 `max_model_len`，该次日志两侧均为 16384 |
| 依赖实际来源 | 私有 Ray runtime-env 与 gRPC 版本不匹配导致准备失败 | 在隔离目录补匹配依赖，核对 driver 和 worker 实际导入路径 |
| 评分与裁判身份 | 固定服务输出随运行时变化，RE 需要独立裁判 | 提前冻结参考，记录模型与镜像；桥接器监听实际 jobs 目录 |

历史静态检查也各有范围：首轮 CPU 回归 53 项通过，后加 3 项专项中有 2 项与首轮重复；修改代码范围的 Ruff、shell 与启动参数捕获通过。全仓 pre-commit 仍有既有文档脚本长行与源码注释问题，完整状态保留在原记录。后续验证按本次修改选必要检查。

## 下一步研究优先级

接入完成后，可以先固定任务、候选程序、策略版本和评分协议，再比较资源策略。建议从**候选反馈的可靠性与总等待时间**开始，然后研究准备复用、训练阶段共享，最后考虑等待裁判时的模型状态释放。

例如服务测试完成后，agent 正在改代码，模型可能仍占显存。继续保留可以省下次加载，释放则能腾给其他任务。研究应先测真实等待与重载成本，再选择策略。推进顺序、比较基线和停止条件统一维护在[研究路线图](02-Science现状_任务覆盖与接入路线.md#后续研究方向与否证条件)。

如果想继续理解，可以分别追问：“独立评分器为什么要重载模型”“TITO 怎样把多轮工具调用变成训练样本”“客户端退出后服务为何仍占显存”。相关入口分别是本文模型交付例子、[会话与训练机制](../../psrl/01-PSRL原理_AgentLoop与工具调用.md)和[GPU 执行与生命周期](../knowledge/01-架构与执行/02-统一执行_EnvRouter与容器及预算策略.md)。
