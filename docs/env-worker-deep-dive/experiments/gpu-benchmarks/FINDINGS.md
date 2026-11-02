# 四类 GPU benchmark 接入验证

> **核心结论**：KernelBench 和 autoresearch 已通过真实 H20、PSRL Harbor runner、EnvRouter 和独立 verifier 的执行与清理。autoresearch 使用官方数据和原始训练脚本，完成真实训练及评估。MLE-bench 被 Kaggle 凭据阻塞，InferenceBench 仍缺真实质量及速度基线、裁判凭据和匹配的 GPU。没有执行语言模型 agent rollout 或 PSRL 策略优化器更新。

## 目录

- [接入范围](#接入范围)
- [KernelBench Results](#kernelbench-results)
- [官方资产准备](#官方资产准备)
- [autoresearch 真实任务](#autoresearch-真实任务)
- [原生 Env 的 episode 长任务验证](#原生-env-的-episode-长任务验证)
- [修复与验证边界](#修复与验证边界)
- [复现入口](#复现入口)

## 接入范围

| 项目 | 实现 | 本次证据 |
|---|---|---|
| KernelBench | 上游题目、Triton 提交、独立 GPU evaluator、PSRL 数据和奖励 | 真实 GPU 执行通过 |
| MLE-bench | 官方 public/private 数据映射、独立 registry/grader、above_median 奖励 | 官方 SDK 认证失败，缺凭据，未取得正式竞赛数据 |
| InferenceBench | 导入官方生成的 Harbor 任务，Docker 本地 handoff，保留质量和 judge 门槛 | 数据集及请求样本已准备，官方任务生成器因缺基线拒绝生成 |
| autoresearch | 固定 prepare.py、缓存挂载、独立训练重放、捕获官方评估调用 | 官方资产准备完成，原始训练脚本在 GPU 上真实通过 |

四项均要求 EnvRouter，不能关闭 Env 后回退裸 Docker。资源策略暂限 episode。公共代码复用 science 的会话与 runner，并新增强制准入路径，避免已有全局 science client 缓存影响新 recipe。

## KernelBench Results

本次日期为 2026-10-29，使用首选实验节点的一张空闲独立 NVIDIA H20。固定动作 agent 仅提交仓库内的 Triton 实现，未调用语言模型。官方题目为 `level1/1_Square_matrix_multiplication_.py`，保持 4096 × 4096 输入，未缩小为教学数据。上游 commit 与参考文件哈希在 [summary.json](summary.json)。

| 检查 | 观测 |
|---|---|
| GPU 后端 | `ManagedDockerEnvironment`，真实 EnvRouter 租约 |
| candidate | recipe 自带 Triton 矩阵乘法，IEEE 精度输入 |
| 编译与正确性 | compiled=true，3/3 检查通过 |
| 性能测量 | 10 次，候选约 7.80 ms，参考约 4.55 ms |
| speedup | 约 0.583，候选更慢，不声称优化收益 |
| reward | 1.0，本 recipe 的正确性奖励 |
| 最终活动租约、声明 GPU 预算 | 0、0 |
| 最终本次所属容器 | 0 |
| 最终 GPU 计算进程 | 0 |
| 清理后显存读数 | 约 3.5 MiB，不写成绝对零 |

完整上游 evaluator 结果在 [kernel-result.json](kernel-result.json)，去除机器标识后的运行摘要在 [summary.json](summary.json)。节点资源快照中的容器计数是缓存遥测，可能保留刚结束时的 2 个容器读数。物理清理结论来自后续 Docker 归属标签查询，而非缓存计数或租约归零。

原始 reward 文件错误地将 runtime 字段命名为 `runtime_us`。核对上游 `timing.py` 中 CUDA event 的 `elapsed_time` 后确认单位为毫秒，最终适配器改为 `runtime_ms`。摘要保留原始值并注明修正，未为了字段改名重跑 GPU 实验。

## 官方资产准备

资产准备和真实验证均使用官方 checkout，版本见 [upstream.json](../../../../examples/gpu_benchmarks/upstream.json)。资产保存在工作区的独立目录，不纳入仓库，也未修改共享 conda 环境。

| 项目 | 官方流程与本次结果 |
|---|---|
| autoresearch | 使用 `uv sync --frozen` 安装 Torch 2.9.1 + CUDA 12.8 等锁定依赖。调用上游下载与 tokenizer 函数，取得训练分片 0、固定验证分片 6542，训练 8192 词表 tokenizer。匹配的 Hopper FlashAttention kernel 已缓存 |
| MLE-bench | 安装其要求的 Kaggle 1.6.17，在认证阶段报缺少 `kaggle.json`。没有取得竞赛数据，也没有禁用 checksum 或使用第三方替代数据 |
| InferenceBench | 官方下载脚本正常完成，完整 Mistral 仓库 15 个文件约 29.0 GB，含两种权重布局。MMLU-Pro 和 LongBench-v2 分别含 12032 个 test 样本和 503 个 train 样本，官方六个种子的样本缓存已生成 |

autoresearch 的数据、tokenizer 哈希与 kernel revision 见 [autoresearch-assets.json](autoresearch-assets.json)。上游默认下载 10 个训练分片，本次选用其支持的单分片配置验证接入，不把结果作为默认数据规模的正式成绩。

InferenceBench 模型清单与完整性结果见 [inference-assets.json](inference-assets.json)。Hugging Face 格式的三个权重分片包含 291 个张量，与索引一致。全部文件大小与官方元数据一致，LFS 文件逐项匹配官方 SHA-256。首次单连接下载较慢，在保留专属缓存的前提下另用 Hub SDK 并行获取权重分片，最终仍由官方脚本完成完整资源清单，未替换模型或数据。

InferenceBench 官方 `run_adapter.py --scenario a --base-model mistral-7b --seed-pairs 21:1337` 已实际执行。它接受已存在的样本文件，但因缺少 Torch 质量基线和 vLLM 场景 A 速度基线而退出。基线必须真实计算，不能填入常数。当前首选节点为 H20，而官方任务声明 `H100!`，且缺少 Anthropic judge 凭据。因此没有启动一个缩减或绕过 judge 的任务来冒充正式验证。

下载与复现命令见 [下载官方资产](../../knowledge/gpu-benchmark-recipes.md#下载官方资产)。

## autoresearch 真实任务

原始结果的公开摘要见 [autoresearch-result.json](autoresearch-result.json)。本次仅执行一个 oracle episode，未重跑已通过的 KernelBench。

| 检查 | 实测结果 |
|---|---|
| 提交 | 与上游 `train.py` 字节完全一致，depth=8，约 50.3M 参数 |
| 数据 | 官方训练分片 0，固定验证分片 6542，真实 tokenizer 与预编译 kernel |
| 训练与评估常量 | `TIME_BUDGET=300`、`MAX_SEQ_LEN=2048`、`EVAL_TOKENS=20971520`，保持原值 |
| 实际训练 | 306 个 optimizer step，训练计时 300.1 秒，约 160.4M token |
| 官方评估 | `val_bpb=1.157410833327473`，由真实 `evaluate_bpb` 调用捕获 |
| 传回 PSRL runner 的奖励 | `1 / (1 + val_bpb) = 0.4635185772464368` |
| GPU | 一张空闲独立 H20，任务声明 65536 MiB |
| Torch 峰值分配 | 45060.2 MiB，区别于包含缓存和驱动的 NVML 读数 |
| 脚本总时长 | 393.6 秒，包含初始化、编译与评估 |
| episode 总时长 | 约 479.8 秒，包含基础设施启动及清理 |
| 清理 | 本次容器 0，GPU 计算进程 0，活动租约 0，声明预算 0，GPU 残余约 0.19 MiB |

oracle 只把原始脚本交给独立 verifier。实际训练与评估发生在 verifier GPU 容器中，整个任务经 EnvRouter 准入并在结束后回收。这里的 optimizer step 属于 benchmark 内部训练的模型，不是 PSRL 策略模型训练。没有验证真实语言模型 agent 的生成轨迹，也不声称模型优化收益。

工作区在运行期间另有 Env 后端重构。本次原始记录保留了运行中的源码状态与 diff，它不是启动前冻结的提交快照，因此该结果用于集成验收，不用作不同 Env 版本的性能对照。

本轮 13 项 recipe 回归全部通过，覆盖新增虚拟环境基础解释器挂载和不完整缓存拒绝。Ruff、格式及 prose 检查通过。全仓 pre-commit 在隔离副本执行仍有其他文件的既有问题，自动修订未回写主目录。训练通过后将重放日志改为直接流式写文件，以便超时也保留输出，该日志改动未触发第二次 GPU 训练。

## 原生 Env 的 episode 长任务验证

2026-10-30 的验收目标是：使用现有官方数据，让原始 autoresearch 训练和评分通过 `EnvWorkerEnvironment`，检查长任务期间 GPU 预算持续保留，以及结束后的物理回收。本轮不推进 Kaggle，也不进行 EnvRouter OFF/ON 性能比较。

入口为 [run.py](../../../../examples/gpu_benchmarks/run.py)，显式指定 `--environment-backend env_worker`。数据仍是官方分片 0、固定验证分片 6542 及原 tokenizer，训练和评估常量保持原值。任务声明 65536 MiB，verifier 上限 900 秒，外层任务上限 1200 秒。

为适配原生后端不解释 Compose 的约束，将已准备的依赖与数据打入预构建运行时镜像，独立 verifier 镜像额外包含 `/tests`。没有让原生后端绕过 EnvRouter，也没有改成宿主直接训练。

执行中发现两项问题，并分别保留失败记录：

1. 首次镜像产物漏掉虚拟环境 Python 和 kernel 缓存的符号链接，验证脚本未能启动，未执行 GPU 训练。改为显式 tar 构建上下文后，镜像内解释器、Torch 导入及离线 kernel 缓存检查通过。
2. 随后的真实训练在底层命令执行约 300 秒时提前中断，整个启动器历时约 332 秒。Harbor 报出的 900 秒是它的外层配置，不是实际耗尽的时限。其 verifier 调用 `environment.exec()` 时未传单条命令超时，原生适配器把 `None` 交给 EnvWorker，触发了后者的默认 300 秒限制。这是长任务实际暴露的接口语义问题。

修复位于 [EnvWorkerEnvironment._execute_command](../../../../psrl/workers/env_worker/harbor.py#L260)：未指定命令超时时不附加 EnvWorker 默认期限，由 Harbor 外层截止时间控制。显式传入的命令超时仍原样执行，外层取消仍触发所属容器清理。没有提高 task.toml 的 900 秒限制。持续 shell 接口的默认期限不受此修复影响。

对应 CPU 回归见 [test_harbor_deadline_does_not_inherit_worker_default](../../../../tests/env_worker/test_harbor_backend.py)。测试把默认期限缩为 0.001 秒，让本地进程运行 0.05 秒，检查省略命令期限时能成功，并检查显式 0.005 秒期限及外层取消仍能终止、清理。它是故障机制缩时复现，不冒充 GPU 测试。

```bash
python -m pytest tests/env_worker/test_harbor_backend.py \
  -k harbor_deadline_does_not_inherit_worker_default -q
```

两次未完成尝试均已核实：本次容器为零，GPU 计算进程为零，活动租约和 GPU 声明预算归零。针对准备器与原生后端的首轮 41 项 CPU 回归通过，超时修复后原生后端 29 项回归通过，其中含新增的两个期限用例。全仓 pre-commit 仍有其他文件问题，隔离副本的自动修订没有回写主目录。

修复后的真实回归通过，完整摘要见 [native-episode-result.json](native-episode-result.json)，去除机器身份后的采样见 [native-episode-samples.json](native-episode-samples.json)。实际 Harbor 配置中的后端类为 `psrl.workers.env_worker.harbor:EnvWorkerEnvironment`。

| 指标 | 实测 |
|---|---|
| 训练脚本 | 与官方 `train.py` 字节一致 |
| 训练 | 306 步，训练计时 300.1 秒，约 160.4M token |
| 官方评估 | `val_bpb=1.1585622078601807` |
| runner 奖励 | `0.4632713369846853` |
| 脚本总时长 | 382.5 秒，包含初始化、编译和评估 |
| 启动器计时 | 约 424.0 秒，包含 Ray、任务及清理 |
| 采样 | 80 个采样记录，其中 75 个观测到 GPU 计算进程 |
| GPU 持卡证据 | 75 个有效 GPU 活跃采样全部对应同一个会话、1 条租约、65536 MiB 声明预算 |
| 可见保留区间 | 首末持租约采样相隔约 388.4 秒，首末 GPU 活跃采样相隔约 378.2 秒 |
| 峰值显存 | NVML 约 49369.1 MiB，Torch 峰值分配 45060.2 MiB |
| 最终清理 | 本次容器 0，GPU 计算进程 0，租约 0，声明预算 0，显存残余约 0.19 MiB |

启动约 17 秒时发生一次账本遥测请求超时，该条按缺失记录，没有补零，也没有当作训练失败。采样频率约每 5 秒一次，上述区间不是精确的 CUDA 活动起止计时，不能证明采样间隙中的每个瞬间。

这证明当前原生 Env 可以在 episode 策略下执行真实长时间训练、评分并回收资源。它不表示 GPU 每秒都满负载，不证明整卡独占、常驻推理服务跨多轮保留、模型训推共存或 EnvRouter 的性能优势。本次没有启动 PSRL 策略模型训练。

## 修复与验证边界

真实执行前后的故障均保留在本地原始结果目录，最终成功以前未完成的尝试不计入任务通过：

1. NVML 旧内存查询把驱动保留量计入占用，启动器误拒绝空闲卡。改用项目已有的 NVML v2 查询方式。
2. 资源池容量与任务申请量都设为 8192 MiB，约 0.19 MiB 驱动常驻量导致准入等待。诊断快照显示零租约、一个等待者。终止等待并核实清理，改用现场设备总容量定义池，任务仍申请 8192 MiB。
3. 预建镜像的独立 verifier 不自动包含 `/tests/test.sh`。显式只读挂载 tests，仅提供给 verifier。
4. 只读 tests 中的脚本缺执行位。准备器设置可执行权限，随后真实 evaluator 通过。

GPU 成功后只做必要的源代码整理与 CPU 检查，包括 runtime 字段单位、离线价格表缓存开关、Python 前缀挂载改为显式选项。它们没有被写成又一次 GPU 验证。任务代码和官方参考数值保持原样。

PSRL 接入回归使用受控 session 数据检查 token 保留、奖励传递、GPU 需求、session 删除和 Env 缺失时拒绝执行。它是接口回归，不是实际 TITO 模型请求。KernelBench 的真实 GPU 证据使用同一 production runner 和 Env 路径，仅执行 kernel 编译与评估。autoresearch 额外执行了任务内模型的反向传播与优化器步骤，二者均未启动 PSRL 策略训练器、模型生成或权重同步。

定向回归最终覆盖 31 个不同测试，其中 20 个为既有 Env runner 回归，11 个为新 recipe、数据隔离、奖励和 session 契约检查。一次全组执行中 29 项通过，另两项因测试文件漏导入 Path 失败，补齐导入后连同相关准备器共 5 项定向复测通过。新增代码 Ruff、格式和 prose 检查通过，Shell 语法及本次文档本地路径检查通过。全仓 pre-commit 在临时副本执行，仍被其他文件的既有行长、格式和注释规范问题阻塞，副本修订未回写主目录。

对于 autoresearch，只有真实调用上游 evaluator 才能生成测量记录，纯打印伪造分数的 CPU 回归会失败。这不构成对任意恶意 Python 的完整安全证明。对于 InferenceBench，缺失 judge 或完整性门槛失败时 PSRL 奖励为零。

## 复现入口

操作命令和资产要求见 [GPU benchmark recipe](../../knowledge/gpu-benchmark-recipes.md)，真实环境验证入口为 [run.py](../../../../examples/gpu_benchmarks/run.py)，固定提交为 [matmul.py](../../../../examples/kernel_bench/solutions/matmul.py)。

运行前现场检查资源和依赖，使用全新输出目录。任务预留、容器 CPU/主存和独立 verifier 都参与 Env 预算。GPU 声明量应覆盖同一 episode 的整个 GPU 容器集合。实验完成后不继续增加任务或启动训练来追求更高分。
