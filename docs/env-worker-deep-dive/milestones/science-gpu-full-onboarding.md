# Science GPU 任务完整接入与本地验收

> **核心结论**：本轮按“独立 verifier → 真实策略模型 rollout → PSRL 后训练更新 → 更新后的继续采样”验收。允许使用本地硬件、小规模真实数据和本地裁判，不要求复现官方榜单协议。任务内部训练模型、oracle 提交与 PSRL 策略更新分别登记，不能互相替代。
>
> 状态：2026-11-10 六类 GPU 代表任务均通过本地完整闭环验收。框架代码基准为 `bc5665f`，收尾时 HEAD 为 `ab74120`，两者已提交差异仅为会话归档文档。验收包含本轮未提交实现及既有工作区内容，结论限定为本文列出的协议、准备资产和代表题，不代表整套官方题库。改动位于本地 `main` 工作区，尚未提交或推送，未宣称远端合并或共享部署。任务总览与后续工作见[当前路线图](science-gpu-bench-status-and-roadmap.md)，持续架构入口为[当前设计](../knowledge/current-design.md)。

## 目录

- [验收目标与执行链](#验收目标与执行链)
- [本地任务协议](#本地任务协议)
- [运行入口](#运行入口)
- [Results 与验证边界](#results-与验证边界)
- [本轮边界修复与回归](#本轮边界修复与回归)
- [下一步研究优先级](#下一步研究优先级)

## 验收目标与执行链

这里有两种不同的“训练”。任务内训练是 agent 在容器中训练数字分类器、语言模型等提交物。PSRL 后训练则是框架用 agent 的工具调用轨迹和奖励，更新负责发出这些命令的策略语言模型。rollout 指用某个版本的策略模型实际采样一次任务交互。TITO（Token-In-Token-Out，输入输出 token 记录）是框架收集模型输入、输出 token 以及对应生成概率的会话机制，用于把多轮交互交给训练器。独立 verifier 只读取提交物并评分，它通过不代表策略模型已经更新。

```text
[1] 真实资产准备 → Harbor task + PSRL Parquet
         ↓
[2] oracle 先检查独立 verifier 与 GPU 工作
         ↓
[3] 策略模型生成工具命令 → 任务执行 → verifier 评分
         ↓
[4] TITO 收集模型 token 与工具交互 → PSRL 收到轨迹和奖励
         ↓
[5] PSRL 训练计算 loss 并调用优化器 → 发布新模型版本
         ↓
[6] 下一批 rollout 使用更新后的模型 → 最终资源清理
```

图例：图里每个部分对应的源码位置。

| 节点 | 入口和状态变化 |
|---|---|
| 1 | [write_dataset](../../../examples/gpu_benchmarks/prepare.py#L47) 生成模型输入和任务资源声明，本轮为不同本地协议保留独立 data_source |
| 2 | [验证运行器](../../../examples/gpu_benchmarks/run.py) 经实际 science runner 与 EnvRouter 执行 oracle，检查评分和清理 |
| 3、4 | [GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L14) 复用 science 的会话和轨迹采集，模型实际生成动作 |
| 5 | [公共训练启动器](../../../examples/gpu_benchmarks/train.sh) 调用 [PSRL 训练入口](../../../psrl/trainer/main_ppo.py)，[完整批次策略](../../../psrl/trainer/ppo/strategies/full_batch.py#L110) 进入 [_update_actor](../../../psrl/trainer/ppo/ray_trainer.py#L3222) |
| 6 | [push_model_state_dict_nixl](../../../psrl/workers/ps/ps_manager.py#L1194) 登记更新后的模型版本，参数实际位于 PS worker，[pull_model_state_dict_nixl](../../../psrl/workers/ps/ps_manager.py#L1269) 协调推理端取回参数 |
| 5 的 GPU 身份核对 | [PSRL_DummyWorker](../../../psrl/trainer/ppo/utils.py#L171) 为融合角色提供实际节点与设备信息，资源检查不因占位角色缺接口而中断 |

跟着图读源码：

| 步骤 | 图中编号 | 真实入口 | 输入、输出或状态变化 |
|---|---|---|---|
| 准备 | 1 | [write_dataset](../../../examples/gpu_benchmarks/prepare.py#L47)，混合任务使用 [prepare](../../../examples/gpu_benchmarks/mix.py#L12) | 任务目录和资源声明变成 Parquet 数据行与 agent 配置 |
| 环境验收 | 2 | [run.py](../../../examples/gpu_benchmarks/run.py) | 固定参考提交进入实际 runner，得到独立评分和清理记录 |
| 策略采样 | 3 | [SciAccelAgentLoop.run](../../../examples/sciaccel_rl/agent_loop.py#L144) 与 [GpuBenchmarkAgentLoop._run_harbor_in_thread](../../../examples/gpu_benchmarks/agent_loop.py#L32) | 建立模型会话并申请任务资源，真实模型决定工具调用 |
| 容器与评分 | 3、4 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275) | 模型和容器反复交互，独立 verifier 在任务结束后评分，runner 在退出路径归还资源 |
| 策略更新 | 5 | [FullBatchStepStrategy 中的更新阶段](../../../psrl/trainer/ppo/strategies/full_batch.py#L110) → [_update_actor](../../../psrl/trainer/ppo/ray_trainer.py#L3222) | 批次轨迹、生成概率和奖励进入训练 worker，实际计算 loss 并调用优化器 |
| 版本回传与后续采样 | 6 | [PS Manager 模型推送](../../../psrl/workers/ps/ps_manager.py#L1194) 与 [模型拉取](../../../psrl/workers/ps/ps_manager.py#L1269) | 推理端取回新版本，后续任务必须留下使用该版本的实际采样证据 |

模型生成、环境工具和资源监控存在异步执行，图中箭头表示数据依赖。5 的策略模型优化与任务内的小模型训练是不同事件，必须看实际训练 step。6 还需要模型版本和新采样记录，不能只凭进程存活作结论。这些代码如有理解上的问题，可以随时追问。

## 本地任务协议

| 家族 | 本地协议和独立验收 | 数据与结果身份 |
|---|---|---|
| KernelBench | 保留原始矩阵乘法题，独立 GPU 验收正确性和速度 | `kernel_bench`，奖励仍为正确性，速度另报 |
| autoresearch | 现有真实缓存与原始限时训练，独立重放并计算验证 BPB | `autoresearch`，不把任务内训练算作 PSRL 更新 |
| MLE-bench 本地任务 | 使用 sklearn 打包的真实手写数字数据，固定训练/隐藏测试划分，agent 在 CUDA 训练并提交 id,label CSV。verifier 按隐藏标签算准确率 | `mle_bench_local`，是 MLE 风格本地竞赛，不是 Kaggle 竞赛成绩，也不是原 MLE-bench 全题库 |
| InferenceBench 本地任务 | agent 提交前台服务脚本，独立 verifier 重启服务并发送两个真实 MMLU-Pro 请求。输出需与固定基线完全相同，再按批请求耗时计分 | `inference_bench_local`，仅固定模型、请求和输出协议，无云裁判，无官方 H100 要求 |
| RE-Bench 本地任务 | 保留真实 GPT-2 XL、Alpaca、偏好训练与官方评价函数，固定本地 Qwen 裁判，只评估两条隐藏提示和有界生成 | `re_bench_local` 的显式 `bounded-local` 协议，非完整 512 条评测。`feedback-smoke` 继续保持零奖励 |
| FlashInfer-Bench | agent 提交官方格式 Solution JSON，独立 verifier 对固定 RMSNorm 子集运行官方评估器 | `flashinfer_bench`，本地奖励要求全部 workload 正确，时延另报，不宣称完整题库成绩 |

具体例子：MLE 本地题把 1797 张真实 8×8 手写数字图像按固定种子划分，训练标签公开，450 个测试标签只挂给 verifier。agent 的 CUDA 分类器生成每个测试 ID 对应的数字类别。verifier 检查 ID 集合完整且不重复，再逐条比较隐藏标签。本轮参考实现得到 437/450，准确率约 0.9711。提交缺行、重复行或非法标签均不得获得有效提交标志。该结果只说明此本地任务的模型训练与评分可运行，不是策略模型学习提升。

已有 CPU 生命周期实验可从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_resident_command_lifetime.py
```

它模拟客户端结束但服务仍持有状态的情况，对应图中 3 的后台服务与 6 的清理边界。预期客户端退出后模拟不可归还，服务停止后才可归还。本轮已实际运行，观察到客户端结束时仍拒绝归还，停止服务后才允许归还。这是 CPU 教学模拟，不代替本轮真实 GPU 验收。

## 运行入口

准备命令的 `--output` 必须指向新的任务目录。`--python` 是任务镜像中可执行的解释器路径，使用只读挂载独立 Python 环境时仍须满足镜像系统库要求。

真实 Terminus agent 需要 `tmux`。公共 science runner [关闭终端录制](../../../examples/sciaccel_rl/runner.py#L499)，因此本轮没有依赖 `asciinema` 录制轨迹，训练轨迹由 TITO 收集。公共镜像层额外包含可选的录制工具。**oracle 不使用这套 agent 初始化，因此 oracle 通过不证明镜像包含 tmux。** 任务容器禁网时必须提前安装，不能依赖运行时联网安装。对 apt 系统的任务基础镜像，可使用[公共 agent 镜像层](../../../examples/gpu_benchmarks/agent.Dockerfile)：

| 环境变量 | 含义 |
|---|---|
| `BASE_IMAGE` | 已准备任务资产与依赖的基础镜像 |
| `AGENT_IMAGE` | 将要构建的新 agent 镜像标签，不能覆盖共享基础镜像 |
| `BUILD_NETWORK` | 可选的 Docker 构建网络，默认 `default`，由当前节点网络条件决定 |

```bash
docker build --network "${BUILD_NETWORK:-default}" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f examples/gpu_benchmarks/agent.Dockerfile -t "$AGENT_IMAGE" .
docker run --rm --network none --entrypoint bash "$AGENT_IMAGE" \
    -c 'tmux -V && asciinema --version'
```

准备任务时将 agent 环境指向新镜像。独立 verifier 可以继续使用包含当前评分代码与私有资产的原镜像，RE-Bench 不得把没有隐藏资产的 runtime 镜像误当 verifier。

| 准备器 | 必要参数 |
|---|---|
| `python -m examples.mle_bench.local` | `--output`、`--image`、`--python`，使用 sklearn 自带公开真实数据，无 Kaggle 凭据 |
| `python -m examples.inference_bench.local` | 上述三项加 `--references`，目录含独立生成的 request-0.json、request-1.json。先冻结基线，不能从待评分候选生成其自己的参考答案 |
| `python -m examples.re_bench.prepare` | 原有资产与镜像参数，加 `--local-judge-model`、`--bounded-local`。verifier 镜像必须包含本轮 verify.py，本地裁判独立部署并启动 judge 桥接 |
| `python -m examples.flashinfer_bench.prepare` | `--repo`、`--subset`、`--cuda`、`--output`、`--image`、`--python`，使用固定官方源码和工作负载 |

训练沿用公共 `examples/gpu_benchmarks/train.sh`。MLE 与 InferenceBench 的本地数据分别使用其原 recipe 的 agent 配置，data_source 保留 local 身份。FlashInfer 新增自己的 agent 配置。混合原生 Env 与 Compose 任务时，在同一个 agent 配置列表中分别声明后端，并在数据表 agent_name 中对应选择。不得把原生任务交给不解析其镜像约束的配置。

[混合任务准备器](../../../examples/gpu_benchmarks/mix.py) 接受 JSON 数组，每项包含以下字段。它读取已经准备好的任务，不更改任务资产，不接受 smoke 冒充完整协议。

| 字段 | 含义 |
|---|---|
| `task_path` | 已准备且执行节点可读的任务目录，包含 instruction.md、task.toml、provenance.json |
| `environment_backend` | `docker` 或 `env_worker`，后者要求无 Compose 依赖的预构建镜像 |
| `data_source` | 上表对应的奖励身份，本地协议保留 `_local` |
| `gpu_memory_mb` | 此任务的显存准入需求，单位 MiB，需根据实际候选与硬件设置 |

先将清单保存到 `task-manifest.json`，在仓库根目录执行：

```bash
python -m examples.gpu_benchmarks.mix \
    --manifest task-manifest.json --output prepared-gpu-suite
```

产物 `prepared-gpu-suite/tasks.parquet` 和 `prepared-gpu-suite/agent.yaml` 必须成对使用。实际训练时设置 `TRAIN_FILES`、`VAL_FILES`、`HF_MODEL_PATH`、`OUTPUT_DIR`，并通过命令行覆盖 agent 配置路径：

```bash
bash examples/gpu_benchmarks/train.sh kernel_bench \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path="${PWD}/prepared-gpu-suite/agent.yaml"
```

这里 `kernel_bench` 选择公共启动模板，具体任务仍由数据行的 `agent_name` 和 `data_source` 决定。节点、独立 GPU unit 和训推布局按[公共配置说明](../knowledge/gpu-benchmark-recipes.md)显式设置，不能把示例命令当作任意节点都能自动选卡。正式实验使用不重叠的训练与验证任务。本轮闭环验收关闭独立验证，只验证任务与框架能衔接，不提供泛化结论。

## Results 与验证边界

当前已确认的环境层结果如下。它们是独立 oracle 验收，不是 PSRL 策略更新的替代证据。

| 本轮任务 | 真实 GPU 结果 | 物理清理 |
|---|---|---|
| MLE 本地题 | 隐藏集 437/450，`valid_submission=1` | 所属容器与 GPU 进程归零 |
| Inference 本地服务 | 两条请求与当前运行时冻结基线一致，`eval_valid=1`、`quality_pass=1`，奖励约 0.0612 | 所属容器与 GPU 进程归零 |
| RE-Bench 有界本地题 | 本地偏好分 0.25，`local_evaluation=1`、`judge_valid=1`、`eval_valid=1`，不是 smoke | 任务所属容器与 GPU 进程归零，独立裁判在策略验收后关闭 |
| FlashInfer RMSNorm | 三项官方 evaluator 结果全部 PASSED，正确性奖励 1 | 所属容器与 GPU 进程归零 |

Inference 的固定输出依赖运行时版本，必须先独立运行基线，再冻结引用。本轮共享依赖与前次记录不同，先以当前运行时重新生成参考输出，再执行上表独立验收，未使用待评分候选生成其自己的参考答案。该短输出协议衡量服务一致性，不证明题目回答正确率。

**六类任务均已完成真实 PSRL 闭环。** 首批 12 条轨迹覆盖六类任务，进入同一次策略更新，产生 11525 个有效训练 token，`masked_sample_fraction=0`，实际策略 loss 约 0.003347、梯度范数约 0.798689。版本 1 已发布且推理端完成参数拉取，路由日志有 `version_tag=1` 的实际模型请求，轨迹目录也分别记录 v0、v1。随后六类任务均至少有一条版本 1 轨迹通过独立 GPU 评分。这些证据证明训练与参数回传执行，不证明策略效果改善。

| 任务 | v0 有效正分 / 已结束 | v1 有效正分 / 已结束 | 本地完整闭环 |
|---|---|---|---|
| KernelBench | 2/2 | 2/2 | 通过 |
| autoresearch | 2/2 | 1/1 | 通过 |
| MLE 本地 digits | 2/2 | 1/1 | 通过 |
| Inference 本地服务 | 1/2 | 1/1 | 通过 |
| FlashInfer RMSNorm | 2/2 | 1/1 | 通过 |
| RE 有界本地任务 | 1/2 | 1/2 | 通过 |

“有效正分”要求实际独立 verifier 返回相应协议标记和正分，不能用模型自述成功替代。共保存 20 条已结束采样，17 条有效正分，另有 3 条失败。v0 的 RE 失败样本未运行起始训练，达到预算后缺失提交物。v0 的 Inference 失败样本额外启动常驻服务，却没有结束，GPU 作业超时。v1 的另一个 RE 样本也缺少提交物，框架标记为 `verifier_error`，不冒充有效零分。达到轮数或 token 预算的轨迹可能已有可评分提交，表格保留这类有效评分，原始 termination 也记录在证据中。它不表示每条 agent 轨迹都自然完成。

原配置上限为两个策略训练步。在一次更新和新版本六类任务覆盖齐备后，按功能验收停止条件结束，未等待第二次优化器更新。剩余 4 条已派发采样被取消，不计成功。训练子进程退出码为 `-15`，对应主动发送 SIGTERM 的收尾，不是“两个训练步正常结束”。本轮不声称收敛、长期稳定性或成功率提升。

清理后检查了任务主容器、独立 verifier 与网络 sidecar 的归属，所属容器为 0，现场八张卡的 GPU 计算进程为空，显存查询均为 0 MiB，独立裁判已关闭。只清理本次进程、私有 Ray 与所属容器，没有停止共享 Ray、重启 Docker 或清理共享镜像。完整的逐任务评分、版本、终止原因、训练指标及源码 SHA256 见[机器可读证据](../experiments/gpu-benchmarks/full-onboarding-20261110.json)。

首个任务派发尝试因 RE 容器集合主存预算比 unit 多 256 MiB 被拒绝。修正配置并逐题核对预算后，真实 agent 初始化又暴露基础镜像缺少 tmux。该异常不影响 oracle verifier，但导致零轨迹，已补充公共 agent 镜像层。后续隔离验收任务明确要求采用公开起始实现，跳过探索性优化，检查真实工具命令、GPU 验收与策略闭环，不以此评价 agent 优化能力。任务内部评分数据和 GPU 运算保持真实。

上下文预算须包含 Terminus 的完整提示、GPU 工具说明、模型回复和工具输出，不能只统计数据表里的任务说明。6144 token 的尝试实际出现 6227 至 7604 token 的请求，被服务拒绝，部分任务只有未完成轨迹和零分，不能据此宣布接入成功。最终有界验收把完整上下文和训练单样本 token 预算设为 16384，回复预算保持 4096，取消额外第三轮数据。启动脚本新增 `MAX_PROMPT_LENGTH` 覆盖入口，本轮设为 12288。`MAX_TOKENS_PER_GPU` 只控制训练打包预算，单独增大它不会扩大推理窗口。已通过无 GPU 的命令捕获检查实际启动参数，并在本轮日志确认两侧 `max_model_len=16384`。验收提示给出公开基线的具体命令，真实 LLM 仍负责生成工具调用与完成提交，verifier 不替 agent 生成提交物。

本轮先后排除私有 Ray runtime-env 连接问题，以及共享 SMG Python 生成代码要求 grpcio 1.85 而共享环境为 1.83 的版本不匹配。匹配依赖只安装在实验独立目录，没有修改共享环境。失败尝试均保留原始日志，不计为 benchmark 通过。训练资源和裁判资源独立分配，实验不进行 Env ON/OFF 性能比较。

停止条件为所选代表任务的 verifier、真实轨迹、策略更新及后续采样证据齐备，或达到预设时间/重试预算。本轮已达到功能停止条件。CPU 针对性回归首轮 53 项通过，增加混合任务校验后的专项 3 项通过，其中 2 项与前轮重复。本轮代码范围 Ruff 检查与格式检查通过，shell 语法及实际启动参数捕获通过。全仓 pre-commit 在临时完整快照执行，仍报既有文档脚本长行和既有源码注释风格问题，未改无关内容，不能写成全仓检查通过。本轮维护的 12 篇索引与正文共检查 477 个本地路径、124 处源码行号和 139 处章节锚点，均有效。

范围外项目须单列：AIRS/MLGym 保留独立 recipe，本轮现场未找到其完整准备资产，不能自动算入上述六类公共任务的 GPU 闭环。SciAccel 作为原有 science 会话来源，已有历史训练记录，本轮未把它的 CPU 科学修复题重复计入新增 GPU 覆盖。上述代表题通过也不等于整套官方题库通过。 2026-11-11 后续扩题没有重跑本节策略更新：快速任务检查通过 Kernel 原尺寸 22 题、小输入档 193 题、FlashInfer 30 个定义与四个本地 MLE 题。原尺寸策略闭环仍保留本节历史条件，新增验证不等于新增训练闭环。数量、题库位置与复现见[路线图统计](science-gpu-bench-status-and-roadmap.md#题库数量位置与逐题覆盖)。

## 本轮边界修复与回归

2026-11-10 在上述 GPU 验收结束后，针对失败归类、提交格式、协议身份与服务清理补充修复。核对范围是 `ab74120` 加主目录未提交改动。**本节是新增 CPU 回归，不是再次完成 GPU 训练**；上节实测、镜像指纹和失败记录保持原始条件。

| 问题与触发条件 | 修复后的行为 | 对应源码 |
|---|---|---|
| RE 没有提交训练模型或 `model.py` | 写出无效提交零分及原因，`judge_valid=0`，不启动模型评价。非法本地裁判配置仍报异常，不伪装成任务零分 | [verify.main](../../../examples/re_bench/verify.py#L12) |
| MLE 提交 CSV 多列、多行、损坏引号或非法 UTF-8 | 评分器拒绝无效提交，限制最多读取隐藏集大小加一行。空隐藏标签属于准备错误并抛异常 | [grade](../../../examples/mle_bench/local_grade.py#L9) |
| manifest 将本地准备任务标成官方数据源 | 根据显式 provenance 标记与 RE 裁判来源检查一致性，在写出数据前拒绝冲突。没有来源标记的历史资产仍需人工核验 | [mix.prepare](../../../examples/gpu_benchmarks/mix.py#L12) |
| Inference 独立评分器的前台 shell 退出，子进程可能仍在运行 | 在回收主进程之外清理整个所属进程组；缺少启动脚本也明确返回零分 | [stop_service](../../../examples/inference_bench/local_verify.py#L11) |

例如，RE 的 agent 花完预算却没有训练模型，旧镜像可能只留下“找不到奖励文件”的 verifier 异常。新脚本直接登记“提交物缺失、奖励零分、未调用裁判”。这既给策略训练保留明确失败反馈，也不会把没有进行的裁判评价写成成功。若裁判模式配置错误，仍应先修复实验环境，不能把它解释为 agent 能力差。

新增 [12 项边界回归](../../../tests/examples/gpu_benchmarks/test_task_boundaries.py)，与已有 MLE、RE 专项一起实际运行，**30 项全部通过**。检查不依赖 GPU，包含一个真实 CPU 服务进程的停止与回收；完整命令见[路线图复现节](science-gpu-bench-status-and-roadmap.md#可用性验收与复现)。它们覆盖上述错误路径，不能代替完整 CUDA 模型评价和修复后训练。

MLE 和 Inference 的准备器会复制评分脚本，需要重新准备任务才能带入修复。RE 将评分器打入独立 verifier 镜像，需重建该镜像后重新准备。原有任务目录和旧镜像不会自动更新，本次未执行这些产物的 GPU 再验收。agent 自己启动常驻服务后没有取消受管作业的失败，也不由 verifier 清理函数自动修复：它发生在独立评分器启动之前，仍须遵守服务生命周期协议。

## 下一步研究优先级

以下是设计提案，不是本轮已取得的性能收益。下一阶段先使用固定任务、固定候选和固定策略模型版本研究环境反馈，避免同时改变模型、任务规模和调度器而无法解释结果。具体研究背景与否证条件见[当前研究规划](science-gpu-bench-status-and-roadmap.md#后续研究方向与否证条件)。

| 优先级 | 问题与适用任务 | 最小有意义的对照 | 何时不继续 |
|---|---|---|---|
| 1 | 多候选评估能否更快且保持可信反馈，使用 KernelBench 与 FlashInfer | 固定一组真实候选，比较串行、直接并行和受控分批，记录完整候选组耗时、正确性和相对独占测量的排序变化 | 原本没有可观等待，或并行收益不足以抵消评分干扰 |
| 2 | 长训练、短算子与服务任务混合时，怎样缩短整组反馈时间和减少无效等待 | 使用 autoresearch、MLE 与 Inference 的同一任务集合，比固定 episode 准入与有界任务分类策略，保留初始化、执行、排队和清理的独立计时 | 仅靠镜像预热和正确资源配置已消除主要成本，或收益来自不等工作量 |
| 3 | 等待裁判时，让出训练状态是否值得，使用 RE 的真实偏好等待 | 先测实际等待分布与卸载恢复成本，再讨论常驻、总卸载和阈值策略，优先使用其他独立 GPU | 等待窗口普遍短于恢复成本，或让位增加关键路径而没有足够任务收益 |

举例说明第一项的验收方式：教学假设有 4 个候选，独占时每个评估 10 秒，串行共 40 秒。如果两两并行，每组因争用变成 16 秒，总共 32 秒，说明整组完成更快。但若候选 A 与 B 的快慢顺序被争用反转，反馈可能误导下一轮 agent。这时要同时比较“整组完成时间”和“排序是否可靠”，不能只报告 40 秒变成 32 秒。这些数字不是本轮实测。当前本地 KernelBench 与 FlashInfer 的训练奖励主要检查正确性，若要研究速度奖励，还需显式定义可靠的性能反馈，不能把现有 reward=1 当作加速证明。
