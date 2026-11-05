# GPU benchmark 接入验证

> **核心结论**：KernelBench、autoresearch 已通过真实 GPU 执行与回收。InferenceBench 固定常驻服务已完成两类 H20 ON/OFF 对照：早期 24 GiB 人工配额只验证排队且更慢；episode 记账修复后采用整卡容量与 7 项真实任务，OFF 一次 OOM，ON 无 OOM，完成相同工作量的任务阶段耗时下降 17.9%。其官方完整评分仍未通过，MLE-bench 暂缓 Kaggle 接入。没有执行语言模型 agent rollout 或 PSRL 策略优化器更新。

> RE-Bench 聊天训练已有原生 Env/PSRL 接入，真实 GPT-2 XL、Alpaca 提示、生成和一次 CUDA 更新已通过。原 Replicate 分支仍缺凭据，本地 Qwen 已完成小样本真实偏好训练与隐藏评分。两种 smoke 的 `reward=0`，不冒充完整成绩。

## 目录

- [KernelBench command 接入验证](#kernelbench-command-接入验证)
- [接入范围](#接入范围)
- [KernelBench Results](#kernelbench-results)
- [官方资产准备](#官方资产准备)
- [autoresearch 真实任务](#autoresearch-真实任务)
- [原生 Env 的 episode 长任务验证](#原生-env-的-episode-长任务验证)
- [InferenceBench 常驻服务 ON/OFF Results](#inferencebench-常驻服务-onoff-results)
- [为什么本次 ON 更慢且 GPU 进程更少](#为什么本次-on-更慢且-gpu-进程更少)
- [episode 记账修复后的整卡压力对照](#episode-记账修复后的整卡压力对照)
- [RE-Bench 原生接入验证](#re-bench-原生接入验证)
- [RE-Bench 本地 Qwen 裁判验证](#re-bench-本地-qwen-裁判验证)
- [评分教学脚本验证](#评分教学脚本验证)
- [修复与验证边界](#修复与验证边界)
- [复现入口](#复现入口)

## 接入范围

| 项目 | 实现 | 本次证据 |
|---|---|---|
| KernelBench | 上游题目、Triton 提交、独立 GPU evaluator、PSRL 数据和奖励 | 真实 GPU 执行通过 |
| MLE-bench | 官方 public/private 数据映射、独立 registry/grader、above_median 奖励 | 官方 SDK 认证失败，缺凭据，未取得正式竞赛数据 |
| InferenceBench | 官方完整 recipe 与固定常驻服务实验入口分开 | 完整任务生成缺官方基线，固定 Mistral 服务与真实样本 ON/OFF 已通过 |
| autoresearch | 固定 prepare.py、缓存挂载、独立训练重放、捕获官方评估调用 | 官方资产准备完成，原始训练脚本在 GPU 上真实通过 |
| RE-Bench | 真实资产下载、原生任务、独立 verifier、宿主裁判传输、PSRL 奖励 | 真实资产 CUDA smoke、本地 Qwen 偏好 smoke 通过，完整 512 条评分未验证 |

这些 recipe 均要求 EnvRouter，不能关闭 Env 后回退裸 Docker。默认资源策略为 episode，KernelBench 新增显式 direct command verifier，见本篇对应 Results。公共代码复用 science 的会话与 runner，并新增强制准入路径，避免已有全局 science client 缓存影响新 recipe。RE-Bench 默认使用原生 EnvWorker。

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

修复位于 [EnvWorkerEnvironment._execute_command](../../../../psrl/workers/env_worker/harbor.py#L266)：未指定命令超时时不附加 EnvWorker 默认期限，由 Harbor 外层截止时间控制。显式传入的命令超时仍原样执行，外层取消仍触发所属容器清理。没有提高 task.toml 的 900 秒限制。持续 shell 接口的默认期限不受此修复影响。

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

## InferenceBench 常驻服务 ON/OFF Results

本次日期为 2026-11-02，验收目标是检查真实模型在多条命令之间驻留，以及 episode 预算是否使超额请求等待、在物理清理后接续运行。先验证 H20、Docker daemon、设备 UUID 与 minor 映射、同一容器内 CUDA 运算和依赖来源。只使用一张现场空闲 H20，没有启动训练或占用其他 GPU。

**负载与边界。** 使用官方 `transformers_openai_server.py`，文件保持原样。模型为官方 `mistralai/Mistral-7B-Instruct-v0.3` 快照 `c170c708c41dac9275d15a8fff4eca08d52bab71`，读取此前已按官方 LFS 哈希核验的三个 HF 权重分片。请求取官方 MMLU-Pro seed 21 缓存中的前两条，sample_id 为 `11920` 与 `9555`。每组同时提交两个 episode，每个 episode 启动独立服务，依次执行请求 1、CPU 等待 8 秒、请求 2，再停止服务。

两组都固定 BF16、上下文上限 2048、温度 0、输出上限 16 token。总计每组 4 次请求、64 个输出 token，全部文本逐样本一致。第二题输出被 16 token 上限截断为解题开头，不能据此宣称答题准确率或质量通过。reward=1 是固定功能 verifier 的完成标志，不是 InferenceBench 官方奖励。

镜像使用本地已有 Torch 2.9.1+cu128 基础运行时，增加官方 Torch server 配方中对应的 Transformers 5.2.0、Accelerate 1.12.0、aiohttp 3.13.3 等依赖。没有构建完整官方 Torch 2.10.0 容器，因此结果只代表这个固定本地运行时。依赖探针先暴露缺少 `async_timeout`，补齐后导入和容器 CUDA 预检通过，GPU 对照每组仅运行一次。

**两组配置。** OFF 使用指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9` 的已有隔离 checkout，沿用静态 GPU Compose 映射，保留进程内 episode 信号量，并发上限设置为 2。沿用的必要补丁包括正确传递 `override_gpus` 和禁止节点级 dangling-image 清理，另有不改变本实验调度的 Ray 地址与 HTTP 事件循环补丁。完整补丁与已记录的 [baseline.patch](../../env-service-verification/results/router-baseline-control-20261020/baseline.patch) 相同，本次哈希另见结果 JSON。

ON 从主目录提交 `8d1fc65230d7bd4c78305e140c41ac8b5ea14fb0` 加本次新增实验脚本启动，核心 runner 无未提交补丁。实际 Harbor 配置为 `EnvWorkerEnvironment`、`mode=episode`，经过生产 `run_harbor_episode` 和 EnvRouter。每个服务申报 20480 MiB，池分配 24576 MiB。主机 CPU、主存和容器槽均能同时容纳两个任务，只有 GPU 预算使第二个等待。H20 实际总容量为 97871 MiB，池预算不是物理容量或硬限制。OFF 没有这个容量账本，两个服务在物理卡上仍能运行，没有人为制造 OOM。

| 观测项 | OFF 固定 baseline | ON 原生 Env episode |
|---|---:|---:|
| 完成 episode / 真实请求 | 2 / 4 | 2 / 4 |
| 每组输出 token | 64 | 64 |
| GPU 计算进程峰值 | 2 | 1 |
| NVML 采样显存峰值 | 28866.625 MiB，约 28.19 GiB | 14435.0625 MiB，约 14.10 GiB |
| 请求间 CPU 等待期间的整卡显存 | 两服务并存，28866.625 MiB | 单服务，14435.0625 MiB |
| 任务阶段总时长 | 46.295 秒 | 76.356 秒 |
| 驱动初始化 | 0.351 秒 | 17.570 秒，含私有 Ray |
| 最终清理阶段 | 0.412 秒 | 2.743 秒 |
| 驱动总计时 | 47.059 秒 | 96.669 秒 |
| 等待证据 | 无容量账本 | 26 条采样出现 1 个等待者，同时只有 1 条活动租约 |
| 最终本次容器 / GPU 进程 | 0 / 0 | 0 / 0 |
| 结束 GPU 残余 | 0.1875 MiB | 3.5 MiB，后续设备查询取整为 0 MiB |
| 最终活动租约 / 声明 GPU 预算 | 不适用 | 0 / 0 |

任务阶段从两条任务提交前计时，到两次 runner 返回后结束，包含各自环境准备、agent、verifier 和 episode 清理。初始化与最后清理单列，镜像构建和 Python 导入不在这张计时表中。服务从启动到 readiness，OFF 为约 12.6 到 13.5 秒，ON 为约 11.3 到 11.4 秒。这不是独立测量的 reload 成本。

**可以得出的结论。** 常驻服务在请求间持续占用显存，CPU 命令结束不能释放它。ON 将两个服务串行准入，把采样显存峰值控制在池预算以内，等待者在前一个释放后完成真实请求。OFF 同时运行两个服务，其总占用超过给 ON 设置的池预算，但未超过物理显存。ON 的任务阶段更慢，说明预算约束会以等待时间换取受控并发。这不是无条件的吞吐提升，也没有验证 OOM 减少、训推同卡共享或跨 driver 的全局协调。

OFF 是旧 Docker 后端，ON 是当前原生 Env 后端，两组版本、初始化及清理实现不同。因此这是一组完整接入路径的固定 baseline 对照，不能把耗时差全部称为“打开一个账本开关的纯开销”。本次不额外启动相同版本的 OFF 消融，也不扩大工作量来追求速度优势。

**遥测边界。** OFF 有 41 条采样，ON 有 62 条。ON 一次 Router 采样超时按缺失保留，还有 19 条容器 CPU/主存资源读取错误或缓存错误字段，其中包含销毁后的 Docker memory sample missing。GPU 记录与请求结果单独存在，没有补零或把监控缺失当成任务失败。最终账本的缓存容器计数仍为 1，缓存年龄约 2.7 秒，不能用它证明资源未回收或已回收。物理清理结论来自按本次 actor 标签的 Docker 查询为空、GPU 计算进程为空及显存回落，随后独立查询再次确认。

证据与复现：

- [resident-comparison.json](resident-comparison.json)：计时、响应、空闲期间显存、源码版本与清理汇总。
- [resident-samples.json](resident-samples.json)：去除节点地址、UUID 与 PID 的相对时间序列，保留遥测缺失。
- [resident-provenance.json](resident-provenance.json)：实际镜像、上游与脚本哈希。
- [resident_ab.py](../../../../examples/inference_bench/resident_ab.py)、[resident_agent.py](../../../../examples/inference_bench/resident_agent.py)、[resident_client.py](../../../../examples/inference_bench/resident_client.py)：有界对照、跨命令常驻服务与真实 HTTP 请求。
- [准备与运行步骤](../../knowledge/gpu-benchmark-recipes.md#command-与常驻推理服务)：镜像、固定 baseline 与源代码阅读路径。

可运行的 CPU 复核如下，`RESULT_DIR` 指向含 `off/`、`on/` 的原始结果目录。它只读取已完成实验，不重新启动 GPU：

```bash
python examples/inference_bench/resident_analyze.py \
  --root "$RESULT_DIR" --output "$AUDIT_OUTPUT" --expect-serialized
```

该脚本已实际通过：检查每组完成 2 个 episode、4 次请求，逐样本输出一致，ON 出现等待且结束租约为零，两组物理清理完成。关于 command 与 offload 的结论来自现有归还协议和服务生命周期源码，本次没有执行服务 offload/reload，也没有测量其代价。

新增脚本的 Ruff、格式、语法编译与 prose 检查通过，本次四篇文档共 139 个本地链接路径和源码行号范围检查通过。全仓 pre-commit 已在隔离副本执行，仍因其他文件的既有行长、格式和注释规则失败，未把自动修改回写主目录。本次没有修改 Env 核心代码，新增脚本、证据和文档保留在主目录未提交，也未推送。baseline 复用既有固定 checkout，未修改或删除其中内容。

## 为什么本次 ON 更慢且 GPU 进程更少

**原因首先是实验配置。** 为触发准入等待，实验主动把 ON 的池预算设为 24 GiB，每个任务申报 20 GiB。OFF 则沿用原有最多两个 episode 的并发保护，没有对应显存池限制。这组实验验证了预算约束下的排队与回收，没有控制一致的有效可用 GPU 预算，不能回答“资源充足时打开 Env 是否提高性能”。把它作为一般性能 ON/OFF 对比是不充分的，应明确承认这项设计局限。

```text
[1] 同时到达两个任务，每个申报 20 GiB，容器槽上限均为 2
       |
       +-- ON: 池预算 24 GiB -> [2] 第一项准入，剩余约 4 GiB
       |                              |
       |                         第二项需要 20 GiB -> 等待
       |                              |
       |                         [3] 第一项清理释放 -> 第二项准入
       |
       +-- OFF: 两个并发槽都有空位 -> 两个容器都映射同一物理 GPU
                                      |
                                 两个模型进程同时驻留
```

图例：图里每个部分对应的源码位置。

| 编号或分支 | 职责与源码 |
|---|---|
| 1 | [resident_ab 的预算参数](../../../../examples/inference_bench/resident_ab.py#L295)、[创建池时的槽上限，旧运行取 2](../../../../examples/inference_bench/resident_ab.py#L153)、[并发提交](../../../../examples/inference_bench/resident_ab.py#L216) |
| 2 | [EnvUnitLedger._gpu_fits_now](../../../../psrl/workers/env_service/ledger.py#L530) 将已有记账占用与新请求相加，超出池预算就拒绝立即准入 |
| 3 | [生产 runner](../../../../examples/sciaccel_rl/runner.py#L275) 执行到清理，再归还会话租约，完整阅读路径见 [recipe 流程图](../../knowledge/gpu-benchmark-recipes.md#command-与常驻推理服务) |
| OFF | [固定 baseline 的运行补丁](../../env-service-verification/results/router-baseline-control-20261020/baseline.patch) 保留静态映射，本次 [驱动配置](../../../../examples/inference_bench/resident_ab.py#L187) 将原有 episode 并发上限设为 2 |

跟着图读源码时，先看驱动传入的池预算和每任务需求，再看 ledger 的容量判断，最后看 runner 的清理与租约释放。两个 episode 由异步任务并发提交，但提交并发不等于 GPU 准入并发。各节点的详细启动到回收步骤已在 recipe 中串联，读者可以随时针对具体函数追问。

**为什么 OFF 有两个 GPU 进程。** 两个 episode 各有一个容器和一份模型服务，它们可以同时使用同一张 GPU。实测每份服务约 14.1 GiB，两份合计约 28.2 GiB，这张卡物理容量约 95.6 GiB，装得下，且未观测到 OOM。两个进程不等于两张卡，也不证明 GPU 算力同时满负荷。ON 只有一个进程，是池预算阻止另一个任务启动，并非 episode 策略天然要求整卡独占。其容器槽上限同样是 2。

**为什么等待期间仍不能接纳第二项。** 8 秒 CPU 等待不代表模型被卸载。第一项服务权重仍在 GPU，episode 声明预算也持续有效，因此账本不能因为没有新推理请求就把它的 20 GiB 归零。20 GiB 申报本身比本轮约 14.1 GiB 的实测峰值保守，但即便按这次实测值算，两份模型也超过人为设置的 24 GiB 池预算。

下表来自已保存的真实 agent 时间戳，时间零点是各组任务阶段开始，区间包括服务启动、请求及中间等待，不含该 agent 之后的 verifier 和环境清理：

| 时间区间 | OFF | ON |
|---|---|---|
| 第一个开始执行的 agent | 6.1 到 30.4 秒 | 22.4 到 43.4 秒 |
| 第二个开始执行的 agent | 6.2 到 29.1 秒，与第一个重叠 | 50.7 到 71.8 秒，前一个结束后执行 |
| 全部任务与 episode 清理完成 | 46.3 秒 | 76.4 秒 |

OFF 重叠了两个模型的加载、等待和请求。ON 串行执行，还存在更长的首任务准备区间与两次 agent 之间的环境清理、启动间隔。已有时间戳能确认这些区间，不能把区间内所有开销精确归到某个组件。单次真实请求耗时则接近：第一题 OFF 为 0.731 到 0.742 秒，ON 为 0.730 到 0.772 秒，第二题 OFF 为 0.323 到 0.325 秒，ON 为 0.317 到 0.319 秒。没有证据显示本次主要问题是模型推理本身变慢。

17.6 秒 ON 驱动初始化包含私有 Ray，它在 76.4 秒任务阶段之外。只有比较完整驱动耗时时才把它计入，即 OFF 47.1 秒、ON 96.7 秒。不能用这 17.6 秒解释 76.4 与 46.3 秒之间的全部差值，也不能把全部差值归为纯 Router 开销，因为两组还有版本与容器后端差异。

**合适的后续性能对照条件。** 在同样一张现场空闲卡上，ON 按实际允许使用的容量配置池，保留合理余量，让两个 20 GiB 请求都能准入。OFF 继续使用指定提交并保持相同任务与并发上限，再比较初始化、任务和清理。资源充足时，合理预期是 ON 不因人为低配预算额外串行，是否更快需重新测量。真正资源紧张的实验另列，用实际冲突验证失败率和完成量。这里是设计纠正，未重新执行 GPU 实验，也不能预告修正后一定加速。

可运行复核仍使用前节 [resident_analyze.py](../../../../examples/inference_bench/resident_analyze.py)，只读取两组实际日志，不启动 GPU。原始配置和结果保持原样，不能通过修改旧 JSON 将这次排队验证改称等资源性能实验。

启动器默认值现已纠正为所选 GPU 的现场总容量，小配额需要显式传 `--pool-gpu-mb 24576`。原始运行脚本已按哈希保存到本地结果目录的 `executed-source/`，原 JSON 与 provenance 不变。本轮只完成默认容量、显式配额与非法容量的 CPU 检查，以及旧压力结果的重新审计，没有重新运行 GPU 对照。当前 unit 模型与保守记账的进一步边界见 [显存准入说明](../../knowledge/gpu-memory-admission.md#8-unit-总容量动态余量与装箱气泡)。

## episode 记账修复后的整卡压力对照

本节为 2026-11-02 重新执行的一组真实对照，旧 24 GiB 数据保持原样。目标是验证 episode 可信归属记账生效，以及真实模型竞争显存时 ON 能否减少 OOM 和重试，不以跑出速度优势作为重试条件。只执行了预先设计的一组 OFF 和一组 ON，没有在结果明确后继续增加任务或重复挑选结果。

**相同条件与到达规则。** 两组使用同一张现场空闲 H20，NVML 报告总容量 97871 MiB，约 95.6 GiB。ON 一个 unit 覆盖整卡，每个 episode 预约 20480 MiB，容器槽和 OFF 原有进程内 episode 信号量上限都设为 7。先启动两个 Mistral 服务，均通过 readiness 后同时提交其余五项。每项执行两条官方 seed 21 MMLU-Pro 请求，中间用独立 CPU 命令等待 30 秒，模型保持驻留。精度 BF16、上下文 2048、每请求最多 16 token，模型、服务、镜像及样本两组相同。

```text
[1] 同一任务队列：2 项服务 ready → 再到达 5 项，每项跨命令驻留
        ├─ OFF：固定 baseline，7 个 episode 槽 → 模型直接竞争整卡
        │      首轮 6 成功 + 1 OOM → 首轮结束后重试该项一次
        └─ ON：[2] EnvRouter 刷新可信 episode 归属 → 单卡预算判断
               同时最多 4 条 lease → 其余等待 → 清理归还后继续
[3] 两组均完成 7 项、14 次请求 → 检查输出一致与 GPU/容器实际回收
```

图例：图里每个部分对应的源码位置。

| 编号 | 源码与职责 |
|---|---|
| 1 | [resident_ab.run](../../../../examples/inference_bench/resident_ab.py#L82)、[提交与重试](../../../../examples/inference_bench/resident_ab.py#L214) 实现相同到达门槛和最多两次尝试 |
| 1 的驻留 | [ResidentInferenceAgent.run](../../../../examples/inference_bench/resident_agent.py#L10) 启动后台模型、两次请求、等待及退出；[client.main](../../../../examples/inference_bench/resident_client.py#L11) 使用真实样本并在启动 OOM 时快速报告 |
| 2 | [Router._check_node](../../../../psrl/workers/env_service/router.py#L515)、[EpisodeMemory](../../../../psrl/workers/env_service/episode_memory.py#L38)、[账本公式](../../../../psrl/workers/env_service/ledger.py#L432) 传递并消费可信归属，详细调用顺序见 [记账修复图解](../../knowledge/gpu-memory-admission.md#9-episode-归属记账修复) |
| 3 | [resident_analyze.analyze](../../../../examples/inference_bench/resident_analyze.py#L13) 读取成功与失败尝试、请求、采样与物理清理结果，不启动 GPU |

跟着图读源码：先看 1 中 `episode` 的 session 标识与结果记录，再看 readiness 门槛和 burst 提交。OFF 调用固定 checkout 的 runner，ON 经 EnvServiceClient 申请 lease 后才进入原生 EnvWorker。2 的观测和其他任务执行并发进行，不能把采样协程解释成顺序任务。最后看 agent 的 `finally` 停服务、runner 清理和驱动的逐 actor 容器核验。源码如果有不清楚的地方，可以继续按图追问。

重试规则在运行前固定：仅确认 CUDA OOM 的失败项可以在首轮全部结束后重试一次，两组一致，其他故障不自动重试。组内任务预算 900 秒，外层 driver 1000 秒，每条 agent 最多 400 秒。没有使用外部显存占位进程，也没有降低 ON 池容量。

| 指标 | OFF，固定 baseline | ON，修复后的 Env |
|---|---:|---:|
| 首轮完成 | 6/7 | 7/7 |
| CUDA OOM 次数 | 1 | 0 |
| 总尝试数 | 8 | 7 |
| 最终完成 / 成功请求 | 7 / 14 | 7 / 14 |
| GPU 计算进程采样峰值 | 7 | 4 |
| GPU 显存采样峰值 | 84.57 GiB | 56.38 GiB |
| 活动 lease 峰值 | 未启用账本 | 4 |
| 有等待者的采样数 | 未启用账本 | 44 |
| 驱动初始化 | 0.27 秒 | 9.93 秒 |
| 任务阶段，含 OOM 重试和各 episode 清理 | 158.05 秒 | 129.79 秒 |
| 最终驱动清理 | 0.15 秒 | 2.00 秒 |
| 驱动总耗时 | 158.48 秒 | 141.73 秒 |
| 最终本次容器 / GPU 计算进程 | 0 / 0 | 0 / 0 |
| 最终 NVML v2 显存占用 | 0.1875 MiB | 0.1875 MiB |

**真实 OOM 原因。** 失败发生在第七个服务加载权重时。错误明确报告要分配 13.50 GiB，当时只有 11.23 GiB 空闲，已有六个模型进程各占约 13.81 到 14.09 GiB。第七个 GPU 进程只建立了约 316 MiB 上下文，未成功加载模型。因此“峰值有七个进程”不等于七份模型全部加载成功，显存采样峰值也无需达到卡容量才会发生一次大分配失败。这是实测容量不足，不需要猜测为碎片化。

**记账修复确实被使用。** 一条 ON 样本中有四条活动 lease、三个等待者，声明量 81920 MiB，整卡实测 57165.6875 MiB，核实归属 57140 MiB，账本有效占用 81945.6875 MiB。计算为 `57165.6875 + 81920 − 57140 = 81945.6875`，约 80.03 GiB。这里各任务实用量均低于其预约，才可以这样简写。旧的无抵扣算法会把这同一状态算成约 135.83 GiB。整个 ON 运行的可信归属采样峰值为 57704 MiB，已取得快照中没有归属错误。不是仅靠静态槽上限限制到四项，槽上限本身是七。

**ON 的本次价值与剩余气泡。** 同样完成 7 项任务，ON 避免了一次失败加载及重试，任务阶段缩短 17.9%，驱动总耗时缩短 10.6%。ON 显存峰值更低，意味着它遵守了各项 20 GiB 的承诺，并不表示已将物理显存利用率最大化。四项预约合计 80 GiB，再加一项是 100 GiB，超过这张卡，所以最多四项符合当前申报。每项实际约 14.1 GiB 与申报 20 GiB 之间的余量是仍有效的增长承诺，不能再当作“重复记账”直接删除。若要进一步装箱，需要有依据地校准任务峰值或设计可撤销预算，本次没有顺带改变这项资源语义。

**计时和因果边界。** 工作时间从首批 episode 提交前到全部逻辑任务返回，包含 readiness、等待、失败尝试和 episode 清理。ON 私有 Ray 初始化另计。shell 环境激活、Python 模块导入、镜像构建在驱动计时之外，其中 OFF 环境激活曾较慢。到达规则相同，但 readiness 的绝对时刻不同，所以剩余五项相对本组工作开始的提交时间为 OFF 19.18 秒、ON 28.57 秒。

OFF 实际提交为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，ON 为 `8d1fc65230d7bd4c78305e140c41ac8b5ea14fb0` 加主目录未提交的记账补丁。OFF 复用先前隔离 checkout 的必要运行补丁，涉及 `override_gpus` 传递、取消全局 dangling-image prune、HTTP loop 归属与 Ray 地址，保留原资源调度语义，无整卡锁。OFF 是旧 Docker 后端，ON 是当前原生 EnvWorker，因此是固定 baseline 的完整路径对照，不能将全部速度差归为单个记账函数，也没有隔离“当前修复前 vs 当前修复后”两版本 GPU 吞吐。CPU 回归分别验证了无归属和可信归属的账本准入结果。

**遥测与验证边界。** ON 一次账本采样超时保留为缺失，17 条容器 CPU/主存资源错误或缓存错误单列，没有补零；这些不是 GPU 归属错误。结束时 lease 归零，驱动逐 actor 查询与后续独立 Docker 查询均无所属容器，GPU 进程为空。没有仅依据账本或缓存容器计数声称清理完成。两组成功请求的逐样本文本和 token 数完全相同。本次验证了真实模型任务、Env、生产 runner 与资源回收，没有启动 PSRL 策略优化器，也没有做 InferenceBench 官方质量评分。

**代码与回归。** 新增 `EpisodeMemory`，连接 node → router → ledger，补齐逐设备抵扣和可观测字段。首批定向 86 项通过。一次较大 Env 回归为 341 通过、5 失败：两项是误用共享 veRL 路径，另两项是既有子进程 3 秒/10 秒启动时限超时，一项是在测试运行期间新增逐设备检查而进程仍载有旧模块。固定 `.verification-deps` 并冻结源码后，包含这五项和归属测试的定向 25 项全部通过；随后新增节点归属缺失/空租约回收检查，episode 文件的 9 项全部通过。原失败日志保留，未声称全仓一次全绿。新增代码 Ruff、格式和 prose 检查通过，全仓 pre-commit 在独立快照执行，仍有本次范围外的既有格式/注释问题，未回写其自动修改。

证据： [comparison](episode-pressure-comparison.json)、[samples](episode-pressure-samples.json)、[provenance](episode-pressure-provenance.json)。原始日志在本地 `benchmark-runs/episode-accounting-20261102/`，包含 ON/OFF 实际源码 diff、执行源码副本、失败日志、Docker/CUDA 预检和物理清理查询。provenance 区分 GPU 运行时脚本哈希与事后新增压力断言的 CPU 审计脚本哈希。

```bash
python examples/inference_bench/resident_analyze.py \
  --root "$RESULT_DIR" --output "$AUDIT_OUTPUT" \
  --prefix episode-pressure --expect-pressure
```

这个命令已经实际通过。它只复核已保存结果，要求相同工作量和预算、OFF 存在 OOM、ON 无 OOM、有可信抵扣和等待，并核验逐样本输出与清理。知识解释和不需要 GPU 的教学脚本见 [episode 记账说明](../../knowledge/gpu-memory-admission.md#9-episode-归属记账修复)。所有改动当前保留在主目录，未提交、未推送、未合并。

## RE-Bench 原生接入验证

2026-11-02 在首选实验节点的一张独立空闲 H20 上运行。PSRL 基准为 `affe7c2` 加本次接入修改，上游 RE-Bench 为 `a0cc6d4a91b20f84ebef505d5c72358f88b3b389`。模型和数据是真实官方资产，下载与来源见 [re-bench-smoke.json](re-bench-smoke.json)。下载首轮因耗时退出，保留缓存续传完成，没有将部分权重当作完整模型。

本次待验证结论是原生 Env 能否接纳并执行此任务的真实 CUDA 工作，以及任务、独立 verifier、奖励字段和清理能否串联。每次 agent 上限 240 秒，运行器总上限 480 秒。连接、Docker、GPU 身份映射、容器 CUDA、小型算子和数据隔离预检已执行。正式模式缺凭据时在启动 GPU 前拒绝，未调用外部裁判。

| 项目 | 本次结果 |
|---|---|
| 模型 | 官方 GPT-2 XL，完整权重，约 15.56 亿非位置嵌入参数 |
| 数据 | 固定 Alpaca revision，51,490 条公开训练提示与 512 条隐藏提示，smoke 使用公开部分前两条 |
| 任务内工作 | 每条最多生成 8 token，然后用一段公开提示 token 做一次 Adam 更新，loss 有限 |
| Env | EnvRouter episode 准入，原生 EnvWorker，独立 verifier，单任务声明 49,152 MiB，unit 按整卡 97,871 MiB 配置 |
| 功能评分 | `smoke_valid=1`，`reward=0`，`official_evaluation=0`，`judge_valid=0` |
| 耗时 | 工作负载内部约 21.21 秒，含初始化与清理的 driver 总耗时约 53.67 秒 |
| 训练后框架内存观测 | PyTorch allocated 19,239,225,856 字节，reserved 33,256,636,416 字节，不当作整轮 NVML 峰值 |
| 清理 | 最终租约 0，本次容器 0，GPU 计算进程 0，GPU 遥测回到约 0.19 MiB |
| CPU 回归 | 25 项通过，覆盖共享 recipe、RE-Bench 数据与评分、PSRL 会话奖励传递、裁判请求预算及错误退出 |

实际 runtime 为 Python 3.10.20、PyTorch 2.9.1+cu128、Transformers 4.44.2，复用隔离镜像依赖，未改共享环境。官方依赖和两张 H100 的资源配置不同，因此这是单 H20 本地接入验证，不是原版环境复现。评分库源码固定 v0.2.1，其元数据要求 Python 3.11 以上，本次只验证了在上述 runtime 中的导入和已运行路径，完整评分仍应单独验收。

**修复及失败记录。** 第一轮 oracle 把 `psrl-gpu` 当作脚本内可执行文件调用而失败，另有独立 verifier 未预置测试脚本。oracle 改为在已获得 episode 预算的容器中执行固定训练命令，公共运行器将任务和输出路径转为绝对路径。随后发现离线缓存缺少固定 revision 对应的 `main` 引用，补齐后离线配置解析通过。独立评分镜像预置 `/tests/test.sh`，通过可信 verifier 环境变量传入模式与超时。修复后真实模型任务通过，失败轮均记录并核验清理，没有重跑成功任务追求更高分。

**验证边界。** 本次一次更新的目标是公开提示的语言模型损失，没有制造裁判偏好。它证明 CUDA 生成与更新能在 Env 中执行，不证明官方 RL 反馈训练完成。外部裁判桥接经过 CPU 测试替身验证，没有真实 Replicate 调用。没有验证实际裁判等待曲线、自动 offload/reload、ON/OFF 性能、语言模型 agent 自主优化或 PSRL 策略训练。接入以最终模型评分，未实现原版中间评分取最佳的完整协议。

**静态检查。** 本次代码的 Ruff、格式检查、prose-style 和启动脚本语法检查通过，相关文档 412 个本地链接及章节、源码行号范围检查通过。全仓 `pre-commit run --all-files` 在临时快照中执行，仍因本次范围外的既有长行、格式和注释风格问题失败。快照内自动修改未回写工作区，未将全仓检查记为通过。

操作入口、执行图与源码路径见 [RE-Bench recipe](../../knowledge/re-bench-recipe.md)。本次验收已达到无裁判条件下的必要边界，停止 GPU 验证并保留日志。

## 评分教学脚本验证

2026-11-02 运行 [评分演示脚本](../../env-service-verification/scripts/explain_inferencebench_score.py)，只使用 CPU 与标准库。性能和质量输入是明确标注的教学数字，PSRL 奖励门槛调用真实 [compute_score](../../../../examples/gpu_benchmarks/reward.py#L6)。

| 教学输入 | 模拟官方奖励 | 真实 PSRL 函数输出 |
|---|---:|---:|
| 参考答对 400/500，TTFT 0.4 秒 | 2.5 | 2.5 |
| 候选答对 390/500，TTFT 0.2 秒 | 5 | 5 |
| 候选答对 360/500，质量比例 0.9 | 0 | 0 |
| 候选质量达标且原始分为 5，但污染标记为 1 | 5 | 0 |

C 场景教学吞吐为 8、27、64 请求/秒，输出几何平均 24。脚本 Ruff 与格式检查通过。这只验证手算例子和本地奖励检查的连接，没有启动 GPU 服务、运行官方质量题集或调用裁判。详细定义见 [评分说明](../../knowledge/inferencebench-scoring.md)。本节评分教学与调研本身不包含 GPU 验证。PostTrainBench 尚未接入，RE-Bench 的独立实机验证见上文，仍未验证完整偏好评分。

本次文档检查覆盖 279 个本地链接及 79 个源码行引用，路径和行号范围通过，相关评分函数与调用位置已人工核对。全仓 `pre-commit run --all-files` 在隔离快照执行，因 `docs/conf.py`、其他实验脚本及已有注释规范等问题失败。快照中的自动格式修改没有回写主目录。此结果不影响上述 CPU 演示结论，也不代表全仓检查通过。

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

## KernelBench command 接入验证

2026-11-04，基于 `cd09114` 加未提交 KernelBench verifier 插件与 runner 接线。使用独立 H20、Compose 提供者、固定官方 level1 矩阵乘法题，输入保持 4096 × 4096，oracle 复制既有 Triton 提交，独立 verifier 经过 direct command 准入。没有语言模型 rollout、PSRL 优化器训练、动态换卡或 OFF/ON 吞吐对照。

首次运行在上传 tests 时失败，原因是 Harbor 自定义 verifier 工厂未透传 skip_tests_upload，尝试覆盖只读测试目录。适配器改为显式 separate、预置测试契约。第二次在只读测试文件的 chmod 上失败，原因是包装层增加了 Harbor 原流程没有的非零退出立即报错语义。最终保留原退出码与评分解析行为，仅统一 GPU 执行和归还。两次失败均未进入正式 GPU 评估，所属资源均已清理。最终修复后只补跑该组，不重跑历史 episode 对照。

| 检查 | 本轮结果 |
|---|---|
| 编译与正确性 | compiled=true，correctness=true，3/3 |
| 性能测量 | 10 次，候选 7.82 ms，参考 4.55 ms |
| speedup | 约 0.582，候选更慢，不是调度加速比 |
| 本地 reward | 1.0，当前 recipe 使用正确性奖励 |
| command 记账 | 采样观测到活动 8192 MiB，以及会话仍存活但活动预算已归零 |
| 任务异常 / 清理异常 | 均无 |
| 本次所属容器 / GPU 计算进程 | 均为 0 |
| 最终显存 | NVML 约 0.19 MiB，独立 nvidia-smi 核验显示 0 MiB |
| CPU 回归 | 第一组 44 项通过，最终专项 10 项通过，存在重叠，不能相加 |

原始证据位于工作区 `benchmark-runs/kernel-command-20261104/`，包含首次 result、retry、最终 final、启动日志和源码副本。测试任务复制历史 task，但 Compose 中 `/tests` 仍挂载历史评分文件，其原始 reward 字段沿用了误命名 runtime_us。上表依据 kernel_result.json 的 runtime 与 CUDA event 毫秒单位解释，未修改原始结果。当前仓库 evaluator 已使用 runtime_ms，此字段修正不触发额外 GPU 运行。

本轮依赖来源已记录。veRL 与 transfer_queue 来自共享安装，任务未调用策略训练或张量往返链路，因此不把本次结果当作隔离 veRL 补丁或训练正确性验证。实验在后台用 setsid、nohup 和每次 480 秒外层期限运行，三次均结束，清理已独立核实。单卡无竞争验证只证明接入可用，没有量化 SM 竞争导致的测速偏差。原生提供者和模型 Agent 的 command 端到端路径未在本轮另跑。

复现参数和源码图见 [KernelBench command 与性能测量](../../knowledge/gpu-benchmark-recipes.md#kernelbench-command-与性能测量)。正式测量仍需实际独占的环境 GPU，当前单会话约束不是平台级独占。

本次全仓 pre-commit 在临时副本执行，未通过：例如 docs/conf.py 的既有行长、env_service/client.py 的 prose 规则及其他文件格式问题。此次新增和修改的运行代码未出现上述 hook 失败或被自动改写，临时副本修改未回写主目录。本次文档本地路径及源码行号有效性检查通过。


## RE-Bench 本地 Qwen 裁判验证

实机执行日期 2026-11-04，交付和清理复核 2026-11-05。源码为 `cd09114` 加本次未提交本地裁判修改，工作区同时存在独立的 KernelBench command 改动，未把它们归入本次实现。结果摘要见 [re-bench-local-qwen.json](re-bench-local-qwen.json)，复现入口见 [部署 recipe](../../knowledge/re-bench-recipe.md#部署独立-qwen-裁判并运行)。

验收目标是独立本地裁判卡、真实反馈训练、独立 verifier 和物理清理接通。先做现场连接、Docker ping、NVML v2、CUDA 容器算子和依赖来源检查。任务使用一张 H20，Qwen3-8B 使用另一张 H20，模型 BF16，vLLM eager，最大上下文 8192，显存预算 45%。没有占用训练/推理中已有 GPU。

裁判进程设 1200 秒寿命，任务每阶段 300 秒，运行器 660 秒，成功轮裁判最多 8 请求、2 并发、每请求 60 秒。训练只运行一轮，两条公开提示、每份候选最多 16 token，按原程序过滤后实际完成一次优化器更新。独立评分用原函数的 max_prompts=2、max_tokens_to_sample=16，对两份固定参考各比较一次，共四个评分请求。

| 指标 | 结果 |
|---|---|
| 成功轮裁判请求 | 6 成功，0 失败，训练 2 次加评分 4 次 |
| 真实训练 | 1 轮，1 次优化器更新，模型保存完成 |
| 训练裁判等待 | 0.824 秒，等待前后 PyTorch allocated 均为 6,456,532,480 字节 |
| 部分隐藏评分 | 2 条提示，对 GPT-2 XL 胜率 1.0，对 GPT-2 Alpaca 胜率 0.5，均值 0.75 |
| 奖励 | reward=0，judge_valid=1，feedback_smoke_valid=1，eval_valid=0，非完整评分 |
| 工作负载时间 | 训练内部 27.52 秒，评分内部 14.30 秒，driver 含初始化和清理 92.27 秒，不包含裁判冷启动 |
| 物理清理 | 任务容器和任务 GPU 进程均为 0。裁判已退出，端点不再监听，现场两卡无计算进程，nvidia-smi 为 0 MiB |

失败证据也保留：NVML v1 预检与现场工具不一致，改用项目既有 v2 查询。共享 vLLM 不接受 UUID 字符串、旧日志参数，启动器改为实际 CUDA UUID 映射及当前参数，没有修改共享依赖。第一轮反馈任务收到真实字符串排名，被整数校验拒绝，未执行更新且清理完成。单请求诊断确认返回内容后，本地分支严格转换 `"1"`、`"2"`，保持真实排名，定向复测通过。没有将无效反馈填成合法偏好。

裁判停止时报告了一个 multiprocessing semaphore 资源追踪警告。随后现场检查没有残留 GPU 进程或本次容器，不将警告当作仍占 GPU，也不声称已审计全部 OS IPC 对象。

这次短等待是一个真实观测，不是稳定分布，更不能证明值得 offload。0.75 是两条隐藏提示的局部结果，不具有完整 benchmark 成绩意义。未执行多裁判副本吞吐测试、完整 512 条评分、PSRL 策略训练、command ON/OFF 或自动卸载收益实验。结论齐备后停止 GPU 验证。

本地裁判最终 CPU 回归 30 项通过，覆盖原有 recipe、本地 HTTP 协议、评分来源隔离、请求预算和排名格式。新增代码 Ruff、格式与 prose-style 检查通过，相关文档 382 个本地链接及锚点检查通过。全仓 pre-commit 在临时快照执行，仍因本次范围外的既有格式与注释问题失败，自动修复未回写工作区。
