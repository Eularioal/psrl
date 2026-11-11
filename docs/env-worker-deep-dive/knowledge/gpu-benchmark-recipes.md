# GPU benchmark 的 Env 与 PSRL recipe

> **核心结论**：GPU recipe 共用 `GpuBenchmarkAgentLoop`，强制通过 EnvRouter 准入，默认支持 Docker 与原生 Env 后端的 episode GPU 持有，KernelBench 另有显式 direct command verifier 入口，见末节。准备器生成 Harbor 任务与 PSRL Parquet。长训练可以包装为一条 command，跨请求常驻模型才更直接体现持续持有的必要性。InferenceBench 提供固定真实请求协议，RE-Bench 提供 CUDA smoke 与显式有界本地评分，均与官方完整协议区分。

先读 [README 导览](../experiments/gpu-benchmarks/README.md) 了解任务目标、具体例子与 command 适用性。验证结果见 [GPU benchmark 接入验证](../experiments/gpu-benchmarks/FINDINGS.md)。官方性能分、质量门槛与本地功能分的区别见 [InferenceBench 评分说明](inferencebench-scoring.md)。

新增的 RE-Bench 聊天模型训练使用原生 Env，镜像、裁判桥接和正式评分准备步骤见 [RE-Bench recipe](re-bench-recipe.md)。下文四类准备命令保留各自入口，不使用通用准备器的 autoresearch 分支处理 RE-Bench。

2026-11-10 对 `bc5665f` 加未提交代码局部核对：六类代表任务均已通过真实 PSRL rollout、一次策略更新及版本 1 后续独立 GPU 评分。本地资源下的 MLE、InferenceBench、RE-Bench 与新增 FlashInfer 接入入口见[完整接入与本地验收](../milestones/science-gpu-full-onboarding.md)。本页原官方协议保持独立，不能用 local 数据源生成官方评分标记。

本轮快速扩题已发布 249 题统一任务池：Kernel 原尺寸 22、本地小输入档 193、FlashInfer 定义 30、本地 MLE 4。FlashInfer 使用 [prepare_suite](../../../examples/flashinfer_bench/prepare_suite.py#L12) 按有原始随机输入的定义准备有界子集；MLE 的 `--dataset` 可选择 `digits`、`iris`、`wine`、`breast-cancer`。具体资源、运行方式、失败边界与数据入口见[快速验证结果](../milestones/science-gpu-bench-status-and-roadmap.md#results扩题与快速-gpu-验证)。

2026-11-10 验收后边界修复经 30 项 CPU 回归：缺少提交物明确零分，MLE 检查损坏和超额 CSV，混合准备器核对显式来源标记，Inference verifier 清理完整服务进程组。六类修复尚未重跑混合策略训练，11-11 快速 GPU 复验仅覆盖上方列出的扩题路径。使用修复需重新准备任务，RE 还需重建 verifier 镜像，见[回归与产物边界](../milestones/science-gpu-full-onboarding.md#本轮边界修复与回归)。

## 目录

- [KernelBench command 与性能测量](#kernelbench-command-与性能测量)
- [入口与边界](#入口与边界)
- [下载官方资产](#下载官方资产)
- [H20 与框架内资源调度对照](#h20-与框架内资源调度对照)
- [Kaggle 下载权限配置](#kaggle-下载权限配置)
- [准备任务](#准备任务)
- [episode 级任务与原生 Env 验证](#episode-级任务与原生-env-验证)
- [command 与常驻推理服务](#command-与常驻推理服务)
- [真实环境验证](#真实环境验证)
- [接入训练](#接入训练)
- [评分与数据隔离](#评分与数据隔离)

## 入口与边界

| 组件 | 入口 | 职责 |
|---|---|---|
| KernelBench | [prepare.py](../../../examples/kernel_bench/prepare.py) | 选择上游题目，保留输入尺寸与参考实现 |
| MLE-bench | [prepare.py](../../../examples/mle_bench/prepare.py) | 使用官方已准备的竞赛数据及 grader |
| InferenceBench | [prepare.py](../../../examples/inference_bench/prepare.py) | 导入上游生成的 Harbor 任务，将 Modal handoff 改为本地 Docker bind |
| autoresearch | [prepare.py](../../../examples/autoresearch/prepare.py) | 使用固定数据缓存，提交训练脚本，独立容器重放 |
| RE-Bench | [prepare.py](../../../examples/re_bench/prepare.py) | GPT-2 XL 聊天训练，原生 Env、宿主裁判请求桥接与独立官方评分 |
| MLE 本地协议 | [local.py](../../../examples/mle_bench/local.py) | 真实 digits 数据、CUDA 分类器、隐藏测试标签独立评分 |
| Inference 本地协议 | [local.py](../../../examples/inference_bench/local.py) | 固定真实请求、冻结参考输出、独立服务重启与评分 |
| FlashInfer-Bench | [prepare.py](../../../examples/flashinfer_bench/prepare.py) | 官方 Solution 格式和 RMSNorm 子集，独立 evaluator |
| 公共准备器 | [prepare.py](../../../examples/gpu_benchmarks/prepare.py) | 记录源码版本、生成任务与 Parquet |
| PSRL 循环 | [agent_loop.py](../../../examples/gpu_benchmarks/agent_loop.py) | 复用会话轨迹收集，强制 EnvRouter，不允许本地无准入回退 |
| 环境验证 | [run.py](../../../examples/gpu_benchmarks/run.py) | 私有 Ray、EnvRouter、任务执行、清理证据 |
| 奖励 | [reward.py](../../../examples/gpu_benchmarks/reward.py) | 读取 verifier 指标，拒绝非有限奖励，检查 InferenceBench 完整性门槛 |

默认使用显式 `episode` GPU unit。KernelBench 可选择末节说明的 direct command 入口，其独立 verifier 已接入受管 GPU 执行。其他 recipe 尚未开放 command，不宣称阶段借还已经覆盖全部任务。这里的 Docker 后端由 EnvRouter 管理，并非绕过 Env 的裸 Docker。运行入口支持显式选择 `docker` 或 `env_worker`。原生 `EnvWorkerEnvironment` 要求预构建镜像，不解析任务 Compose 文件。两种路径均使用 EnvRouter 准入。

上游核验版本见 [upstream.json](../../../examples/gpu_benchmarks/upstream.json)。准备时要求干净 checkout 并记录实际 commit。依赖装入独立镜像或任务专属环境，不修改共享 conda 安装。

## 下载官方资产

设置 `BENCH_REPO` 为对应上游 checkout 的绝对路径，`ASSET_ROOT` 为任务专属资产目录的绝对路径。下载也设置外层超时，失败后保留缓存和日志，不把部分文件当作完成。

autoresearch 使用官方锁文件安装依赖，把 Python 本体和虚拟环境都放在执行节点可见的存储。以下命令在 PSRL 仓库根目录运行：

```bash
mkdir -p "$ASSET_ROOT"
UV_PYTHON_INSTALL_DIR="$ASSET_ROOT/python" \
UV_PROJECT_ENVIRONMENT="$ASSET_ROOT/autoresearch-env" \
  timeout 600 uv sync --project "$BENCH_REPO" --frozen
TASK_PYTHON="$ASSET_ROOT/autoresearch-env/bin/python"
RAYON_NUM_THREADS=4 timeout 600 "$TASK_PYTHON" examples/autoresearch/download.py \
  --repo "$BENCH_REPO" --data "$ASSET_ROOT/autoresearch-data" \
  --hf-cache "$ASSET_ROOT/autoresearch-hf" --num-shards 1
```

[下载器](../../../examples/autoresearch/download.py) 调用上游 `download_data` 和 `train_tokenizer`，只将缓存位置移到指定目录。`--num-shards 1` 选取官方训练分片 0，仍下载固定验证分片 6542。它适合接入验证，不代表默认 10 个训练分片的完整运行配置。300 秒训练预算、2048 上下文和固定验证 token 数不变。默认 kernel 为 Hopper 使用的 `varunneal/flash-attention-3`，非 Hopper 按上游选择传入 `--kernel-repo kernels-community/flash-attn3`。`assets.json` 记录版本及文件哈希，数据可复用，无需每次重训 tokenizer。

MLE-bench 在其独立依赖环境中执行官方命令：

```bash
timeout 600 mlebench prepare -c aerial-cactus-identification \
  --data-dir "$ASSET_ROOT/mle-bench" --keep-raw
```

先按 [Kaggle API 说明](https://github.com/Kaggle/kaggle-api#api-credentials) 配置凭据。实际搜索目录以该环境的 SDK 为准，或显式设置 `KAGGLE_CONFIG_DIR`。缺少凭据时停止，不使用第三方重打包数据绕开官方校验，也不关闭 checksum 验证。

InferenceBench 按官方资源清单下载完整 Mistral 仓库及两个数据集：

```bash
export HF_HOME="$ASSET_ROOT/inference-hf"
timeout 1800 python "$BENCH_REPO/containers/download_hf_cache/download_resources.py" \
  --resources-file "$BENCH_REPO/containers/download_hf_cache/resources.json" --workers 2
(
  cd "$BENCH_REPO"
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    timeout 180 bash src/commit_utils/precache_seeds.sh
)
```

模型访问权限以实际下载结果为准。官方资源清单含 `Mistral-7B-Instruct-v0.3`、`TIGER-Lab/MMLU-Pro`、`THUDM/LongBench-v2`，样本缓存写入上游已忽略的 `baselines/samples`。下载完成仍不等于任务可运行：还需真实计算质量及速度基线，具备 judge 凭据，并满足官方任务的 GPU 类型要求。不能用伪造基线或关闭 judge 获得通过。完整说明见 [官方准备流程](https://github.com/aisa-group/InferenceBench#download-models-and-datasets)。

## H20 与框架内资源调度对照

本节针对框架内资源调度实验。正式 recipe 导入官方任务，保留官方评分条件。另一个入口 [resident_ab.py](../../../examples/inference_bench/resident_ab.py) 使用官方服务、模型和真实请求，专门比较资源准入。它不产生官方质量或 judge 分数，实机结果见 [FINDINGS](../experiments/gpu-benchmarks/FINDINGS.md#inferencebench-常驻服务-onoff-results)。

两个未通过的 benchmark 都停在准备阶段。MLE-bench 缺少 Kaggle 账户凭据。InferenceBench 的官方任务生成器因缺质量和速度基线而退出，后续还要求 judge 凭据及 H100。此前将这些条件概括为 H20 无法继续的阻塞过于宽泛，它们不是框架调度实验全部必需的条件。

| 名称 | 作用 | 框架内实验如何处理 |
|---|---|---|
| 官方速度基线 | 官方评估所需的参考性能文件 | 不与外部成绩比较时，可采用两组各自实测的绝对耗时和吞吐，无需照搬 H100 参考值 |
| 官方质量基线 | 防止以降低回答质量换取速度 | 固定模型、精度、服务实现和输入时，保留真实输出、完成率与正确性检查。若允许 agent 改模型或服务策略，仍需质量约束 |
| 官方 judge | 检查 agent 自由修改方案的合规性等 | 固定受控代码的资源调度实验可不使用 LLM 裁判，结果单独标为框架 workload，不能输出伪造的官方 judge_pass |
| 框架 baseline | 衡量 EnvRouter 等资源管理机制的影响 | 必须保留相同工作量、到达顺序、资源和计时边界的对照 |

H20 可以作为本地实验硬件，官方模板中的 `H100!` 属于配置约束。固定服务实验已在 H20 执行官方 Transformers 服务和真实 Mistral 请求，但采用本地预构建镜像及较短请求，不代表官方镜像、长上下文场景与完整评分都已验证。

当前实现的相关位置：

- [run.py:57](../../../examples/gpu_benchmarks/run.py#L57) 核对任务声明的 GPU 类型，不匹配就拒绝运行。这是配置校验，不是 CUDA 运行失败。
- [reward.py:16](../../../examples/gpu_benchmarks/reward.py#L16) 要求官方完整性与质量门槛。固定负载实验不能沿用这个入口，再把缺失的裁判字段补成成功。
- 官方 checkout 的 `src/harbor_adapter/adapter.py` 中 `baselines()` 实际检查质量和速度文件，缺失即拒绝生成任务。上游源码不随 PSRL 仓库分发，版本见本文的 upstream.json 链接。

可以随时针对这些源码继续追问。实验应固定模型、服务代码、数据、精度及请求序列，比较任务完成量、总耗时、排队、失败/OOM 和物理回收。官方场景 A 含 128 个请求、名义输入 8192 token、输出上限 1024 token，存在长度浮动配置。本次常驻服务采用另一组有界的真实 MMLU-Pro 请求，不能标成完整场景 A。

单个空闲 GPU 上的一次请求只能证明执行链路，不能证明跨 worker 资源账本的价值。教学假设：一张可用 24 GiB 的 GPU，两个不同 worker 同时提交各需 16 GiB、运行 60 秒的任务。若没有跨 worker 准入，它们可能同时运行并导致 OOM。有容量账本时第一项占用 16 GiB，剩余 8 GiB，第二项等待第一项归还。最终可能用约 120 秒完成两个任务。收益可以是减少失败，不必表现为单任务更快。这些数字是教学假设，不是实测。

以下可运行的 CPU 教学模拟只演示容量计算，不代表真实 EnvRouter、CUDA 分配或硬隔离：

```bash
python - <<'PYEXAMPLE'
capacity, request = 24, 16
used = request
print('第一项准入后剩余 GiB:', capacity - used)
print('第二项可以立即准入:', used + request <= capacity)
used -= request
print('第一项归还后第二项可准入:', used + request <= capacity)
PYEXAMPLE
```

预期输出为剩余 8、False、True。真实实验还必须触发跨 worker 等待，并核实容器清理与物理资源归还。既有 baseline 可能已有进程内 episode 信号量，不能把它描述为完全不限制并发。

固定对照仍遵循既有约定：使用提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，记录必要的任务接入补丁并保留原有资源调度语义。同版本仅关闭 EnvRouter 的额外消融实验不能替代这一 baseline。当前强制 EnvRouter 的 recipe 本身不是 OFF 组启动器。

## Kaggle 下载权限配置

SDK 是调用 Kaggle 服务的软件，安装 SDK 不会获得数据下载权限。需要两层条件：账户 API 凭据证明身份，目标竞赛的规则接受状态允许该账户访问数据。当前实测只确定缺少凭据，尚未验证账户是否已接受规则。

本次 MLE-bench 固定依赖 Kaggle 1.6.17，因此采用旧版可识别的 `kaggle.json`。新 CLI 支持的 OAuth 或单字符串 token 不能直接当作旧版 SDK 已兼容的凭据。

1. 登录 Kaggle，打开 [API 设置](https://www.kaggle.com/settings/api)。
2. 在 **Legacy API Credentials** 下选择 **Create Legacy API Key**，取得 `kaggle.json`。它含用户名与密钥，不要写入仓库或聊天。
3. 打开 [Aerial Cactus Identification](https://www.kaggle.com/competitions/aerial-cactus-identification/rules)，按页面要求加入并接受规则。
4. 将文件放在本机私有目录，设置 `KAGGLE_CONFIG_DIR` 指向包含该文件的目录。无需为此向 PSRL 申请权限或部署额外服务。

设置 `KAGGLE_CONFIG_DIR` 后可在 MLE 专属环境执行：

```bash
chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
kaggle competitions files -c aerial-cactus-identification
```

第一行限制凭据文件读取权限，第二行只列出竞赛文件，用于先检查身份和访问权限。通过后再运行本文的 `mlebench prepare`，执行正式下载、校验与公开/隐藏数据划分。凭据只用于准备数据，不应挂入 agent 任务容器。后续只需要提供本机文件路径，无需发送密钥内容。

参考：[官方认证说明](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication)、[官方 API 入门中的竞赛规则要求](https://www.kaggle.com/getting-started/524433)。

## 准备任务

以下命令在仓库根目录运行。先设置变量：

| 变量 | 含义 |
|---|---|
| `BENCH_REPO` | 对应 benchmark 的干净源码 checkout |
| `TASK_DIR` | 不存在的新输出目录，生成任务及同名旁置 Parquet |
| `TASK_IMAGE` | 已在执行节点准备好的镜像，含 benchmark 依赖 |
| `TASK_PYTHON` | 容器中的 Python 绝对路径。显式增加 `--mount-python-prefix` 时只读挂载任务环境及其基础 Python 前缀，两者都必须在执行节点可见 |
| `DATA_ROOT` | MLE 官方数据根目录或 autoresearch 缓存根目录 |
| `UPSTREAM_TASK` | InferenceBench 官方 adapter 已生成的单个任务 |

KernelBench：

```bash
python -m examples.kernel_bench.prepare \
  --repo "$BENCH_REPO" --output "$TASK_DIR" \
  --image "$TASK_IMAGE" --python "$TASK_PYTHON" \
  --level 1 --problem 1_Square_matrix_multiplication_.py --timeout 240
```

这个题目自带一个固定 Triton 矩阵乘法提交，供 oracle 检查执行链路。它不是最优解，不要求速度超过 PyTorch。其他题目不会自动生成 oracle。上游输入尺寸不做缩小。

2026-11-11 新增 [prepare_suite](../../../examples/kernel_bench/prepare_suite.py#L29) 批量入口，默认枚举 L1 至 L4，固定本地题库共 270 题，已实际生成 270 个任务和配套 PSRL 数据表。调用方式与所有题库的数量、位置见[路线图逐题统计](../milestones/science-gpu-bench-status-and-roadmap.md#题库数量位置与逐题覆盖)。批量准备与快速 GPU 验证分开登记：本轮原尺寸通过 22 题、小输入档通过 193 题，后者使用 `kernel_bench_local` 与 `--bounded-inputs`，不能冒充原尺寸结果。小输入档将相同维度覆盖实现打包进独立 verifier，奖励须携带本地 profile 标记。策略更新闭环仍仅有此前的方阵乘法代表题，不能把快速验证写成 270 题闭环通过。

MLE-bench：先用官方 `mlebench prepare` 得到 `DATA_ROOT/competition/prepared/public` 与 `private`，再执行：

```bash
python -m examples.mle_bench.prepare \
  --repo "$BENCH_REPO" --output "$TASK_DIR" \
  --image "$TASK_IMAGE" --python "$TASK_PYTHON" \
  --data "$DATA_ROOT" --competition aerial-cactus-identification
```

数据授权和 Kaggle 访问由官方准备流程处理。agent 只能访问 public，私有数据根目录只挂入独立 verifier。没有数据时准备器报错，不生成模拟竞赛。

InferenceBench：先按照上游 `src/harbor_adapter/README.md` 准备官方质量基线、请求样本、模型访问与 judge 凭据，并用上游 `run_adapter.py` 生成任务，然后执行：

```bash
python -m examples.inference_bench.prepare \
  --repo "$BENCH_REPO" --harbor-task "$UPSTREAM_TASK" \
  --output "$TASK_DIR" --gpu-memory-mb 80000
```

保留原始 agent/verifier Dockerfile、质量门槛和 judge，不调用 Modal。执行节点要有构建依赖及访问上游所需资产的条件。官方 H100 要求和长评分预算不能当作已在其他卡验证。该 recipe 默认 agent 预算 7200 秒、verifier 43200 秒，仅在明确做完整评测时使用。

autoresearch：先使用其 `prepare.py` 准备 cache，安装 `rustbpe`、`kernels` 和上游固定 Torch 版本及 GPU kernel 缓存，再执行：

```bash
python -m examples.autoresearch.prepare \
  --repo "$BENCH_REPO" --output "$TASK_DIR" \
  --image "$TASK_IMAGE" --python "$TASK_PYTHON" --data "$DATA_ROOT" \
  --hf-cache "$ASSET_ROOT/autoresearch-hf" --mount-python-prefix \
  --gpu-memory-mb 65536 --timeout 900
```

保留上游单次 300 秒训练预算和固定评估。任务外层有超时，指令要求最多两次开发实验。实验次数本身不是后端强制配额。离线容器需要提前缓存所需数据和第三方 kernel。

这个 recipe 自带 oracle，将未修改的上游 `train.py` 交给独立 verifier 实际训练和评估。这里的 65536 MiB 是该基线的预留配置，实际需求还受候选脚本影响。不要对它套用下文 KernelBench 的 600 秒外层示例，应给运行器 1200 秒预算，并保持同一 GPU 申请量。

## episode 级任务与原生 Env 验证

Episode 是一次完整任务执行。`gpu_reservation: episode` 表示 GPU 预算从任务准入开始保留，直到任务清理并归还，而非每条命令结束就释放。这不等于 GPU 始终满负载，也不等于所有任务阶段都有必要占卡。

| 工作负载 | 持卡特点 |
|---|---|
| InferenceBench 常驻推理服务 | 模型权重跨多次请求、命令驻留。KV cache 是否跨请求保留取决于服务实现，不能一概而论 |
| autoresearch | 一次训练及评估需要连续保留模型、优化器和 CUDA 上下文，适合验证长时间持卡。准备代码阶段未必必须保留 GPU |
| MLE-bench | 训练期间连续持卡，但数据处理、训练、提交之间是否释放取决于任务实现。官方 Kaggle 数据待凭据，本地 digits 协议可独立准备和评分 |
| KernelBench | 单次编译和测量通常较短，可以按命令安排。默认 episode，可显式选择本篇 KernelBench command 入口 |

原生 Env 验证优先选择已有官方数据的 autoresearch。验证目标是原生容器启动、真实训练、官方评分、租约保留和物理归还，不以策略模型训练或资源管理性能提升为验收条件。

原生任务必须提前构建两个镜像：`TASK_IMAGE` 含 Python、依赖、`/opt/upstream`、`/root/.cache/autoresearch` 及离线 Hugging Face kernel 缓存，`VERIFIER_IMAGE` 在相同运行时基础上额外含准备器生成的 `/tests`。Python 位置与 `TASK_PYTHON` 一致。独立 verifier 不自动上传这些测试文件。镜像打包必须保留虚拟环境解释器及 Hub 缓存中的符号链接，在执行节点先验证解释器、依赖导入和缓存文件存在。

在已经设置本文变量的仓库根目录准备任务：

```bash
python -m examples.autoresearch.prepare \
  --repo "$BENCH_REPO" --output "$TASK_DIR" \
  --environment-backend env_worker \
  --image "$TASK_IMAGE" --verifier-image "$VERIFIER_IMAGE" \
  --python "$TASK_PYTHON" --data "$DATA_ROOT" \
  --gpu-memory-mb 65536 --timeout 900
```

准备后，将生成的 tests 目录打入 `VERIFIER_IMAGE`。原生模式不会生成 Compose 或挂载宿主 Python、数据缓存，镜像内容必须自行具备上述目录。在目标节点运行：

```bash
python -m examples.gpu_benchmarks.run \
  --task "$TASK_DIR" --output "$RESULT_DIR" \
  --environment-backend env_worker \
  --node-ip "$ENV_NODE_IP" --gpu "$GPU_INDEX" \
  --gpu-memory-mb 65536 --agent oracle --timeout 1200
```

[run.py](../../../examples/gpu_benchmarks/run.py) 每 5 秒采样账本及 NVML，写入 `samples.jsonl`，最终记录 `evidence.json`。采样用于观察持卡区间，不能证明采样间隙内不存在瞬时变化。遥测错误单独记录，不补零。运行结束要同时检查任务评分、活动租约、本次容器、GPU 进程和显存，不只看退出码。

PSRL agent 配置中的 `harbor.environment_backend` 同样支持 `env_worker`，仍要求显式 episode unit 和启用 EnvRouter。此验证使用 oracle 与真实 runner，未调用语言模型 agent，不能据此声称完整 PSRL 策略训练已通过。

## command 与常驻推理服务

先区分“能用一条命令启动”和“每次请求结束可以归还 GPU”。前者几乎总能做到，后者取决于命令结束后是否仍存在 GPU 状态。autoresearch 的整个训练脚本可以是一条长 GPU command，因此长达数分钟不能证明它只能使用 episode。除显式 KernelBench command 配置外，公共 recipe 仍限制 episode，这是接入边界而非任务的物理限制。 动态放置不能通过只删除 episode 校验来启用。KernelBench 已接入受管 verifier，动态换卡收益仍需另做对照，见[按 benchmark 的必要性评估](../research-proposals/dynamic-command-placement.md#9-已接入-benchmark-的必要性评估)。

常驻推理的实际路径如下。箭头表示调用或生命周期，两个请求通过独立环境命令发出。

```text
[1] 同时提交两个 episode
       |
       +-- OFF: 固定旧 runner，静态 GPU 映射，保留原有 episode 并发保护
       |
       +-- ON: [2] EnvRouter 申请整段预算，额度不足的 episode 等待
                         |
                    [3] 原生 EnvWorker 创建 GPU 容器
                         |
                    [4] 启动官方推理服务，加载模型一次
                         |
                    [5] 请求 1 -> CPU 等待 8 秒 -> 请求 2
                         |        服务权重仍在 GPU
                    [6] 停服务 -> verifier 校验 -> 删容器 -> 归还租约
                         |
                    等待的另一个 episode 得到预算
```

图例：图里每个部分对应的源码位置。

| 编号或箭头 | 职责 | 函数与源码 |
|---|---|---|
| 1 | 提交固定任务，可配置 warm 阶段和 OOM 重试，OFF 校验提交 | [resident_ab.run](../../../examples/inference_bench/resident_ab.py#L82)、[episode](../../../examples/inference_bench/resident_ab.py#L194) |
| 2 | 在创建环境前申请资源，附带 GPU 预算 | [EnvServiceClient.acquire](../../../psrl/workers/env_service/client.py#L159)、[run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275) |
| 3 | 创建原生容器并应用租约设备映射 | [EnvWorkerEnvironment.start](../../../psrl/workers/env_worker/harbor.py#L146) |
| 4、5 | 服务在后台持续运行，多个命令访问同一服务 | [ResidentInferenceAgent.run](../../../examples/inference_bench/resident_agent.py#L10)、[resident_client.main](../../../examples/inference_bench/resident_client.py#L11) |
| 6 | 停服务、检查结果、清理容器和租约 | [agent finally](../../../examples/inference_bench/resident_agent.py#L45)、[prepare 的 verifier](../../../examples/inference_bench/resident_ab.py#L48)、[EnvWorkerEnvironment.stop](../../../psrl/workers/env_worker/harbor.py#L182)、[EnvLeaseHandle.release](../../../psrl/workers/env_service/client.py#L69) |

跟着图读源码：先从 `resident_ab.run` 的 asyncio 任务和 warm 阶段看请求如何并发进入 runner，再沿 ON 的 `client.acquire` 看等待发生在容器启动之前。准入后进入 `EnvWorkerEnvironment.start`，再由固定 agent 启动服务。agent 等待 readiness 后依次发送两次 HTTP 请求，中间的 `sleep 8` 是另一条 CPU 命令。最后 agent 发 SIGTERM，Harbor 执行 verifier，runner 清理环境并释放租约。采样协程与任务并发运行，不能把采样顺序误读成任务的串行调用顺序。上述源码如有理解上的问题，可以随时沿图中某一步继续追问。

**具体例子。** 给一个任务池分配 24 GiB 显存预算，每个服务声明 20 GiB，同时提交 A、B 两个 episode。ON 先接纳其中一个，因为 `20 + 20 > 24`，另一个等待。A 在两个请求间没有计算，也仍持有模型权重，所以它的 20 GiB 账本额度不能提前归零。A 的容器清理完成后，B 才能获得预算。这里的 24 GiB 是部署分给任务池的预算，不是 H20 的物理显存，也不是 CUDA 显存硬上限。声明量与实际峰值应分别观测，不能混用。

把这个服务改成 command 模式有三种不同含义：

| 做法 | 能否在两次请求之间归还 |
|---|---|
| 启动服务、全部请求、停止服务包装在一个长 GPU job 中 | 不能在中间归还，但可以作为一条 command 执行，状态持有区间几乎等于整个服务生命周期 |
| 后台服务继续常驻，每条 HTTP 请求单独算 GPU command | 不能直接成立，HTTP 客户端退出没有释放后台服务的模型及 GPU 句柄 |
| 每轮结束后卸载 GPU worker，下轮重建并加载模型，或实现完整可恢复卸载协议 | 可以设计成细粒度借还，但引入状态保存、搬运和重新初始化成本，当前官方服务未实现这种接口 |

Offload 指把状态移到 CPU 内存或其他存储，reload 指下一次使用时恢复到 GPU。对于只读推理，权重可从已有模型文件重新加载，不一定要先保存一份新权重。若要恢复训练，则还需保存优化器状态等。KV cache 是保存已计算注意力键和值的缓存，本次官方 Transformers 服务没有验证跨请求复用 KV cache，不能把持续显存都称为 KV cache。

**当前 Env 的实际限制比“显存变少”更严格。** [CommandGpuReservation.wait_return](../../../psrl/workers/env_service/gpu_execution.py#L67) 要等待 Router 确认归还。[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L207) 撤销新的设备访问后，仍检查已有 GPU 设备句柄和 CUDA 进程。CUDA 上下文是进程与 GPU 驱动维持的执行状态，权重移到 CPU 并调用 `torch.cuda.empty_cache()` 不等于这个状态和设备句柄也消失。仅这样改服务，不能保证符合当前归还协议。保守的适配方法是退出拥有 GPU 的工作进程，下次新建进程加载模型。归还失败会保留隔离状态，而不是把仍占用的容量宣布可用，见 [GpuCommandScheduler._return](../../../psrl/workers/env_service/command_gpu.py#L215)。

**运行入口。** [resident_image.py](../../../examples/inference_bench/resident_image.py) 打包官方服务、HF 模型快照和官方 seed 21 的前两条 MMLU-Pro 样本。`BASE_IMAGE` 必须已经包含可运行的 Python 3.10、Torch 及其基础依赖，`TASK_PYTHON` 指向对应版本的宿主解释器。先设置各路径变量为自己的绝对路径，再在 PSRL 根目录执行：

```bash
mkdir -p "$BUILD_CONTEXT"
uv pip install --python "$TASK_PYTHON" --no-deps \
  --target "$BUILD_CONTEXT/python-extra" -r examples/inference_bench/resident-requirements.txt
python examples/inference_bench/resident_image.py \
  --upstream "$BENCH_REPO" --model-snapshot "$MODEL_SNAPSHOT" \
  --context "$BUILD_CONTEXT" --base-image "$BASE_IMAGE"
timeout 420 docker build --network none -t "$TASK_IMAGE" "$BUILD_CONTEXT"
python examples/inference_bench/resident_ab.py --prepare --root "$RESULT_DIR" \
  --node-ip "$ENV_NODE_IP" --gpu "$GPU_INDEX" --image "$TASK_IMAGE"
```

启动前按集群运行指南验证同一镜像内 CUDA 可执行。`--prepare` 生成固定任务与 OFF 静态 GPU 映射，不会运行请求。OFF 使用指定 baseline 的隔离 checkout，保留其原有静态分配和进程内并发保护。现有必要运行补丁包括传递 `override_gpus`、取消节点级镜像清理，必须保留并披露。运行阶段将脚本目录加入导入路径，避免从当前目录误导入 ON 的源码：

```bash
REPO="$PWD"
cd /tmp
PYTHONPATH="$BASELINE_REPO:$VERIFICATION_DEPS:$REPO/examples/inference_bench" \
  timeout 1000 python "$REPO/examples/inference_bench/resident_ab.py" \
  --root "$RESULT_DIR" --node-ip "$ENV_NODE_IP" --gpu "$GPU_INDEX" --mode off
PYTHONPATH="$REPO:$VERIFICATION_DEPS:$REPO/examples/inference_bench" \
  timeout 1000 python "$REPO/examples/inference_bench/resident_ab.py" \
  --root "$RESULT_DIR" --node-ip "$ENV_NODE_IP" --gpu "$GPU_INDEX" --mode on
```

两组之间必须确认前组物理清理通过。脚本默认每组两个 episode、没有重试，可显式配置任务数和最多两次尝试，组内上限 900 秒，结果保存源码来源、账本、GPU 采样与清理状态。每个 episode 使用固定服务完成两条真实请求，输出上限改为 16 token，上下文上限 2048，精度 BF16。这是有界框架实验，不是官方完整评测，也没有语言模型 agent 或 PSRL 策略优化器。reward=1 只表示两次请求及服务复用的功能校验成功，不代表答题全对。

默认 unit 预算读取所选空闲 GPU 的现场总容量。只有要复现 24 GiB 小配额排队验证时，才给两组运行命令显式增加 `--pool-gpu-mb 24576`，其结果不作为等资源性能结论。旧压力数据使用 `resident_analyze.py --expect-serialized` 审计，普通容量对照不要求 ON 串行。总容量与实时余量见 [unit 容量说明](gpu-memory-admission.md#8-unit-总容量动态余量与装箱气泡)，新 episode 可信抵扣路径见 [记账修复](gpu-memory-admission.md#9-episode-归属记账修复)。


**整卡高压力协议。** 使用新目录，在上面的 `--prepare` 和两组运行命令中都增加同一组选项：

```bash
--episodes 7 --warm-episodes 2 --resident-idle-s 30 --attempts 2 \
--gpu-identity-backend ebpf
```

这组参数先启动两个模型服务，两者 readiness 通过后同时提交其余五项。每项完成两道固定题，中间保持服务驻留 30 秒。并发槽均设为 7，ON 的 unit 容量取现场整卡容量，每项预约 20 GiB，不传低配池参数。仅当首轮失败原因明确包含 CUDA OOM 时，在首轮全部结束后最多重试一次，其他故障不自动重试。ON/OFF 使用完全相同的到达和重试规则。eBPF 是可选的宿主身份核验后端，使用前必须通过节点支持检查，不能在不支持的平台上照抄。

两组物理清理后进行纯 CPU 审计：

```bash
python examples/inference_bench/resident_analyze.py \
  --root "$RESULT_DIR" --output "$AUDIT_OUTPUT" --prefix episode-pressure --expect-pressure
```

该审计要求每组最终完成 7 项、14 次成功请求，保留 OOM 次数，比较逐样本文本与 token 数，还检查等待、归属抵扣和物理回收。它不会用失败尝试的部分请求充当额外完成量。计时区分驱动初始化、任务及最终清理，环境激活和模块导入发生在驱动计时开始前，不能把这些时间隐藏地归为账本开销。具体一次实测与边界见 [压力实验记录](../experiments/gpu-benchmarks/FINDINGS.md#episode-记账修复后的整卡压力对照)。

## 真实环境验证

先执行项目的[集群预检](../../psrl/psrl-run-guide/cluster-preflight.md)，核对 Python 来源、目标节点、GPU UUID/minor、Docker 与依赖。准备器不是镜像安装器，现有镜像缺库时应在独立环境补齐。

设置 `ENV_NODE_IP` 为实际执行节点地址，`GPU_INDEX` 为现场核验的空闲独立卡，`RESULT_DIR` 为新输出目录。在该节点的仓库根目录运行：

```bash
python -m examples.gpu_benchmarks.run \
  --task "$TASK_DIR" --output "$RESULT_DIR" \
  --node-ip "$ENV_NODE_IP" --gpu "$GPU_INDEX" \
  --gpu-memory-mb 8192 --agent oracle --timeout 600
```

默认 oracle 只适用于具有 solution 的任务。`--agent nop` 验证未提交时的评分行为。真实模型可指定 `--agent terminus-2 --model-url "$MODEL_URL" --model "$MODEL_NAME" --max-turns 4`，模型推理资源另行分配。

这条命令使用现有 PSRL runner 与真实 EnvRouter，但不启动训练器或 TITO 模型服务。因此 oracle 通过证明容器、任务、评分和清理，不代表真实策略轨迹或优化器步骤通过。`evidence.json` 区分原始任务 reward 与基础设施异常，记录来源、Ray 节点、租约和物理清理。

`--gpu-memory-mb` 是任务声明量。池容量从空闲卡现场总容量读取，不能把任务声明量同时作为池容量，否则驱动常驻占用可能让满额请求永久等待。启动前拒绝已有计算进程，结束后检查本次所属容器以及 GPU 进程，不停止共享 Ray 或清理共享镜像。

## 接入训练

准备互不重叠的训练和验证 Parquet。设置 `TRAIN_FILES`、`VAL_FILES`、`HF_MODEL_PATH`、任务输出目录，以及现有训练 launcher 所需的节点、模型角色和 Env 单元参数。必须显式设置 `AGENT_NODE_IPS` 为允许执行容器的节点地址列表，逗号分隔。显式空值表示允许所有存活节点：

```bash
bash examples/gpu_benchmarks/train.sh kernel_bench \
  psrl.env_service.node_policy=any \
  "psrl.env_service.units=$ENV_UNITS"
```

`ENV_UNITS` 是 Hydra 列表字符串，每项声明 `unit_id`、`node_ip`、`cpu_quota`、`memory_mb`、`gpu_indices`、`gpu_memory_mb`、`max_containers` 和 `gpu_reservation: episode`。环境卡与模型训推卡必须现场核验无重叠。公共 launcher 复用 science 的训练配置，只替换数据、agent loop、奖励入口和项目输出名。recipe 可选择 `kernel_bench`、`mle_bench`、`inference_bench`、`autoresearch`、`re_bench`、`flashinfer_bench`。混合任务使用[混合准备器](../../../examples/gpu_benchmarks/mix.py)，并显式覆盖其生成的 agent 配置路径，见[本地协议入口](../milestones/science-gpu-full-onboarding.md#运行入口)。

CPU 与主存需求包含主容器、独立 verifier 和预留 sidecar，不能只按 task.toml 的主容器数值设置 unit。准备后用 [demand_from_task](../../../psrl/workers/env_service/harbor_resources.py#L21) 核对单个任务能否装入空 unit，再初始化策略模型，避免到真实派发阶段才发现资源形状不匹配。

为了避免题目元数据和预算脱节，准备的 task 路径必须可被实际环境 worker 访问。若调整任务的 agent/verifier 时限，同步调整对应 `agent_config.yaml`，否则外层 runner 预算可能更早结束任务。只用一题做集成检查时不要把同题 train/val 写成独立泛化评测。

## 评分与数据隔离

KernelBench 使用上游 `eval_kernel_against_ref`，3 次随机正确性检查与 10 次性能测量，保留完整 kernel 结果。训练默认奖励为正确性，速度另存，未声称这个工程检查等同完整 `fast_p` 排行榜。

MLE 使用官方 registry 和 `grade_csv`，训练奖励为 `above_median`，同时保留原始分数与奖牌字段。选择这个有界奖励是 recipe 约定，不覆盖官方分数。

InferenceBench 保留上游 reward，PSRL 侧要求质量、有效性和 judge 通过，并拒绝污染或禁用行为标记。缺 judge 输出不能只凭速度训练。

autoresearch 将提交脚本放到新 verifier，使用固定 `prepare.py` 重放并捕获上游真实评估调用，忽略 agent 打印的分数。奖励映射为 `1/(1+val_bpb)`。此类候选代码与评估在解释器内交互，沿用上游代码执行的信任边界，不声称已完成对恶意 Python 的防篡改安全审计。

## KernelBench command 与性能测量

**局部核对范围**：2026-11-04，`cd09114` 加本次未提交代码。KernelBench 新增显式 direct command 入口，独立 verifier 也经过 GPU 准入和归还。其他 benchmark 保留默认 episode。实现存在不代表远端已合并、集群默认部署或动态换卡已完成。

```text
[K1] command CLI / AgentLoop 配置
  ↓ 独立单卡、direct、max_containers=1
[K2] science runner → EnvRouter 租约 → 受限设备容器
  ├→ Agent 的 psrl-gpu 命令使用已有受管执行接口
  └→ [K3] Harbor 独立 KernelCommandVerifier
           ↓ 文件操作委托原环境
         [K4] verifier shell 调用 → exec_gpu → Router 准入、设备授权
           ↓ 原 KernelBench evaluator 执行，确认物理归还
         [K5] Harbor 解析奖励、收集日志 → 清理容器及会话租约
```

图例：图里每个部分对应的源码位置。

| 编号 | 实现与职责 |
|---|---|
| K1 | [command 配置](../../../examples/kernel_bench/command_agent_config.yaml)、[validate_units](../../../examples/kernel_bench/verifier.py#L8) 限定独立单卡、direct、单会话，拒绝模型共享与 MPS |
| K2 | [run](../../../examples/gpu_benchmarks/run.py#L33) 选择持有方式与 verifier 插件，science runner 把 HarborConfig.verifier_import_path 交给 JobConfig |
| K3 | [KernelCommandVerifier](../../../examples/kernel_bench/verifier.py#L43) 使用 Harbor 正式插件入口，保留原日志和评分解析 |
| K4 | [ManagedVerifierEnvironment.exec](../../../examples/kernel_bench/verifier.py#L35) 把 shell 调用交给 exec_gpu，并使用 overlap_return=False 等待归还 |
| K5 | [evaluate](../../../examples/gpu_benchmarks/evaluate.py#L10) 返回正确性与速度指标，原 Harbor 逻辑解析奖励，runner 收尾 |

这些代码如果有理解上的问题，可以随时追问。

**跟着图读源码**：K1 显式选择 command 并检查资源条件，K2 获得租约后才创建容器。K3 仅包装独立评分环境，上传下载等文件操作继续委托原实现，K4 的命令执行必须获准并完成物理归还才返回。退出码沿用 Harbor 的处理语义，不把 chmod 等准备命令改成新的评分规则。K5 保留原 evaluator 的正确性、速度测量和奖励解析。取消与超时由现有 GPU 执行器和 Harbor 外层期限共同处理。

插件要求 separate verifier，以及其预置或只读挂载的 `/tests`。当前 Harbor 自定义 verifier 工厂没有透传默认 verifier 的 skip_tests_upload 参数，因此插件明确跳过重复上传。此接口不用于共享 Agent 环境、需要动态上传测试的 verifier。

在前文已准备 TASK_DIR、RESULT_DIR、ENV_NODE_IP、GPU_INDEX 后，从仓库根目录运行有界 oracle 验证：

```bash
python -m examples.gpu_benchmarks.run --task "$TASK_DIR" --output "$RESULT_DIR" \
  --node-ip "$ENV_NODE_IP" --gpu "$GPU_INDEX" --gpu-reservation command --timeout 360
```

模型 Agent 的 PSRL 接线使用：

```bash
GPU_RESERVATION=command bash examples/gpu_benchmarks/train.sh kernel_bench
```

后者仍需前文的模型、TRAIN_FILES、VAL_FILES、输出与 Env units 配置。所有 unit 应显式配置 command、direct、一张独立环境 GPU 和 max_containers=1，不允许模型借卡或共享绑定。默认配置保留 episode。首版限制单会话，避免同一 Router 内该 unit 上其他会话干扰测速。会话名额仍被占用，因此目前不能让该 unit 的多个 KernelBench 会话交替填补空档；此次先完成准入与归还链路，尚未实现多会话之间独立的性能测量锁，也没有动态换卡。

**显存够用不等于适合测速。** SM 是流式多处理器，负责执行 GPU 线程块。其他进程不仅可能竞争 SM，也可能争用显存带宽、缓存与功耗预算。少量显存的任务也能占满计算资源，相同显存占用既不是公平测速的必要条件，也不是充分条件。

GPU utilization 主要反映采样区间内设备是否忙于执行工作，不等于剩余 SM 百分比。低利用率快照不能保证整个评测期间都没有竞争，不能用 `耗时 / (1 - 利用率)` 简单校正。还应控制 GPU 型号、运行栈、输入与精度、预热、频率、温度和功耗条件。[NVIDIA CUDA 性能指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

教学假设：参考程序 10 ms，候选 5 ms，speedup=2。如果只有候选测量时受干扰变成 15 ms，speedup 变成约 0.667。同样的代码会被误判为变慢。即使参考与候选都受干扰，也不保证同比例变慢，二者可能分别受计算和带宽限制。

```bash
python -c 'print("clean:", 10/5); print("contended:", 10/15)'
```

该 CPU 算术演示不模拟 GPU 调度。当前本地 evaluator 的 reward 是 correctness，speedup 单独记录。竞争直接污染速度指标及 Agent 的优化判断，不能说本地奖励已经按速度计算。官方性能分析还包括正确性与速度阈值指标，见[KernelBench 评测说明](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md)。

**保护边界**：单会话只限制同一 Router 管理的资源，不构成平台级独占。外部进程、其他 Router 仍可能进入。正式测速应使用实际独占环境 GPU，记录设备身份和运行栈，预热并重复测量。发现外部竞争应标记性能结果受污染，在有限预算内重测或报告无有效性能分，不将污染自动当作模型失败。此次没有实现集群独占、自动污染检测或频率控制。

以下 CPU 回归用测试替身验证 K3→K4 的委托、准入调用参数、归还等待选项和取消传播，并验证真实 Harbor 插件工厂的构造契约。它不证明实际设备隔离或 GPU 性能。预期专项 10 项通过；真实实验与失败记录见[本轮 Results](../experiments/gpu-benchmarks/FINDINGS.md#kernelbench-command-接入验证)。

```bash
python -m pytest tests/examples/gpu_benchmarks/test_kernel_command.py -q
```
