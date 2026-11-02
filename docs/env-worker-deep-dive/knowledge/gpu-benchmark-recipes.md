# GPU benchmark 的 Env 与 PSRL recipe

> **核心结论**：四个 recipe 共用 `GpuBenchmarkAgentLoop`，强制通过 EnvRouter 准入与 ManagedDockerEnvironment 执行，支持整次 episode 持有 GPU。准备器生成 Harbor 任务与 PSRL Parquet。真实运行结果单独记录，存在 recipe 不代表四个 benchmark 都已实机通过。

验证结果见 [四类 GPU benchmark 接入验证](../experiments/gpu-benchmarks/FINDINGS.md)。

## 目录

- [入口与边界](#入口与边界)
- [下载官方资产](#下载官方资产)
- [H20 与框架内资源调度对照](#h20-与框架内资源调度对照)
- [Kaggle 下载权限配置](#kaggle-下载权限配置)
- [准备任务](#准备任务)
- [episode 级任务与原生 Env 验证](#episode-级任务与原生-env-验证)
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
| 公共准备器 | [prepare.py](../../../examples/gpu_benchmarks/prepare.py) | 记录源码版本、生成任务与 Parquet |
| PSRL 循环 | [agent_loop.py](../../../examples/gpu_benchmarks/agent_loop.py) | 复用会话轨迹收集，强制 EnvRouter，不允许本地无准入回退 |
| 环境验证 | [run.py](../../../examples/gpu_benchmarks/run.py) | 私有 Ray、EnvRouter、任务执行、清理证据 |
| 奖励 | [reward.py](../../../examples/gpu_benchmarks/reward.py) | 读取 verifier 指标，拒绝非有限奖励，检查 InferenceBench 完整性门槛 |

当前只支持显式 `episode` GPU unit。服务和独立 verifier 可能跨命令使用 GPU，因此这版不开放 command 模式，不宣称阶段借还已经覆盖全部任务。这里的 Docker 后端由 EnvRouter 管理，并非绕过 Env 的裸 Docker。运行入口支持显式选择 `docker` 或 `env_worker`。原生 `EnvWorkerEnvironment` 要求预构建镜像，不解析任务 Compose 文件。两种路径均使用 EnvRouter 准入。

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

本节针对框架内资源调度实验。已有正式 recipe 与后续实验设计必须分开：当前 InferenceBench recipe 导入官方任务，保留官方评分条件。下述固定负载对照是调整方向，尚未实现或验证，不能把已有下载成功称为 H20 推理任务通过。

两个未通过的 benchmark 都停在准备阶段。MLE-bench 缺少 Kaggle 账户凭据。InferenceBench 的官方任务生成器因缺质量和速度基线而退出，后续还要求 judge 凭据及 H100。此前将这些条件概括为 H20 无法继续的阻塞过于宽泛，它们不是框架调度实验全部必需的条件。

| 名称 | 作用 | 框架内实验如何处理 |
|---|---|---|
| 官方速度基线 | 官方评估所需的参考性能文件 | 不与外部成绩比较时，可采用两组各自实测的绝对耗时和吞吐，无需照搬 H100 参考值 |
| 官方质量基线 | 防止以降低回答质量换取速度 | 固定模型、精度、服务实现和输入时，保留真实输出、完成率与正确性检查。若允许 agent 改模型或服务策略，仍需质量约束 |
| 官方 judge | 检查 agent 自由修改方案的合规性等 | 固定受控代码的资源调度实验可不使用 LLM 裁判，结果单独标为框架 workload，不能输出伪造的官方 judge_pass |
| 框架 baseline | 衡量 EnvRouter 等资源管理机制的影响 | 必须保留相同工作量、到达顺序、资源和计时边界的对照 |

H20 可以作为本地实验硬件。当前证据没有证明其存在不可运行 InferenceBench 的硬件障碍，官方模板中的 `H100!` 属于配置约束。仍需实际确认官方容器内服务启动、CUDA 算子、上下文及 KV cache 显存需求。其他 benchmark 已在 H20 成功，不等于该推理服务已经验证。

当前实现的相关位置：

- [run.py:57](../../../examples/gpu_benchmarks/run.py#L57) 核对任务声明的 GPU 类型，不匹配就拒绝运行。这是配置校验，不是 CUDA 运行失败。
- [reward.py:12](../../../examples/gpu_benchmarks/reward.py#L12) 要求官方完整性与质量门槛。固定负载实验不能沿用这个入口，再把缺失的裁判字段补成成功。
- 官方 checkout 的 `src/harbor_adapter/adapter.py` 中 `baselines()` 实际检查质量和速度文件，缺失即拒绝生成任务。上游源码不随 PSRL 仓库分发，版本见本文的 upstream.json 链接。

可以随时针对这些源码继续追问。下一步实验设计应保留已下载的 Mistral 权重和场景负载，在 H20 上固定服务代码、数据、精度及请求序列，比较任务完成量、总耗时、排队时间、失败/OOM 数和物理回收。原场景 A 含 128 个请求、名义输入 8192 token、输出上限 1024 token，存在长度浮动配置。若采用更小验证子集，必须标注，且对照两组保持一致。

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
| InferenceBench 常驻推理服务 | 模型和 KV cache 可能跨多次请求、命令驻留，是典型的跨命令保留场景。该原生后端场景尚未验证 |
| autoresearch | 一次训练及评估需要连续保留模型、优化器和 CUDA 上下文，适合验证长时间持卡。准备代码阶段未必必须保留 GPU |
| MLE-bench | 训练期间连续持卡，但数据处理、训练、提交之间是否释放取决于任务实现。当前暂缓接入数据 |
| KernelBench | 单次编译和测量通常较短，可以按命令安排。当前 recipe 仍统一使用 episode 策略 |

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

`ENV_UNITS` 是 Hydra 列表字符串，每项声明 `unit_id`、`node_ip`、`cpu_quota`、`memory_mb`、`gpu_indices`、`gpu_memory_mb`、`max_containers` 和 `gpu_reservation: episode`。环境卡与模型训推卡必须现场核验无重叠。公共 launcher 复用 science 的训练配置，只替换数据、agent loop、奖励入口和项目输出名。其他三项将最后的 recipe 名替换即可。

为了避免题目元数据和预算脱节，准备的 task 路径必须可被实际环境 worker 访问。若调整任务的 agent/verifier 时限，同步调整对应 `agent_config.yaml`，否则外层 runner 预算可能更早结束任务。只用一题做集成检查时不要把同题 train/val 写成独立泛化评测。

## 评分与数据隔离

KernelBench 使用上游 `eval_kernel_against_ref`，3 次随机正确性检查与 10 次性能测量，保留完整 kernel 结果。训练默认奖励为正确性，速度另存，未声称这个工程检查等同完整 `fast_p` 排行榜。

MLE 使用官方 registry 和 `grade_csv`，训练奖励为 `above_median`，同时保留原始分数与奖牌字段。选择这个有界奖励是 recipe 约定，不覆盖官方分数。

InferenceBench 保留上游 reward，PSRL 侧要求质量、有效性和 judge 通过，并拒绝污染或禁用行为标记。缺 judge 输出不能只凭速度训练。

autoresearch 将提交脚本放到新 verifier，使用固定 `prepare.py` 重放并捕获上游真实评估调用，忽略 agent 打印的分数。奖励映射为 `1/(1+val_bpb)`。此类候选代码与评估在解释器内交互，沿用上游代码执行的信任边界，不声称已完成对恶意 Python 的防篡改安全审计。
