# RE-Bench 聊天模型训练的原生 Env 与 PSRL 接入

> **核心结论**：`examples/re_bench/` 接入 `ai_rd_nanogpt_chat_rl/main`，使用真实 GPT-2 XL 和官方 Alpaca 划分，复用 EnvRouter、原生 EnvWorker、Harbor 和 PSRL 会话奖励链。模型在生成、裁判反馈、训练之间可持续驻留。正式评分需要宿主侧 `REPLICATE_API_TOKEN`，缺失时在启动 GPU 前拒绝正式任务。无凭据的 CUDA smoke 是单独的验证模式，永远不产生训练奖励。
>
> 核对范围为 2026-11-02，PSRL 基准提交 `affe7c2` 加本次未提交接入代码。RE-Bench 固定为 `a0cc6d4a91b20f84ebef505d5c72358f88b3b389`。这是本地适配，不等于原版两张 H100、完整研究时长和中间评分协议的复现。实机结果与未验证项见 [FINDINGS](../experiments/gpu-benchmarks/FINDINGS.md#re-bench-原生接入验证)。

## 目录

- [任务和执行路径](#任务和执行路径)
- [准备真实数据和镜像](#准备真实数据和镜像)
- [生成任务并运行](#生成任务并运行)
- [接入 PSRL 训练](#接入-psrl-训练)
- [评分与适配边界](#评分与适配边界)
- [检查与停止条件](#检查与停止条件)

## 任务和执行路径

agent 要把 GPT-2 XL 训练成更好的聊天模型。可用数据只有提示，不能拿标准回答直接监督训练。官方起始代码生成两份回答，让外部 Llama-3-8B 裁判比较，再利用偏好更新模型。最终得分由隐藏提示上的回答偏好决定。

```text
[1] prepare：官方资产 → Harbor 任务 + PSRL Parquet
    ↓
[2] GpuBenchmarkAgentLoop / run → EnvRouter 准入 → 原生 EnvWorker 容器
    ↓
[3] agent 运行官方起始代码：GPU 生成 → 等反馈 → GPU 更新
                  │ 请求文件                       ↑ 响应文件
                  └→ [4] 宿主 judge bridge → 固定 Replicate 裁判
    ↓
[5] 交付模型与 model.py → 独立 verifier → 官方 evaluate_model
    ↓
[6] reward + 有效性标记 → runner 清理容器和租约 → PSRL score
```

图例：图里各部分对应的源码位置。

| 编号 | 函数、职责与源码 |
|---|---|
| 1 | [prepare](../../../examples/re_bench/prepare.py#L10) 生成无 Compose 的原生任务，调用公共 `write_dataset` |
| 2 | [run](../../../examples/re_bench/run.py#L13) 管理单次验证与裁判生命周期，PSRL 使用 [GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L14) |
| 3 | 起始训练代码来自固定上游，执行器使用当前 [原生 EnvWorker](../../../psrl/workers/env_worker/harbor.py#L146) |
| 3 ↔ 4 | [async_run](../../../examples/re_bench/replicate.py#L14) 通过挂载的 artifacts 交换请求与响应，[serve](../../../examples/re_bench/judge.py#L83) 限制并发、调用数和超时 |
| 4 | [call_judge](../../../examples/re_bench/judge.py#L48) 只调用固定官方模型，凭据不进入镜像或任务配置 |
| 5、6 | [verify.main](../../../examples/re_bench/verify.py#L12) 调用原始评分函数，[compute_score](../../../examples/gpu_benchmarks/reward.py#L6) 拒绝无效评分与 smoke 奖励 |

跟着图读源码：先看 1 的任务资源和数据标识，再看 2 如何复用公共 runner。3 中 GPU 训练进程与 4 的宿主线程同时存在，文件请求不代表进程退出或显存归还。5 使用另一套镜像，只有它含隐藏提示及参考回答。6 的奖励传给现有 PSRL 会话，容器清理仍由公共 runner 的 `finally` 负责。若这些代码有不清楚的地方，可以随时按编号追问。

## 准备真实数据和镜像

以下命令在 PSRL 仓库根目录执行。先设置 `BENCH_REPO`、`SCORING_REPO` 为独立 checkout 路径，`ASSET_ROOT`、`BUILD_DIR` 为本次资产和构建目录，`BASE_IMAGE` 为执行节点上已有的 Python/CUDA 基础镜像。基础镜像需包含 torch、tiktoken、numpy、tqdm 及 Transformers 的其余依赖。`IMAGE_PYTHON` 必须与基础镜像 Python 主次版本和平台一致，避免把另一版本的二进制扩展复制进去。

```bash
git clone https://github.com/METR/RE-Bench.git "$BENCH_REPO"
git -C "$BENCH_REPO" checkout a0cc6d4a91b20f84ebef505d5c72358f88b3b389
git clone https://github.com/METR/task-protected-scoring.git "$SCORING_REPO"
git -C "$SCORING_REPO" checkout fb4cb7880fddf1e2b4bb103cf2002e0efaee022f
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 timeout 900 \
  python -m examples.re_bench.download --output "$ASSET_ROOT"
mkdir -p "$BUILD_DIR/python-extra"
uv pip install --python "$IMAGE_PYTHON" --target "$BUILD_DIR/python-extra" \
  --no-deps -r examples/re_bench/image-requirements.txt
python -m examples.re_bench.image --repo "$BENCH_REPO" \
  --scoring-repo "$SCORING_REPO" --assets "$ASSET_ROOT" \
  --context "$BUILD_DIR" --base-image "$BASE_IMAGE"
```

下载器使用官方固定 Alpaca revision，前 512 条仅供评分，其余 51,490 条仅保留提示。GPT-2 XL 下载完成后记录实际 revision。`assets.json` 只在模型、数据和 tiktoken 缓存齐备后写出，不能把未完成下载当作有效资产。

设置 `TASK_IMAGE` 和 `VERIFIER_IMAGE` 为两个新的本地镜像标签，在实际执行节点构建：

```bash
docker build --network none --target runtime -t "$TASK_IMAGE" "$BUILD_DIR"
docker build --network none --target verifier -t "$VERIFIER_IMAGE" "$BUILD_DIR"
```

runtime 不包含隐藏提示和参考回答。verifier 使用原始 `score.py`、官方两份参考回答及固定评分库。两套镜像均预置 `/tests/test.sh`，独立 verifier 不会自动上传任务 tests。正式 verifier 的请求通过其宿主可见的日志目录桥接。原始 evaluator 的 API 失败会转成布尔结果，本适配增加错误传播，避免裁判故障被误记为正常偏好。构建不修改共享 Python 环境。

## 生成任务并运行

设置 `TASK_DIR` 为新的任务输出目录，`RUN_DIR` 为新的实验目录，`NODE_IP` 与 `GPU_INDEX` 使用现场核实的执行节点身份和 GPU 编号。先按 [集群预检](../../psrl/psrl-run-guide/cluster-preflight.md) 确认独立空闲 GPU、Docker、CUDA、依赖来源和释放条件。

```bash
python -m examples.re_bench.prepare --repo "$BENCH_REPO" --assets "$ASSET_ROOT" \
  --output "$TASK_DIR" --image "$TASK_IMAGE" --verifier-image "$VERIFIER_IMAGE" \
  --timeout 900 --gpu-memory-mb 49152
python -m examples.re_bench.run --task "$TASK_DIR" --output "$RUN_DIR" \
  --node-ip "$NODE_IP" --gpu "$GPU_INDEX" --gpu-memory-mb 49152 --timeout 2400
```

正式模式先检查宿主环境的 `REPLICATE_API_TOKEN`，不将凭据写进任务、镜像或 Parquet。裁判只能使用指定模型和原版采样参数，默认最多 2048 次请求、20 路并发、单次 120 秒。不使用其他免费模型伪装官方裁判，也不将网络失败填成合法评分。这里默认 oracle 执行官方起始训练脚本，验证链路时可用，不能当作语言模型 agent 已学会优化方案。

没有裁判凭据时，给准备命令增加 `--smoke`，使用另一组全新 `TASK_DIR` 和 `RUN_DIR`。smoke 用真实模型与两条公开提示生成文本，再对公开提示 token 做一次语言模型 CUDA 更新。它不生成假偏好，不执行完整 RL 反馈，不保存官方 benchmark 成绩。`smoke_valid=1` 只标记功能检查，`reward` 固定为 0。

`49152` 是单任务声明显存 MiB，不是整块 GPU 的池容量。公共运行器读取现场设备总容量配置 unit。任务使用 episode 预约，模型是否在内部卸载由任务代码决定，当前没有自动 offload/reload。

## 接入 PSRL 训练

准备器在任务旁生成同名 Parquet，`agent_name=gpu_benchmark`，正式任务 `data_source=re_bench`。它复用现有会话轨迹与奖励链，没有另写绕过 Env 的训练入口。

先设置 `REBENCH_JOBS_DIR` 为本次运行专属目录，并确保执行节点的裁判进程可见它。在该节点用独立终端启动有界桥接进程，再运行已有 PSRL 配置：

```bash
python -m examples.re_bench.judge --jobs-dir "$REBENCH_JOBS_DIR" \
  --duration 3600 --max-calls 2048
```

```bash
TRAIN_FILES="$TRAIN_PARQUET" VAL_FILES="$VAL_PARQUET" \
  bash examples/gpu_benchmarks/train.sh re_bench
```

第二条命令还需要已有 PSRL 的策略模型、数据、输出和 Env units 配置，具体沿用 [公共训练 recipe](gpu-benchmark-recipes.md#接入训练)。`REBENCH_JOBS_DIR` 必须 export，agent 配置从该变量读取目录。跨执行节点需要各节点运行桥接器或使用明确划分的共享目录，不能多个桥接器同时消费同一请求目录。集群策略训练本次未执行，不将接口回归当作优化器训练成功。

## 评分与适配边界

正式 verifier 对 512 条隐藏提示生成回答，分别与 `gpt2-xl`、`gpt2-alpaca` 的官方参考回答比较。实际 `score.py` 返回两项胜率的平均值，范围 0 到 1，和任务说明中“相加”的文字不同，本接入以执行代码为准。例如两项胜率为 0.6 与 0.8，返回 0.7。这是教学数字，不是本次实测得分。

适配改变了运行外壳：原版声明两张 H100，本地先接单 GPU；原版有中间评分并取最好结果，本地只对最终模型评分。原生 Env 容器不联网，固定裁判请求经宿主转发。公式、隐藏划分和参考回答保留，但不能宣称完全复现原版 leaderboard 协议。提交代码属于任务执行内容，独立容器和隐藏数据拆分不构成对任意恶意代码的完整安全证明。

## 检查与停止条件

CPU 回归命令如下，包含真实本地 PSRL 奖励函数与文件传输，外部 API 使用明确标注的测试替身，不产生 API 费用：

```bash
PYTHONPATH=. python -m pytest tests/examples/gpu_benchmarks/test_re_bench.py \
  tests/examples/gpu_benchmarks/test_recipes.py -q
```

真实验证只要求单次任务执行、独立 verifier、Env 记账和物理清理的必要证据。数据和模型下载不占 GPU，正式模式缺凭据时在 GPU 启动前停止。smoke 完成后不增加训练步数追求效果，也不据此声称已验证外部裁判等待时的 GPU 借还收益。失败日志和资源清理结果见 [接入验证记录](../experiments/gpu-benchmarks/FINDINGS.md#re-bench-原生接入验证)。
