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
- [API token 与命令边界](#api-token-与命令边界)
- [卸载成本与一个可运行例子](#卸载成本与一个可运行例子)
- [已有研究与工程机制](#已有研究与工程机制)
- [对 Env 的研究价值与验证边界](#对-env-的研究价值与验证边界)

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

## API token 与命令边界

本节及以下研究讨论于 2026-11-02 对照本地 `96acfcc` 核对，只补充解释，不改变 Env 实现。前文实验仍保持原提交和验证条件。

`REPLICATE_API_TOKEN` 是 Replicate 云端模型服务的账户访问凭据，环境变量是把凭据交给宿主进程的方式。它不是 GPU 驱动权限、Hugging Face 下载权限，也不是 PSRL 的训练 token。此任务使用本地 GPT-2 XL 生成两份候选回答，再请求固定的远程 Llama-3-8B 裁判返回偏好，隐藏集评分也使用该裁判。缺凭据意味着当前宿主无法认证这些请求，不能推断用户没有账户。

在 [Replicate 账户](https://replicate.com/account/api-tokens) 创建凭据，通过自己的私密配置注入宿主环境即可，不要写入仓库或聊天。服务还有模型可用性、配额与计费条件，配置 token 不等于已验证调用成功。官方说明见 [API 认证](https://replicate.com/docs/reference/http) 和 [计费](https://replicate.com/docs/topics/billing)。本地裁判桥只从宿主读取凭据，不把它交给任务容器。若改用本地裁判，需要单列实验配置并验证行为差异，不应直接称为原版评分。

沿前文全局图读源码，以下是需要补充的关键节点：

| 步骤与图中编号 | 输入、状态变化与源码 |
|---|---|
| 入口 2 | [run:22](../../../examples/re_bench/run.py#L22) 检查宿主凭据，缺失时正式模式在 GPU 启动前退出 |
| 训练 3 | 固定上游 [main:51](https://github.com/METR/RE-Bench/blob/a0cc6d4a91b20f84ebef505d5c72358f88b3b389/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L51) 初始化 CUDA 模型和 Adam，在同一 Python 进程循环生成、等待、训练 |
| 并发 3 ↔ 4 | [async_run:23](../../../examples/re_bench/replicate.py#L23) 写文件并异步等待宿主结果，等待不是退出进程，也不是卸载模型 |
| 资源 2 | [EpisodeGpuReservation:9](../../../psrl/workers/env_service/gpu_execution.py#L9) 命令结束不归还会话预算，[CommandGpuReservation:34](../../../psrl/workers/env_service/gpu_execution.py#L34) 每次调用申请，在结束后请求物理归还确认 |
| 收尾 6 | [run:156](../../../examples/gpu_benchmarks/run.py#L156) 的 finally 清理本次容器、租约并记录物理资源状态 |

上游代码已在本工作区独立 checkout 核对，表中用固定提交链接方便仓库外的读者访问。这些代码若有不清楚的地方，可以按节点追问。

需要区分三个时间范围：episode 是完整任务会话，command 是一次受管理的 GPU 调用，计算阶段是命令内部的生成或更新。官方起始程序的一次 `python rl_finetune.py` 已包含所有阶段，`await evaluate(...)` 并没有结束这条命令。

| 执行组织 | 能否释放 GPU | 需要付出的代价 |
|---|---|---|
| 当前 recipe 的 episode 预约 | 命令间继续保留预算，进程是否占显存由实际状态决定 | 简单，但等待阶段没有自动借还 |
| 整个训练包装成一个 command | 语义上可行，内部裁判等待时仍未到归还点，本 recipe 未验证此配置 | 不会仅因改成 command 就填补内部空窗 |
| 拆成生成命令、裁判请求、更新命令 | 各 GPU 命令真正退出后可释放 | 保存与恢复模型、优化器、随机状态等，或接受显式改变算法 |
| 保持进程，安全点卸载和恢复 | 可在一个命令甚至一个 episode 内归还部分资源 | 需要卸载机制、物理确认、再准入和恢复协议，目前未实现 |

因此“只能 episode”不准确。“当前起始实现具有跨等待阶段的常驻状态，现有 command 完成边界看不到这个内部空窗”更准确。把工作放到后台并让 shell 返回，也不能把仍占显存的进程当作已释放。

## 卸载成本与一个可运行例子

offload 指把需要保留的 GPU 状态搬到 CPU 内存或存储，reload 指恢复到 GPU。checkpoint 是能继续运行的状态快照，可包含模型、优化器、随机数、训练步数及程序状态，完整进程恢复还涉及 CUDA 上下文等。三者不是同一个范围。Adam 优化器会保存历史梯度的一阶与二阶统计，因此只移动模型权重可能仍留下大量显存。暂停 Python 进程也不会自然释放它的 CUDA 内存。已分配给活跃张量的内存不能靠清空缓存消除。

本任务在生成完成、等待文本偏好时存在一个值得检查的安全点：下一阶段输入是文本与裁判结果，而不是必须保留生成计算图的反向传播。但这只是源码层面的候选，仍须逐项核查活跃张量、优化器状态、异步 CUDA 工作以及恢复后的数值语义。

以下全部为教学假设，不是 H20 实测。A 和 B 的显存需求不能同时满足，B 已排队且不可中断。A 生成用 10 秒，裁判等待 W 秒，再更新 10 秒。假设搬出的有效状态为 32 GB，有效单向带宽 16 GB/s，忽略固定开销，则卸载和恢复各 2 秒。B 计算 15 秒，完成后立即恢复 A，恢复操作可与剩余裁判等待重叠。

不卸载时，A 完成时间为 `20 + W`，B 为 `35 + W`。强制卸载时，B 在 `10 + 2 + 15 = 27` 秒完成，A 在 `10 + max(W, 2 + 15 + 2) + 10` 秒完成。

| W | 策略 | A 完成 | B 完成 | 全部完成 |
|---|---|---|---|---|
| 40 秒 | 常驻 | 60 | 75 | 75 |
| 40 秒 | 卸载后借给 B | 60 | 27 | 60 |
| 2 秒 | 常驻 | 22 | 37 | 37 |
| 2 秒 | 强制卸载 | 39 | 27 | 39 |

长等待时可让 B 提前完成且不拖慢 A。短等待时，即便 B 提前，A 反而慢 17 秒，全体也慢 2 秒。这解释了为什么需要代价感知决策。`W >= 卸载 + B运行 + 恢复` 是此简化模型中不延迟 A 的充分条件，不是通用调度定理。真实系统还有队列、并发搬运、CPU 内存预算、等待预测误差和恢复争用。即使空窗长，没有等待任务或本就能共存，也未必值得卸载。

运行标准库教学模拟，不需要模型、凭据或 GPU：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/episode_offload_cost_demo.py
```

脚本比较 resident、always、cost_aware 三种决策，分别表示不卸载、每次卸载、只有足够空窗才卸载。预期长等待时后两者全部完成时间为 60，短等待时 always 为 39，另外两者为 37。本次已运行脚本，六组输出与上表及阈值策略预期一致。它假定精确知道 W，只做时间计算，不调用 Env，不验证物理释放，也不能证明实际策略能预测外部 API 延迟。

## 已有研究与工程机制

以下为查阅原始论文或官方文档后的相关工作筛选，不是穷尽检索，也不是对这些系统在 H20 上的复现。

| 工作 | 已解决的问题 | 对本项目的意义与边界 |
|---|---|---|
| [Salus，MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/d9cd83bc91b8c36a0c7c0fcca59228f2-Abstract.html) | 快速作业切换、内存共享与迭代级调度 | 证明共享并非新问题，其 TensorFlow 执行服务不能直接当作任意容器的透明卸载器 |
| [PipeSwitch，OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/bai) | 利用模型分层，把参数传输和执行流水重叠，降低训练与推理切换成本 | 与卸载恢复延迟高度相关，论文毫秒结果不能外推到本任务完整优化器状态 |
| [Singularity，Microsoft 2022](https://arxiv.org/abs/2202.07848) | 透明抢占、迁移与弹性调整 AI 作业 | 说明无需应用重写的状态管理和调度已有系统方案，不应将通用暂停恢复作为全新贡献 |
| [CRIUgpu，2025](https://arxiv.org/abs/2502.16631) | 利用驱动能力进行透明 GPU 检查点与恢复 | 可参考容器和 GPU 状态保存的机制，支持范围须按实际版本核验 |
| [GCR，FAST 2026](https://www.usenix.org/conference/fast26/presentation/zeng) | 控制与数据分离、增量检查点，降低保存恢复延迟与正常运行开销 | 直接涉及状态搬运成本，提示全量复制不是唯一基线 |
| [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) | level 1 把权重卸载到 CPU 并丢弃 KV cache，唤醒时恢复 | 已有协作式模型休眠，但不是任意 PyTorch 训练进程的卸载接口，优化器状态不在其保障范围内 |
| [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) | 在运行中的 Linux 进程内保存恢复 CUDA 状态，可结合 CRIU | 更接近透明机制，但驱动版本、IPC、容器和多进程支持需现场验证，不能只凭 H20 型号判断 |

KV cache 是推理时缓存历史 token 的注意力键值，丢弃后可能需要重新计算上下文。vLLM 的 level 2 还会丢弃权重，需要重新提供权重，不能把它当作保留训练状态的通用恢复方式。

[ZeRO-Offload](https://arxiv.org/abs/2101.06840) 也相关，但核心目标是在训练中将状态与部分计算放到 CPU，让模型在有限显存下运行。它与“外部等待期间临时让出 GPU，回来时保持会话状态”有共同技术，也有不同调度目标。

## 对 Env 的研究价值与验证边界

有研究价值，但“episode 可以 offload”本身不足以构成新的研究贡献。更具体的候选问题是：面对 agent 产生的异构任务、不可预测的工具等待和有限主存，何时让出多少显存，如何让排队任务获得收益，同时控制原任务恢复延迟并保持执行语义。这里 episode 是本项目的生命周期抽象，文献通常用 GPU preemption、context switching、checkpoint/restore、memory sharing 等术语。

可研究的协议为 `持有 → 到达安全点 → 卸载 → 物理释放确认 → 账本让出 → 排队恢复准入 → reload → 继续`。这是设计提案，当前 Env 没有这个完整状态机。当前 episode 的预算不会因任务自行 `model.cpu()` 自动缩小。要保持会话存活同时借出 GPU，需要明确把会话身份、CPU/容器预算与可回收 GPU 份额分开，且预留恢复所需的预算或定义排队上界。

值得比较的策略是常驻、每次等待卸载、阈值决策，以及在条件允许时只清临时缓存或部分卸载。指标应包括全部任务完成时间、单任务尾部延迟、有效 GPU 工作量、搬运量、CPU 内存、失败与恢复正确性，而非只看显存读数降低。若做 EnvRouter ON/OFF，还须沿用用户指定的固定提交 baseline，不能拿当前关闭开关替代。

当前已验证的是原生 Env 的真实模型 CUDA smoke，完整外部偏好训练仍待凭据，因此尚无该任务的真实等待分布或 offload 收益证据。本次仅运行上述 CPU 教学模拟，没有追加 GPU 实验。研究优先级建议先测真实阶段时间和可释放状态，再决定是否需要协作式安全点或透明进程检查点，不能为了显示优势人工延长裁判等待并当作真实结果。
