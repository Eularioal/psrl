# PSRL 基础运行与示例脚本

> **核心结论**：以 DAPO 示例说明安装、模型与数据准备、启动参数和单机 Demo；执行前先核对集群检查专题中的环境约束。

## 目录

- [项目结构速览](#项目结构速览)
- [安装](#安装)
- [运行前准备](#运行前准备)
- [逐段拆解示例脚本 `examples/dapo_trainer/qwen2.5_3b_fsdp.sh`](#逐段拆解示例脚本-examplesdapo_trainerqwen25_3b_fsdpsh)
  - [1. 基本变量与路径](#1-基本变量与路径)
  - [2. 集群拓扑（几节点、几卡，谁做生成谁做训练）](#2-集群拓扑几节点几卡谁做生成谁做训练)
  - [3. 算法超参（DAPO/GRPO 相关）](#3-算法超参dapogrpo-相关)
  - [4. 核心 PSRL 命令（真正的启动命令）](#4-核心-psrl-命令真正的启动命令)
  - [5. 运行时会发生什么](#5-运行时会发生什么)
- [监控指标](#监控指标)
- [实操建议](#实操建议)
- [延伸阅读](#延伸阅读)
- [单机 Demo 脚本与常用 bash 语法：`qwen2.5_3b_fsdp_singlenode_demo.sh`](#单机-demo-脚本与常用-bash-语法qwen25_3b_fsdp_singlenode_demosh)
  - [脚本做了什么](#脚本做了什么)
  - [涉及的 bash 语法](#涉及的-bash-语法)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。旧的中文章节编号已改为主题标题，正文引用已改为可点击链接。

[返回专题索引](../psrl-run-guide.md)

运行前先阅读[共享环境与集群检查](cluster-preflight.md)；文中的历史地址、依赖和机器状态需要重新核对。

## 项目结构速览

- 代码位于 `psrl/`（本仓库根目录下的 `psrl/psrl` 才是 Python 包 `import psrl` 对应的目录，`psrl/` 顶层是仓库）。
- 训练入口是 `python -m psrl.trainer.main_ppo`，用 **Hydra** 管理配置，配置基类在 `psrl/trainer/config/ppo_trainer.yaml`（FSDP）和 `ppo_megatron_trainer.yaml`（Megatron）。
- `examples/` 下按算法/任务分目录（`dapo_trainer`、`ppo_trainer`、`grpo_trainer`、`retool`、`mini_swe` 等），每个目录里是拼好命令行参数的 shell 脚本，直接 `bash` 运行即可。
- `examples/ray/ray_start.sh` 是拉起多机 Ray 集群的辅助脚本。

## 安装

**Docker（推荐，省去环境搭建）：**

```bash
docker build --progress=plain -f docker/Dockerfile -t psrl:latest .

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/home/psrl" \
  -it psrl:latest
```

**源码安装：**

```bash
# Rust（编译 SMG 的 gRPC gateway 需要）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

conda create -n psrl python=3.12
conda activate psrl

bash scripts/install_basic.sh      # PyTorch/vLLM/veRL/SMG 等核心依赖
bash scripts/install_nixl.sh       # 可选：RDMA 权重同步
bash scripts/install_megatron.sh   # 可选：Megatron 训练后端
bash scripts/install_lmcache.sh    # 可选：KV cache 多实例复用

python -m pip install -e .
```

安装完成后可用下面的命令逐项验证（PyTorch/CUDA、`psrl`、`vllm`、`ray`、`verl` 等能否正常 import），装完确认无报错再进入下一步。

## 运行前准备

0. **`PSRL_WORKSPACE`**：跑任何示例脚本前必须自己 `export` 好这个变量，脚本不会替你猜。它指向的目录要同时包含三样东西：
   - `env/`：一般是软链接，指向团队共享的环境激活脚本所在目录（见下方「`env/psrl.sh` 是什么」）；
   - `models/<模型名>`：本地模型权重；
   - `data/<数据集>/{train,test}.parquet`：训练/验证数据。

   `checkpoint/`、`logs/` 这两个子目录不需要预先创建，脚本运行时会自动 `mkdir -p`。

   **为什么不直接在脚本里写死这个路径**：`qwen2.5_3b_fsdp.sh` 是进仓库、给所有用户共用的正式示例脚本，不同人跑的 workspace 必然不同，写死就意味着别人 checkout 下来直接跑不了、还得手改代码——这也是仓库不允许硬编码绝对路径这条约束的直接体现。同理，`qwen2.5_3b_fsdp_singlenode_demo.sh` 里之所以能直接写死 `PSRL_WORKSPACE=`，是因为那份脚本本来就是某个人自用的本地 demo，不追求跨用户复用。另外，脚本用 `: "${PSRL_WORKSPACE:?msg}"` 强制要求显式赋值、不给默认值，也是刻意为之：[多机启动脚本修复](startup-incident-reviews.md#让-qwen25_3b_fsdpsh-跑通-2-节点实验脚本改了什么)的那次 `FileNotFoundError` 根因正是上游脚本给了一个"隐式默认值"、悄悄覆盖了调用者的意图，所以这里选择"缺失就立刻报错"而不是"猜一个默认路径"。
1. **模型**：脚本里用的是 `Qwen2.5-3B-Instruct`，下载到 `${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct`。
2. **数据**：需要 `${PSRL_WORKSPACE}/data/gsm8k/{train,test}.parquet`，PSRL 复用了 veRL 的数据预处理脚本（GSM8K/MATH 等），按 veRL 文档转换即可。
3. **Ray 集群**：准备一个 hostfile（第一行是 head 节点），再执行：

   ```bash
   bash examples/ray/ray_start.sh ${PSRL_WORKSPACE}/hosts/16GPUs
   ray status   # 确认看到 2 节点 / 16 GPU
   ```

   `ray_start.sh` 做的事：先用 `pssh` 在所有节点上 `ray stop --force` 清理残留进程，再在 head 节点跑 `ray start --head --port=8887 --dashboard-port=8265`，其余节点跑 `ray start --address=<head>:8887` 加入集群。

## 逐段拆解示例脚本 `examples/dapo_trainer/qwen2.5_3b_fsdp.sh`

这是官方 Quick Start 用的脚本，跑的是 DAPO 算法在 GSM8K 上训练 Qwen2.5-3B，用法：

```bash
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh        # 默认 staleness=3
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh 2      # 指定 staleness=2
```

脚本内容可以分成 5 块：

### 1. 基本变量与路径

```bash
staleness=${1:-3}                 # 第1个位置参数：staleness（异步程度），默认3
experiment_name=DAPO-Qwen2.5-3B-fsdp-staleness_${staleness}

: "${PSRL_WORKSPACE:?...}"        # 要求调用者已 export，未设置则直接报错退出
_psrl_workspace=${PSRL_WORKSPACE}
source ${PSRL_WORKSPACE}/env/psrl.sh   # 见下方「env/psrl.sh 是什么」，会把 PSRL_WORKSPACE 覆盖掉
export PSRL_WORKSPACE=${_psrl_workspace}   # 所以要恢复回调用者设的值

LOCAL_IP=$(python3 -c "...")      # 脚本内自算可路由 IP，无需调用者手动 export
HF_MODEL_PATH=... TRAIN_FILE=... TEST_FILE=...   # 都基于 ${PSRL_WORKSPACE} 拼路径
```

**`env/psrl.sh` 是什么**：它不是 PSRL 项目自己的代码，是团队共享的一份环境激活脚本，纯环境变量设置、没有业务逻辑，主要做三件事：
1. `source` 团队公共的 `activate` 脚本（负责切 conda 环境、清代理设置，细节见 [[activate-script-notes|activate 脚本说明]]），再 `conda activate` 到项目用的具体 env；
2. 设置一批日志/调试开关（`HYDRA_FULL_ERROR`、`PSRL_LOGGING_LEVEL`、`VLLM_LOGGING_LEVEL` 等）和 `LD_LIBRARY_PATH`（保证能找到 conda env 里的 `libstdc++`/cudnn/torch 动态库）；
3. 设置一大批 UCX/NCCL 调优参数（网卡设备列表、算法选择等），这些是跟机器/机房网络拓扑绑定的，用来让 NIXL/NCCL 走 RDMA 网络而不是慢速以太网。

**坑点**：`activate` 脚本里会无条件把 `PSRL_WORKSPACE` 重新导出成它自己默认的工作区路径——这是它原本设计给自己用的默认值，不知道你想跑的是哪个 workspace。所以任何脚本只要 `source` 了 `env/psrl.sh`，就必须像上面这样在 `source` 前后手动保存/恢复 `PSRL_WORKSPACE`，否则后面所有基于它拼出来的模型/数据/日志路径都会指向错误目录（详见[多机启动脚本修复](startup-incident-reviews.md#让-qwen25_3b_fsdpsh-跑通-2-节点实验脚本改了什么)）。

### 2. 集群拓扑（几节点、几卡，谁做生成谁做训练）

```bash
NNODES=2; NGPUS_PER_NODE=8          # 总共2节点×8卡=16卡
GEN_NNODES=1                        # 1节点专门做 rollout(生成)
TRAIN_NNODES=1                      # 1节点专门做训练
GEN_TP=1; GEN_PP=1                  # 生成侧张量并行/流水并行都是1（每张卡独立跑一个vLLM实例）
GEN_INSTANCES = (GEN_NNODES*NGPUS_PER_NODE)/(GEN_TP*GEN_PP)  # =8，即8个独立生成实例
```

这体现了 PSRL 的核心设计：**训练集群和生成(rollout)集群物理分离**，通过 Parameter Server 异步同步权重，而不是像同步 RL 那样两者互相等待。

### 3. 算法超参（DAPO/GRPO 相关）

```bash
adv_estimator=grpo                  # 用 GRPO 计算 advantage
clip_ratio_low=0.2; clip_ratio_high=0.28   # DAPO 的非对称 clip 区间
enable_overlong_buffer=True         # 对超长回复做惩罚（DAPO 特有机制）
train_prompt_bsz=64; n_resp_per_prompt=8   # 64个prompt，每个采样8条回复
```

### 4. 核心 PSRL 命令（真正的启动命令）

```bash
python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \        # Parameter Server 地址
    psrl.staleness=${staleness} \           # 异步staleness上限
    psrl.ps_mode=nixl_cpu \                 # 权重同步走 NIXL(RDMA) + CPU中转
    psrl.deployment.n_rollout_instances=... \   # 上一步算出的生成实例数
    psrl.rollout_coordination.partial_rollout.enable=True \   # 开启“部分rollout”优化长尾延迟
    gen_actor_rollout_ref.rollout.* \       # 生成侧（vLLM）参数：显存占用率、TP/PP、temperature等
    train_actor_rollout_ref.actor.* \       # 训练侧（FSDP2）参数：学习率、梯度裁剪、offload等
    reward.active_managers='[dapo]' \       # 用DAPO的reward逻辑（含超长惩罚）
    data.train_files=... data.val_files=... \
    trainer.total_training_steps=200 \
    2>&1 | tee ${experiment_name}.log
```

这一整条命令本质是通过 **Hydra 的 CLI override 语法**，在 `psrl/trainer/config/ppo_trainer.yaml` 的默认配置基础上覆盖参数。参数前缀分组含义：

| 前缀 | 含义 |
|---|---|
| `psrl.*` | PSRL 自己的调度层：Parameter Server、staleness、部署拓扑、rollout 协调策略 |
| `gen_actor_rollout_ref.*` | 生成（rollout）侧模型/vLLM 配置 |
| `train_actor_rollout_ref.*` | 训练侧模型/FSDP(或Megatron)配置 |
| `reward.*` | reward 计算逻辑 |
| `data.*` | 数据集路径与格式 |
| `algorithm.*` | RL 算法（adv_estimator、KL 等） |
| `trainer.*` | 日志、保存频率、总步数等训练循环控制 |

`+xxx=yyy`（带加号）表示往配置里**新增**一个原本 yaml 里没有的字段，而不是覆盖已有字段，Hydra 的语法约定。

### 5. 运行时会发生什么

按 Quick Start 文档，一次调用的内部流程是：

1. `AgentLoopWorkers` 驱动 rollout，把生成请求通过 **SMG RolloutGateway** 转发；PSRL 的 worker selector 向 `PSManager` 查询版本/准入状态后，再分派给某个 vLLM gRPC 实例。
2. 选中的 vLLM 实例用当前策略版本生成轨迹。
3. 轨迹写入 **TransferQueue**，训练进程读写字段完成 DAPO 各阶段计算。
4. 每步训练结束后，训练侧把新权重推给 **Parameter Server**；rollout 侧在 `RolloutCoordinator` 协调下拉取新版本权重。

## 监控指标

默认用 W&B 记录，重点看：

- `critic/score/mean`、`critic/rewards/mean`：应随训练逐步上升
- `critic/advantages/mean`、`critic/returns/mean`：GRPO 的 advantage/return
- `response_length/mean`：回复长度，收敛后应趋于稳定
- `actor/entropy`：策略熵，缓慢下降代表探索→利用正常过渡
- `actor/grad_norm`：突然飙高往往意味着更新不稳定
- `perf/throughput`、`perf/mfu/actor`：吞吐与 MFU
- `training/global_step`：当前步数

## 实操建议

1. `qwen2.5_3b_fsdp.sh` 需要 **2 节点 × 8 GPU（共16卡）**，资源门槛不低。如果只是想先摸一遍流程、验证环境装对了，可以：
   - 先跑仓库自带的单测（例如 `tests/` 下的 CPU 测试）确认 `import psrl`、`ray`、`verl` 等都正常；
   - 如果只有单机多卡，可以把脚本里的 `NNODES=1`、`GEN_NNODES=1`、`TRAIN_NNODES=1`，并相应调小 `NGPUS_PER_NODE`、`GEN_INSTANCES` 等派生变量，做一次小规模验证跑通（显存不够可以调小模型或 `gpu_memory_utilization`）。
2. `examples/` 下还有 `ppo_trainer/`、`grpo_trainer/`、`retool/`（数学+代码工具）、`mini_swe/`（软件工程 agent）等更复杂的示例，跑通 DAPO 之后可以对比着看其他脚本，套路是一致的（Ray集群 → Hydra命令行）。
3. `env/psrl.sh`（脚本里 `source ${PSRL_WORKSPACE}/env/psrl.sh`）不在仓库里，需要自己按环境创建，通常放 `conda activate psrl`、`export LOCAL_IP=...`、`export PSRL_WORKSPACE=...` 等。

## 延伸阅读

- `psrl/docs/tutorial/quickstart.md`、`psrl/docs/tutorial/installation.md`、`psrl/docs/tutorial/configuration.md`
- `psrl/docs/design/staleness_control.md`（异步staleness机制）、`parameter_server.md`（权重同步）

## 单机 Demo 脚本与常用 bash 语法：`qwen2.5_3b_fsdp_singlenode_demo.sh`

### 脚本做了什么

它是 `qwen2.5_3b_fsdp.sh`（2 节点×8GPU 的 Quick Start 示例）的**单机缩小版**：只用 1 台机器上的 4 张卡（8 张里的 4 张），跑通同一套 DAPO 训练流程，目的是快速验证环境装对了、训练循环能端到端跑起来，而不是训出好模型（`trainer.total_training_steps=3`，只跑 3 步）。拓扑上 4 张卡分两组：2 张做 rollout 生成（vLLM），2 张做训练（FSDP2）；数据是 GSM8K，模型是 Qwen2.5-3B-Instruct，算法是 DAPO/GRPO。

### 涉及的 bash 语法

| 语法 | 脚本里的例子 | 含义 |
|---|---|---|
| `#!/bin/bash` | 第1行 | shebang，指定用 bash 解释器执行 |
| `set -xeuo pipefail` | 第5行 | `-x` 打印每条命令再执行（方便调试）；`-e` 任何命令出错立即退出；`-u` 用到未定义变量就报错退出（防止拼错变量名）；`-o pipefail` 管道中任一环节失败，整个管道就算失败（默认只看最后一个命令的返回值） |
| `${1:-3}` | `staleness=${1:-3}` | 参数默认值：`$1` 是第1个位置参数，没传时用 `3` |
| `export VAR` | `export PSRL_WORKSPACE` | 导出为环境变量，供后面 `python -m psrl...` 启动的子进程读取 |
| `$(...)` 命令替换 | `LOCAL_IP=$(hostname -I \| awk '{print $1}')` | 执行括号里的命令，把标准输出赋给变量（这里取本机第一个 IP） |
| `$((...))` 算术展开 | `GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / (${GEN_TP} * ${GEN_PP}) ))` | bash 内置整数运算，不需要 `expr`/`bc` |
| `unset VAR` | `unset http_proxy https_proxy` | 清除环境变量，让 Ray/gRPC 走本机直连，不走代理 |
| 行尾 `\` | 大段 `python -m psrl.trainer.main_ppo \` | 反斜杠+换行＝逻辑上一整行命令，只是拆行方便阅读 |
| `${VAR:-default}` 允许外部覆盖 | `CKPT_ROOT=${CKPT_ROOT:-${PSRL_WORKSPACE}}` | 外部已设置 `CKPT_ROOT` 就用外部值，否则用默认值 |
| `2>&1 \| tee file` | 命令最后一行 | `2>&1` 把标准错误合并到标准输出，再交给 `tee` 同时打印到终端并写入日志文件 |
