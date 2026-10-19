# PSRL 框架与相关知识索引

一句话结论：这里收录 PSRL 训练框架本身的运行/源码知识，以及理解它所必须懂的强化学习算法与推理系统知识（GAE、PPO/GRPO/DAPO、RLVR、vLLM、NIXL 等）——两者放在一起，因为看 PSRL 代码经常要跳到这些底层概念才能看懂。

## 目录

- [PSRL 框架本身（运行、源码地图、Agent Loop）](#psrl-框架本身运行源码地图agent-loop)
- [RL 算法与训练机制（PSRL 用到的算法知识）](#rl-算法与训练机制psrl-用到的算法知识)
- [推理与系统机制（PSRL 依赖的底层框架）](#推理与系统机制psrl-依赖的底层框架)
- [延伸](#延伸)

## PSRL 框架本身（运行、源码地图、Agent Loop）

| 文档 | 内容 |
|---|---|
| [psrl-run-guide.md](psrl-run-guide.md) | 已拆分为 5 篇子文档；原文件为阅读入口。 |
| [PSRL 基础运行与示例脚本](psrl-run-guide/getting-started.md) | 以 DAPO 示例说明安装、模型与数据准备、启动参数和单机 Demo；执行前先核对集群检查专题中的环境约束。 |
| [PSRL 的 Ray 调度与生成长尾](psrl-run-guide/ray-and-rollout.md) | 分别说明 Ray 的角色、资源调度流程和 rollout 长尾，区分集群调度职责与模型计算职责。 |
| [PSRL 日志监控、批量配置与进程排查](psrl-run-guide/monitoring-and-batches.md) | 从训练日志提取指标，理解 train_prompt_bsz 的配置映射，并确认训练进程所在节点及运行状态。 |
| [PSRL 单机与多机启动故障复盘](psrl-run-guide/startup-incident-reviews.md) | 保留 IP、数据格式、环境变量覆盖及上下文长度问题的历史故障链、修复过程与验证边界。 |
| [PSRL 共享环境与集群检查](psrl-run-guide/cluster-preflight.md) | 运行前核对源码与依赖来源、IP/Ray、临时路径、Docker、GPU 身份、数据和验证预算；2026-10-13 的证据只代表当时的条件。 |
| [psrl-training-script-source-map.md](psrl-training-script-source-map.md) | 跑一次 `fsdp_qwen2.5_3b_laps.sh` 涉及的全部源码文件地图：启动脚本/Hydra 配置、Trainer 核心（`main_ppo.py`/`ray_trainer.py`）、Parameter Server（NIXL 权重同步）、Rollout Gateway/SMG/RolloutCoordinator、Agent Loop、FSDP2/Ulysses/GRPO/rollout correction（本仓库 vs 外部 `verl`）、reward.py/DAPO/数据集，逐文件标注角色与本仓库/外部归属 |
| [psrl-agent-loop-harbor-tool-calling.md](psrl-agent-loop-harbor-tool-calling.md) | rollout 阶段模型怎么"调用工具"：Agent Loop 架构（`PSRL_AgentLoopWorker`/`SessionAgentLoop`/`SciAccelAgentLoop`）、第三方框架 Harbor（`Terminus2`，纯文本 JSON 协议驱动 Docker 容器里的真实终端）、TITO（Token-In-Token-Out）怎么在旁路记录 token 重建训练样本、`multi_thinking` 模板作用 |
| [dapo-training-collapse-diagnosis.md](dapo-training-collapse-diagnosis.md) | 一次 DAPO+Qwen2.5-3B 训练在 step~115 发生熵崩溃→长度打满→reward 归零的不可逆连锁崩溃复盘，含完整时间线和各阶段指标特征 |
| [psrl-env-service-envworker-design.md](psrl-env-service-envworker-design.md) | 从一道修代码题理解 EnvWorker：谁执行命令、谁分配资源、为什么两个进程各限 8 个任务仍会超载；用数字解释 unit、显存预算、两个 Gate 和容器清理，再区分 SMG、SessionRouter 与环境服务，以及现有实现和提案 |
| [../sciaccel-grpo-oom-analysis.md](../sciaccel-grpo-oom-analysis.md) | SciAccel GRPO+Qwen3.5-4B 训练在第一步 backward 撞 OOM 的根因分析（多轮轨迹累积长度 + 过大的训练 token 预算），以及后续用 `TRAIN_SP` 4→8 + 压低 `max_response_length`/`max_model_len` 重跑 5 步验证修复有效的结果（全程无 OOM，显存稳定在 82GB/95GB；但验证集 `acc/mean@1=0.0`，暴露出独立的训练效果问题） |

## RL 算法与训练机制（PSRL 用到的算法知识）

| 文档 | 内容 |
|---|---|
| [gae-bias-variance-tradeoff.md](gae-bias-variance-tradeoff.md) | GAE 用参数 $\lambda$ 把"只信任 1 步 TD 残差"（低方差、高偏差）和"信任整条轨迹的蒙特卡洛回报"（低偏差、高方差）两种优势函数估计方式做指数加权平均，在两个极端之间连续调节偏差与方差 |
| [ppo-grpo-dapo-sample-requirements.md](ppo-grpo-dapo-sample-requirements.md) | PPO / GRPO / DAPO 核心区别在 advantage 怎么算，决定了"凑齐多少样本才能做一次更新"：PPO 用 critic 给每条独立轨迹算 baseline；GRPO/DAPO 没有 critic，baseline 是组内统计量，必须凑齐一个 prompt 的完整 `n` 条 response |
| [rlvr-explained.md](rlvr-explained.md) | RLVR（可验证奖励的强化学习）与 RLHF 的核心区别：reward 来自确定性规则函数还是训练出的 Reward Model |

## 推理与系统机制（PSRL 依赖的底层框架）

| 文档 | 内容 |
|---|---|
| [vllm-and-training-parallelism.md](vllm-and-training-parallelism.md) | 已拆分为 5 篇子文档；原文件为阅读入口。 |
| [vLLM 推理机制：缓存、调度与合批](vllm-and-training-parallelism/inference-mechanisms.md) | 从 prefill/decode 的计算差异出发，理解 KV 缓存、PagedAttention、前缀复用、连续合批和选择性合批；原有算例与性能边界保留。 |
| [训练并行、通信量与带宽分析](vllm-and-training-parallelism/training-parallelism.md) | 按切分对象理解训练并行，再用每卡通信量和互联带宽判断并行策略应如何排布。 |
| [PSRL 训推协作与 PPO、GRPO 基础](vllm-and-training-parallelism/rl-training-workflow.md) | 串联轨迹生成、概率重算、优势估计、参数更新与权重同步，并保留 PPO/GRPO 的逐步推导和长尾分析。 |
| [PSRL 分组采样、DAPO 与训练损失](vllm-and-training-parallelism/rl-loss-and-sampling.md) | 解释同一 prompt 的分组采样、DAPO 改动、熵与 KL 项，以及多轮 response_mask、奖励标签和 rollout correction。 |
| [推理与训练并行：公式速查及实验记录](vllm-and-training-parallelism/experiments-and-reference.md) | 集中查阅符号、公式、机制对应的实验脚本与历史测量条件；历史结果不代表当前环境的重新验证。 |
| [hf-config-context-window.md](hf-config-context-window.md) | HuggingFace `config.json` 字段逐一解释（`max_position_embeddings`/`rope_theta`/`rope_scaling`/GQA 相关字段等），RoPE 与长上下文扩展原理；`max_prompt_length`/`max_response_length`/`max_model_len` 三者的关口原理与为什么要留余量；三次复盘：① `max_model_len > max_position_embeddings` 导致 vLLM 启动崩溃，② 改参数后 `max_model_len` 刚好卡满上限、没给 multi-turn 累积长度留余量导致运行时持续报错训练卡死，③ 留了余量之后训练侧（backward）又撞了显存 OOM，牵出 `max_turns` 如何通过序列总长度影响训练激活显存 |
| [nixl-explained.md](nixl-explained.md) | NIXL（NVIDIA Inference Xfer Library）是什么：底层基于 UCX 自动挑选最快传输方式（共享内存/GPU IPC/RDMA）的点对点内存搬运库，PSRL 用它做训练/生成集群间的权重同步与 KV cache 迁移，不占用 Ray 对象序列化开销 |

## 延伸

- [知识文档总索引](../README.md)
- [PSRL 项目自身文档目录](../overview/index.md)（`psrl/docs/`，不重复收录到这里）
- vLLM 机制验证实验脚本（本地资料，未随仓库发布）
