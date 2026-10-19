# `fsdp_qwen2.5_3b_laps.sh` 涉及的源码文件地图

## 一句话结论

跑一次 `examples/sciaccel_rl/fsdp_qwen2.5_3b_laps.sh` 实际触达的代码分四层：**bash 脚本拼 Hydra CLI 参数 → `psrl` 仓库自己的 Trainer/Worker 层（Ray actor 编排、PS 权重同步、rollout gateway、reward/dataset 适配）→ 外部第三方库 `verl`（真正的 FSDP2/Ulysses/GRPO loss/checkpoint 底层实现，不在这个仓库里）→ 外部第三方库 `harbor`（rollout 阶段的终端 agent，已有专门文档）**。`psrl` 本身更像一个"编排 + 适配"层，重的模型并行/优化器/loss 计算几乎全部委托给 `verl`。

## 总体分层

```
examples/sciaccel_rl/fsdp_qwen2.5_3b_laps.sh   (bash：拼所有 Hydra CLI 覆盖参数)
  └─ python3 -m psrl.trainer.main_ppo           (Hydra @hydra.main 入口)
        │  config_path="config", config_name="ppo_trainer"
        │  → psrl/trainer/config/ppo_trainer.yaml 及其 defaults 组合出的完整配置
        │
        ├─ run_ppo() → Ray.init → 起 TaskRunner (Ray actor)
        │      └─ TaskRunner.run(): 注册各 role 的 worker 类 → 建 ResourcePoolManager
        │             → 构造 PSRL_RayPPOTrainer（日志里的 <MainRayTrainer>）
        │             → trainer.init_workers() → trainer.fit()
        │
        ├─ PSRL_RayPPOTrainer.init_workers()
        │      ├─ 建各 Ray 资源池（train_pool / rollout_pool_i / validate_pool_i）
        │      ├─ 起 PS（Parameter Server）：PSManager + N×PSStorageWorker
        │      ├─ 起 RolloutGateway（Ray actor）
        │      │      ├─ 起 SMG（Rust 路由进程）—— 真正在 N 个 vLLM rollout 实例前做负载均衡/准入
        │      │      └─ 起 SessionRouter（FastAPI）—— TITO session 代理，转发给 SMG
        │      ├─ 起 RolloutCoordinator（权重版本同步/迁移）
        │      └─ 起 Agent Loop Workers（跑 SciAccelAgentLoop → Harbor，见专门文档）
        │
        └─ PSRL_RayPPOTrainer.fit()  ——— 主训练循环
               batch 拉取（PSRLRLHFDataset）
                 → agent loop 并发 rollout（Harbor 终端 agent，产出 TITO token 数据）
                 → reward_manager 打分（DAPORewardManager → reward.py::compute_score）
                 → rollout_correction 重要性采样修正
                 → compute_advantage（GRPO）
                 → actor_wg.update_actor()（FSDP2 前反传，委托给 verl）
                 → _save_checkpoint()
```

## 1. 启动脚本与 Hydra 配置

| 文件 | 角色 | 注解要点 |
|---|---|---|
| `examples/sciaccel_rl/fsdp_qwen2.5_3b_laps.sh` | 唯一的用户入口，把环境变量翻译成上百个 Hydra CLI 覆盖参数 | ① 所有 `max_*` 长度参数在这里拼出来（[hf-config-context-window.md](hf-config-context-window.md) 分析过的 `max_model_len` 余量问题就出在这一层）；② `total_epochs` 之前脚本里根本没有暴露，默认吃 Hydra 里 `trainer.total_epochs=30`，而这个任务的训练集正好等于 `train_batch_size`（`steps_per_epoch=1`），导致训练在第 30 步被这个隐藏上限拦停，跟 `total_training_steps=200` 无关——这是本次运行诊断出的真实 bug，修法是显式传 `trainer.total_epochs=${TOTAL_EPOCHS:-1000}`；③ 末尾 `2>&1 \| tee ...log` 决定了日志既进 stdout 也落盘到 `${experiment_name}.log` |
| `psrl/trainer/config/ppo_trainer.yaml` | Hydra 根配置，`@hydra.main(config_path="config", config_name="ppo_trainer")` 指向它 | `defaults:` 列表在这里把下面几行的配置组合并成最终配置树，脚本里的 `gen_actor_rollout_ref.*`/`train_actor_rollout_ref.*`/`data.*` 等键名都是由这个组合结构决定的 |
| `psrl/trainer/config/psrl/*.yaml`（`psrl.yaml`/`deployment.yaml`/`agentic_rl.yaml`/`checkpoint.yaml`/`nixl.yaml`/`rollout_gateway.yaml`/`env_worker.yaml`） | 提供 `psrl.*` 命名空间下所有键的默认值 | 脚本里 `psrl.ps_mode=nixl_cpu`、`psrl.deployment.*`、`psrl.rollout_gateway.*` 等覆盖的都是这些文件里的默认值 |
| `psrl/trainer/config/rollout/psrl_rollout.yaml` | `gen_actor_rollout_ref.rollout` 和 `train_actor_rollout_ref.rollout` 共用的同一份 rollout 默认配置 | 两处 `rollout.max_model_len=...` 覆盖的是同一份 schema 的两个实例（生成引擎 vs 训练侧校验引擎） |
| `psrl/trainer/config/data/data.yaml`、`psrl/trainer/config/reward/reward.yaml` | `data.*`、`reward.*` 命名空间默认值 | `data.truncation`/`filter_overlong_prompts`、`reward.active_managers`/`launch_reward_fn_async` 等键的默认值来源 |
| `psrl/trainer/config/hydra_plugins/psrl_searchpath.py` | Hydra 搜索路径插件 | 让 Hydra 能在这些分散的 `config/` 子目录里找到上面所有 `defaults:` 引用的文件 |

## 2. Trainer 核心（Ray 编排 + 训练主循环）

| 文件 | 角色 | 注解要点 |
|---|---|---|
| `psrl/trainer/main_ppo.py` | `python -m psrl.trainer.main_ppo` 的真正入口 | `main(config)`：设 device、强制 `transfer_queue.enable=True`、`validate_config`、`run_ppo()`；`run_ppo()`：`ray.init` 后把 `TaskRunner` 作为 Ray actor 起起来并阻塞等待；`TaskRunner.run()`：注册 actor/critic/ref/reward_model/dummy 各 role 的 worker 类，调 `init_resource_pool_mgr()` 把 `psrl.deployment.*`（`n_rollout_instances`/`train_nnodes`/`colocate_validate_and_train` 等）翻译成 `resource_pool_spec` 字典（train_pool、每个 `rollout_pool_i`、`colocate_validate_and_train=True` 时才建的 `validate_pool_i`，train/validate 池按 0.9/0.1 分 GPU bundle 避免抢卡），再构造 `PSRL_RayPPOTrainer` 并调 `init_workers()`/`fit()` |
| `psrl/trainer/ppo/ray_trainer.py` | **`PSRL_RayPPOTrainer(RayPPOTrainer)`**——日志里 `<MainRayTrainer>` 的真身，继承 verl 的 `RayPPOTrainer` | `__init__`：起 ReplayBuffer、HTTP 连接池、NIXL 记账，早起 rollout gateway（和后续初始化并行）；`init_workers()`：真正建各资源池的 worker group（日志 "Creating worker group for resource pool"）、解析 gateway/session router 的 URL（"Rollout gateway launched"/"Session router launched"）、按 `agent.node_ips` 限制 agent loop worker 调度节点；`fit()`：batch 拉取 → agent loop rollout → reward → rollout correction → GRPO advantage → `actor_wg.update_actor()` → `_save_checkpoint()`；checkpoint 相关方法都在这个文件：`_save_checkpoint`、`_load_checkpoint`（`trainer.resume_mode` 的 `disable`/`auto`/`resume_path` 三种分支，"Found checkpoint: %s" 就是这里打的）、`_resolve_resume_checkpoint_paths`、`_resume_load_and_push_to_ps`（NIXL 模式下"先加载 checkpoint 再推给 PS"的续训路径） |
| `psrl/trainer/ppo/utils.py` | `ResourcePoolManager`：把 `main_ppo.py` 算出的 `resource_pool_spec` 变成真正的 Ray 资源池 | `create_resource_pool()` 逐个 spec 建 `RayResourcePool`（Ray placement group）；`_check_resource_available` 校验集群 GPU 够不够；也是 `rollout_is_weights`（重要性采样权重）从 rollout_correction 传到 actor 的中转点 |

## 3. Parameter Server（PS）—— 训练权重和 rollout 引擎之间的同步层

`PS` 在这个仓库里指的不是传统同步 SGD 的参数服务器，而是**训练侧权重和 N 个 vLLM rollout 实例之间的 CPU 常驻权重中转站**，`ps_mode=nixl_cpu` 表示权重放在主机内存里、推拉都走 NIXL 传输（不是 Ray object ref、也不是阻塞式 CPU tensor 拷贝）。

| 文件 | 角色 | 注解要点 |
|---|---|---|
| `psrl/workers/ps/ps_manager.py` | `PSManager`（Ray actor，全局单例式）—— PS 的控制面 | 管 `staleness_inventory`（rollout 请求的 staleness 记账）、`ModelStore`（当前模型版本+payload）、NIXL 控制面；`push_model_state_dict_cpu`/`pull_model_state_dict_*` 按 `ps_mode` 分支到不同传输路径；`_coordinate_broadcast_init` 做初始 checkpoint 的树形广播 |
| `psrl/workers/ps/ps_storage_worker.py` | `PSStorageWorker`——每个 rank 一个的存储 actor，日志里 `<PSStorageWorker_R1>` | 持有 train/gen 两套 meta（空权重）HF 模型，把 NIXL 可寻址的 tensor 注册好、把 checkpoint shard 加载进去；`train_model_dtype != gen_model_dtype` 时还要把 train buffer 拷贝成单独的 gen buffer；只支持 `nixl_cpu`/`nixl_gpu` 两种 `ps_mode` |
| `psrl/workers/ps/ps_worker_group.py` | 定义 `PSResourceSpec`（单个 PS worker 该放哪个节点/GPU）、`PSResourcePool`（spec 列表）、`PSWorkerGroup`（按 spec 逐个起 `PSStorageWorker`） | 日志里 `PSResourcePool(ps_spec_list=[PSResourceSpec(node_ip=..., ...)])` 就是这几个 dataclass 的 repr |
| `psrl/workers/gen/vllm_rollout.py` | `PSRL_ServerAdapter(ServerAdapter)`——每个 rollout 副本对应的 Ray actor，包了一层 verl 的 `ServerAdapter`（真正跑 vLLM/SGLang OpenAI 兼容 server 进程的类） | 只加了 `get_node_id()`/`get_runtime_gpu_ids()` 两个方法，供 trainer 建 NIXL 的节点/GPU 映射表；`/dev/shm/verl-cache/.../<model>` 那条 warning 其实来自 verl 的 `copy_to_local(use_shm=True)` 共享内存模型缓存，不是 psrl 自己的代码，"already exists" 只是提示上次遗留的缓存还在、会重新校验而不是重新拷贝 |

## 4. Rollout Gateway / 路由与权重协调

| 文件 | 角色 | 注解要点 |
|---|---|---|
| `psrl/workers/gen/rollout_gateway.py` | `RolloutGateway`（Ray actor）——同时负责起 SMG 和 SessionRouter，两者是**叠加关系不是替代关系** | `launch_router()` 用 `multiprocessing.Process` 起 SMG（Rust 路由进程，真正在 N 个 vLLM 实例前做负载均衡/准入控制，日志 "Rollout gateway launched at ..."）；`launch_session_router()` 另起一个子进程跑 `uvicorn` 挂 `SessionRouter.app`（日志 "Session router launched at ..."），这个 SessionRouter 转发请求给 SMG 并加上 TITO/session-turn 语义 |
| `psrl/workers/gen/session_router.py` | `SessionRouter`（FastAPI）—— TITO session 的代理层，[已有专门文档](psrl-agent-loop-harbor-tool-calling.md)详细讲过 | 这里只补一句：它是 `RolloutGateway` 起的两个子进程之一，不是独立顶层组件 |
| `psrl/workers/gen/smg_adapter.py` | `build_rollout_router_args`：把 `psrl.rollout_gateway.*` 和 `psrl.rollout_coordination.routing_strategy.*` 翻译成 SMG 的 Rust `RouterArgs` | `routing_strategy.max_concurrent_seqs_per_instance` 实际在这里被读取并原样交给 SMG，做的是**每个 rollout 实例的准入上限**（超了就排队/拒绝），跟下面的 `RolloutCoordinator` 是两层不同的机制 |
| `psrl/workers/gen/rollout_coordination/coordinator.py` | `RolloutCoordinator`（Ray actor）——权重版本同步/迁移的协调者，日志 "Initialized RolloutCoordinator" | 跟踪每个 rollout 副本的模型版本（`instance_to_model_version`）、引擎状态、KV cache 容量，驱动权重版本推送/迁移循环；跟 SMG 的准入控制是分工关系，不是它去做准入限流 |
| `psrl/workers/gen/rollout_coordination/command_loop.py`、`status_loop.py`、`sync_and_migrate/*.py` | 组成 `RolloutCoordinator` 的各个 mixin | abort/sync 命令处理、状态轮询、按负载在各 rollout 实例间做重平衡 |

## 5. Agent Loop / rollout 阶段的"工具调用"

这部分已经有一篇专门文档做了完整拆解，这里只列文件名做索引，细节直接看 **[psrl-agent-loop-harbor-tool-calling.md](psrl-agent-loop-harbor-tool-calling.md)**：

- `psrl/workers/agent_loop/worker.py` — `PSRL_AgentLoopWorker`
- `psrl/workers/agent_loop/manager.py` — `PSRL_AgentLoopManager`
- `psrl/workers/agent_loop/loops/base_agent_loop.py` — `AgentLoopBase`
- `psrl/workers/agent_loop/loops/session_agent_loop.py` — `SessionAgentLoop`
- `psrl/workers/agent_loop/loops/utils.py` — `AGENT_LOOP_REGISTRY`
- `examples/sciaccel_rl/agent_loop.py` — `SciAccelAgentLoop`
- `examples/sciaccel_rl/agent.py` — `TruncatingTerminus2`
- `examples/sciaccel_rl/runner.py` — `run_harbor_episode`，构造 Harbor `JobConfig`
- `examples/sciaccel_rl/config/sciaccel_agent_config.yaml` — 注册 `sciaccel` agent loop
- `examples/sciaccel_rl/config/qwen35_acc_thinking.jinja2` — `multi_thinking` 模式用的累积式 chat template（脚本里 `+gen_actor_rollout_ref.rollout.chat_template=...` 指向它）
- `psrl/utils/tito/training_data.py` — `build_training_data()`，TITO token 还原训练样本
- `psrl/utils/agent/thinking.py` — `MULTI_THINKING` 等模板策略

## 6. 训练侧：FSDP2 / Ulysses SP / GRPO loss / rollout correction

**这一块的重活基本都在外部库 `verl` 里**（本仓库不含 `verl` 源码，是装在 conda 环境里的独立可编辑安装，路径 `.../third_party/verl/verl`；本仓库只有 `patch/verl/*.patch` 这几个打给它的 diff）。`psrl` 侧只有薄薄一层子类做 NIXL 权重流转和显存 sleep/wake 的适配。

| 文件 | 归属 | 角色 | 注解要点 |
|---|---|---|---|
| `psrl/workers/train/engine_train_worker.py` | **本仓库** | `PSRL_EngineTrainWorker(ActorRolloutRefWorker, PSRL_BaseTrainWorker)` | 只在 `strategy in {"fsdp","fsdp2"}` 时插入 NIXL 权重流转、显存 sleep/wake（`_sleep_fsdp_model`/`_wake_up_fsdp_model`）、非持久 buffer 恢复；真正的 forward/backward/optimizer 全部调用父类 `ActorRolloutRefWorker.update_actor`/`init_model` |
| `verl/workers/engine_workers.py`（外部） | verl | `ActorRolloutRefWorker` | 读 `use_dynamic_bsz`/`ppo_max_token_len_per_gpu`，实现 `update_actor`/`compute_log_prob` |
| `verl/workers/engine/fsdp/transformer_impl.py`（外部） | verl | `FSDPEngine` | `strategy=="fsdp2"` 分支：`fully_shard` 真正做 FSDP2 切分，`fsdp_config.optimizer_offload` 通过 `CPUOffloadPolicy` 生效，`forward_backward_batch` 是真正的前反传循环 |
| `verl/utils/ulysses.py`（外部） | verl | Ulysses 序列并行 | `get_ulysses_sequence_parallel_group/world_size/rank`、`all_to_all_tensor`（真实 `dist.all_to_all`）、`SeqAllToAll` autograd 函数、`FSDPUlyssesShardingManager`（从 `device_mesh["sp"]` 取 SP 组、对 `DataProto` 做分片前后处理）；`train_actor_rollout_ref.actor.ulysses_sequence_parallel_size=4` 直接决定这里的分组大小 |
| `verl/trainer/ppo/core_algos.py`（外部） | verl | GRPO advantage + PPO-clip loss | `compute_grpo_outcome_advantage`（`norm_adv_by_std_in_grpo` 控制是否做标准差归一化）；`clip_ratio_low/high`、`loss_agg_mode="token-mean"`、`use_kl_loss` 门控的 loss 组装逻辑都在这里 |
| `verl/workers/config/actor.py`（外部） | verl | `PolicyLossConfig` schema | 定义 `clip_ratio_low/high`/`loss_agg_mode`/`use_kl_loss` 这些键的合法取值 |
| `verl/trainer/ppo/rollout_corr_helper.py`（外部） | verl | rollout correction（重要性采样修正）的真正实现 | `log_ratio = old_log_prob(训练侧FSDP) - rollout_log_prob(vLLM)`；`rollout_is="token"` 时算逐 token 权重 `w_t = exp(clamp(log_ratio))`，再用 `rollout_is_threshold=2.0` 做截断重要性采样（只截上界，因为传的是单个 float 不是 `"lower_upper"`）；权重 detach 后作为 loss 的重要性权重，用来修正训练策略（FSDP fp32）和 rollout 策略（vLLM bf16、可能有 staleness）之间的偏差；同时算一堆 `rollout_corr/*` 诊断指标（KL、k3_kl、χ²、ESS、perplexity ratio） |
| `psrl/trainer/ppo/ray_trainer.py`（本仓库，第 41 行 import，~2907-2935 行调用） | 本仓库 | 调用桥 | `import compute_rollout_correction_and_add_to_batch from verl...`；读 `algorithm.rollout_correction`，在算 GRPO advantage **之前**调这个函数，把 `rollout_is_weights` 塞进 batch 供 actor 用 |
| `verl/utils/seqlen_balancing.py`（外部） | verl | `rearrange_micro_batches` | 按 `ppo_max_token_len_per_gpu` 把不同长度的序列打包进 token 预算内的 micro-batch；脚本里那条 "rearrange_micro_batches requires max_token_len >= max_seq_len" 报错文案其实是脚本自己的护栏（见下面第7节），不是这个函数直接抛的，但校验的正是这里的前提条件 |

## 7. Reward 与数据集

| 文件 | 归属 | 角色 | 注解要点 |
|---|---|---|---|
| `examples/sciaccel_rl/reward.py` | 本仓库 | `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)` | **完全不看** `solution_str`/`ground_truth`，纯粹是个"取字段"的薄适配器：从 `extra_info["harbor_rewards"]`（Harbor verifier 容器在 rollout 时算出的打分字典）里取 `extra_info["reward_key"]`（默认 `"reward"`）对应的值，返回 `{"score": ..., "reward_extra_info": {...}}`——真正的任务成功判定发生在 Harbor 的 verifier 容器里，这里只是接口对接 |
| `psrl/workers/reward/reward_loop/dapo.py` | 本仓库 | `DAPORewardManager.run_single` | 解码有效（非 padding）response token → 调 `compute_score` → 取 `result["score"]` 当 reward；`overlong_buffer_cfg.enable=True` 时才会用 `max_resp_len` 对超长响应做惩罚（本脚本里 `enable=False`，所以 `max_resp_len=16384` 这个参数当前实际没生效） |
| `psrl/workers/reward/reward_loop/registry.py` | 本仓库 | `@register("dapo")` 注册表 | 把配置字符串 `reward.active_managers=[dapo]` 映射到 `DAPORewardManager` 类 |
| `psrl/workers/reward/reward_manager.py` | 本仓库 | 顶层 reward 协调器 | 读每条样本的 `reward_model_dicts`，解析出该用哪个 `RewardSpec`/manager，`reward.launch_reward_fn_async=True` 对应这里用 `asyncio.create_task` 逐样本异步打分而不是同步阻塞循环，最后按 `reward_coef` 加权聚合回每条 rollout |
| `psrl/utils/dataset/rl_dataset.py` | 本仓库 | `PSRLRLHFDataset(RLHFDataset)` | 只重写了 `maybe_filter_out_long_prompts`，在父类的长度过滤逻辑跑之前先注入 PSRL 加载的工具 schema（因为工具 schema 也占 prompt token 长度）——这是训练用的数据集类 |
| `psrl/utils/dataset/utils.py` | 本仓库 | `create_rl_dataset` | 除非配了 `data.custom_cls.path`，否则默认选 `PSRLRLHFDataset`；trainer 真正读 `data.train_files`/`val_files` 用的就是这个函数产出的对象 |
| `verl/utils/dataset/rl_dataset.py`（外部） | verl | `RLHFDataset` 基类 | 解析 parquet/json/jsonl（`_read_files_and_tokenize`），`max_prompt_length`/`truncation`（`"error"` 表示超长直接抛异常而不是静默截断）/`filter_overlong_prompts`/`prompt_key`/`return_raw_chat=True`（多存一份原始 chat 列表供 agent loop 用）/`reward_model_dicts`（挂到每行 `row_dict` 上，供下游按样本路由 reward manager）都在这个基类里处理；日志里 "Using dataset class: RLHFDataset" 其实来自另一个调用点（`agent_loop/worker.py` 里 `get_dataset_class` 拿到的默认类，只给多模态辅助函数用，跟训练用的 `PSRLRLHFDataset` 是两个不同的对象） |

## 8. 外部依赖位置小结

- **`verl`**：独立可编辑安装在 `.../third_party/verl/verl`，**不在这个 `lw/psrl` 仓库里**；本仓库只保留了 `patch/verl/*.patch`（几个打给它的 diff）和 `docker/patch/verl.patch`。第 6、7 节里标"外部"的文件都在这里，仅供理解机制参考，不属于本仓库要维护的代码。
- **`harbor`**：独立 pip 包（`harbor==0.22.0`），跟 rollout 阶段"工具调用"相关的所有细节见 [psrl-agent-loop-harbor-tool-calling.md](psrl-agent-loop-harbor-tool-calling.md) 的"关键文件路径一览"。

## 延伸

- `max_model_len`/`max_prompt_length`/`max_response_length` 的关口原理和两次训练失败复盘，见 [hf-config-context-window.md](hf-config-context-window.md)
- rollout 阶段模型如何"调用工具"（Harbor/TITO/multi_thinking），见 [psrl-agent-loop-harbor-tool-calling.md](psrl-agent-loop-harbor-tool-calling.md)
- vLLM 推理机制、训练并行维度、PPO/GRPO loss 组装的更系统性讲解，见 [vllm-and-training-parallelism.md](vllm-and-training-parallelism.md)
