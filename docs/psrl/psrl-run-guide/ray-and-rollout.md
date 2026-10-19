# PSRL 的 Ray 调度与生成长尾

> **核心结论**：分别说明 Ray 的角色、资源调度流程和 rollout 长尾，区分集群调度职责与模型计算职责。

## 目录

- [Ray 的基础职责与 Actor 协作](#ray-的基础职责与-actor-协作)
  - [一句话结论](#一句话结论)
  - [1. 启动：`ray.init`](#1-启动rayinit)
  - [2. 两类 Actor：控制面单例 vs. 水平扩展 worker 组](#2-两类-actor控制面单例-vs-水平扩展-worker-组)
  - [3. 节点/GPU 级调度](#3-节点gpu-级调度)
  - [4. Object Store 做"零拷贝"权重交接（对应 `ps_mode=cpu_ref`）](#4-object-store-做零拷贝权重交接对应-ps_modecpu_ref)
  - [5. 日常跨 Actor 通信：同步 RPC 风格](#5-日常跨-actor-通信同步-rpc-风格)
  - [训练结果保存在哪](#训练结果保存在哪)
- [Ray 的资源调度与应用启动流程](#ray-的资源调度与应用启动流程)
  - [一句话结论](#一句话结论-1)
  - [1. 资源层：`ray_start.sh` 拉起物理集群](#1-资源层ray_startsh-拉起物理集群)
  - [2. 应用层：`main_ppo.py` 里的 `ray.init` 和 `TaskRunner`](#2-应用层main_ppopy-里的-rayinit-和-taskrunner)
  - [3. 资源池管理：把物理 GPU 切成"角色专属"的资源池](#3-资源池管理把物理-gpu-切成角色专属的资源池)
  - [4. Worker 即 Ray Actor](#4-worker-即-ray-actor)
  - [5. Ray 不负责什么](#5-ray-不负责什么)
  - [6. 落到 `qwen2.5_3b_fsdp.sh` 的 2 节点实验上](#6-落到-qwen25_3b_fsdpsh-的-2-节点实验上)
- [为什么 RL rollout 会有“生成耗时不均”（long-tail latency）](#为什么-rl-rollout-会有生成耗时不均long-tail-latency)
  - [结论](#结论)
  - [为什么 LLM 的 RL 特别容易出现](#为什么-llm-的-rl-特别容易出现)
  - [一个直观例子](#一个直观例子)
  - [PSRL 怎样处理，而不是消灭差异](#psrl-怎样处理而不是消灭差异)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。旧的中文章节编号已改为主题标题，正文引用已改为可点击链接。

[返回专题索引](../psrl-run-guide.md)

运行前先阅读[共享环境与集群检查](cluster-preflight.md)；文中的历史地址、依赖和机器状态需要重新核对。

## Ray 的基础职责与 Actor 协作

### 一句话结论

Ray 是 psrl 的**跨机器进程管理 + 调度基础设施**——psrl 不自己写 RPC/进程编排代码，而是把训练侧、rollout(vLLM)侧、reward 侧、Parameter Server 等所有"角色"都建成 Ray Actor，靠 Ray 的调度器分布到具体节点/GPU 上，再通过 Actor 方法调用（`.remote()`/`ray.get()`）互相通信。

### 1. 启动：`ray.init`

入口在 `psrl/trainer/main_ppo.py:135-149`：

```python
if not ray.is_initialized():
    ray.init(**OmegaConf.to_container(ray_init_kwargs))
```

集群参数（地址、`runtime_env` 里的环境变量等）来自 Hydra 配置 `config.ray_kwargs.ray_init`，每个训练任务只在启动时初始化一次。

### 2. 两类 Actor：控制面单例 vs. 水平扩展 worker 组

区分标准：这个 Actor 管的是"权威状态/协调决策"，还是"具体干活"。

- **控制面单例**：全集群只有 1 个实例，维护"唯一真相"（谁拿到了锁、权重当前版本号、请求该派给谁）。之所以必须是单例——这类状态如果有多份拷贝就需要互相同步，等于又造出一个新的一致性问题，所以只放一份，其他人都来问它。它本身不干重活：`PSManager` 不搬权重字节（那是 NIXL 干的，见[nixl-explained.md](../nixl-explained.md)），`RolloutGateway`/`RolloutCoordinator` 不生成文本（那是 vLLM 实例干的）。
- **水平扩展的 worker 组**：同一种角色有 N 个实例，N 随 GPU 数增减，彼此地位对等、可互换——任意一个 rollout 实例都能处理任意生成请求，任意训练 rank 都在跑同一套模型的不同 shard。它们只干活，不维护全局状态，需要协调时才去问对应的控制面单例。对应到 [示例脚本拆解](getting-started.md#逐段拆解示例脚本-examplesdapo_trainerqwen25_3b_fsdpsh) 里 `GEN_INSTANCES=8` 这个派生变量，这 8 个 vLLM 实例就是"rollout worker 组"的具体实例数。

一个类比：控制面单例像调度室的调度员，只有一个人，手握全局账本，负责派活；worker 组像一线跑活的司机，可以有很多个，人数随车队规模变化，谁跑哪一单由调度员说了算，司机之间不需要互相商量。

**控制面单例**（每种只有 1 个 Actor，负责协调/记账）：

- `TaskRunner`（`main_ppo.py:185`，`@ray.remote(num_cpus=1)`）：真正的"驱动大脑"，构建 `role_worker_mapping`、`ResourcePoolManager`，然后调 `trainer.fit()`。注释里特意写了"确保不要调度到 head 节点"。
- `PSManager`（`psrl/workers/ps/ps_manager.py:54`）：Parameter Server 的控制面，只管版本号和 push/pull 锁，不管字节数据本身（跟 NIXL 是配套的——PSManager 是控制面，NIXL 是数据面，详见[nixl-explained.md](../nixl-explained.md)）。
- `RolloutGateway`（`psrl/workers/gen/rollout_gateway.py:36`）、`RolloutCoordinator`、`RewardModelGateway`（`psrl/workers/reward/reward_model/gateway.py:18`）：分别负责生成请求路由、rollout 协调、reward 请求路由。

**水平扩展的 worker 组**：训练 actor（FSDP/Megatron）、rollout actor（vLLM 实例）、reward model actor，各自绑定到自己的资源池（`train_pool`、`rollout_pool_i`、`reward_pool_*`，见 `main_ppo.py:236-352`），数量随 GPU 数扩展，不是单例。

### 3. 节点/GPU 级调度

- **精确钉住某个节点**：`PSManager` 必须跑在 `config.psrl.ps_manager_ip` 对应的那台机器上（因为所有 rollout/train 侧都要连这个固定地址），靠 `NodeAffinitySchedulingStrategy` 实现（`ray_trainer.py:472-478`，`RolloutCoordinator` 同样处理见 `:611-618`）。
- **占坑防调度**：`_GPUSlotReserver`（`main_ppo.py:58`，`@ray.remote(num_gpus=1, num_cpus=0)`）是个什么都不干的空 Actor，纯粹用来占住"多余"GPU 节点，防止 Ray 把别的任务调度上去（`main_ppo.py:90-112` 遍历 `ray.nodes()` 找出这些多余节点）。
- **同卡合租**：validate 和 train worker 共享同一批 GPU 时，把 bundle 的资源比例设成 `0.9` 而不是 `1.0`（`main_ppo.py:235-242`），否则两个都申请"整卡"会导致调度冲突。
- **Placement Group 紧密打包**：弹性 reward model 调度用 `get_placement_groups(strategy="STRICT_PACK", ...)`（`ray_trainer.py:1470`），保证同一批 reward worker 尽量挤在同一批节点上，减少跨机通信。

### 4. Object Store 做"零拷贝"权重交接（对应 `ps_mode=cpu_ref`）

这是最能体现"Ray 不只是调度器，也是数据搬运通道"的例子：

```python
# 训练侧 psrl/workers/train/engine_train_worker.py:457-458
object_ref = ray.put(full_state_dict)
ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(next_version, [object_ref])
```

```python
# rollout 侧 psrl/workers/gen/vllm_async_server.py:1455
object_ref = await ps_manager_handle.pull_model_state_dict_cpu_ref.remote(rollout_instance_ids)
```

训练侧把整份 state dict `ray.put` 进 Ray 的 plasma 共享内存对象存储，只把一个轻量的 `ObjectRef` 传给 `PSManager`；rollout 侧再用同一个引用去 resolve，本质是"传指针不传数据"，避免了一次显式的整份权重序列化搬运。这是 `nixl_cpu` 之外更轻量的兜底同步模式。

### 5. 日常跨 Actor 通信：同步 RPC 风格

除了上面这个零拷贝特例，大部分场景就是最朴素的 `ray.get(actor.method.remote(...))`（例如 `ray_trainer.py:480, 503, 770, 1107`）——Actor 方法调用被当成远程函数调用，`.remote()` 发起异步调用返回 future，`ray.get()` 阻塞等结果。

### 训练结果保存在哪

脚本里 `trainer.save_freq=-1`，是 veRL/PSRL 里"关闭定期保存 checkpoint"的约定值。`default_local_dir` 虽然配了 `${PSRL_WORKSPACE}/checkpoint/${experiment_name}`，但因为 `save_freq=-1` 从未触发保存逻辑，这个目录实际上不会被创建，权重训完就随进程释放，磁盘上不留任何 checkpoint。唯一留下的产物是 `tee` 出来的日志文件（`${PSRL_WORKSPACE}/logs/${experiment_name}.log`），里面有每步的训练指标和最终验证指标。想要真正落盘权重，需要把 `trainer.save_freq` 改成正整数（比如每 N 步存一次），FSDP 场景下用的是 PyTorch Distributed Checkpoint 格式（配置里 `checkpoint.use_dcp_save=True`）。

## Ray 的资源调度与应用启动流程

### 一句话结论

Ray 在 PSRL 里承担两层职责：① **资源层**——`examples/ray/ray_start.sh` 把多台物理机组成一个共享 CPU/GPU 资源池的 Ray 集群；② **编排层**——PSRL 在这层资源之上，用 Ray Actor 把训练 worker、生成(vLLM) worker、Parameter Server、路由网关等逻辑角色实例化并分配到具体节点/GPU 上，同时提供跨进程的控制流 RPC。真正的大流量（权重、KV cache）传输**不走** Ray，走的是 NIXL/RDMA 或共享内存。

### 1. 资源层：`ray_start.sh` 拉起物理集群

```bash
# head 节点
ray start --head --port=8887 --dashboard-port=8265 --num-cpus=32
# 其余节点
ray start --address=<head_ip>:8887 --num-cpus=32
```

`ray_start.sh` 先用 `pssh` 在所有目标机器上 `ray stop --force` 清残留进程，再按 hostfile 第一行为 head、其余为 worker 依次拉起。此时得到的是一个"裸"集群：一堆节点把自己的 CPU/GPU 数量注册进 GCS，谁来用、怎么切分完全还没决定。

### 2. 应用层：`main_ppo.py` 里的 `ray.init` 和 `TaskRunner`

- `ray.init(**ray_init_kwargs)`：连接到上一步已经存在的集群（多节点场景），或者用 `+ray_kwargs.ray_init.address=local` 在本机现起一个单机集群（单机 demo 用的是这种模式，见[单机 Demo](getting-started.md#单机-demo-脚本与常用-bash-语法qwen25_3b_fsdp_singlenode_demosh)）。
- `TaskRunner` 本身也被包成一个 `ray.remote(num_cpus=1)` 的 Actor（代码注释特意说明"要避免被调度到 head 节点"），也就是说驱动整个训练循环的主控逻辑，运行在 Ray 集群里某个 worker 节点上，而不是本地跑脚本的那个 shell 进程里——脚本进程只是提交任务、等它跑完。

### 3. 资源池管理：把物理 GPU 切成"角色专属"的资源池

`main_ppo.py` 里的 `init_resource_pool_mgr` 会读 `psrl.deployment.*`（`train_nnodes`、`train_ngpus_per_node`、`n_rollout_instances`、`rollout_ngpus_per_node_per_instance` 等），拼出一份 `resource_pool_spec`：训练用的 `train_pool`、每个生成实例各自的 `rollout_pool_i`、每个验证实例的 `validate_pool_i`、reward 模型的 `reward_pool_i`……本质是对 **Ray Placement Group** 的封装（来自 verl 的 `single_controller.ray` 机制）：每个 pool 对应集群里一组资源 bundle（例如 1 GPU = 1 bundle），Ray 按这些 bundle 的节点归属，把之后创建的 Actor 钉死在该 pool 名下的物理节点/GPU 上。

### 4. Worker 即 Ray Actor

`role_worker_mapping` 把逻辑角色映射到具体的 `ray.remote` 类：

| 逻辑角色 | 对应 Ray Actor 类 | 资源声明 |
|---|---|---|
| 训练（Actor/Critic） | `PSRL_TrainWorker` / `TrainingWorker` | 绑定训练 pool 的 GPU |
| 生成/验证（Rollout/Validate） | `PSRL_ServerAdapter`（内部跑 vLLM） | 绑定各自 rollout/validate pool 的 GPU |
| Reference Policy | `ActorRolloutRefWorker` | 按需绑定 GPU |
| Parameter Server 管理 | `PSManager`（`@ray.remote(num_cpus=1)`） | 纯 CPU，常驻 head 节点，做版本管理和 Reserve/Occupy/Consume 协调 |
| 路由网关 | `RolloutGateway`（`@ray.remote(num_cpus=0)`） | 纯 CPU，转发生成请求 |

需要"贴着谁跑"的场景（比如 PS Worker 要和它服务的 rollout 实例同机，走本机 PCIe 而不是跨机网络）会用 `NodeAffinitySchedulingStrategy` 显式指定调度到某个节点，而不是让 Ray 自动选。

### 5. Ray 不负责什么

- **大数据搬运**：模型权重同步、KV cache 传输走 NIXL over UCX/RDMA（跨节点）或共享内存（同节点），不经过 Ray 的 object store/plasma——这是 PSRL 为了拿到 RDMA 级别带宽和延迟，特意绕开 Ray 默认数据通道的设计选择。
- **框架内部的分布式**：FSDP2/Megatron 的进程组通信（AllReduce 等）、vLLM 内部的 TP/PP，都是各自框架自己的事；Ray 只负责把"跑 FSDP 的这个 worker 进程"或"跑 vLLM 的这个 worker 进程"作为一个 Actor 拉起来、分配资源、管生命周期，进程内部怎么并行 Ray 不掺和。

### 6. 落到 `qwen2.5_3b_fsdp.sh` 的 2 节点实验上

1. `ray_start.sh <hostfile>` 先在 2 台机器上组出一个 16 GPU 可见的 Ray 集群。
2. 脚本跑 `python -m psrl.trainer.main_ppo`（**没有**加 `+ray_kwargs.ray_init.address=local`），所以 `ray.init()` 连接的是第 1 步已经存在的集群，而不是现起一个本地集群。
3. 脚本里的 `psrl.deployment.train_nnodes=1`、`n_rollout_instances=8` 等数字，就是喂给 `ResourcePoolManager` 用来在这 2 个物理节点、16 张 GPU 上切资源池的依据——训练/生成物理分离正是通过"分别声明成不同 resource pool"实现的。
4. [单机启动故障复盘](startup-incident-reviews.md#单机-demo-踩坑记录两处根因与修复)记录过的报错 `PSManager IP xxx not found in ray nodes`，本质就是 Ray 集群里节点注册用的 IP 和脚本传给 `psrl.ps_manager_ip` 的 IP 没对上——提醒我们 Ray 认的"节点身份"是它自己记录的 IP，脚本参数必须和它严格一致，不能各算各的。

## 为什么 RL rollout 会有“生成耗时不均”（long-tail latency）

### 结论

这里的“生成耗时不均”是指：同一轮 RL 采样中，不同 prompt/轨迹完成 rollout 的时间差很大；少量极慢的轨迹（stragglers）形成长尾。同步式训练必须等到这一批要求的样本全部完成，所以本轮的等待时间由最慢者决定，而不是平均耗时决定。

若一轮采样有 `m` 条轨迹，其完成时间为 `t_1, ..., t_m`，同步边界近似为：

```text
rollout_step_time = max(t_1, ..., t_m)
```

因此，哪怕 95% 的请求已经结束，只要最后 5% 还在生成，其余生成 GPU、reward 计算和训练侧都可能处于等待或低利用率状态。

### 为什么 LLM 的 RL 特别容易出现

1. **输入和输出长度本来就不同。** Prefill 成本随 prompt/token 上下文增长；decode 成本又随生成 token 数、采样到 EOS 的时机而不同。一个立即答完的数学题和一个一路生成到 `max_response_length` 的题，耗时可以相差很多。
2. **RL 采样具有随机性。** 相同 prompt 在不同采样下会得到不同长度的回复；PPO/GRPO/DAPO 常为每个 prompt 采样多条 response，组内也可能只有一条特别长，仍会拖住整组。
3. **Agentic RL 的轨迹长度更不规则。** 一条轨迹可能直接回答，另一条可能经历多轮「生成 -> 工具调用/环境交互 -> 再生成」。工具响应、代码执行、网页或沙箱 I/O 的差异会进一步放大尾部延迟。
4. **推理服务是共享资源。** vLLM 会做 continuous batching、KV cache 管理和请求排队。长上下文、缓存未命中、不同 GPU 上的排队长度、显存压力或实例性能差异，都使相同 token 数的请求实际完成时间不同。
5. **训练早期常有异常长轨迹。** 模型可能重复、走错工具调用分支，或迟迟不输出 EOS，直到长度上限；这类少数低质量轨迹尤其容易成为长尾。

### 一个直观例子

假设一步 GRPO 需要 `64` 个 prompt，每个采样 `8` 条回复。大多数回复在 3--8 秒结束，但有两条进入重复循环，分别需要 45 秒和 60 秒。若算法要求完整 batch，训练更新只能在约 60 秒后开始；前面已完成的绝大多数结果不能让这一步立刻完成。这就是“平均耗时不高，但吞吐很差”。

### PSRL 怎样处理，而不是消灭差异

PSRL 不假定所有轨迹会同速完成，而是尽量减少慢请求造成的全局等待：

- **异步与 staleness**：允许 rollout 比训练领先有限个模型版本，把某个 buffer 的长尾隐藏在训练其他已完成 buffer 的计算期间。代价是样本相对当前参数略旧，因此需要限制 staleness，并可采用 rollout correction。
- **Fine-grain overlap**：已完成的小块样本先进入训练侧的可提前计算阶段，不必等整个 buffer 填满。
- **Partial rollout**：当实例需要同步新权重时，中断尚未完成的长生成，但保留已生成 token；请求随后继续路由，避免“等它结束”或“全部丢弃”。
- **Redundant rollout**：多生成一些 prompt 或每 prompt 多生成几条 response；足够组成算法所需 batch 后终止最慢的冗余请求。它以额外算力换尾延迟，但若长轨迹本身更有学习价值，系统性丢弃会带来训练偏差。
- **智能路由与迁移**：根据排队数、吞吐预测、KV cache 命中等状态分配请求，避免把长任务或拥堵任务进一步集中到同一实例。

所以，PSRL 要解决的核心不是让每条回复等长，而是把 RL 中不可避免的随机、长尾 rollout 从“全体都必须等最慢者”的同步障碍，变成可被异步、分块和调度机制吸收的系统波动。
