# PSRL 日志监控、批量配置与进程排查

> **核心结论**：从训练日志提取指标，理解 train_prompt_bsz 的配置映射，并确认训练进程所在节点及运行状态。

## 目录

- [如何阅读训练日志、抓取每步指标](#如何阅读训练日志抓取每步指标)
  - [日志的结构：按 Ray actor 交织打印](#日志的结构按-ray-actor-交织打印)
  - [每步指标是"一整条长行"](#每步指标是一整条长行)
  - [实操：把指标从日志里抠出来](#实操把指标从日志里抠出来)
  - [排查问题时看什么](#排查问题时看什么)
  - [更好的可视化](#更好的可视化)
- [`train_prompt_bsz` 对应哪个配置、是什么意思](#train_prompt_bsz-对应哪个配置是什么意思)
  - [一句话结论](#一句话结论)
  - [映射关系](#映射关系)
  - [语义与推导量](#语义与推导量)
  - [记忆点](#记忆点)
- [查一个训练 pid 的运行情况：去哪儿看、看什么](#查一个训练-pid-的运行情况去哪儿看看什么)
  - [一句话结论](#一句话结论-1)
  - [第一步：确认自己在不在正确的机器上](#第一步确认自己在不在正确的机器上)
  - [分层检查清单](#分层检查清单)
  - [判断"卡住了" vs. "只是慢"的经验法则](#判断卡住了-vs-只是慢的经验法则)
  - [延伸](#延伸)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。旧的中文章节编号已改为主题标题，正文引用已改为可点击链接。

[返回专题索引](../psrl-run-guide.md)

运行前先阅读[共享环境与集群检查](cluster-preflight.md)；文中的历史地址、依赖和机器状态需要重新核对。

## 如何阅读训练日志、抓取每步指标

### 日志的结构：按 Ray actor 交织打印

日志是 `2>&1 | tee xxx.log` 得到的，本质是多个 Ray actor 的输出交织在一个文件里，每行前面的 `(XxxActor pid=xxxx)` 标签说明这行是谁打的：

| 标签 | 角色 | 关注点 |
|---|---|---|
| `(TaskRunner pid=...)` | **主控进程**，训练循环、指标、阶段切换都在这打印 | ⭐ 想看训练进展只看这一路就够了 |
| `(FusedWorker_actor_dummy pid=...)` | FSDP 训练 worker | 模型加载、显存占用、NIXL 权重收发 |
| `(PSRL_vLLMHttpServer pid=...)` | vLLM rollout 引擎 | 生成引擎启动、CUDA graph capture，多为初始化噪音 |
| `(PSManager/PSStorageWorker pid=...)` | 参数服务器 | 权重同步协议（NIXL protocol step 1~8） |
| `(RolloutGateway/RolloutCoordinator ...)` | 生成请求路由/调度 | 排查长尾/调度问题时才需要看 |

### 每步指标是"一整条长行"

真正的每步训练指标不是分行打印的，是一条超长行，格式为：

```
step:1 - global_seqlen/min:... - actor/entropy:... - actor/loss:... - training/global_step:1 - critic/score/mean:... - ...（几十个 key:value 用 " - " 连接）
```

关注哪些 key：

| 关心什么 | 看哪些 key | 怎么解读 |
|---|---|---|
| 训练进度 | `training/global_step` | 与 `trainer.total_training_steps` 对比 |
| 模型是否在学 | `actor/loss`、`actor/pg_loss`、`actor/grad_norm` | 关键是不发散、不是 NaN/Inf；`grad_norm` 突然飙高说明更新不稳定 |
| 探索程度 | `actor/entropy` | 应缓慢下降；一开始就骤降到接近0可能是坍缩 |
| 奖励信号 | `critic/score/mean`、`critic/rewards/mean` | 训练有效应震荡上升；步数太少看不出趋势很正常 |
| 采样效率 | `group/zero_variance_fraction` | 一组内奖励全相同→advantage=0→这条数据白采，比例越高效率越差 |
| 回复长度 | `response_length/mean`、`response_length/clip_ratio` | `clip_ratio>0` 说明有回复被截断，`max_response_length` 可能不够 |
| 效率 | `perf/mfu/actor`、`perf/throughput`、`timing_s/step`、`timing_s/wait_for_gen`/`update_actor` | 判断瓶颈在生成还是训练 |
| 验证效果 | `val-core/openai/gsm8k/acc/mean@1` | 只在触发验证的步才出现 |

### 实操：把指标从日志里抠出来

日志行带 ANSI 颜色码，直接 grep 容易漏配，建议先去掉颜色码再处理。

快速看某几个指标随 step 变化：
```bash
LOGF=.../psrl_demo/logs/xxx.log
grep -oP "step:\d+ - .*?actor/loss:\S+" "$LOGF" | sed -E 's/\x1b\[[0-9;]*m//g'
```

解析成表格（更推荐，方便画图/筛选）：
```python
import re
keys = ["training/global_step", "actor/loss", "actor/entropy", "actor/grad_norm",
        "critic/score/mean", "response_length/mean", "perf/mfu/actor",
        "timing_s/step", "val-core/openai/gsm8k/acc/mean@1"]
ansi = re.compile(r"\x1b\[[0-9;]*m")
rows = []
with open(LOGF, errors="ignore") as f:
    for line in f:
        line = ansi.sub("", line)
        if re.search(r"\bstep:\d+ - global_seqlen", line):
            fields = dict(kv.split(":", 1) for kv in line.split(" - ") if ":" in kv)
            rows.append({k: fields.get(k, "-") for k in keys})
```

### 排查问题时看什么

- `grep -in "error\|traceback\|assert\|exception\|nan\|inf"`：定位报错或数值异常。
- `Training Progress: NN%|...`（tqdm 进度条行）：看每步耗时和整体进度，卡住不动说明某一步卡死了。
- 某个 actor 长时间没有新日志时，结合日志文件 mtime、`ps aux`、`nvidia-smi` 判断是真卡住还是正常在算。

### 更好的可视化

`trainer.logger='["console"]'` 时指标只打印到文本日志，没有曲线图。长期训练建议改成 `trainer.logger='["console","wandb"]'`（需配 `WANDB_API_KEY`）或 `tensorboard`，避免只能啃文本日志。

## `train_prompt_bsz` 对应哪个配置、是什么意思

### 一句话结论

`train_prompt_bsz` 只是示例脚本里的 shell 变量名，会被传给 Hydra 配置项 `data.train_batch_size`；它表示**每个 RL 训练 step 采样的 prompt 数量**，不是总样本数。

### 映射关系

```bash
train_prompt_bsz=64
...
    data.train_batch_size=${train_prompt_bsz} \
```
（`psrl/examples/dapo_trainer/qwen2.5_3b_fsdp.sh:56,162`）

### 语义与推导量

- 每个 prompt 会再采样 `n_resp_per_prompt`（对应配置 `rollout.n`，脚本里是 8）条 response。
- 这一步实际的轨迹总数（代码里叫 `real_train_batch_size`）：

$$
\text{real\_train\_batch\_size} = \text{train\_batch\_size} \times \text{rollout.n} = 64 \times 8 = 512
$$

  该值会在 `validate_config`（`psrl/psrl/utils/config.py:144`）里校验是否能被最小可用 batch（GPU 数，或 Megatron 下 `megatron_dp * ppo_micro_batch_size_per_gpu`）整除。
- 这 512 条轨迹是这一轮训练**可用**的全部数据（global batch），不是"攒满 512 条才触发一次更新"；真正触发一次参数更新的粒度是下面的 mini-batch。训练时会再逐级切分：
  - `ppo_mini_batch_size`：把 global batch 切成 `全局条数 / ppo_mini_batch_size` 个 mini-batch，**每个 mini-batch 触发一次 `optimizer.step()`**（`verl/workers/engine_workers.py` 的 `train_mini_batch` 对 mini-batch 做 `for` 循环，每次迭代调用一次 `train_batch`，一次 `train_batch` 就是一次参数更新）。所以同一轮 rollout 的数据可能被拆成好几个 mini-batch、做好几次梯度更新——这是 PPO 允许"同批数据多次更新、靠 clip 限制偏离"的地方。
  - `ppo_micro_batch_size_per_gpu`：mini-batch 再按单卡显存切成 micro-batch，纯粹是梯度累积（多次 forward+backward，只在**这个 mini-batch 的所有 micro-batch 都跑完后**才触发唯一一次 `optimizer.step()`），micro-batch 边界本身不独立触发更新。
  - 落到本脚本：`train_prompt_mini_bsz=64` 乘以 `rollout.n=8` 后等于 512，正好等于 global batch，所以这份配置下一轮 rollout 只切出 1 个 mini-batch，等价于"整批只更新 1 次"——这是这份具体配置的结果，不是机制本身；把 `train_prompt_mini_bsz` 调小会得到多次更新。脚本用的是 `use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu` 按 token 数动态打包，角色和固定条数的 `ppo_micro_batch_size_per_gpu` 等价，都不影响"mini-batch = 一次更新"这个边界。
- Dataloader 默认也按它取数据：`data.get("gen_batch_size", data.train_batch_size)`（`psrl/psrl/utils/dataset/data_processor.py:248`）；只有开启 redundant rollout 时才改用 `redundant_train_prompt_bsz`（对应 `psrl.rollout_coordination.redundant_rollout.redundant_global_batch_size`）——多采样一些 prompt/response 来对冲长尾延迟，机制见[生成长尾](ray-and-rollout.md#为什么-rl-rollout-会有生成耗时不均long-tail-latency)。

### 记忆点

`train_prompt_bsz` 计的是 **prompt 数**，不是 response 数；真正喂进优化器的样本数要再乘 `rollout.n`。改这个值时，脑子里要同时算一下 `train_prompt_bsz * rollout.n` 是否还能被 GPU 数整除，否则 `validate_config` 会直接 assert 失败。

## 查一个训练 pid 的运行情况：去哪儿看、看什么

### 一句话结论

没有单一入口能回答"这个训练还好吗"，需要从**进程存活、Ray 集群资源、Ray actor 明细、训练日志、GPU 显存**五个不同层面交叉验证，因为多机场景下跑命令的本机 shell 通常不是训练进程实际所在的机器，必须先 ssh 到正确节点。

### 第一步：确认自己在不在正确的机器上

多机拓扑下，起 Ray 集群/跑脚本的 head 节点、真正跑训练 driver 的节点、终端/Bash 工具所在的本机，往往是三台不同机器（甚至 head 节点自己的 GPU 也可能被 `_GPUSlotReserver` 占位而完全不跑这个任务的实际 worker，见[Ray 调度流程](ray-and-rollout.md#ray-的资源调度与应用启动流程)"占坑防调度"）。本地直接 `ps`/`nvidia-smi` 大概率什么都看不到，第一件事永远是先确认 pid 是在哪台机器上启动的（看启动命令里的 `ssh <ip> "nohup ... &"`，或者查 Ray 节点列表逐个 ssh 排查），再 `ssh <该节点IP> "..."` 执行后面所有检查。

### 分层检查清单

| 层面 | 命令 | 回答什么问题 |
|---|---|---|
| 进程本身活不活 | `ssh <node_ip> "ps -o pid,etime,cmd -p <pid>"` | 进程还在不在、跑了多久；不在了就该看它是怎么退出的（退出码、有没有留 core dump） |
| Ray 集群资源汇总 | `ssh <head_ip> "ray status --address=<head_ip>:8887"`（先 `source .../env/psrl.sh` 激活对应 conda env） | 集群还有几个节点存活、GPU/CPU 总用量是否符合预期（比如 2 节点任务该用 16/24 卡，不是全 0 也不是意外占满） |
| Ray actor 明细 | `ssh <node_ip> "ps -ef \| grep 'ray::'"`，对比各 actor 的启动时间（`ps` 的 `etime`/`STIME` 列） | 定位"卡在哪一步"：`_GPUSlotReserver` → `TaskRunner.run` → `PSManager` → `RolloutCoordinator`/`PSRL_AgentLoopWorker` 这几个控制面 actor 是按顺序陆续起来的，哪个迟迟没出现就是卡在它的前一步 |
| 训练日志 | `tail -f`/`grep` 脚本里 `tee` 出来的 `${experiment_name}.log`（外加 `nohup ... > xxx.nohup.log` 的重定向文件，内容基本重复） | 看每步指标（[训练日志与指标](#如何阅读训练日志抓取每步指标)的长行格式）、报错关键字（`error/traceback/exception/oom`）、Agent Loop 里单条 episode 的完成情况（`Harbor episode completed: reward=...`） |
| GPU 层面 | `ssh <node_ip> "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"`，训练涉及的**每一台**节点都要查 | 显存/利用率是不是真的在算，还是进程活着但 GPU 空转（启动阶段正常是 0，模型加载完之后应该稳定爬升到接近 `gpu_memory_utilization` 设的比例） |
| 训练质量曲线 | wandb 网页端（`trainer.logger` 配了 `wandb` 才有），或 Ray dashboard `http://<head_ip>:8265` | 进程"活着"不等于"训练健康"（参见[多机训练检查清单](startup-incident-reviews.md#lapsrepair_easy-多机训练踩坑复盘从-max_model_len-崩溃到重启给下一次训练任务的检查清单)熵崩溃案例），这两者能看到 reward/entropy 等指标随 step 变化的趋势图，比 grep 文本日志直观 |

### 判断"卡住了" vs. "只是慢"的经验法则

单看训练日志最后一行没有更新，不能立刻断定卡死——`main_ppo.py` 的启动阶段（Ray 连接 → GPU slot 预留 → 各控制面 actor 依次起来 → 模型真正加载进显存）本身可能持续几分钟到十几分钟，尤其模型更大、拓扑更复杂（比如 TP>1、多 rollout 实例）时更慢。正确的做法是同时看"Ray actor 明细"里最新出现的 actor 时间戳——只要还在陆续冒出新 actor，就是正常推进，而不是卡死；如果连最新一个 actor 之后也过了远超预期的时间（比如几十分钟）都没有新 actor、GPU 显存还是 0，才需要怀疑真的卡住了，再去查 Ray dashboard 或对应 actor 的日志定位。

### 延伸

- 关于自动化巡检（不用自己一直盯着看）：用 `CronCreate` 建一个定时任务，让它定期跑上面这套分层检查，只在真的有问题（进程死了、日志有报错、GPU 该有显存却是 0）或有里程碑（比如跑出第一个 training step）时才主动通知，正常就不打扰——见[多机训练检查清单](startup-incident-reviews.md#lapsrepair_easy-多机训练踩坑复盘从-max_model_len-崩溃到重启给下一次训练任务的检查清单)检查清单第 7 条。
- 每步指标怎么从日志里抓、关注哪些 key，见[训练日志与指标](#如何阅读训练日志抓取每步指标)。
