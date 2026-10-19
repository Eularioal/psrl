# PSRL 单机与多机启动故障复盘

> **核心结论**：保留 IP、数据格式、环境变量覆盖及上下文长度问题的历史故障链、修复过程与验证边界。

## 目录

- [单机 Demo 踩坑记录：两处根因与修复](#单机-demo-踩坑记录两处根因与修复)
  - [启动全流程（跑通之后回看）](#启动全流程跑通之后回看)
  - [报错一：`AssertionError: PSManager IP xxx not found in ray nodes`](#报错一assertionerror-psmanager-ip-xxx-not-found-in-ray-nodes)
  - [报错二：DataLoader worker 里 `AttributeError: 'str' object has no attribute 'get'`](#报错二dataloader-worker-里-attributeerror-str-object-has-no-attribute-get)
  - [教训](#教训)
- [让 `qwen2.5_3b_fsdp.sh` 跑通 2 节点实验：脚本改了什么](#让-qwen25_3b_fsdpsh-跑通-2-节点实验脚本改了什么)
  - [一句话结论](#一句话结论)
  - [根因](#根因)
  - [三处修改](#三处修改)
  - [验证](#验证)
  - [教训](#教训-1)
- [laps/repair_easy 多机训练踩坑复盘：从 `max_model_len` 崩溃到重启，给下一次训练任务的检查清单](#lapsrepair_easy-多机训练踩坑复盘从-max_model_len-崩溃到重启给下一次训练任务的检查清单)
  - [一句话结论](#一句话结论-1)
  - [时间线](#时间线)
  - [改脚本 vs. 改调用参数：为什么选后者](#改脚本-vs-改调用参数为什么选后者)
  - [检查清单：下次启动一个新的训练任务前](#检查清单下次启动一个新的训练任务前)
  - [记忆点](#记忆点)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。旧的中文章节编号已改为主题标题，正文引用已改为可点击链接。

[返回专题索引](../psrl-run-guide.md)

运行前先阅读[共享环境与集群检查](cluster-preflight.md)；文中的历史地址、依赖和机器状态需要重新核对。

## 单机 Demo 踩坑记录：两处根因与修复

跑 `qwen2.5_3b_fsdp_singlenode_demo.sh` 时依次遇到两个报错，都是环境/数据问题而非代码逻辑 bug，记录下来避免下次重复排查。

### 启动全流程（跑通之后回看）

1. 探测 `LOCAL_IP` → 本地起 Ray（`ray_init.address=local`）→ Hydra 把脚本参数覆盖进 `ppo_trainer.yaml`。
2. `TaskRunner` 建 worker pool：2 卡跑 `PSRL_vLLMHttpServer`（rollout，显存占用高，因为 `gpu_memory_utilization=0.9`），2 卡跑 `FusedWorker_actor_dummy`（FSDP2 训练）。
3. 加载 GSM8K parquet 数据集，建 DataLoader。
4. 训练侧 FSDP 初始化完成后，通过 **NIXL** 把权重推给 `PSManager`，rollout 侧的 vLLM 实例再拉取对齐。
5. 训练循环 `trainer.total_training_steps=3`：rollout 生成 → reward → GRPO advantage → FSDP 反向更新 actor → NIXL 把新权重推给 rollout。第 3 步夹一次验证。
6. 3 步跑完，停 PSManager/NIXL，打印 `Training completed successfully!`，退出码 0。

### 报错一：`AssertionError: PSManager IP xxx not found in ray nodes`

- **现象**：`ray_trainer.py` 里 `assert self.config.psrl.ps_manager_ip in ip_to_node_id` 断言失败。
- **根因**：脚本用 `LOCAL_IP=$(hostname -I | awk '{print $1}')` 取本机 IP，依赖 `hostname -I` 输出顺序。这台机器有上百个虚拟/桥接网卡，真实可路由 IP 没有排在最前面，导致取到了一个虚拟网卡 IP，而 Ray 节点注册用的是真实可路由 IP，两者对不上。
- **修复**：改用不依赖网卡枚举顺序的路由查询法——建一个 UDP socket "连"外部地址（不会真发包，只走内核路由选择），读出内核选定的本地出口 IP：
  ```bash
  LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
  ```

### 报错二：DataLoader worker 里 `AttributeError: 'str' object has no attribute 'get'`

- **现象**：`verl` 的 `rl_dataset.py` 里 `row_dict.get("extra_info", {}).get("index", 0)` 对着字符串调 `.get()` 报错，且发生在 DataLoader 子进程里被上层重试吞掉，刷出几千万行日志。
- **根因**：`psrl_demo/data/gsm8k/{train,test}.parquet` 的 `prompt`/`reward_model`/`extra_info` 三列本该是 Arrow 原生 struct/list，实际存成了 JSON 字符串，`datasets.load_dataset` 读出来仍是字符串，与 veRL 期望的 dict/list 契约不符。
- **修复**：不改第三方 `verl` 代码加防御判断，而是修数据源——用 pandas 把这三列 `json.loads` 解码回原生类型再落盘（原文件备份到 `.bak/`）。

### 教训

两处都属于"环境/数据契约"层面的坑，而不是训练算法逻辑问题：
- 依赖 `hostname -I`/`ip addr` 之类会受网卡数量和顺序影响的命令来定位"本机 IP"不可靠，应该用路由表查询（`ip route get` 或 socket connect 技巧）。
- 用第三方框架的数据 loader 时，数据文件的列类型必须严格匹配其假设（这里是"嵌套字段必须是原生 Arrow 类型，不能是 JSON 字符串"），生成数据集的脏活最好在预处理脚本里保证类型，而不是等运行时报错才发现。

## 让 `qwen2.5_3b_fsdp.sh` 跑通 2 节点实验：脚本改了什么

### 一句话结论

脚本原本要求调用者自己导出一堆环境变量才能跑（`PSRL_WORKSPACE`、`LOCAL_IP` 等），且这些变量会被脚本内部 `source` 的公共环境脚本悄悄覆盖；改成脚本自身负责计算和保护这些变量后，调用方只需要设置 `PSRL_WORKSPACE` 一个变量即可运行。

### 根因

脚本里 `source ${PSRL_WORKSPACE}/env/psrl.sh` 会连带 source 到上级的公共 `activate` 脚本，而那个脚本会无条件地把 `PSRL_WORKSPACE` 重新导出成它自己默认的工作区路径，覆盖掉调用者原本设置的值。脚本后面所有依赖 `PSRL_WORKSPACE` 算出来的路径（模型路径、训练/测试数据路径、日志路径）都因此指向了错误的目录，报 `FileNotFoundError`。

### 三处修改

1. **保护 `PSRL_WORKSPACE` 不被 `source` 链路覆盖**：
   ```bash
   : "${PSRL_WORKSPACE:?PSRL_WORKSPACE must be set...}"
   _psrl_workspace=${PSRL_WORKSPACE}
   source ${PSRL_WORKSPACE}/env/psrl.sh
   export PSRL_WORKSPACE=${_psrl_workspace}   # 恢复被覆盖的值
   ```
   `source` 前把值存进临时变量，`source` 后立刻恢复；同时用 `:${VAR:?msg}` 强制要求调用者显式传入，缺失时给出清晰报错而不是静默用错路径。

2. **`LOCAL_IP` 改为脚本内自计算**，复用单机 demo 脚本的做法（见[单机启动故障复盘](#单机-demo-踩坑记录两处根因与修复)报错一）：
   ```bash
   LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
   ```
   不再要求调用方手动导出，避免因取错网卡 IP 导致和 Ray 实际注册的节点 IP 对不上而报 `PSManager IP not found`。

3. **日志/checkpoint 路径改为跟着 `PSRL_WORKSPACE` 走，而不是跟着 Python 包安装路径走**：删掉了通过 `python -c "import psrl..."` 反推安装目录的写法（以及从未真正生效的 `HOME=` 赋值），`logging_path`、最终 `tee` 的日志文件、`CKPT_ROOT` 默认值统一改成基于 `${PSRL_WORKSPACE}`，并补上 `mkdir -p ${PSRL_WORKSPACE}/logs`。

### 验证

2 节点（16 GPU：8 卡训练 + 8 卡生成）实验用改后的脚本跑完全部训练步数，训练/验证/wandb 上报均正常退出，过程中无报错。

### 教训

和[单机启动故障复盘](#单机-demo-踩坑记录两处根因与修复)的单机坑同一类问题：多层 `source` 链路里，下游脚本对"某个环境变量已经是我要的值"的假设很容易被上游脚本悄悄打破。凡是脚本要 `source` 一个不受自己控制的公共环境脚本，且自身行为依赖某个变量的具体值时，都应该在 `source` 前后做保存/恢复，不能假设 `source` 只会新增变量、不会覆盖变量。

## laps/repair_easy 多机训练踩坑复盘：从 `max_model_len` 崩溃到重启，给下一次训练任务的检查清单

### 一句话结论

`examples/sciaccel_rl/fsdp_qwen2.5_3b_laps.sh` 是从另一个模型的脚本（`fsdp_qwen35_4b.sh`）改造来的，改造时只调了模型路径，没有重新核对**目标模型自己的上下文窗口上限**，导致 `max_prompt_length + max_response_length`（67584）超过 Qwen2.5-3B 的 `max_position_embeddings`（32768），8 个 vLLM server 在启动阶段全部 `ValueError` 崩溃；修复只用脚本本身已经预留的 `MAX_RESPONSE_LENGTH` 环境变量覆盖（改成 `30720`，使 `2048+30720=32768` 刚好等于上限），不改脚本默认值，重新以相同拓扑重启后 `[validate_config] All configuration checks passed successfully!`，顺利通过了此前崩溃的那一步。详细的字段含义和 RoPE 原理见 [hf-config-context-window.md](../hf-config-context-window.md)。

### 时间线

| 时刻 | 事件 |
|---|---|
| 2026-09-17 | 首次启动，`max_response_length=65536`（继承自 `fsdp_qwen35_4b.sh` 的默认值），8 个 vLLM rollout/validate server 在 `__init__` 阶段全部抛出 `ValueError: max_model_len (67584) should be less than or equal to max_position_embeddings (32768)`，级联触发 `ActorDiedError`，训练 driver 退出 |
| 2026-09-18 | 确认 Ray 集群仍空闲（`ray status` 显示 3 节点 24 GPU 全部 0 占用，`main_ppo` 无残留进程）后，用 `MAX_RESPONSE_LENGTH=30720` 覆盖重启，`validate_config` 通过，进程正常拉起 rollout/validate/PS 各角色 |

### 改脚本 vs. 改调用参数：为什么选后者

`fsdp_qwen2.5_3b_laps.sh` 里 `max_response_length` 本来就写成了 `${MAX_RESPONSE_LENGTH:-65536}`——脚本作者已经预留了外部覆盖的口子，只是默认值是从别的模型抄过来的、没人替 Qwen2.5-3B 校正过。既然口子已经在，就没必要改脚本本身（改了会影响这份脚本以后被用在别的、真的需要大 `max_response_length` 的模型上），直接在调用时传 `MAX_RESPONSE_LENGTH=30720` 即可——这也是仓库里"能用参数覆盖就不碰脚本默认值"的一贯做法（同一份脚本很多变量都是 `${VAR:-default}` 形式）。

### 检查清单：下次启动一个新的训练任务前

1. **核对上下文窗口预算，尤其是"照抄别的模型脚本"时**：`max_prompt_length + max_response_length`（进而 `max_model_len`）必须 `≤` 目标模型 `config.json` 里的 `max_position_embeddings`（没有配 `rope_scaling` 时是硬上限）。换模型 = 重新核对这一项，不能假设别的模型脚本里的默认值天然适用。
2. **改脚本前先看有没有现成的环境变量覆盖口子**（`grep '\${.*:-' script.sh`）；有就传参数，没有再考虑改默认值——改默认值会影响脚本对其他场景的复用性。
3. **复用集群前先确认它真的空闲**：`ray status`（3 节点/GPU 数是否符合预期、`Total Usage` 是否接近 0）+ `ps aux | grep main_ppo`（确认没有残留进程）+ `nvidia-smi`（两台机器 GPU 利用率/显存是否都是 0），三者都确认了再重用，不要凭"看起来没人用"就直接抢占共享集群资源。
4. **`LOCAL_IP` 多机场景必须显式传**：单机脚本能自动探测（见[单机启动故障复盘](#单机-demo-踩坑记录两处根因与修复) UDP socket 技巧），但 `fsdp_qwen2.5_3b_laps.sh` 这类多机脚本默认 `LOCAL_IP=127.0.0.1`，必须显式传成头节点的真实可路由 IP，否则 worker 节点连不上 `ps_manager`（`PSManager IP xxx not found` 类报错，见[多机启动脚本修复](#让-qwen25_3b_fsdpsh-跑通-2-节点实验脚本改了什么)同源问题)。
5. **后台启动用 `nohup setsid ... & disown` 三件套**，并把 stdout/stderr 重定向到固定路径的日志文件，这样断开 SSH 连接、甚至当前 Claude 会话结束都不影响训练进程存活；重启前记得把旧崩溃日志 `mv` 走而不是覆盖（`tee` 不是 `tee -a`，直接重跑会连崩溃现场一起冲掉）。
6. **别只看"进程还活着"就判定训练健康**：这次崩溃发生在启动阶段（`validate_config`/vLLM 初始化），下一类常见故障发生在训练中途——[dapo-training-collapse-diagnosis.md](../dapo-training-collapse-diagnosis.md) 记录过一次 DAPO+Qwen2.5-3B 训练在 step~115 由**熵崩溃**触发的不可逆连锁崩溃（`entropy`→0、`response_length` 打满、`reward`→-1、`grad_norm` 爆炸），过程中进程本身没有退出、只是指标全废。所以监控除了"进程存活 + GPU 利用率"，还必须盯 `actor/entropy`、`response_length/clip_ratio`、`group/zero_variance_fraction`、`grad_norm` 这几个指标（见[训练日志与指标](monitoring-and-batches.md#如何阅读训练日志抓取每步指标)指标表），越早发现越有救（该案例是过了 40% 训练量才发现，等于白烧了近一半算力）。
7. **建立自动化的定期巡检**，不要指望自己全程盯着：本次用 `CronCreate` 设了一个每 20 分钟检查一次日志/GPU/进程的定时任务，正常就极简汇报一句，异常或训练结束才详细汇报并自动清理，比人工不定期看一眼更不容易错过第 6 条里说的那类"进程没死但指标已经废了"的情况。

### 记忆点

崩溃分两大类，排查思路不一样：**启动期崩溃**（配置/资源不匹配，比如这次的 `max_model_len`，或第八/十二节的 IP/数据类型问题）通常几秒内就能从日志报错定位到根因，修好参数重启即可；**训练中期崩溃**（熵崩溃、reward 坍缩这类）不会报错、进程也不退出，只能靠持续盯几个关键指标的变化趋势发现，且往往不可逆（回滚到出问题前的 checkpoint 重新调超参，而不是"再等等它会不会自己恢复"）。
