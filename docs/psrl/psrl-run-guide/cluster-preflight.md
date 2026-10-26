# PSRL 共享环境与集群检查

> **核心结论**：运行前固定验收目标、预算、源码依赖和实际节点；运行中区分监控故障与训练故障；结束后核实物理资源释放。2026-10-18 补充真实 science 对照复盘：占位要衔接实际任务，账本正确不等于性能更好，历史机器状态不能当作当前保证。

> 2026-10-20 排查补充：本环境 Docker 残留线程卡在内核挂载清理锁等待，Docker 的 `fuse-overlayfs` 同时卡在写回等待。已确认 SIGKILL 退出状态，最初触发者和信号发送者仍未知，不能直接归因于 OOM，详见第 19 节。

## 目录

- [共享 conda env 里 `pip install -e` 是全局的：为什么日志/checkpoint 会悄悄跑到别人的目录下](#共享-conda-env-里-pip-install--e-是全局的为什么日志checkpoint-会悄悄跑到别人的目录下)
  - [一句话结论](#一句话结论)
  - [背景：`pip install -e` 为什么是"全局单一状态"](#背景pip-install--e-为什么是全局单一状态)
  - [排查过程](#排查过程)
  - [解决方案：临时 vs. 根治](#解决方案临时-vs-根治)
  - [记忆点](#记忆点)
- [集群踩坑总表与最小复现（2026-10-13）](#集群踩坑总表与最小复现2026-10-13)
  - [16.1 本次真正检查了什么](#161-本次真正检查了什么)
  - [16.2 环境激活与源码来源：命令没错，跑的却是另一份代码](#162-环境激活与源码来源命令没错跑的却是另一份代码)
  - [16.3 多网卡 IP：`hostname -I` 第一个地址不等于 Ray 节点地址](#163-多网卡-iphostname--i-第一个地址不等于-ray-节点地址)
  - [16.4 Ray：隔离了临时目录，子进程仍可能加入旧集群](#164-ray隔离了临时目录子进程仍可能加入旧集群)
  - [16.5 临时路径过长，以及共享文件系统不适合放运行时 socket](#165-临时路径过长以及共享文件系统不适合放运行时-socket)
  - [16.6 Docker：socket 文件存在，不代表服务存活](#166-dockersocket-文件存在不代表服务存活)
  - [16.7 GPU 身份和远程遥测：申请的卡、监控的卡、挂载的卡必须是同一张](#167-gpu-身份和远程遥测申请的卡监控的卡挂载的卡必须是同一张)
  - [16.8 清理：租约账本归零，不等于容器已经退出](#168-清理租约账本归零不等于容器已经退出)
  - [16.9 Parquet：JSON 字符串与真正的字典不是一种数据类型](#169-parquetjson-字符串与真正的字典不是一种数据类型)
  - [16.10 上下文越界与训练 OOM：两个不同阶段，分别算预算](#1610-上下文越界与训练-oom两个不同阶段分别算预算)
  - [16.11 张量形状：为什么只有加载补丁依赖才真正修好](#1611-张量形状为什么只有加载补丁依赖才真正修好)
  - [16.12 全零 mask 与无限补采：有轨迹也可能没有可训练 token](#1612-全零-mask-与无限补采有轨迹也可能没有可训练-token)
  - [16.13 不报错也会浪费算力：奖励基线、指标与验收边界](#1613-不报错也会浪费算力奖励基线指标与验收边界)
  - [16.14 下次怎么最快执行检查](#1614-下次怎么最快执行检查)
  - [16.15 短响应验证先核奖励配置](#1615-短响应验证先核奖励配置)
  - [16.16 独立 Ray 不等于物理 GPU 独占](#1616-独立-ray-不等于物理-gpu-独占)

- [17. Science 训练复盘与下次执行规则](#17-science-训练复盘与下次执行规则)
- [18. 跨任务池压力实验与事件循环归属](#18-跨任务池压力实验与事件循环归属)
- [19. 本机 Docker 挂起原因排查（2026-10-20）](#19-本机-docker-挂起原因排查2026-10-20)
- [20. Docker 恢复尝试与 inode 补充（2026-10-20）](#20-docker-恢复尝试与-inode-补充2026-10-20)

本篇由原长文按章节迁移，保留原有例子、推导和历史证据。本次仅整理结构，未重新验证技术结论或重跑实验。旧的中文章节编号已改为主题标题，正文引用已改为可点击链接。

[返回专题索引](../psrl-run-guide.md)

运行前先完成本文对应的检查；文中的历史地址、依赖和机器状态需要重新核对。

## 共享 conda env 里 `pip install -e` 是全局的：为什么日志/checkpoint 会悄悄跑到别人的目录下

### 一句话结论

`fsdp_qwen35_4b_laps.sh` 等启动脚本用 `python3 -c "import psrl; ..."` 反查自己仓库的根目录（`PSRL_PATH`），但这条反查依赖的是**共享 conda env 里 `psrl` 当前被 `pip install -e` 到谁的 checkout**——这是一个全局单一状态，不跟"你在哪个目录下敲命令"绑定，谁最后装的就全局生效到谁，导致脚本明明是从 `lw` 这份代码里跑的，日志、checkpoint、`reward.py`、chat template 全部却悄悄写到了 `lhy` 的目录树下，且**不报错、没有任何提示**。修复思路是不去碰共享安装（会影响同一个 env 的其他用户），而是给这次启动单独加一个 `PYTHONPATH` 环境变量，让 Python import 优先命中 `lw` 自己的代码；根治则是把脚本自身改成"用脚本文件自己的路径定位仓库根目录"，不再依赖 `import psrl` 这条不稳定的反查链。

### 背景：`pip install -e` 为什么是"全局单一状态"

`pip install -e <dir>` 装的是"可编辑安装"——它不复制代码到 site-packages，而是让 `import psrl` 直接指向 `<dir>` 里的源码。这个指向关系记在 conda env 自己的 site-packages 元数据里（新版 pip 是 `__editable__.psrl-*.pth`/`finder` 文件），是**这个 conda env 的全局属性**，不是"当前工作目录"或"当前脚本"的属性。如果 `lw` 和 `lhy` 两个用户共用同一个 conda env（这里是 `psrl_smg_new`），谁最后跑了一次 `pip install -e .`，之后**任何人**在这个 env 里 `import psrl`，不管他此刻站在哪个目录、跑的是哪个仓库里的脚本，解析到的都是那"最后一次安装"指向的那份代码。

### 排查过程

1. 第二次启动 Qwen3.5-4B 训练（把 `MAX_MODEL_LEN` 从 `65536` 调到 `131072` 之后）看着一切正常，直到检查日志才发现：`main_ppo.py` 的报错行前缀是 `${PSRL_WORKSPACE}/psrl/trainer/main_ppo.py`，而不是预期的 `lw` 路径。
2. 对比脚本 stdout 里 `+ PSRL_PATH=...` 那一行：上一次（第一次崩溃前的）启动解析出来是 `lw`，这次变成了 `lhy`——同一份脚本、同一个人操作，中间只是隔了几个小时，解析结果就变了，说明这个值不是脚本自己决定的。
3. `pip show -f psrl` + `python3 -c "import psrl; print(psrl.__file__)"` 确认：conda env `psrl_smg_new` 当前的可编辑安装指向 `${PSRL_WORKSPACE}`。大概率是这段时间内 `lhy` 在自己的 checkout 里重新跑过一次 `pip install -e .`，把这个共享 env 的指向从 `lw` 改成了 `lhy`（这个操作对当事人自己往往毫无感觉——他只是在自己的目录里装了一下自己的包）。
4. 用 `diff` 确认两份 `reward.py` 当时内容一致，所以这次训练结果本身没受影响，但checkpoint/日志路径已经写错了地方，且这个"写错地方"没有任何报错，纯靠肉眼对比日志路径才发现。

### 解决方案：临时 vs. 根治

**临时**（不改脚本，只改这一次的启动命令）：给 `python3 -m psrl.trainer.main_ppo` 之前的 shell 加一个 `PYTHONPATH`，指向想要生效的那份 checkout：

```bash
PYTHONPATH=${PSRL_WORKSPACE} \
  LOCAL_IP=... AGENT_NODE_IPS=... MAX_MODEL_LEN=131072 \
  nohup bash fsdp_qwen35_4b_laps.sh > fsdp_qwen35_4b_laps.nohup.log 2>&1 &
```

原理：Python 解析 `import psrl` 时，显式设置的 `PYTHONPATH` 目录会排在 site-packages 的可编辑安装记录前面被优先命中，不需要动共享 env 里的任何东西，也不会影响同一个 env 的其他用户（比如 `lhy`）。这条路径只对这一次 shell 里跑起来的进程生效。

**根治**（改脚本本身，写进了 `fsdp_qwen35_4b_laps.sh` / `fsdp_qwen2.5_3b_laps.sh` / `fsdp_qwen35_4b.sh` / `eval_qwen35_4b.sh` 这四个脚本）：把 `PSRL_PATH` 的来源从"反查共享 env 里 `import psrl` 解析到谁"，换成"脚本自己的文件路径在哪"，从根上切断对共享可编辑安装状态的依赖：

```bash
# 用脚本自身文件路径定位仓库根目录，不依赖共享 env 里 psrl 当前指向谁
PSRL_PATH=${PSRL_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PYTHONPATH="${PSRL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
```

`${BASH_SOURCE[0]}` 是当前脚本自己的路径，不管这个 conda env 里可编辑安装当前指向谁、也不管是谁最后跑过 `pip install -e`，只要你执行的是 `lw` 这份 checkout 里的脚本，它就一定用 `lw` 自己的代码——每份 checkout 自带自己的路径解析，互不干扰，以后也不需要每次手动补 `PYTHONPATH`。

### 记忆点

**共享 conda env + 可编辑安装是一个隐藏的"全局单例"陷阱**：只要多个人（或多份 checkout）共用同一个 env 且都对同一个包做过 `pip install -e`，"我在跑哪份代码"这件事就不再由你自己的操作决定，而是由"最后一次全局安装是谁做的"决定，而且不会报错、只会安静地把产出（日志/checkpoint/被 import 的模块代码）写到别的地方。排查这类问题的信号是：**同一条命令、同一个人跑，结果路径却变了**——这几乎总是外部共享状态被别人动过，第一反应应该是 `pip show -f <pkg>` 或 `python3 -c "import pkg; print(pkg.__file__)"` 确认当前解析到哪，而不是去怀疑脚本本身的逻辑。长期修法永远是让脚本"自举"定位自己的依赖（用 `${BASH_SOURCE[0]}`/`__file__` 之类锚定在自身路径上），而不是依赖任何外部、全局、可被别人悄悄改变的状态。

## 集群踩坑总表与最小复现（2026-10-13）

**一句话结论：先确认“实际运行哪份代码、连哪个 Ray、在哪台机器用哪张卡”，再确认数据与训练预算。当前最直接的阻塞是本机 Docker 无响应，以及普通共享环境仍加载未修复的 veRL 张量代码。**

这节汇总本工作区现存日志、当前源码和本次最小复现。它不是全体集群节点的健康证明。检查基于 `90a4178` 加当时未提交改动，工作树快照保存在探针 JSON 中。未提交的修复和外置补丁依赖不属于上游提交自带能力。

### 16.1 本次真正检查了什么

为了快速得到结论，本次使用 CPU 小样本、带超时的只读系统查询和已有回归测试，没有重新启动完整训练、制造 GPU OOM、重启 Docker 或清理共享 Ray。

| 类型 | 本次结果 | 能说明什么 |
|---|---|---|
| 现场环境探针 | 完整一次约 61.6 秒，含环境激活、导入及 Docker 超时 | 本节点当前状态及几个故障的最小触发条件 |
| 默认依赖回归 | 62 项中 60 通过、2 失败，约 51.0 秒 | GPU overlay、Gate B 基础行为和清理逻辑回归通过，但默认 veRL 的张量问题仍存在 |
| 指定补丁依赖回归 | 42 项全部通过，约 50.5 秒 | 补丁张量路径、空 mask、补采熔断和超长过滤相关单测通过 |
| 历史实机证据复核 | 容器、多节点和单步训练已有原始记录 | 只能证明那些运行及其配置，不等于本次重新做了实机训练 |

上面的耗时分别来自探针和 pytest，不是整个整理任务的总耗时。两批测试覆盖范围不同，不能写成“同一批 62 个测试全部修好了”。

复现材料：

- [最小探针源码](../cluster-pitfall-checks/probe.py) 与 [结构化结果](../cluster-pitfall-checks/results-2026-10-13.json)。JSON 中存在 `error` 是被记录的预期故障，不能只看探针退出码判断环境健康。
- [源码核对记录](../cluster-pitfall-checks/source-checks-2026-10-13.json) 保存当前 Ray 临时目录选择函数及依赖位置。
- 默认依赖测试日志（本地资料，未随仓库发布） 与 补丁依赖测试日志（本地资料，未随仓库发布）。
- [早期完整验证记录](../../env-worker-deep-dive/env-service-verification/README.md)，以及本节后文链接的较新 `fixed-*` 原始证据。早期文档里的“尚未修复”不自动代表当前状态。

### 16.2 环境激活与源码来源：命令没错，跑的却是另一份代码

**现象一：模型、数据或日志突然指向别人的工作区。** 本次在子 shell 中先设 `PSRL_WORKSPACE` 为本工作区，再 source 共享脚本，实际值被改为共享环境维护者的工作区。`source` 会在当前 shell 执行赋值，所以它可以覆盖变量，不只是“打开 Python 环境”。

**现象二：改了自己的代码，实验仍表现得像没改。** 从 `/tmp` 启动 Python、不设 `PYTHONPATH` 时，`psrl`、`verl`、`transfer_queue` 都来自共享环境的另一份 checkout。仅加入自己的 PSRL 根目录后，`psrl` 来源正确，但后两个依赖仍来自共享安装。

例如，你的 `psrl` 调用了 `verl.utils.tensordict_utils`。即便前者是新代码，后者仍可能是旧函数，因此 PSRL 层的修复和依赖层的修复必须分别确认。当前 `examples/sciaccel_rl/fsdp_qwen35_4b.sh` 仍有通过 `import psrl` 反推默认 `PSRL_PATH` 的代码，不能照搬前文“脚本已全部自定位”的历史结论。

启动前在 source 后恢复实验工作区，显式设置 `PSRL_PATH`、`PYTHONPATH`、模型及输出目录，并打印三个包的真实来源。多机实验还要在 worker 上检查同样信息。不要通过修改共享环境的 editable install 来修复个人实验。

### 16.3 多网卡 IP：`hostname -I` 第一个地址不等于 Ray 节点地址

本次 `hostname -I` 返回 326 个地址，第一个为 `192.168.1.254`。默认外出路由选择的地址则是 `192.168.1.11`，接口为 `bond1`。这复现了“取第一个 IP 会选到不同地址”的前提，没有为了触发断言而启动 Ray。

PSManager 是参数服务器的管理进程。训练器会用其配置 IP 在 Ray 节点列表中查找目标节点，地址不匹配就可能报 `PSManager IP ... not found in ray nodes`。跨节点传 `127.0.0.1` 也不行，因为每台机器都会把它理解为自己。

正确顺序是：查询到目标集群节点的路由，核对 Ray 的 `NodeManagerAddress`，再确定 PSManager 放在哪台节点。`ip route get 1.1.1.1` 适合确认本机默认出口，但多机通信还应查询实际对端地址，不能保证默认外网出口就是集群通信网卡。

### 16.4 Ray：隔离了临时目录，子进程仍可能加入旧集群

Ray 负责把 worker 进程分配到节点。历史实验中主进程创建了隔离集群，但 vLLM 子进程没有得到明确的 `RAY_ADDRESS`，读到 `/tmp/ray/ray_current_cluster` 的旧地址，结果一直连接超时。主进程连对集群，不代表它启动的所有进程都会自动连对。

运行时应把同一地址传到 driver、Ray runtime environment（worker 的运行环境）及 vLLM 子进程。当前工作树中的 `main_ppo.py` 已有向 TaskRunner 传递实际 GCS 地址的修改，仍需确认执行的是这份代码。GCS 是 Ray 保存节点与任务元信息的服务。

另一个小坑发生在配置阶段：OmegaConf 的 struct 模式禁止增加未声明键。本次用十几行 CPU 代码重新触发了加入 `TRANSFER_QUEUE_ENABLE` 时的 `ConfigKeyError`，转成普通字典后赋值成功。当前 `main_ppo.py` 已有对应转换。这个错误和 Ray 网络不通是两件事，不能都用“重启集群”处理。

证据见历史 显式 Ray 地址运行日志（本地资料，未随仓库发布）。本次仅复现配置错误，没有新建 Ray 集群。

### 16.5 临时路径过长，以及共享文件系统不适合放运行时 socket

Ray 的进程间通信会用 Unix socket，也就是以文件路径命名的本机通信端点。Linux 的普通 pathname socket 路径最多约 107 字节。长工作区路径再拼接 session、socket 名称，很容易越界。

本次真实调用 `socket.bind()`：长路径得到 `AF_UNIX path too long`，同一临时目录下的短路径绑定成功。这里只需要 CPU 和 `/tmp`，不必等 Ray 启动后报错。

运行时 socket、编译 scratch 和短期缓存放节点本地短路径。需要跨节点读取的数据、模型与结果放共享存储，因为 A 节点的 `/tmp/example` 不等于 B 节点的同名路径。旧 runbook 还记录过共享 FUSE 挂载在负载下出现 `ENOSPC`，本次未制造磁盘或 inode 耗尽，不能据此断言当前共享存储仍有该故障。

**纠正旧说明：当前安装的 Ray 的 `ray._common.utils.get_user_temp_dir()` 会先读取 `RAY_TMPDIR`，其次读取 Linux 的 `TMPDIR`。** 因而只改 `TMPDIR=/tmp` 仍可能被已有的 `RAY_TMPDIR` 覆盖。应检查两个变量，或明确配置隔离的 `_temp_dir`。不要把某一版本的临时目录行为写成永久事实。

### 16.6 Docker：socket 文件存在，不代表服务存活

本机本次检查得到：`docker info` 在 6 秒内未返回，向 Docker Unix socket 请求 `/_ping` 也在 3 秒后超时，进程列表中 dockerd 与 containerd 为 `<defunct>`。后者表示进程已经退出，但进程表条目尚未回收。

所以，在本节点继续运行 Harbor 容器任务会先卡在基础设施。磁盘上有 `/var/run/docker.sock`，只证明这个路径还在，不能证明背后有服务处理请求。此现象不能确定 dockerd 最初为什么退出。

最快的避坑方式是带超时运行 `docker info` 和 `/_ping`，失败即阻止在该节点启动容器实验，再选经检查正常的节点或单独处理服务。不要靠无限增加 Harbor 重试次数等它恢复，也不要未经范围确认重启共享 daemon。

其他 Docker 历史问题来自 [节点准备 runbook](../../../examples/sciaccel_rl/prepare/internal/NEW_NODE_RUNBOOK.md)，本次没有重新搭建环境触发：

- shell 的代理可用不代表 Docker BuildKit 构建器拿到了代理。`curl` 成功但 `apt-get update` 卡住时，要检查构建进程的网络配置。
- 默认地址池在原配置下约能分出 31 个网络。若一个 trial 需要两个网络，约 16 个并发 trial 就可能超过容量。这是该历史配置的例子，不是 Docker 固定的并发上限。
- 镜像构建缓存是逐节点的，共享任务目录不会共享 BuildKit 缓存。预热要覆盖实际执行节点。`docker builder prune` 会删除构建缓存，不应作为日常实验清理步骤。
- 初次 MoE 模型启动可能长时间编译 CUDA 内核。MoE 是“混合专家”模型。先看编译进程、缓存和日志是否推进，再判断为挂死，历史约 27 分钟不能作为所有模型的固定启动时间。

### 16.7 GPU 身份和远程遥测：申请的卡、监控的卡、挂载的卡必须是同一张

`nvidia-smi` 的 index 是显示序号，设备 minor number 决定 `/dev/nvidiaN` 中的 N，UUID 是设备身份标识。本次本机直接读到以下映射：

| 显示 index | 设备 minor | 正确设备文件 |
|---|---|---|
| 0 | 0 | `/dev/nvidia0` |
| 1 | 2 | `/dev/nvidia2` |
| 2 | 3 | `/dev/nvidia3` |
| 3 | 1 | `/dev/nvidia1` |

例如租约申请 index 1，直接拼 `/dev/nvidia1` 就挂到了 index 3。这会使账本检查一张卡的显存、容器占用另一张卡。当前 overlay 已读取映射再挂载，本次对应回归用例通过。映射必须每节点现场获取，不能把上表当成所有节点的固定映射。

跨节点还有另一个维度：Gate A 是创建容器前的资源准入，Gate B 是模型回复发出前等待 GPU 空闲的检查。路由服务在 A、容器在 B 时，A 上执行 `nvidia-smi` 不能说明 B 的卡空闲。旧实验已复现错误地立即放行，较新的 [双节点证据](../../env-worker-deep-dive/env-service-verification/results/fixed-multinode/evidence.json) 记录 `remote_gate_b_passed=true`。当前代码传递节点遥测 URL 与 GPU index。本次基础 Gate B 单测通过，但没有重做两台物理机的并发实验。

声明“每任务 512 MiB，卡预算 1200 MiB”，两个任务声明共 1024 MiB，第三个达到 1536 MiB 应被挡住。这是准入规则，不是硬件强制限额，容器实际超用仍需监测，也要给未知进程和 CUDA 上下文留余量。

驱动也不能只看 `nvidia-smi` 正常。应验证容器实际加载的驱动库、GPU UUID 和小型 CUDA 计算。旧实验里 570 用户态库与 535 内核驱动的对照没有触发预期 803 错误，因此不能记录为“版本号不同必然失败”。真实 GPU 内核执行成功才是该配置可用的证据。

### 16.8 清理：租约账本归零，不等于容器已经退出

租约是资源使用登记，TTL 是心跳丢失后租约到期的时间。它们本身不会让 Docker 容器消失。一个任务还可能包含 agent 主容器、独立 verifier（评分程序）以及网络 sidecar（辅助容器），只给主容器加标签不足以覆盖整组清理。

旧实现中独立 verifier 丢弃额外 Compose overlay，标签随之消失。worker 被杀后，回收器找不到这些无标签容器，账本空了但容器仍占资源。较新的 [崩溃恢复证据](../../env-worker-deep-dive/env-service-verification/results/fixed-crash-recovery/evidence.json) 显示 verifier 与 sidecar 已有归属标签，`after_worker_reaper=[]`，即该次实验已回收。

本次测试验证了“Docker 删除失败或仍有容器时保持清理待办，确认消失后才完成”的路径，未再次杀真实 worker。实验收尾必须同时看账本、按本次租约/actor/Compose project 查询的容器、GPU 进程和本次 Ray 进程。清理只针对本次实验的归属，不使用全局 `ray stop --force` 或不加范围的 Docker prune。

### 16.9 Parquet：JSON 字符串与真正的字典不是一种数据类型

Parquet 是列式数据文件格式，Arrow schema 描述每列的实际类型。文本 `'{"index": 7}'` 看起来像字典，但在 Python 中仍是字符串。DataLoader 调用 `extra_info.get("index")` 时，字符串没有 `.get()`，就会失败。

本次分别写出一个含 JSON 字符串和一个含原生字典的临时 Parquet。前者读回后准确复现 `'str' object has no attribute 'get'`，后者读回可取得 `7`。当前 `psrl_demo/data/gsm8k` 的 train/test 文件已是正确的嵌套 struct/list，这个真实数据集问题没有再次发生。

启动前查看 `prompt`、`reward_model`、`extra_info` 的 schema 和一条读回样本。错误应在数据预处理阶段修正，不要靠 DataLoader 无限重试。SciAccel 某个辅助函数能解析 JSON，不代表训练数据入口也接受全部 JSON 字符串。

### 16.10 上下文越界与训练 OOM：两个不同阶段，分别算预算

上下文窗口是模型单次请求能处理的总 token 数。此次读取本地 Qwen2.5-3B 配置，上限仍为 32768。历史失败配置为 prompt 2048 加 response 65536，共 67584，超过该上限。修正示例为 `2048 + 30720 = 32768`。这是对该模型配置的核对，没有再次加载 vLLM 触发启动失败。不同模型、RoPE 扩展设置及 vLLM 配置要分别检查，不能套用同一个上限。

多轮 agent 还会累计工具输出和消息模板。初始 prompt 只有 2048，不代表第 10 轮仍只有 2048。历史较新的单步日志就有 prompt 长 35043、服务上限 34816 的报错。应给实际 token 化后的上下文与下一次生成留空间。

OOM 是显存不足。要先看堆栈发生在生成初始化、旧策略 log-prob 计算还是 `loss.backward()`。此前 SciAccel 的 OOM 位于首次 actor 反向传播，历史记录中已分配约 83 GiB，只有约 582 MiB 空闲，却还需要申请约 1.64 GiB。这个数量关系直接说明剩余显存不够，不是“进程活着就再等一会”能解决的。

`micro_batch_size=1` 只是不把多条样本合成一个小批次，一条 7 万 token 的轨迹自身仍可能超限。动态 token 预算也不能自动把单条序列变短。应联合限制累计轨迹长度、工具输出、turn 数和每卡 token 预算，按需启用序列并行并留共卡余量。序列并行是把长序列的部分计算和激活分摊到多张卡。

已有 [OOM 分析及五步验证](../../sciaccel-grpo-oom-analysis.md) 记录了减小 token/turn 预算、增大序列并行后的成功运行。那是特定配置的历史验证，本次没有申请大显存重新制造 OOM，也不能据此声称五步验证准确率低的问题已解决。

### 16.11 张量形状：为什么只有加载补丁依赖才真正修好

`position_ids` 是模型用来确定 token 位置的数组。这里每条轨迹有 4 行坐标和 L 列 token，形状为 `[4, L]`。长度不同的轨迹用 jagged tensor（不等长张量）打包，通过 offsets 记录各条边界。

本次只用 CPU 构造一条 `[4, 2407]` 的数组，经实际 TransferQueue 存储打包得到 offsets `[0, 4]`，此时边界描述坐标轴。旧函数只把“不等长轴”标记改到 token 轴，未重建边界。于是训练拆包想在长度 2407 的轴上按 `[4]` 切分，必然报：

```text
split_with_sizes expects split_sizes to sum exactly to 2407
(input tensor's size at dimension 1), but got split_sizes=[4]
```

修法是按 token 轴重建数组与 offsets，使这里的边界为 `[0, 2407]`，并在构造时正确指定 `jagged_dim`，不能只修改私有标记。测试还检查重复执行修正不会改变数值，以及多条等长、变长轨迹倒序选择后仍保持对应关系。

**本次默认共享 veRL 在单条和多条等长用例上仍失败，变长用例通过。指定 `.verification-deps` 后三个用例全部通过。** 所以只测不同长度的两条样本会漏掉这个缺陷。修复补丁文件见 [d613affc.patch](../../../patch/verl/d613affc.patch)，当前实际有效副本位于工作区 `.verification-deps/verl`，该目录是包目录，其父目录 `.verification-deps` 才应加入 `PYTHONPATH`。

这不是要求永久依赖一个临时目录。正式实验需要在隔离依赖环境中应用补丁并保留来源版本，而不是覆盖团队共享 veRL。worker 的 `PYTHONPATH` 也必须接收同一个依赖路径。

### 16.12 全零 mask 与无限补采：有轨迹也可能没有可训练 token

`response_mask` 指定哪些 token 参与训练，1 表示参与，0 表示忽略。例如两条长度 4 的回复若 mask 都是 `[0, 0, 0, 0]`，总共有 8 个文本 token，但有效训练 token 是 0。把它们继续送入需要有效 token 的 rollout correction（新旧策略概率修正）会报 `response_mask must contain at least one valid token (1)`。

这个错误存在于 较早的修复后单步日志（本地资料，未随仓库发布）。将交互回合压到很小，可能导致轨迹因截断等规则被全部屏蔽，因此“只跑一个回合”未必是有效的训练冒烟测试。

当前已有空组过滤、补采和失败熔断相关实现。本次配套测试通过：空组应重新申请样本，部分有效的组不能被误丢，补采无法派发或持续失败时应通知训练等待者退出，不能让 trainer 永久等待凑不齐的 batch。GRPO 是按同一 prompt 的多条回复比较奖励的训练方法，零奖励本身不等于无有效 token，二者不能混为一谈。

较新的 fixed-one-step-training-v3 日志（本地资料，未随仓库发布） 已包含 `step:1`、训练 actor 更新、权重推送及 `Training completed successfully!`。因此“从未跑完一步”已不是完整的历史结论。但这来自先前使用补丁依赖的运行，不代表本次普通共享环境已经可直接照跑，也不证明训练质量收敛。

### 16.13 不报错也会浪费算力：奖励基线、指标与验收边界

历史 [DAPO 诊断](../dapo-training-collapse-diagnosis.md) 记录过进程继续运行，但 entropy 接近零、回复打满上限、奖励恶化和梯度异常同步出现。entropy 是输出概率分布的熵，可帮助观察策略是否过度确定。单看熵低不能判失败，应联合奖励、截断比例、梯度和验证集结果看趋势。那份记录支持异常之间的关联，未通过本次消融实验证明每一条因果链。

SciAccel 在正式训练前应做 nop 和 oracle 小对照。nop 不修改任务，用来检查“不交付是否仍被奖励”。oracle 用已知解验证评分链路。任务要求零基线时 nop 应为 0、oracle 应达到预期值，还要确认没有基础设施异常。容器创建成功、verifier 结束、奖励等于零、优化器更新成功和模型效果提高，是五个不同的验收层次。

第一次有效 step 后检查 `global_step`、有效 token 数、有限的 loss/grad_norm、reward、截断比例、组内零方差比例及验证指标。零方差组表示同一 prompt 的回复奖励完全相同，通常无法提供组内相对学习信号。需要保存权重时还应检查 `save_freq`，不能只因配置了 checkpoint 目录就认为实际存了文件。

### 16.14 下次怎么最快执行检查

从本工作区根目录运行。以下命令只做环境探针和回归测试，不启动训练。`PSRL_PATH` 指仓库根目录，`PSRL_WORKSPACE` 指放模型、数据、日志的实验工作区，二者不必相同。

```bash
task_root="$PWD"
source ${PSRL_ENV_SCRIPT}
export PSRL_WORKSPACE="$task_root/psrl_demo"
export PSRL_PATH="$task_root/psrl"
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp
check_dir="$task_root/docs/psrl/cluster-pitfall-checks"
run_tag="$(date +%Y%m%d-%H%M%S)"

python "$check_dir/probe.py" \
  --env-script ${PSRL_ENV_SCRIPT} \
  --output "$check_dir/results-$run_tag.json"

cd "$PSRL_PATH"
PYTHONPATH="$PSRL_PATH:$task_root/.verification-deps" python -m pytest -q \
  tests/env_service/test_position_ids_roundtrip.py \
  tests/env_service/test_node_policy_overlay_placement.py \
  tests/env_service/test_gate_b.py \
  tests/env_service/test_node_cleanup.py \
  tests/env_service/test_runner_env_service.py \
  tests/agent_loop/test_empty_training_group.py \
  tests/agent_loop/test_manager_refill_breaker.py \
  tests/agent_loop/test_overlong_filtering.py
```

探针刻意分别检查“不固定路径”和“只固定 PSRL”的导入，并记录默认依赖的张量故障。若要检查指定依赖的张量结果，可给探针增加 `--dependency-path "$task_root/.verification-deps"`。它只将指定依赖用于张量检查，仍保留不固定路径的环境对照。输出中的原始 IP、路径和 UUID 是本地排障记录，不能原样搬入公开仓库。

真正开跑前还要补齐目标节点的 Docker、空闲 GPU、Ray 节点地址、worker 导入路径检查，以及本任务 nop/oracle 和一个有效训练 step。通过后再放大并发与长度。收尾检查实验所属容器和进程确实消失，保留退出码、参数及日志。

### 16.15 短响应验证先核奖励配置

2026-10-16 的 science 最小训练尝试发现：把 `max_resp_len` 缩到 512，却沿用默认 `overlong_buffer_cfg.len=3072`，会在奖励模块创建时断言失败。即使 `enable=False`，当前构造函数仍要求前者不小于后者。实验配置已显式设 `reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=0`，并在启动 Ray/GPU 前构造真实奖励模块进行 CPU 预检。旧配置失败、新配置通过，但本次按预算没有再次启动训练，不能将构造校验说成完整训练修复通过。

另发现旧 science runner 漏传 `HarborConfig.override_gpus=0`，导致真实 Harbor 在容器创建前拒绝 GPU 任务；已补上传参，38 项回归与单 GPU 容器对照通过。后续类似验证先做奖励配置构造检查和最小 Harbor 任务检查，再付出模型初始化成本。完整原因、源码、脚本与实机边界见[Science 资源账本对照](../../env-worker-deep-dive/science-env-router-comparison.md)。

### 16.16 独立 Ray 不等于物理 GPU 独占

独立 Ray 地址和 namespace 可以隔离 actor、任务和逻辑资源账本，但不会使显卡拒绝其他进程。启动前 GPU 空闲也不代表运行期间别人不会占用。`CUDA_VISIBLE_DEVICES` 只是进程可见性设置，同样不提供跨作业互斥。

2026-10-16 的原始 LAPS 训练在开始时 GPU 空闲，随后同卡外部进程占用约 76 GiB，训练进程在首步前向阶段 OOM。四条任务已产生评分，但没有完成 optimizer step。应在调度平台获得独占 GPU 时间窗口，或让所有竞争者共享统一分配机制，再做性能对照。不能仅靠 EnvRouter 约束那些未接入其账本的外部进程，也不能把外部进程造成的 GPU 高利用率当作本方案收益。

本次还修复了实验监控超时误杀训练、science runner 隐式清理全机无标签镜像的问题。每组开跑前再次检查 GPU，一组失败后停止配对实验。遥测缺失应记录为缺失，不应变成零利用率或触发训练退出。原始日志、显存冲突图、源码与清理边界见 [真实 LAPS 训练对照尝试](../../env-worker-deep-dive/science-training-router-ab.md)。


## 17. Science 训练复盘与下次执行规则

2026-10-18 晚间实际 GPU 压力实验的新约束见 [第 18 节](#18-跨任务池压力实验与事件循环归属)，尤其是不能跨事件循环复用 HTTP 连接池。

**记录日期：2026-10-18；实验日期：2026-10-16。** 本节复核已有结果与当前脚本，整理可复用的决策规则；没有重新启动训练，也没有把历史空闲状态当作当前状态。完整过程、角色协作图及源码阅读路径见 [真实 LAPS 对照报告](../../env-worker-deep-dive/science-training-router-ab.md)，最终成功运行见其 [第 9 节](../../env-worker-deep-dive/science-training-router-ab.md#9-释放占位后重新运行真实训练)。

**先承认执行上的问题。** 用户最初要验证功能，早期工作出现了范围扩大、耗时偏长；后来用户明确要求真实训练对比，才有必要付出模型加载与训练的成本。另一次只完成了显存占位，尚未启动实验，需要用户再次催促。这些属于任务推进方式的问题，不能都归因于 Docker 或集群环境。以后先写验收边界，再执行已授权的实际工作；占位、进程启动和完成训练分别报告。

| 遇到的问题或容易误判的现象 | 原因与影响 | 下次的动作和停止条件 | 当前状态 |
|---|---|---|---|
| 验证被扩大成较长训练；占卡后没有立即实验 | 没将验收目标与资源动作分开，用户付出额外等待 | 先写问题、证据、资源/时间/步数/重试上限；只验证容器就不加载模型。占位设置到期时间，准备完成后释放并启动已授权实验；证据齐备即停止 | 已补充执行规则；不是调度器自动保证 |
| 本机 GPU 空闲但 Docker 无法使用 | socket 存在，daemon 却不响应；曾有僵尸进程 | 在实际节点限时探测 docker info 和 /_ping；失败就换可用节点或报告阻塞，不能继续反复启动 Harbor | 历史节点问题，未通过本轮重启 daemon 修复；每次重查 |
| 首步训练显存不足，即 OOM | 同卡外部进程占约 76 GiB；独立 Ray 只隔离自己的逻辑调度，不锁物理卡 | 预检目标 UUID 和计算进程，运行中记录外部占用；要求独占时使用平台保证。若对照受干扰而失效，保存依据后停止，不杀未知进程 | 有预检和失败停止；未解决任意外部进程进入同卡 |
| 一次 GPU 查询超时导致整场训练被清理 | 监控异常逸出，进入启动器 finally 清理路径 | 启动前关键探活失败则拒绝启动；运行中单次遥测失败记 gpu_error，训练由独立的退出码和时间预算控制，缺失值不补零 | 已修复 sample_gpu；有历史定向检查 |
| OFF 正常退出，ON 却被判 GPU 被占用 | CUDA 驱动释放有延迟，进程消失不等于显存立刻归零 | 清理自己的进程后有界轮询显存，再做下一组预检；保留拒绝记录，只继续尚未实际启动的 ON | 已增加释放核验；最终 ON 收尾实际等待约 12.57 秒 |
| 任务结束时执行了节点级无标签镜像清理 | 实验生命周期调用了全局 prune，超出自己的资源归属 | 清理必须按 Compose project、actor、lease 和 PID/创建时间核对；不因镜像“无标签”就认定属于本实验 | runner 已移除全局调用，保留本次项目清理；有历史回归 |
| 通用 reward=0.5，但训练 score=0 | 数据指定的是 reward_repair，和通用 reward 不是同一个字段 | 保存 reward_key、评分器原始字典、最终训练 score，按实际键解读 | 字段选择是现有正确行为，不需要为匹配图表改奖励 |
| loss/梯度为零，或者有轨迹却没有有效更新 | 同一任务的组内奖励可能相同；有效 token、优化器调用、梯度非零是不同条件 | 检查训练 mask、组内奖励差异、global_step、有限 loss/grad_norm 和退出码；不为了得到非零梯度无限补采 | 最终两组都各有一个零梯度 step 和一个非零梯度 step，仍完成训练 |
| 主训练配置检查通过，代理创建仍因 null 类型失败 | 代理 YAML 在 worker 内另行按 dataclass 合并，GPU override 是 str，不接受 None | 将无 GPU override 写为空字符串，在创建 Ray 和模型前调用实际 build_runtime_config 校验代理配置。任一组失败就停止后续组 | [配置故障修复](../../env-worker-deep-dive/science-single-gpu-training-control.md#9-results不注入压力的原生任务对照)已在续跑中通过真实代理创建，双方完成两步，第三步另遇镜像请求故障，见[续跑记录](../../env-worker-deep-dive/science-single-gpu-training-control.md#10-results无压力续跑与共同完成的两步) |
| 镜像预热成功，运行中独立评分器构建仍失败 | Harbor 源码任务每次仍走 compose build，基础镜像标签元数据查询可能访问远端，缓存层不能保证离线 | 训练前核对主任务和评分器实际使用预构建镜像的分支。若仍需构建，保留网络失败并触发有界退出，不把异常零分算成成功 | [无压力续跑](../../env-worker-deep-dive/science-single-gpu-training-control.md#10-results无压力续跑与共同完成的两步)第三步出现镜像请求 EOF，未修改共享 Docker 配置或追加长训练 |
| 同请求重试设为 1，仍出现额外开始的请求 | 请求内 retry 与组级 refill 不同，失败阈值唤醒训练器后，并发回调仍可能派发替补组 | 核对开始标记、最终记录和补采日志，未落盘的中断轨迹不算成功。严格零补采需验证组级停止分支，不能只看 retry_limit | 本轮 ON 有 26 个开始标记、24 份最终记录，退出期间两个替补请求失败，已释放资源 |
| 两组都跑完，但没有速度或利用率优势 | OFF 已有四任务信号量；CPU 同构任务不触发 GPU 借还或装箱；仅两个 step 且生成长度不同 | 设计时写清机制触发条件、基线保护、同等工作量与计时口径；结果不支持优势就如实结束，不改弱基线、不反复跑到出现优势 | 最终对照已完成，未测出性能优势 |
| 声明 GPU 或启动许多协程，被误当成高显存压力 | CPU oracle 不会因任务声明 gpus=1 而变为 GPU 程序，轻量模型的同卡并发也可能远低于设备容量 | 先确认原生 CUDA 计算、任务代码与真实输入，再观察实际进程重叠和整卡显存，用户未明确要求时，不额外占位或放大输入来制造 OOM | [真实 GPU 多协程实验](../../env-worker-deep-dive/science-natural-gpu-concurrency.md)记录任务选择、原始模型和边界 |
| 同容器很多命令在等，但 Router 等待计数为零 | exec_gpu 在 Router 申请之前取得同容器异步锁，等待者可能尚未到达 Router | 同时看代理提交时间、锁前等待、Router 授权和 CUDA 进程，不把本地串行当作跨 worker 排队证据 | 当前接口行为，见[执行链路](../../env-worker-deep-dive/science-natural-gpu-concurrency.md#1-执行链路与源码对应) |
| 每个训练命令的准备时间远大于十步训练 | 新 Python 进程重复导入依赖，共享存储上的安装文件清单检查会放大等待，完整 exec_gpu 还把 CPU 准备纳入串行区 | 先抓导入栈并分段计时。依赖补丁只放隔离目录，核验包名映射一致。可控 Python 程序用显式 prepare 接口并行准备，不删除同容器 GPU 归还保护 | [启动耗时与修复](../../env-worker-deep-dive/science-natural-gpu-concurrency.md#7-启动耗时定位与修复机制) |
| CPU 准备后拿到 GPU，进程仍报告 CUDA 不可用 | Trainer 等导入可能在 GPU 授权前探测 CUDA，失败初始化或依赖缓存不能视为授权后状态 | 将会探测 CUDA 的导入推迟到授权后。在无 GPU 容器中用调用即报错的 CUDA 检查入口验证 prepare。不能用伪造可用性或放开设备权限绕过 | 分阶段接口只接受协作程序，不自动拆分任意 shell 命令 |
| 绝对路径 Python 的 CUDA 预检成功，受管理作业却退出 127 | 作业监督器通过容器 PATH 查找 python3，训练命令的绝对路径成功不代表监督器可启动 | 同时检查容器内 python3 和 bash，再执行 GPU 作业。挂载独立 Python 环境时将 bin 加入容器 PATH，不修改共享解释器 | [统一后端实机验证](../../env-worker-deep-dive/unified-gpu-execution.md#4-真实-docker-gpu-任务验证)保留首次失败、环境修复和两种模式通过证据 |
| 显存占位结果被误写成原生压力或小显卡性能 | 外部分配只改变剩余可用显存，没有改变设备算力、带宽或原生任务需求 | 仅在用户明确要求时注入占位，保存 PID/创建时间/UUID，限制持有时间，监控确认两组全程保持占位，按相同模型、输入和并发比较。失败更快不是提速 | [高低可用显存对照](../../env-worker-deep-dive/science-natural-gpu-concurrency.md#10-固定-off-与修复后-on-的容量压力对照) |
| 无计算进程，NVML 却显示数百 MiB used | 本节点旧接口 used 包含驱动保留内存，新接口分离 used 与 reserved | 用 NVML v2 区分驱动保留量和实际进程占用，并核对 UUID、进程和物理释放。不得按 index 直接拼设备 minor | 本次空闲 used 约 0.19 MiB，另有约 586 MiB reserved，已按接口区分 |
| 账本不超额，却不能保证所有容器物理资源都受限 | 每个租约可含主容器、独立评分器及网络辅助容器；sidecar 没有单独 CPU/内存上限 | 同时核对账本和 Docker inspect 的真实限制与归属；未覆盖的开销、排队和跨 worker 行为必须列为边界 | 主容器限制已现场验证；sidecar 缺少单独硬上限仍未解决 |
| 结果文件读取失败或图中出现不属于本轮的故障说明 | 读到了尚未写完整的 JSON；旧绘图逻辑写死历史 OOM | 运行中将不完整文件视为未完成；收尾核对完整性。缺失数据不补造，故障标签从本轮日志提取，保存运行时脚本快照 | 绘图硬编码已修复；读取规则补入 AGENTS |

**两个必须记住的具体例子。**

1. **显存交接：进程退出只是中间状态。** OFF 的两步已成功完成，下一组启动前却读到 GPU 0/1 仍有62087 / 58523 MiB的占用，因此 ON 预检拒绝。随后显存清空。正确动作是保留这个拒绝记录、核实释放、继续未开始的 ON；重跑已经成功的 OFF 会浪费另一轮模型初始化和采样成本。此处只允许等待本次清理的驱动延迟，不能假定任何占用都会自动消失。
2. **账本验证：观测到释放，不等于观测到排队。** 每任务声明 4 CPU、8 GiB，四个任务记账为 16 CPU、32 GiB；一个结束后降为 12 CPU、24 GiB，最后归零。最终 782 份快照中记账一致、没有越界，但最多只有四个请求，等待峰值为 0。它证明实际记账和释放有效，却没有触发“第五个请求等待”的条件；更不能证明任务容器使用 GPU 时显存碎片会减少。

**下次开跑前填写的最小验收卡。** 以下是执行约束模板，不是已经自动实现的全阶段 watchdog；现有训练脚本的 1440 秒主要约束训练子进程，环境激活、Ray 初始化和清理也需要纳入总等待预算。

```text
待验证结论：功能接通 / 预算准入 / 超额排队 / GPU 借还 / 训练性能（选择实际目标）
触发条件：哪个输入会让被测机制真正发生？OFF 已有哪些保护？
必要证据：原始任务评分、目标训练 step、账本/容器/设备观测、退出码
固定配置：模型、数据、奖励键、种子、并发、长度、设备 UUID、代码与依赖来源
预算：环境与初始化时间；任务与评分时间；训练步数；总墙钟；清理时间；重试次数
通过/失败停止条件：写明，不能运行中追加“必须效果更好”
资源归属与释放：本次进程/actor/lease/Compose project；物理释放如何核验
统计口径：初始化、采样与评分、优化更新、清理分别计时；缺失观测怎样处理
```

进度更新应交代“正在初始化 / 已启动任务 / 已有评分 / 已完成第几个更新 / 正在清理”，并给出日志证据；不能只说“在跑”。临时占位要另列状态和自动到期时间，不能算训练进度。节点身份要用路由源地址、GPU UUID 等现场依据确认，相同 hostname 不能当作同一台机器。

**现有修复的源码与可复核证据。** 以下源码已经实际读取核对；它们若有理解上的问题，可以随时追问。真实系统的全局协作图和逐步阅读路径沿用上述实验报告，不在本节另造一套架构。

| 规则对应位置 | 源码或证据 |
|---|---|
| 运行中监控失败不直接杀训练 | [sample_gpu](../../env-worker-deep-dive/env-service-verification/scripts/science_training_ab.py#L197)、[历史遥测回归](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-retry/telemetry-regression.json) |
| 启动前拒绝占用、结束后有界观察释放 | [require_idle_gpus / wait_for_gpu_release](../../env-worker-deep-dive/env-service-verification/scripts/science_training_ab.py#L212)、[拒绝记录](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/on-preflight-rejected/launch.json)、[定向回归](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/handoff-regression.json) |
| 先清理本次容器，再归还租约 | [runner finally](../../../examples/sciaccel_rl/runner.py#L534)、历史清理回归（本地资料，未随仓库发布） |
| 训练奖励字段 | [compute_score](../../../examples/sciaccel_rl/reward.py#L44)、[实际数据键](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/training-reward-keys.json) |
| 最终成功与边界 | [结果统计](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/experiment-status.json)、[容器限制](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/on/docker-admission-evidence.json)、[物理清理](../../env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live/cleanup-verified.json) |

需要学习准入机制时，可运行已有的 [unit_fragmentation_demo.py](../../env-worker-deep-dive/env-service-verification/scripts/unit_fragmentation_demo.py)：它用教学需求调用真实账本，只需要 CPU，不启动 Docker 或训练；运行方式和预期结果见 [unit 粒度说明](../../env-worker-deep-dive/science-env-router-comparison.md)。本次没有重跑这个演示，也没有把历史回归成功写成今天重新执行通过。

**持久化方式。** 根目录 AGENTS.md（本地资料，未随仓库发布） 保存运行前必须读本节的入口与简明规则；当天 memory（本地资料，未随仓库发布） 保存复盘日期、范围和链接。以后新增教训优先维护本指南，避免让日记、实验报告和运行规则各写一套不一致的结论。


## 18. 跨任务池压力实验与事件循环归属

这轮使用相同 science 任务副本、同等 CUDA 压力比较 OFF/ON，详细数值、图解、源码和可运行复现统一维护在 [GPU 压力实验报告](../../env-worker-deep-dive/science-gpu-pressure-ab.md)。与第 17 节此前的 CPU 容器对照是不同实验。

| 下次必须检查的点 | 本次发现的机制 | 执行动作与验证边界 |
|---|---|---|
| 基线必须读当前实现 | OFF 已有整卡锁，但锁池按 jobs_dir 分组，不能保证跨目录互斥 | 保留原有基线，不移除保护来制造优势；不同池必须共同接入同一个 Router 才共享账本 |
| 每一个采集脚本都要容错 | 短对照 helper 沿用了未处理 nvidia-smi 超时的逻辑，任务完成却丢失汇总 | 不能只修训练主脚本而漏掉 helper；保留原始 trial 结果，监控失败与任务失败分开记录 |
| 自定义代理遵循 Harbor 退出契约 | 普通异常会跳过正常 verifier 流程；stderr 还可能是 None | 真实非零命令退出用标准代理退出异常上报，空输出安全处理，不手工填零奖励 |
| 网络连接归属事件循环 | actor 循环和 Harbor 线程共用全局 aiohttp client，可把传输对象带到错误线程 | 已改成每循环独立连接池；CPU 双线程真实 HTTP 回归通过，完整 GPU 训练尚未回归 |
| 原生异常不能只看 Ray 通用提示 | ON worker core 显示 uvloop/libuv 读回调 SIGSEGV，不是已证实的 CUDA OOM | 保存 stderr、core 栈和依赖来源；确定下游无法推进就停止占用，不空等到超时 |
| 不等长实验不能伪装成完整配对 | OFF 四步完成、ON 两步后崩溃 | 展示等量前缀及全部实际点，同时披露崩溃；不补造缺失 step，不据此前缀宣称长期稳定 |

具体例子：同一个进程中循环 A 创建 client P，循环 B 调用旧全局获取函数也拿到 P，但 `P._loop` 仍是 A。修复后 A、B 分别获得自己的连接池，A 关闭时 B 的实际 HTTP 请求仍成功。该 CPU 检查验证了对象归属缺陷及修复；它没有再次触发或排除所有可能的原生段错误。完整流程与源码映射见 [报告第 8 节](../../env-worker-deep-dive/science-gpu-pressure-ab.md#8-on-段错误调查与-http-连接池修复)。

优先运行 `PYTHONPATH=. python -m pytest tests/utils/test_http_loop_ownership.py -q`（在激活的项目环境、`psrl/` 仓库目录内）。本次三个测试通过，包含 asyncio 与 uvloop 两种组合共 162 次本地 HTTP 请求。不要把该结果写成修复后四步训练通过，也不要为验证连接池修复默认重启整套长训练。

## 19. 本机 Docker 挂起原因排查（2026-10-20）

**结论分两层：当前不能恢复退出的直接阻塞已定位到内核文件系统挂载清理与写回等待。Docker 存储链路存在挂起的证据很强，但最初为何挂起、哪个锁由谁持有、谁发送了终止信号，现存材料不足以确定。** 不能将这次故障写成已经证实的 OOM、磁盘损坏或某个人误操作。

这里检查的是当前计算环境内部的 Docker 服务。PID 1 的名称是 `taijirun`，不是 systemd。这不代表宿主机供平台使用的全部容器运行服务也已停止。用户之前看到的 `<defunct>` 是重要现象，但只看主线程会漏掉尚未退出的其他线程。

### 19.1 Results：日志与线程证据

下表时间来自保留的服务日志，时区为 UTC+8。线程状态则是本次现场读取，两类证据不能混写成同一时刻的观测。

| 时间或观察项 | 证据 | 能得出的结论 |
|---|---|---|
| 9 月 7 日 07:22 左右，前一份 Docker 日志 | 容器执行连接被重置，停止超过时间上限，最终出现 `trying direct SIGKILL` | 重启之前已经有容器停止异常，不能只把后来的启动失败看成独立网络问题 |
| 9 月 7 日 07:25:26 | 新日志出现 `Starting up`，随后启动 containerd，PID 与当前残留进程一致 | 当前残留实例来自这次启动 |
| 9 月 7 日 07:25:41 | `using prior storage driver: fuse-overlayfs` | 本次 Docker 使用该容器存储驱动 |
| 9 月 7 日 07:25:53 | `Loading containers: start.` | 启动进入恢复已有容器阶段 |
| 之后约 91 分钟 | 该日志未记录加载完成或 API 就绪 | 不能认为这次启动已成功，具体停在哪一步需结合线程栈 |
| 9 月 7 日 08:57:25 | `Processing signal 'terminated'` | Docker 确实收到 SIGTERM，也就是请求终止信号 |
| 当前 dockerd、containerd 的 proc 状态 | 僵尸线程的退出状态均为 9，按 wait 状态解码是 SIGKILL | 结束过程涉及强制终止，不能仅凭这个值认定是 OOM killer 或人工操作 |
| 当前 dockerd 的另一个线程 | D 状态，栈含 `cleanup_mnt → deactivate_super → rwsem_down_write_slowpath` | 有线程在退出时的挂载清理中等待内核读写锁 |
| 当前一个 `fuse-overlayfs` 进程 | D 状态，参数指向 Docker 存储层，栈含 `fuse_release_end → iput → evict → inode_wait_for_writeback` | Docker 相关文件系统进程在等待文件节点写回完成 |
| 当前两个 `sync` 进程 | D 状态，等待点为 `iterate_supers` | 文件系统同步也存在阻塞，现象超出了 Docker 命令行客户端 |

完整线程栈、信号状态、日志关键行和容量检查保存在 [本次证据 JSON](../cluster-pitfall-checks/docker-forensics-2026-10-20.json)。已有的最小探活结果见第 16.6 节，本次再次确认本机 socket 请求超时。

### 19.2 为什么主进程显示 Z，服务却没有彻底退出

Linux 中一个进程可以包含多个线程。Z 状态是 zombie，即某个线程已经结束、退出记录仍保留。D 状态则表示线程正处于不可中断等待，常见于内核 I/O 或锁等待。D 不等于已经死亡，也不必然意味着物理磁盘损坏。

本次 dockerd 正好提供了一个实际例子：主线程 PID `3763725` 是 Z，但同进程的线程 `3763750` 是 D。后者在信号处理和返回用户态的路径中执行挂载清理，随后停在 `rwsem_down_write_slowpath`。`rwsem` 是读写信号量，这里可以理解为保护文件系统对象的读写锁，`down_write` 表示申请独占的写权限。

用一个教学例子理解等待：假设 A 正持有某个文件系统对象的读锁，B 要卸载它，需要拿写锁。只要 A 不释放，B 就不能继续。这个例子解释锁为什么阻塞，不表示本次已经查明 A 是哪一个线程，也不能据此断言形成了环形死锁。

SIGKILL 的用途是强制结束进程，但它不能安全地把任意内核临界区直接拆掉。线程陷入这种等待时，重复发送信号不保证立刻消失。本次 `fuse-overlayfs` 还存在待处理的 SIGKILL 位，但它仍是 D 状态，说明“信号已发出”和“物理资源已释放”是不同状态。

因此，之前“只有一个死掉且未回收的 daemon”的描述不够完整。准确说法是：**主线程已经退出，同时还残留未完成内核清理的线程，存储相关进程也被阻塞。** 当前 API 请求超时已实测，socket 为什么仍表现为超时而非立即拒绝连接，没有进一步追踪其全部引用者。

### 19.3 为什么重点怀疑存储挂载链路，以及哪些解释尚未成立

`fuse-overlayfs` 是用户态的容器分层文件系统实现，Docker 用它把镜像的只读层和容器可写层组合为容器看到的文件树。FUSE 表示文件系统的一部分处理在用户进程中完成，需要与内核交换请求。写回是把缓存中的文件修改完成落盘或相关回写操作，文件节点释放会等待必要的写回结束。

当前证据将几个现象联系起来：Docker 明确选用了这个存储驱动，相关进程的参数明确属于 Docker 的存储目录，它在文件节点写回路径等待，dockerd 在文件系统挂载清理等待，还有文件系统同步进程等待。这支持“存储和挂载清理链路挂起”的判断。

但栈里没有直接提供所有锁对象的身份与持有者，不能证明它们都在等同一个 superblock，也不能确定当前退出清理的等待点就是 9 月 7 日启动时最初卡住的地方。superblock 是内核表示一个已挂载文件系统实例的核心对象。进一步确认原始根因需要当时的内核栈、锁状态或平台事件记录，本轮没有这些材料。

| 候选解释 | 本次判断 |
|---|---|
| 当前磁盘或 inode 用尽 | 检查时可用空间约 4.0 TiB，空间使用率约 67%，inode 使用率约 3%，不支持当前容量耗尽。不能反推故障发生当天的容量 |
| OOM 杀死 Docker | 当前内核日志覆盖最早到 9 月 14 日，晚于 9 月 7 日事件。可见的较新 OOM 记录不能当作这次 Docker 故障的证据 |
| NVIDIA runtime 缺失 | 启动日志确实有该警告，但之后继续进入加载容器阶段，现在线程栈也指向文件系统。不能将警告直接当作这次主故障根因 |
| 普通应用异常或 Go panic | 这份启动日志未见相应 fatal/panic 证据，已看到的是终止信号和内核等待 |
| 人工或清理脚本结束进程 | 收到终止信号是事实，信号发送者未知。没有可用的对应 journal 或 audit 记录，不能归责到具体人或脚本 |
| systemd 自动恢复失败 | 当前环境 PID 1 不是 systemd，journal 为空。没有发现自动拉起后的健康实例，不能套用常规宿主机的 systemd 排障结论 |

### 19.4 可运行的只读证据采集

在已激活项目环境的 PSRL 仓库根目录执行：

```bash
python docs/psrl/cluster-pitfall-checks/docker_forensics.py \
  --output /tmp/psrl-docker-forensics.json
```

[采集脚本](../cluster-pitfall-checks/docker_forensics.py) 只使用 Python 标准库和系统只读命令，查询进程、线程栈、日志及 Docker ping。外部命令有 8 秒上限，ping 自身有 3 秒上限，最多保存 64 个匹配线程，不发送信号、不挂载或卸载文件系统、不遍历容器合并目录、不启动 daemon。本次已实际运行并保存上面的 JSON。

预期观察是多个线程各自的 `state`、`stack` 和 `wchan`，以及 Z 线程的 `termination_signal`。如遇权限不足或进程已消失，脚本记录读取错误，不能把读取失败解释为健康。它是证据采集，不是教学模拟或故障注入，也不会重现最初的卡死过程。

这次用到的 `cleanup_mnt`、`deactivate_super` 等函数来自正在运行的 Linux 内核，不是 PSRL Python 函数。本工作区未提供匹配的内核实现源码，因此不编造可跳转的本地源码位置。需要理解某一段线程栈或采集代码时，可以按函数继续追问。

### 19.5 修复方向与本轮停止条件

本轮用户要求排查原因，已完成只读证据保存，没有重启 Docker、删除数据目录、发送 kill、执行 sync、强制卸载或重启节点。

后续恢复应先由平台侧确认当前计算环境及相关挂载的归属，评估残留 D 状态任务影响。在存储等待未解除前，简单删除 pid/socket 文件并反复启动 dockerd 不能保证恢复，也可能制造更多残留。若需要终止或重建计算环境，应先确认本环境其他任务与结果保存情况。宿主机重启属于更大范围的操作，不能仅凭当前这些栈就直接执行。

恢复后的验收应限定为：服务能返回 ping/info、最小容器能运行退出、相关资源能清理。是否恢复 PSRL 训练是另一个任务，不需要为了证明 Docker 修复额外启动模型训练。

## 20. Docker 恢复尝试与 inode 补充（2026-10-20）

本轮用户授权尝试修复。本节的结论是：**原存储链路的内核阻塞仍存在，但新的本地数据目录配合 `overlay2` 能启动独立 Docker，并实际运行、读写和删除最小容器。默认服务入口尚未切换。**

### 20.1 尝试解除旧连接的边界

在临时私有挂载命名空间内挂载 FUSE 控制接口，只读取连接状态，退出前卸载。观察到 18 个连接，其中连接 1527 有 289 个等待请求。随后检查了 211 个可读进程的挂载表，没有找到能够把该连接可靠对应到故障 Docker 挂载的记录。

FUSE 的 `abort` 控制文件可以让对应连接上的等待操作以错误结束，但误中止共享文件系统会影响其他任务。因为归属未确认，本次没有向任何连接写入 abort，也没有批量终止 FUSE 进程。[内核 FUSE 文档](https://docs.kernel.org/filesystems/fuse/fuse.html)

这一步没有修好旧链路，也不能把等待请求最多的连接直接认定为 Docker。它明确了继续直接处理旧挂载的证据边界。

### 20.2 Results：新的数据目录与存储驱动测试

恢复测试使用本地 XFS 挂载中的新目录，指定 `overlay2`，不加载旧 `fuse-overlayfs` 元数据。通过独立 Unix socket 提供服务，使用单独的 pidfile 和 exec-root，禁用 bridge、iptables 修改和 IP forwarding 修改，因此没有替换默认 Docker 或调整现有容器网络。

| 检查 | 结果 |
|---|---|
| daemon 初始化 | 完成，日志包含 `Loading containers: done.` 与 API 就绪 |
| Docker info | 返回 Docker 26.1.3 和存储驱动 `overlay2` |
| 镜像准备 | 本地 shell 与所需动态库组成小型镜像，无网络拉取 |
| 真实容器执行 | 输出 `DOCKER_RECOVERY_OK`，退出码 0 |
| 容器可写层 | 写入临时文件并通过 shell 读回验证成功 |
| 资源边界 | 容器使用 network none、64 MiB 内存、32 进程上限、移除 capabilities |
| 容器清理 | 使用 `--rm`，随后 `docker ps -a` 返回空 |

初次镜像夹具缺少正确的动态库搜索布局，shell 报 `libtinfo.so.6` 不存在。修正镜像库路径后上述容器测试通过。这是测试镜像构造问题，不是第二个 Docker 内核故障。首次失败和修正后的结果都保存在 [恢复证据](../cluster-pitfall-checks/docker-recovery-2026-10-20.json)。

测试结束后停止本次独立 daemon，核对其 socket 消失且没有本次 daemon/containerd 活进程。保留恢复目录中的镜像、启动参数与日志供后续接入，不删除旧数据目录。没有为了测试恢复而重跑 PSRL 训练。

验证范围仅包括服务启动、CPU 容器可写层与容器回收。没有验证 GPU 透传、镜像下载、bridge 网络或 Harbor 全流程，也没有解除旧线程的 D 状态。不能据此声称旧环境完整恢复。

### 20.3 为什么切换默认服务需要明确选择

新的 data-root 相当于使用另一个 Docker 镜像与容器数据库。旧数据保留在原目录，不会被清空，但新服务默认不会展示旧镜像、旧容器及其可写层。数据目录也不能直接在 `fuse-overlayfs` 和 `overlay2` 之间互换，不能把两个目录粗暴合并。

本轮已向用户说明该行为变化，等待是否接受“保留旧数据，切换新目录恢复服务”的选择。确认之前不替换默认 socket、pidfile 或系统 Docker 配置。若继续恢复默认入口，还需针对实际网络配置做最小验证，不能把 network none 的结果当作 bridge 已验证。

### 20.4 inode 概念与本故障的关系

inode 是文件系统记录文件身份、属性及内容映射的索引节点。inode 使用率很低，不代表某一个 inode 的写回无法卡住。当前问题是 `inode_wait_for_writeback` 等待，不是没有空闲 inode。

详细图解、真实六字节文件实验、硬链接与删除后打开引用的机制见 [inode 基础](../../tools/linux/inode-basics.md)。该实验已执行并清理临时文件，与 Docker 恢复测试是两个独立验证。
