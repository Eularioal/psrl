# EnvWorker 与 Harbor 共用容器后端：实现、边界与真实验证

> **核心结论**：EnvWorker 已经直接管理 Docker，因此自主容器管理不需要从零重写。本次新增 `EnvWorkerEnvironment`，让 Harbor 保留任务编排、标准答案执行、评分与产物收集，由 EnvWorker 实现容器生命周期、命令和文件传输。预构建 CPU 单容器的完整 Harbor oracle 已通过真实 Docker 验证。
>
> 统一的是执行实现，不是所有资源账本。此适配器目前在调用进程内创建 EnvWorker，没有接入共享 Ray worker 池或 science 的 GPU 租约。需要 GPU、Compose 构建、多服务或复杂网络策略的任务会被拒绝，不能替换现有 science 默认路径。

> **当前修复状态**：准备阶段对外拒绝命令、清理失败保留待回收状态已修复。CPU 后端已接 science runner 的整次任务准入，并通过真实 LAPS oracle 与固定 baseline 对照。第 7 节保留修复前复现，最新结果与未完成项见[修复与 science 验证](../experiments/native-backend-fixes-and-science.md)。

## 目录

- [1. 全局关系：统一哪一层](#1-全局关系统一哪一层)
- [2. 具体难点：相同容器，不同执行语义](#2-具体难点相同容器不同执行语义)
- [3. 文件、资源与清理契约](#3-文件资源与清理契约)
- [4. 跟着图读源码](#4-跟着图读源码)
- [5. 可运行验证与 Results](#5-可运行验证与-results)
- [6. 自主管理的自由度、成本与后续边界](#6-自主管理的自由度成本与后续边界)
- [7. 后续修复优先级与生命周期故障复现](#7-后续修复优先级与生命周期故障复现)

## 1. 全局关系：统一哪一层

Harbor 是任务运行框架，负责读任务定义，让 agent 操作环境，再调用 verifier 评分。Agent 是执行任务的角色，verifier 是检查结果的角色。后端则是实现“创建环境、执行命令、传文件、销毁环境”的组件。替换后端不要求重新实现整套任务框架。

```text
[1] Harbor Job / Trial
    读取任务 → oracle 或其他 agent → verifier → 收集产物 → finally 清理
                      │ start / exec / upload / download / stop
                      ↓
[2] EnvWorkerEnvironment（新适配器）
    校验能力，转换资源和命令参数，保持 Harbor 返回值契约
                      │
                      ↓
[3] EnvWorker（同一份容器管理实现）
    ├→ create_sandbox → [4] SandboxSpec → docker run → 一个所属容器
    ├→ exec：持久 shell，供需要保留终端状态的调用方使用
    ├→ exec_process → [5] run_process → docker exec → 每次独立 shell
    ├→ copy_path → [5] run_process → docker cp
    └→ destroy_sandbox → docker rm -f → 确认结果 → 释放本地记录

[1] 的命令结果/文件从同一路径返回，再驱动下一步任务或评分。
[3] 的两个命令入口共享容器文件系统，但不共享 shell 局部状态。
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 职责 | 对应函数与源码 |
|---|---|---|
| 1 | 调用真实 Harbor Job，装配环境插件并验收评分 | 本地入口 [real_env_worker_harbor.py:142](../env-service-verification/scripts/real_env_worker_harbor.py#L142)，Harbor 内部属于安装依赖，不冒充 PSRL 源码 |
| 2 | 环境契约适配与不支持能力的显式拒绝 | [harbor.py:18，`EnvWorkerEnvironment`](../../../psrl/workers/env_worker/harbor.py#L18) |
| 3 创建 | 管理容器所有权、启动 shell | [worker.py:224，`create_sandbox`](../../../psrl/workers/env_worker/worker.py#L224) |
| 3 命令 | 持久命令与独立命令入口 | [worker.py:278，`exec`](../../../psrl/workers/env_worker/worker.py#L278)、[worker.py:352，`exec_process`](../../../psrl/workers/env_worker/worker.py#L352) |
| 3 文件 | 同一容器的上传下载 | [worker.py:386，`copy_path`](../../../psrl/workers/env_worker/worker.py#L386) |
| 3 清理 | 删除容器成功后释放本地所有权 | [worker.py:431，`destroy_sandbox`](../../../psrl/workers/env_worker/worker.py#L431) |
| 4 | 把镜像、挂载、网络及资源要求变为 Docker 参数 | [sandbox.py:141，`build_docker_run_argv`](../../../psrl/workers/env_worker/sandbox.py#L141) |
| 5 | 异步执行 Docker 客户端，分别读取两个输出流，限制输出与等待 | [process.py:17，`run_process`](../../../psrl/workers/env_worker/process.py#L17) |

这些代码如果有理解上的问题，可以随时追问。

这对应原提案所说的 **L2：替换 Harbor 环境实现**，见[原设计的接入深度说明](../../psrl/psrl-env-service-envworker-design.md#9-当前代码是什么样新方案准备怎样接入)。它沿用 Harbor 的环境插件入口，不修改共享安装的 Harbor 包。已有 [ManagedDockerEnvironment](../../../psrl/workers/env_service/docker.py#L29) 则是在 Harbor Docker 后端之上增加资源管理。

当前 science runner 的 [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L100) 已支持显式选择 CPU 原生后端并传入整次任务预算，GPU 仍使用原有路径。此次没有把 GPU 接线替换完成，也没有把两个资源调度器的账本合并。每个新适配器目前持有一个本地 EnvWorker 实例，说明的是共用实现，不是多任务已经共用一个远程 actor。

## 2. 具体难点：相同容器，不同执行语义

容器是文件系统、进程和资源边界，shell 是容器里面解释命令的进程。同一个容器可以有多个 shell，它们能看到同一份文件，但 `export` 和 `cd` 这样的进程局部状态不天然共享。

| 行为 | EnvWorker 持久 `exec` | Harbor 适配器 `exec` |
|---|---|---|
| shell 生命周期 | 创建容器时保留，跨命令使用 | 每条命令 `docker exec ... /bin/bash -c` |
| `cd`、`export` | 影响后续持久命令 | 只影响本条命令，后续重新设置 cwd/env |
| 输出 | 持久终端协议的合并观察 | stdout、stderr、退出码分开返回 |
| 文件 | 写入容器文件系统 | 可读取同一个容器内持久命令写入的文件 |
| 不确定超时 | 销毁容器，避免污染下一次观察 | 同样销毁容器，避免只杀 Docker 客户端却留下真实命令 |

例如初始容器有 `/work`，没有 `PERSIST` 环境变量。第一次通过持久接口执行 `export PERSIST=kept; cd /tmp; echo 42 > /work/answer`。第二次持久命令打印变量和目录，得到 `kept:/tmp`。

接着同一个容器通过 Harbor 接口执行命令，传 `cwd=/work`、`INPUT=a b`。这次看到 `PERSIST` 未定义，当前目录为 `/work`，但能从文件 `answer` 读到 42。向标准错误输出的 `err` 单独返回，不能混进标准输出。

**这不是为历史版本保留两套 Docker 后端。** 两种接口表达两个仍然存在的需求：有状态交互终端与无状态单次进程。统一的是底层生命周期和文件系统所有权，不能为了统一函数名而改变任务观察结果。

## 3. 文件、资源与清理契约

[适配器 `_spec`](../../../psrl/workers/env_worker/harbor.py#L74) 把 Harbor 的 CPU、内存、宿主机 bind 挂载及启动环境变量转换成 `SandboxSpec`。Bind 挂载表示把宿主机目录映射到容器目录，评分输出由此返回任务结果目录。CPU 和内存最终成为 Docker 限制，不只留在 Python 配置中。

目录上传容易出现一个很小但真实的错误：把 `inputs` 复制到 `/work`，可能得到 `/work/inputs/seed.txt`，而任务期望 `/work/seed.txt`。适配器保留 Docker 源路径末尾的 `/.`，明确表达“复制目录内容”。真实验证会检查最终文件位置。

命令由 [run_process](../../../psrl/workers/env_worker/process.py#L17) 并行读取 stdout/stderr，保留真实退出码并调用 Harbor 输出回调。两个流合计输出上限为 8 MiB，防止一个无限输出命令拖垮宿主进程。超时或取消会结束 Docker 客户端。**结束客户端不等于结束容器内命令**，所以 `exec_process` 的异常分支还会等待删除所属容器。

这个清理策略有明确代价：一条命令超时，整个 sandbox 的未保存内存状态都会丢失，其他在其中运行的命令也会结束。它不是“只取消该作业且其他作业无感”的细粒度隔离。本次不支持在这一后端里承诺同容器独立作业取消。文件拷贝失败向上抛错，由任务生命周期清理收尾，不声称文件拷贝具有事务回滚。

能力校验在创建容器前拒绝以下配置，避免默默忽略要求后继续运行：

- 没有预构建镜像，或者任务包含 Dockerfile、Compose 定义、强制构建要求。
- GPU 资源或原有 GPU command/lease 等绑定参数。
- 非禁网策略、动态阶段网络策略及不支持的挂载类型。
- 磁盘配额要求，以及 `stop(delete=False)` 的保留容器要求。

镜像还需要支持原生 EnvWorker 的 Bash 启动协议。此验证使用已缓存、入口兼容的 Linux 镜像，没有宣称任意镜像的 entrypoint、默认用户或全部挂载扩展选项已兼容。

## 4. 跟着图读源码

1. **入口 → 图 1。** [实验脚本 `Job.create`:142](../env-service-verification/scripts/real_env_worker_harbor.py#L142) 把 `import_path=psrl.workers.env_worker.harbor:EnvWorkerEnvironment` 传给 Harbor。输入是任务与插件配置，输出是可运行 Job。
2. **实例化 → 图 2。** [构造函数与 `_validate_definition`:19](../../../psrl/workers/env_worker/harbor.py#L21) 初始化所有权和生命周期锁，Harbor 基类调用能力校验。不支持的任务在启动前失败。
3. **启动 → 图 2→3→4。** [适配器 `start`:93](../../../psrl/workers/env_worker/harbor.py#L106) 转换 spec，调用 `create_sandbox`。随后准备工作目录、挂载目录权限及环境文件。部分启动失败会销毁已创建容器。
4. **命令 → 图 2→3→5。** [适配器 `exec`:133](../../../psrl/workers/env_worker/harbor.py#L202) 合并 Harbor scoped 环境变量，解析用户，把 cwd/env/timeout 交给 `exec_process`。两个输出流并发读取，回调也在这一过程中发生，不能理解成全部命令结束后才第一次反馈输出。
5. **文件 → 图 2→3→5。** [上传下载方法:147](../../../psrl/workers/env_worker/harbor.py#L219) 调用 `copy_path`，使用同一所属容器名，不另开一个与任务无关的容器。
6. **评分 → 图 1。** Harbor 的 oracle 写入答案，verifier 通过相同环境接口读取答案并写奖励。适配器不决定题目对错。
7. **清理 → 图 2→3。** [适配器 `stop`:124](../../../psrl/workers/env_worker/harbor.py#L142) 调用 `destroy_sandbox`，重复 stop 可收敛。命令超时/取消是另一条进入销毁的异常分支，不等待下一条命令才处理残留。

## 5. 可运行验证与 Results

**验收问题与停止条件。** 本次只验证能否把 Harbor 的完整 CPU 单容器任务交给 EnvWorker，并保住命令语义、资源约束和物理清理。完整 oracle 评分成功且所属容器全部释放即可停止。没有把训练 step、模型效果或吞吐提升追加为验收门槛。

在已配置的项目 Python 环境、仓库根目录运行 CPU 回归，无需 Docker 或 GPU：

```bash
python -m pytest tests/env_worker/test_harbor_backend.py tests/env_worker/test_worker.py tests/env_worker/test_sandbox.py -q
```

实机脚本为 [real_env_worker_harbor.py](../env-service-verification/scripts/real_env_worker_harbor.py)。它调用真正的 Docker 与安装的 Harbor，不是教学模拟。运行前按[集群预检指南](../../psrl/psrl-run-guide/cluster-preflight.md)核验目标节点。交互输入的 `HARBOR_TEST_IMAGE` 必须是节点已缓存、包含 Bash 的兼容 Linux 镜像，`HARBOR_TEST_OUTPUT` 必须是尚不存在的输出目录：

```bash
read -r -p 'Cached compatible image: ' HARBOR_TEST_IMAGE
read -r -p 'New output directory: ' HARBOR_TEST_OUTPUT
CUDA_VISIBLE_DEVICES='' timeout --signal=INT --kill-after=30 240 \
  python docs/env-worker-deep-dive/env-service-verification/scripts/real_env_worker_harbor.py \
  --image "$HARBOR_TEST_IMAGE" --output "$HARBOR_TEST_OUTPUT"
```

预期结果是 `evidence.json` 的 `passed=true`、`timeout_confirmed=true`、一个 trial 的 reward 为 1、`remaining_containers` 为空。脚本内部工作流上限 150 秒，一次 oracle 尝试，最多三个依次使用的容器，不涉及模型、GPU 或训练。脚本以任务标签回收自己的容器，外层超时只是兜底，异常退出后仍须按 `ownership.json` 核验物理释放。

**Results。** 本轮在用户指定的优先实验节点后台脱离连接执行。环境激活、导入来源、Docker 探活和镜像缓存检查通过。实际观测如下：

| 检查 | 实测结果 |
|---|---|
| CPU 回归 | 29 项通过，覆盖新接口和原有 worker/sandbox 路径 |
| 后端真实工作流 | 4.54 秒，退出码 0，含本脚本的创建、执行与清理，不含环境激活 |
| Docker 限制 | `NanoCpus=1000000000`，`Memory=268435456`，`NetworkMode=none`，即 1 CPU、256 MiB、禁网 |
| 同容器不同命令语义 | 持久接口得到 `kept:/tmp`；Harbor 接口得到 `unset:/work:a b42`，stderr 单独为 `err` |
| 文件传输 | 环境目录内容上传位置正确，下载读到 `seed` 与答案 `42` |
| 超时清理 | `sleep 20` 在 0.2 秒超时预算下触发异常，随后确认所属容器不存在；0.2 秒不包含清理耗时 |
| 真实 Harbor | oracle → verifier → 产物收集完整结束，1 个 trial、0 错误、0 重试、reward=1 |
| 收尾 | 清理错误列表为空，脚本与独立后检均未发现本轮所属容器，后台实验进程已退出 |

原始证据位于本地 `env-service-verification/results/native-harbor-20261028/`，包含入口提交、脚本快照、导入来源、源码 SHA256、原始 Job/Trial 结果和退出码。结果目录默认不发布。这是接口可行性验证，**不是 science 训练，不是 Ray 远程派发，不是 GPU 租约验收，也不是 Harbor 两种后端的性能对照**。4.54 秒不能拿来声称新后端更快。

## 6. 自主管理的自由度、成本与后续边界

这轮验证支持的判断是：**单容器后端的最小闭环难度可控，完整替换 Harbor Docker 的工程量仍然不小。** 创建、执行、复制、销毁已经有可复用实现，主要成本转移到“哪些任务行为必须原样兑现”和“异常时能否确认清理”。

| 想获得的自由度 | 为什么统一实现有帮助 | 还要付出的工作 |
|---|---|---|
| 直接控制容器生命周期 | 生命周期入口由 PSRL 持有，资源标签和删除确认可以共用 | 进程重启后的持久所有权与恢复、远程 worker 失联回收 |
| 精确控制命令阶段 | Harbor 命令统一经过 `exec_process`，便于接 GPU 作业入口和超时监督 | 接入现有 GPU lease、设备授权、归还确认，不能绕过安全预算 |
| 镜像预热、容器复用 | 可在统一创建入口制定策略 | 镜像构建缓存、多服务依赖；容器复用还需彻底清理任务状态，不能只复用一个脏 shell |
| 统一观测与取消 | 命令开始、输出、结束和异常都有明确入口 | 容器内逐作业隔离、进程归属、独立回收，当前仍以整个容器删除兜底 |
| 跨节点放置 | 后端 API 可以作为远程调用契约 | 远程文件传输、回调流量、重试幂等性和统一资源账本，远程部分尚未接入，CPU 整次任务预算已接入 |
| 更自由的网络和挂载 | 由后端明确决定执行参数 | 精确兑现白名单、阶段切换、代理 sidecar、多服务卷语义，而不是“能启动就算支持” |

因此可继续沿“保留 Harbor 任务层、由 PSRL 管理执行后端”的方向推进。CPU science 接线与真实 LAPS oracle 验证已完成，下一步仍需接入 GPU 资源协议与远程 worker 生命周期，之后才考虑替换默认路径。不能因为 CPU oracle 通过就删除仍然承担 GPU、网络与 Compose 能力的已有后端。

这与[执行路径审查](../experiments/router-paired-latency-and-design-review.md)及 [GPU 统一执行后端](unified-gpu-execution.md)是不同层面的统一：GPU 作业协议统一解决 submit/wait/cancel；本文解决 Harbor 环境接口到 EnvWorker 容器实现的复用。二者最终应接起来，但本轮只完成了后者的 CPU 范围。


## 7. 后续修复优先级与生命周期故障复现

本节保留修复前版本 `8d4de8f` 的故障复现，图示和原始输出是历史证据。源码链接更新到当前对应函数入口，旧逻辑已被修复，不能把图中的提前 ready 当作当前行为。结论是先修就绪状态，再接统一资源准入和回收，随后测量共享计算干扰。本节仅记录审查、CPU 故障注入和建议，没有修改后端运行逻辑。前面的正常 CPU oracle 成功仍成立，但不覆盖初始化与清理同时失败。

先看确认的问题如何发生。A、B 是两种分别触发的路径，不要求同时发生：

```text
[1] start → create_sandbox 成功 → _started=True
                                 │
            A：环境上传仍在等待 ─┼→ [2] 并发 exec 通过检查，进入后端
                                 │
            B：环境上传失败 ────┴→ [3] destroy_sandbox 再次失败
                                          │
                         _started=False / _closed=True 没有执行
                                          ↓
                              [4] 再次 start 直接返回成功
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 代码与职责 |
|---|---|
| 1 | [harbor.py:106，start](../../../psrl/workers/env_worker/harbor.py#L106)，创建后立即设置 `_started`，随后才准备目录并上传环境 |
| 2 | [harbor.py:202，exec](../../../psrl/workers/env_worker/harbor.py#L202)，只检查 `_started` 和 `_closed`，不等待初始化结束 |
| 3 | [harbor.py:135，异常处理](../../../psrl/workers/env_worker/harbor.py#L135)，先 await 清理，再更新状态，清理抛错会跳过状态更新 |
| 4 | [harbor.py:110，启动检查](../../../psrl/workers/env_worker/harbor.py#L110)，已有 `_started=True` 时直接返回 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，从 1 的正常启动读到上传等待，分支 A 跳到并发调用 2。分支 B 则从上传异常进入 3，清理再次异常后返回调用者，调用者重试才进入 4。第二条调用链不是“容器成功初始化两次”。

### 7.1 第一优先：修复生命周期的就绪状态

假设容器已创建，但题目数据上传到一半失败，Docker 又暂时无法执行删除。当前适配器会留下 `_started=True`、`_closed=False`。调用者第二次要求启动时，没有重新准备环境，却收到正常返回。这会让上层误以为环境可用，掩盖第一次初始化失败。

另外，数据还没上传完时，有另一个协程调用 exec，会进入后端执行。正常 Harbor 顺序调用通常先等待 start，因此本次不能声称生产 science 已发生该并发故障。但后端接口确实没有防住它，扩展并发执行前应该修复。

**修复前的建议方案：** 显式区分 `starting`（准备中）、`ready`（可用）、`closing`（清理中）、`cleanup_failed`（清理未确认）与 `closed`（已关闭）。只有目录、权限、环境文件全部准备成功才发布 ready。初始化失败时先撤销可用状态，再尝试清理。清理失败保留所有权和重试入口，不能把环境伪装成 ready，也不能未经确认就记为 closed。exec、上传和下载统一检查就绪状态，内部启动步骤通过专用内部方法执行，避免因公共接口的新门禁使启动自己卡住。

这不是简单地把 `_started=True` 挪到函数最后：当前启动中的 `ensure_dirs` 和上传过程也会调用适配器方法，需要一并理清“内部准备”和“对外可用”。命令超时导致底层销毁后，也应同步适配器状态。收尾复查发现并行工作已在未提交改动中加入 `_invalidate` 和 `exec_shell`，本节不将这部分算成本轮修复；上述 start 提前发布状态及清理失败问题仍未被覆盖。

**最小验收：** CPU 注入初始化失败、清理失败、并发 exec，检查拒绝错误就绪且能够重试清理。修复后再用一个真实 CPU 容器验证启动中断及删除结果，不需要训练或 GPU。

### 7.2 第二优先：接入现有资源准入和回收，再扩展 science GPU

修复前每个 `EnvWorkerEnvironment` 自行实例化一个 EnvWorker。单容器 Docker CPU/内存上限已经生效，但当时这些实例没有共同参加 EnvRouter 的节点预算核算。当前 science 入口已补齐整次任务准入。这是明确的接入缺口，不是本轮复现的新 OOM。

教学例子：一台机器可用于环境的内存是 32 GiB，8 个任务分别配置 6 GiB。每个容器的 6 GiB 限制都正确，不妨碍总申请达到 `8 × 6 = 48 GiB`。单容器限额回答“一个容器最多用多少”，节点准入回答“现在还能不能再启动一个”。二者都需要。

建议复用既有资源租约，而不是再增加一份独立总账：先申请 CPU/内存/容器名额，创建时绑定所有权，销毁并确认后释放。再让 GPU 命令复用既有 command 的设备授权、预算和物理归还协议。现有 science 接线入口见 [runner.py:100](../../../examples/sciaccel_rl/runner.py#L100)，不能直接解除适配器拒绝 GPU 参数的检查，就宣称接入完成。

最小验收先做受控容量下的两个 CPU 容器：第二个必须等第一个物理回收后才能取得名额。之后根据目标 science 任务实际需要补齐镜像或网络能力，再验证一条真实 GPU 命令与正常归还。跨节点 worker 池和多服务不必同时全部实现。可信进程归属沿用另一项工作的接口，不在这里重复建设。

### 7.3 第三优先：把同卡共享的性能边界测出来

这是测量与保护策略的缺口，不是已确认的错误调度。现有功能实验主要持有显存，不能量化环境持续计算使模型变慢多少。详见[计算竞争与控制等待](vllm-env-opportunistic-sharing.md#11-计算竞争与控制等待已经验证到哪里)。

先把排队、GPU 命令、归还、模型控制门禁的时间边界分开记录。使用一个模型、小量固定输入输出、有限轮数，比较模型独占、环境仅持有显存、环境持续计算三组。分别报告生成耗时和控制等待，不用包含模型加载的总时间替代。训练吞吐不是这一轮必要验收目标。

有了数据，再决定哪些环境命令可以进入共享卡、命令期限要多短、是否需要限制共享并发。缺少测量时，不能通过减小显存安全余量来解决计算竞争，更不应把主动 offload 当作必然更快的方案。

### 7.4 暂缓：队列回填与主动 offload

[command_gpu.py:82](../../../psrl/workers/env_service/command_gpu.py#L82) 当前按 FIFO，即先到先服务，选择队首等待者。教学例子：只剩 8 GiB，队首需要 12 GiB，后面需要 2 GiB，也可能跟着等待。允许小请求绕过队首称为回填，确实可能提高利用率，但必须限制绕过次数或保留预约，避免大任务一直没有机会。这是优化取舍，不等同于当前队列实现错误。

相比之下，就绪状态错误能直接让调用者误判环境成功，应先修。主动 offload 还需要空窗判断、恢复成本和提前返回处理，当前也没有充分性能证据支持优先扩展。

### 7.5 Results 与可运行复现

配套 [review_native_harbor_lifecycle.py](../env-service-verification/scripts/review_native_harbor_lifecycle.py) 调用真实适配器，用测试替身代替 Docker 创建、删除和上传，内部 5 秒预算，不触碰共享服务：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/review_native_harbor_lifecycle.py
```

本次实际运行得到：

```json
{
  "exec_accepted_before_upload_finished": true,
  "first_start_error": "docker unavailable",
  "second_start_returned_success": true,
  "create_calls_after_two_starts": 1,
  "started_after_failed_cleanup": true,
  "closed_after_failed_cleanup": false
}
```

第一项对应图 A，后几项对应图 B，明确证明接口的状态问题。`docker unavailable` 是人为注入的异常，不表示实验节点的 Docker 真的故障。本次没有启动真实容器、模型或训练，也没有把上述接入建议写成已完成修复。
