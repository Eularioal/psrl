# 原生容器后端修复与真实 science 短验证

> **核心结论**：修复初始化期间过早开放命令的问题，清理失败保留待回收状态，并将原生 CPU 后端接入 science runner 的整次任务资源准入。真实 LAPS repair oracle 与独立 verifier 完成，两组 `reward_repair=1`。单次任务调用缩短约 21.7%，但包含新建 Ray／EnvRouter 的整轮时间增加约 14.6%，不能宣称短任务端到端提速。
>
> 这是缓存镜像、禁网 oracle 对照，不是模型生成或 PSRL 训练 step。新后端 GPU command、原生 allowlist 网络与远程容器 worker 池仍未接入。vLLM 计算竞争测量在模型加载前被 gRPC 依赖冲突阻断。

## 目录

- [修复范围与调用链](#修复范围与调用链)
- [实验设计与版本](#实验设计与版本)
- [Results 与收益边界](#results-与收益边界)
- [验证失败与未完成项](#验证失败与未完成项)
- [复现入口与收尾](#复现入口与收尾)

## 修复范围与调用链

```text
[1] science runner 读取整次任务预算
        ↓ 先预留 agent + 独立 verifier，避免各占名额后互相等待
[2] EnvRouter 准入，runner 持有租约与心跳
        ↓ 将 lease/unit 标签和单容器预算交给后端
[3] EnvWorkerEnvironment 启动
        创建 → 目录/权限 → 上传 → 最后发布 ready
        初始化内部操作只允许启动协程访问
        外部 exec / 上传 / 下载在准备阶段被拒绝
        ↓
[4] oracle 修复真实 LAPS → 独立 verifier 检查两个物理算例
        ↓
[5] 停止对外服务 → 删除容器 → 确认清理
        runner 释放租约，既有节点回收协议处理待清理资源
```

图例：图里每个部分对应的源码位置。

| 部分 | 函数与源码 |
|---|---|
| 1 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L273)，选择 `harbor.environment_backend=env_worker`，要求已配置 EnvRouter |
| 2 | [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L100)，CPU 原生后端传租约标签和预算，不生成 Compose overlay |
| 3 | [start](../../../psrl/workers/env_worker/harbor.py#L106)、[_require_running](../../../psrl/workers/env_worker/harbor.py#L166)，分别管理内部准备和对外就绪 |
| 4 | [真实 science 脚本](../env-service-verification/scripts/science_native_backend_check.py)，通过 PSRL runner 调用 Harbor oracle 与原任务 verifier |
| 5 | [_cleanup](../../../psrl/workers/env_worker/harbor.py#L150)，等待所属删除任务，失败保留 `_cleanup_pending`，允许 stop 重试 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，先读 1 的预算申请，再读 2 的参数转换，然后进入 3 的启动及异常分支。4 的命令与文件操作必须等 ready。最后读 5 和 runner 的 finally，不能把适配器禁止继续使用的 `_closed` 标记理解为物理删除已经确认，后者由 `_cleanup_pending` 与实际删除结果区分。

预算复用的是既有整次任务计划，没有引入另一份独立总账。教学例子：每个任务的 agent 和 verifier 各需要 4 CPU。如果节点只有 8 CPU，两个任务各先占 4 CPU 再等待自己的 verifier，会互相卡住。当前接线在启动前按两类容器合计预算准入。单容器限制超过已准入的主容器预算会被拒绝。

## 实验设计与版本

- 固定 baseline：`0eeead861ffafed6bf31c962d259fd847775dcf9`，使用已有隔离基线目录。当前组：`8d4de8f` 加本轮未提交补丁。
- baseline 已有运行补丁单独保存：透传 GPU override、禁用节点级 dangling-image prune、Ray 地址/runtime-env 处理、HTTP 连接池按事件循环归属。没有增加整卡锁或改变静态 GPU 调度语义。本次 CPU oracle 不进入训练和 GPU 调度路径。
- 同一真实任务：`sciaccel/laps-repair-coef-2d-rktmod-l18k4`，原标准答案修复 Fortran 缺陷并运行物理算例，独立 verifier 检查 `aw-2d-256` 和 `aw-2d-512`。
- 两组均使用已有缓存的 agent/verifier 镜像，移除任务副本中的 Dockerfile 构建入口，将运行时策略从 allowlist 收紧到 no-network。科学代码、答案、算例和评分文件不改，保存原始文件 SHA256。原任务目录不改。
- 本轮每组一个 oracle episode，无模型、无训练 step、无环境 GPU。native 先运行、baseline 后运行，不随机化顺序，不重复采样。
- 每组内部 episode 上限 480 秒，外层 Python 上限 570 秒。GPU 功能／延迟脚本另设上限，依赖失败发生在其工作流之前。
- 执行在用户指定的优先节点，现场 Docker 探活、镜像缓存和 Ray 节点身份检查。后台脱离连接运行，结果与启动脚本保留。

## Results 与收益边界

| 指标 | 固定 baseline | 新后端 |
|---|---:|---:|
| PSRL runner 调用，含该调用内清理 | 50.58 秒 | 39.60 秒 |
| 整轮脚本计时，含 native 独立 Ray／Router 启动和收尾，不含 Python 导入及任务复制 | 50.73 秒 | 58.16 秒 |
| Harbor 主环境准备 | 5.28 秒 | 2.03 秒 |
| oracle 执行 | 10.05 秒 | 10.41 秒 |
| Harbor verifier 阶段记录 | 19.92 秒 | 3.80 秒 |
| `reward_repair` | 1 | 1 |
| 两个算例、等价性检查 | 全部通过 | 全部通过 |
| 任务异常 | 无 | 无 |
| 本轮所属容器残留 | 无 | 无 |

各阶段字段来自 Harbor 原始 Trial。独立 verifier 的准备与传输可能跨越阶段边界，这些行不能简单相加替代 runner 总时间。runner 减少 `50.5796 - 39.5983 = 10.9813` 秒，约 21.7%。整轮反而增加约 7.43 秒，约 14.6%。所以本轮支持“容器任务路径有改善”，不支持“每个短任务从零启动服务也更快”。

长期训练若复用已有 Ray／Router，服务初始化开销可能被多次任务分摊，这是推断，本轮没有测量训练吞吐或稳态多轮效果。单次样本还受缓存、宿主机负载和执行顺序影响，不能把 21.7% 作为稳定性能承诺。

任务返回的 `speedup=1.133/1.135` 是科学任务自身的评分指标，不是后端提速，不能拿来替代上述计时对照。

另一个真实 Docker 故障实验通过：容器创建后阻塞环境上传，外部命令被拒绝；注入上传失败和第一次删除失败后，原租约仍占用唯一名额，第二请求返回 wait；重试删除成功、释放并确认后，第二请求得到 granted。这个实验使用进程内真实 EnvRouterCore 和实际 Docker，故障异常人为注入，不宣称验证了远程节点故障恢复。

相关 runner、配置、后端回归最终 **46 项通过**。新增检查覆盖准备阶段拒绝外部命令、清理失败后的重试、超过容器预算的拒绝、整次任务预算传递而不创建 Compose overlay。

## 验证失败与未完成项

1. 最初的后端回归有一个真实 Bash 命令在 2 秒预算内超时。单独复跑通过，之后相关 46 项一起通过。保留首轮日志，没有把首轮写成全通过。
2. 首轮生命周期实验错误地期望进程内 EnvRouterCore 在 retire 后具备 Ray EnvRouter 的节点清理围栏，断言失败，容器已清理。纠正实验为实际 runner 的“持有租约至清理完成，再释放”顺序后通过，没有因此修改或弱化生产准入逻辑。
3. vLLM 干扰脚本增加了模型独占、环境持有显存、环境持续 CUDA 负载三组短请求，以及唤醒请求到完成的计时。但依赖加载时报错：运行时 grpcio 为 1.83.0，SMG 生成代码要求至少 1.85.0。隔离安装目标目录的尝试未在 60 秒预算内完成，没有修改共享 Python 环境，没有启动模型或取得 GPU 延迟数据。
4. 唤醒请求到完成包含归还等待和恢复本身，不能把该字段直接称为纯门禁等待。短请求样本也不等于持续训练或尾延迟统计。新测量代码未完成 GPU 实机验收，不能沿用旧的共存实验作为它已通过的证据。
5. 新后端 GPU 设备授权／command、网络 allowlist、多服务 Compose 和远程容器池仍未实现。science 默认后端未切换。

## 复现入口与收尾

在项目已配置的 Python 环境、仓库根目录运行：

```bash
python -m pytest tests/env_service/test_runner_env_service.py tests/env_service/test_science_native_config.py tests/env_worker/test_harbor_backend.py -q
python docs/env-worker-deep-dive/env-service-verification/scripts/science_native_backend_check.py --help
python docs/env-worker-deep-dive/env-service-verification/scripts/real_native_lifecycle.py --help
```

science 脚本要求显式指定原任务、已缓存的两种镜像、节点 IP、模式与新输出目录。baseline 模式必须使用固定版本源码的 PYTHONPATH，不能只在当前代码关闭 EnvRouter 代替。生命周期脚本只需要兼容 Bash 的 CPU 镜像与新输出目录，最多一个真实容器，内部 60 秒预算。

本地证据目录为 `env-service-verification/results/native-fixes-20261029/`，含两组原始任务、奖励、Trial、退出码、baseline 补丁、当前补丁、回归和生命周期故障日志。GPU 依赖失败日志单独位于 `results/sharing-latency-20261029/`。结果目录默认不随 Git 发布。

脚本按本轮 actor 标签清理资源，并核对无所属容器。未进行节点级 Docker/Ray 重启或缓存 prune。代码与本轮文档保持未提交，既有其他会话改动未覆盖。
