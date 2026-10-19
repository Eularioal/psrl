# AIRS-Bench 已知问题修复与小实验对比

## TL;DR

- 修复了普通 CSV 任务的标签挂载、独立评分、shell 超时串线、排队占用 RPC、提前创建会话、持有者崩溃后的回收，以及 runner 线程无法取消的问题。
- 实际 MLGym `BaseAgent` 三步回放通过。三行预测 `[0, 1, 0]` 对照标签 `[0, 1, 1]`，`validate` 与 `submit` 都返回 `2/3`。
- 全组 101 项中先有 100 项通过。修正剩下的远端重置问题后，正常和故障两项真实 MLGym 定向回归均通过。随后增加评分故障分类的两个用例，最终覆盖 103 个不同用例，不把它写成一次整组零失败的运行。
- **2026-10-15 已补齐 APPS 独立代码执行容器并移除入口禁用。** 见 [APPS 隔离实现与新实验](apps-isolation.md)。完整 benchmark 及训练链路仍未验收。
- **2026-10-16 已修复后续发现的评分异常分类问题。** 正式 AIRS 归档镜像上完成四题小样本新旧对照和真实 MLGym 单 episode 回放，详见 [后续验收](apps-isolation.md#评分修复与正式镜像补验)。全量吞吐仍未解决，分数反馈问题按用户要求搁置。
- **第二轮复核又修复了最终奖励沿用旧分数、APPS 类型适配、shell I/O 阻塞和回收队列阻塞等问题。** 本轮 145 项 CPU 回归全部通过，见 [第二轮接入复核](#第二轮接入复核)。没有重新运行 Docker、集群或训练。
- 本机 Docker 探活在 5 秒后超时。随后在健康远端节点完成真实 Docker 小实验，验证标签不可见、独立评分返回 2/3、超时物理删除和资源清理。使用合成数据与小型测试镜像，未启动 Ray、GPU、模型或训练。

## 目录

- [修复后的完整路径](#修复后的完整路径)
- [跟着图读源码](#跟着图读源码)
- [问题与修复选择](#问题与修复选择)
- [修复前后实验对比](#修复前后实验对比)
- [如何重跑小实验](#如何重跑小实验)
- [远端 Docker 实机补验](#远端-docker-实机补验)
- [逐项验收状态与交付入口](#逐项验收状态与交付入口)
- [APPS 是什么以及为什么暂时禁用](#apps-是什么以及为什么暂时禁用)
  - [Pass@5 的小例子](#pass5-的小例子)
  - [APPS 评分源码阅读路径](#apps-评分源码阅读路径)
  - [为什么当前普通 CSV 评分隔离还不够](#为什么当前普通-csv-评分隔离还不够)
- [验收边界与剩余限制](#验收边界与剩余限制)
- [当前剩余问题复核](#当前剩余问题复核)
  - [已复现的评分错误分类缺陷](#已复现的评分错误分类缺陷)
  - [全量 APPS 的耗时与评分兼容性](#全量-apps-的耗时与评分兼容性)
  - [私有测试反馈与评测协议](#私有测试反馈与评测协议)
  - [生命周期与正式运行的缺口](#生命周期与正式运行的缺口)
- [第二轮接入复核](#第二轮接入复核)

## 修复后的完整路径

本报告承接 [修复前审查第 17 节](../implementation-and-gaps.md#17-airs-bench-接入批评核对与最小复现)。旧结论对应的源码已保存为 `baseline/`，其中五个关键文件与旧结果记录的 SHA-256 完全一致。本次是对修复前后实现的比较，不将历史记录改写为本次实测。

```text
[1] EnvWorkerManager：注册远端 worker 与资源预算
                    |
[2] MLGymAgentLoop：任务预检查 -> 客户端轮询 reserve
                    | 无槽位就返回等待，不创建 TITO session
                    | 获准后启动持有者心跳
                    v
[3] EnvWorker：创建 agent 容器 -> 就绪探测
                    | 只挂 train / validation / test
                    v
[4] 创建 TITO session -> 启动可终止的 MLGym 子进程
                    |
[5] BaseAgent 工具调用 -> 进程间请求 -> handle -> EnvWorker.exec
                    | 每个 shell 串行执行，每条命令有独立结束标识
                    | 超时：销毁沙箱，本条轨迹按基础设施错误丢弃
                    v
[6] validate / submit -> 拷贝 submission.csv 的有界快照
                    -> 同一远端 worker 上的新 verifier 容器
                       私有标签 + 可信评分代码，agent 不能访问
                    -> 只返回有限数值指标，删除 verifier
                    v
[7] 正常结果 -> 获取 TITO 训练数据 -> 计算奖励 -> 关闭会话
                    |
[8] finally：回收 runner -> 请求远端删除沙箱 -> 归还资源
                    ^
       并行分支：心跳停止 -> coordinator watchdog -> 同一清理路径
                 删除失败：保留资源占用，继续重试
```

图例：图里每个部分对应的源码位置。

| 节点 | 实际实现 |
|---|---|
| [1] actor 创建、CPU 和 RAM 预算 | [manager.py:52](../../../psrl/workers/env_worker/manager.py#L52) |
| [2] 整个 episode 的超时范围、会话顺序 | [MLGymAgentLoop.run:109](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L109) |
| [2] 客户端轮询和心跳 | [acquire_sandbox:164](../../../psrl/workers/env_worker/coordinator.py#L164)、[SandboxHandle._beat:124](../../../psrl/workers/env_worker/coordinator.py#L124) |
| [3] 公共数据挂载与沙箱启动 | [build_sandbox_spec:60](../../../psrl/environments/mlgym_env.py#L60)、[create_sandbox:222](../../../psrl/workers/env_worker/worker.py#L222) |
| [4] 子进程与实际 MLGym 调用 | [run_in_process:57](../../../examples/airs_bench/process_runner.py#L57)、[run_mlgym_agent:123](../../../examples/airs_bench/runner.py#L123) |
| [5] 环境契约适配与命令结束标识 | [build_psrl_mlgym_env_class:142](../../../psrl/environments/mlgym_env.py#L142)、[exec:276](../../../psrl/workers/env_worker/worker.py#L276)、[wrap_command:22](../../../psrl/workers/env_worker/shell.py#L22) |
| [6] 私有评分挂载、指标校验与容器归属 | [build_verifier_spec:32](../../../examples/airs_bench/grading.py#L32)、[parse_metrics:86](../../../examples/airs_bench/grading.py#L86)、[evaluate:402](../../../psrl/workers/env_worker/worker.py#L402) |
| [7] 训练数据与奖励 | [run:109](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L109) 中的 session 内分支 |
| [8] 删除后释放、失联回收 | [release:284](../../../psrl/workers/env_worker/coordinator.py#L284)、[reap_expired:302](../../../psrl/workers/env_worker/coordinator.py#L302)、[destroy_sandbox:377](../../../psrl/workers/env_worker/worker.py#L377) |

这些代码如果有理解上的问题，可以随时追问。这里的 verifier 是独立评分容器。TTL 是“心跳多久未更新就判定持有者失联”的时限，不是命令允许执行多久，也不是容器空闲多久。

## 跟着图读源码

按实际调用顺序阅读，不要把后台心跳误认为每条命令后的串行步骤。

| 步骤 | 图中位置 | 输入、调用和状态变化 |
|---|---|---|
| 1 | [1] | `EnvWorkerManager.__init__` 读取节点资源，给每个 worker 注册预算，入口见上表 manager.py |
| 2 | [2] | `run` 启动 episode 总预算，`env.reset` 检查 task，预检查 verifier 配置 |
| 3 | [2] | `acquire_sandbox -> reserve`，按 CPU、RAM、槽位和 GPU 共同准入，资源不足就在客户端等 |
| 4 | [3] | 获准后启动心跳，再调用 `worker.create_sandbox`，探测失败也走清理 |
| 5 | [4] | 创建 session，`run_in_process -> _run_child -> run_mlgym_agent`，LLM URL 仍由 TITO session 提供 |
| 6 | [5] | `BaseAgent -> PSRLMLGymEnv._communicate -> sync_exec -> RemoteHandle -> handle.exec -> worker.exec` |
| 7 | [6] | `validate/submit -> grade_submission -> handle.evaluate -> worker.evaluate`，只快照 CSV，重新启动可信评分环境 |
| 8 | [7] | 正常结束才获取训练数据。`exit_error` 不进入奖励计算，未知/空退出状态不能伪装成提交成功 |
| 9 | [8] | `run_in_process.finally` 终止并 join 子进程，episode finally 释放 handle，再执行远端物理清理 |
| 并行 | [2]/[8] | `_beat` 定期续期，`_reap_loop` 每五秒检查失联或未完成的删除，二者与 agent 执行并发 |

所有函数的源文件入口均在上面的图例中。TITO 仍承担模型 token 与概率记录，子进程只隔离同步 runner 的生命周期，没有改动 MLGym agent 的推理循环。

## 问题与修复选择

| 问题 | 修复后的行为与原因 |
|---|---|
| `test_with_labels` 可见 | agent 只挂载明确命名的公开 split，禁止把任务父目录整体挂进去。标签只进入 verifier |
| agent 修改评分脚本或 Python 环境 | verifier 使用干净镜像、只读可信代码和 CSV 快照，不运行 agent 工作区里的 `evaluate.py` |
| 评分器执行故障被当成坏提交 | 评分器超时和非 2 的非零退出码按基础设施错误丢弃轨迹，避免缺少依赖时惩罚 agent。退出码 2 返回格式错误。该判断依赖评分脚本的退出码约定，不声称能诊断所有评分器异常 |
| 超时输出串线 | 超时销毁沙箱，不承诺恢复同一个 shell。唯一命令标识防止旧标记或普通同形输出提前结束本次读取 |
| 持续输出或进程秒退 | 每次读取都检查总截止时间，管道对应进程退出立即报错，不再空等启动预算 |
| 先开会话再排队 | 先准入并启动沙箱，再开 session。外层 episode 总超时覆盖排队和执行 |
| 等待 RPC 堵住 release | `reserve` 只做一次即时判断，等待移到客户端，不把长等待驻留在 actor 并发池中 |
| agent worker 消失 | 不依赖调用方 finally。心跳到期由 coordinator 向实际容器所在 worker 发起回收 |
| 删除失败误还槽位 | reservation 保持 closing 和计费，确认 Docker 删除成功后才释放。重复 release 按唯一 id 幂等 |
| runner await 超时但线程存活 | 改为每 episode 独立进程，取消时 terminate、join，必要时 kill，临时目录由父进程负责清理 |
| MLGym 直接依赖 Docker 对象 | 显式适配 reset、工具写入、命令执行和评分。禁止沿上游路径重置失效的远端容器 |
| 原版接口参数错误 | 修正 `agent_config_path`，提供轨迹目录，按实际 TaskConfig schema 转换 staged YAML 和绝对资源路径 |
| 配置无消费者 | 删除 env_worker 的死 `sandbox_cpus/memory` 和 runtime 的死 observation 预算。容器规格由 runtime 管，截断由 env_worker 管。`exec_default_timeout_s` 现在用于未显式传入命令超时时的缺省值 |
| 资源吞吐配置 | agent 和 verifier 一起做资源预留，槽位配置只是上限。不把 56 个待运行 episode 当成 56 套物理资源 |
| host 网络、代理凭据和 sudo apt | AIRS 默认无网络，不继承代理，镜像移除免密 apt 权限，运行时关闭提权并移除 capabilities |

**超时策略有明确代价。** 训练命令超过时限会损失本 episode 的工作，交给上层按基础设施错误处理。它不会以“发了 Ctrl-C 所以一定已经恢复”为假设复用 shell。需要保留超时后环境状态，必须另做可验证的进程组中断和恢复协议，本次没有声称完成这项能力。

资源手算例子：默认 agent 为 8 CPU、32 GiB，verifier 为 1 CPU、4 GiB，所以每个 episode 预留 9 CPU、36 GiB。8 个 episode 是 72 CPU、288 GiB。若某个 worker 预算只有 36 CPU、144 GiB，即使槽位上限设为 8，也最多准入 4 个。以上是配置算术，不是本次集群吞吐实测。

## 修复前后实验对比

| 验证对象 | 修复前证据 | 修复后实际观察 | 验证边界 |
|---|---|---|---|
| shell 超时 | A 超时，B 拿到 A-end/退出码 0，C 拿到 B-output/退出码 1 | 超时返回后 shell 已删除，B 被拒绝，不产生下一条错位观测 | 真 Bash，Docker 创建与删除替换成本地进程组操作 |
| 进程秒退 | exit 7 后仍耗尽 0.2 秒读取预算 | 秒退触发 RuntimeError，测试要求小于 1 秒 | 真 Bash，非真实缺失镜像 |
| 活跃输出 | 旧读循环只在没数据时检查 deadline | 持续输出在短截止时间后停止并销毁 shell | 真 Bash，0.1 秒命令预算 |
| 持有者消失 | worker reaper 删除后 coordinator used_cpu 仍为 1 | 注入过期，删除成功后 live 集合为空，used_cpu 为 0 | Docker 删除为测试替身，不是实际杀 Ray actor |
| 删除失败 | 正常 handle finally 可无条件还计数 | 注入删除失败时 used_cpu 仍为 1，重试成功后为 0 | 调用真实 coordinator 生命周期代码 |
| 超额等待 | 并发池模型中 release 被长等待挡住 | 80 个客户端等待者期间 release 完成，取消后无残留 reservation | 不是真实 Ray 压测，不声称测得 56 路吞吐 |
| runner 取消 | 等待 future 被取消，线程仍运行 | 启动阻塞子进程，取消后 PID 不存在，进程已 join | 真实独立 OS 进程，无模型请求 |
| MLGym 正常路径 | 最小真实方法重放在 `_init_scripts` 断言失败 | 真 `BaseAgent` 写 CSV、validate、submit，两个分数都为 2/3 | Docker、conda 激活和评分容器用明确替身，未连接 TITO |
| MLGym 故障路径 | 原 runner 可能把空退出状态当 submitted | 注入远端执行失败，最终为 exit_error，无训练分数 | 实际上游 agent 的异常捕获与返回逻辑 |
| 数据隔离 | 整个 dataset 父目录映射进 agent | agent spec 只有 train/test，verifier spec 才有私有标签父目录，代理为空 | 配置回归，不是 Docker 内攻击实验 |
| 评分失败分类 | 非零退出统一作为坏提交 | 真 Python 子进程缺依赖退出 1，被转为 RuntimeError。argparse 拒绝输入退出 2，被转为 ValueError | 验证退出码分类，不是全部任务评分脚本的故障注入 |

完整记录：

- [修复前结果](results.json)、[本次快照重放结果](baseline-replay.json) 与 [可重放的修复前脚本](reproduce.py)，脚本现在读取 [baseline 源码快照](baseline/psrl/workers/env_worker/worker.py)。
- 整组回归日志（本地资料，未随仓库发布） 与 JUnit 结果（本地资料，未随仓库发布），保留了 100 通过、1 失败的真实中间状态。
- 修正远端重置后的定向日志（本地资料，未随仓库发布） 与 JUnit 结果（本地资料，未随仓库发布），正常和故障回放均通过。
- shell 定向回归日志（本地资料，未随仓库发布） 与 JUnit 结果（本地资料，未随仓库发布），默认命令超时配置接入后 20 项通过。
- 评分隔离与故障分类日志（本地资料，未随仓库发布） 与 JUnit 结果（本地资料，未随仓库发布），10 项通过，其中两个新增用例使用真实 Python 子进程。
- [最终汇总与源码摘要](after-results.json)，按测试身份合并最终结果，不覆盖原始失败日志。
- [Docker 探活结果](docker-probe.json)，5 秒超时后停止，不修改 daemon。
- 全库静态检查日志（本地资料，未随仓库发布），存在 SciAccel 等本次范围外文件的既有问题。本次文件另做针对性 Ruff、格式和 prose 检查。

全库 pre-commit 使用原配置的只检查副本：移除 Ruff 的 `--fix`，formatter 加 `--check`，防止顺手改动工作区已有的其他修改。它没有因为这些既有问题而得到全库通过结论。

## 如何重跑小实验

从本工作区根目录重放修复前逻辑，预算数秒，不需要安装训练栈：

```bash
python docs/env-worker-deep-dive/airs-bench-audit/reproduce.py \
  --mlgym-repo "$MLGYM_REPO" --output /tmp/airs-before.json
```

`baseline/` 是实验夹具，不是可部署的实现。不要将它加到正常训练的 PYTHONPATH。

从 PSRL 仓库根目录验证修复后的关键路径：

```bash
python -m pytest tests/env_worker/test_airs_regressions.py \
  tests/examples/airs_bench/test_grading_isolation.py \
  tests/examples/airs_bench/test_process_runner.py \
  tests/examples/airs_bench/test_mlgym_agent_loop.py -q
PYTHONPATH="$MLGYM_REPO:${PYTHONPATH:-}" \
  python -m pytest tests/examples/airs_bench/test_upstream_mlgym.py -q
```

测试脚本分别对应图 [5]/[8]、[3]/[6]、[4] 和 [2]/[7]。真实上游回放对应 [4] 到 [7]，使用上游 ReplayModel 提供固定动作，不做网络 LLM 请求。实验夹具只有本地 Bash、三行 CSV 和少量 Python 对象。

本次 MLGym checkout 为 `9d40c1b5035202018cd7091fb4e83a9c68b377c0`。缺失的 gymnasium 1.0.0 与 simple-parsing 0.1.7 仅安装到仓库忽略的 `.verification-deps-airs/`，测试时显式加入 PYTHONPATH，没有更改共享环境。正式 controller 仍需正确安装上游依赖。

## 远端 Docker 实机补验

2026-10-14，用户授权改用其他服务器后，现场探测 `192.168.1.10` 与 `192.168.1.13`，两台的 `docker info` 与 `/_ping` 均成功。选择前者运行，路由检查确认实际执行节点为 `192.168.1.10`。该节点 Docker 版本为 26.1.3，共享工作区可见，PSRL 导入来源是当前工作区。依赖路径和五个关键源文件的 SHA-256 保存在结果 JSON 中。

**这是受控的真实容器实验。** 使用节点已有的 `python:3.13-slim` 加一个小层，创建非 root 的 `agent` 用户、工作目录及评分 Python 路径。没有安装额外包，没有下载 AIRS 数据或构建完整 AIRS 镜像。真实调用当前 `build_sandbox_spec`、`build_verifier_spec`、`grade_submission` 以及 `EnvWorker` 的创建、执行、文件读写、评分和删除方法，未替换 Docker 调用。

实验最多同时两个容器，每个限制 1 CPU、256 MiB，网络为 `none`。脚本总预算 210 秒，场景预算 90 秒，构建预算 60 秒。实际脚本耗时 75.603 秒，含依赖导入和镜像构建。

| 检查 | 真实容器中的观察 |
|---|---|
| 公共数据可读 | agent 成功读取三行 test 输入，Docker inspect 确认仅 train/test 两个只读挂载 |
| 私有标签不可见 | `data/test_with_labels/labels.csv` 不存在，可信评分脚本和 Docker socket 同样不在 agent 中 |
| 公共数据不可写 | 尝试追加 test 文件失败，返回 `READ_ONLY` |
| 持续 shell 正常 | 环境变量与工作目录跨 exec 保留，`false` 正确返回退出码 1 |
| 独立评分不使用 agent 的评分脚本 | agent 写入声称满分的假 `evaluate.py`，真实 verifier 仍用可信脚本将 `[0,1,0]` 对照 `[0,1,1]`，返回 `0.6666666666666666` |
| verifier 及时回收 | 评分返回后仅 agent 容器存在，worker 内部计数也为 1 |
| 超时不再串线 | `echo A-start; sleep 3; echo A-end` 设置 0.3 秒预算，约 0.404 秒后返回 timed_out，只含 A-start。此时容器已删除，下一条 exec 被 KeyError 拒绝 |
| 物理清理 | 按本次专属标签查询容器为空，临时镜像删除返回 0。独立 SSH 核对确认没有容器、镜像标签、脚本进程或临时目录残留 |

这次观测对应上方全局图的 [3]、[5]、[6] 和 [8]。没有把合成评分器的成功解释为所有 AIRS 任务的评分依赖都可用，也没有执行容器逃逸或任意攻击面的全面安全审计。

| 图中步骤 | 可重跑实验的对应位置 |
|---|---|
| [3] 公私数据构造和实际挂载 | [remote-docker-smoke.py:28](remote-docker-smoke.py#L28) |
| [5] 数据访问和 shell 状态 | [remote-docker-smoke.py:92](remote-docker-smoke.py#L92) |
| [6] 独立评分和评分容器回收 | [remote-docker-smoke.py:109](remote-docker-smoke.py#L109) |
| [5]/[8] 超时销毁与下一条拒绝 | [remote-docker-smoke.py:135](remote-docker-smoke.py#L135) |
| 实验预算、镜像准备和最终清理 | [remote-docker-smoke.py:154](remote-docker-smoke.py#L154) |

证据：[结构化结果](remote-docker-results.json)、完整输出（本地资料，未随仓库发布）、独立清理核对（本地资料，未随仓库发布）。原先的 103 项 CPU 回归汇总保持为独立批次，不将本次容器实验混入其测试数量。

在选定且 Docker 探活正常的服务器上，激活项目环境后执行。`PSRL_PATH` 是该节点可见的 PSRL 仓库路径，`AUDIT_DIR` 是本报告所在目录，基础 Python 镜像需要已在该节点缓存：

```bash
export PYTHONPATH="$PSRL_PATH${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=10 210 python "$AUDIT_DIR/remote-docker-smoke.py" \
  --repo "$PSRL_PATH" --run-id "airs-$(date +%Y%m%d-%H%M%S)-$$" \
  --output "$AUDIT_DIR/remote-docker-results-new.json"
```

正常与场景异常路径都会按本次标签回收资源。若整个进程被外层强制 kill，仍需在该节点按实际 run-id 查询并清理本次容器，不能用全局 prune。脚本只删除自建镜像标签，不清理 Docker 共享构建缓存。

## 逐项验收状态与交付入口

**不能将本次结果写成“全部原问题已修复并完成生产验收”。** 可以验收的是故障机制修复，以及明确覆盖到的 CPU、真实进程、真实容器路径。尚未验收的是全部 AIRS 任务、Ray 分布式故障恢复和真实训练链路。下表把实现完成与验证完成分开，避免用测试数量代替覆盖范围。

| 原问题 | 当前实现 | 已有证据 | 尚缺的验收 |
|---|---|---|---|
| P0 测试标签可见 | 普通 CSV 任务只挂公共 split，独立 verifier 读私有标签 | 远端真实容器：私有路径不可见、公开挂载不可写、伪造评分脚本不影响 2/3 分数 | 全部真实任务的数据与评分依赖。APPS 的后续隔离验证见 [新报告](apps-isolation.md) |
| P0 超时输出串线 | 唯一命令标识，超时销毁并拒绝复用 | 修复前 Bash 复现、修复后 Bash 与真实 Docker 对比，约 0.404 秒完成超时和删除 | 未验收保留状态后继续执行，该能力没有实现 |
| P1 会话先开、排队不受总超时约束 | 先准入再开会话，总截止时间覆盖等待 | [会话次序测试:43](../../../tests/examples/airs_bench/test_mlgym_agent_loop.py#L43) | 真实 TITO 服务上的会话生命周期 |
| P1 等待 RPC 阻塞释放 | 等待移到客户端，即时 reserve | [80 个等待者测试:141](../../../tests/env_worker/test_airs_regressions.py#L141) | 真实 Ray 并发池压力。不是实测 56 路吞吐 |
| P1 持有者消失导致槽位泄漏 | 心跳、失联回收、远端删除确认后释放 | [过期和删除失败注入:120](../../../tests/env_worker/test_airs_regressions.py#L120)，远端 Docker 另证实删除动作 | 杀真实 agent actor 后的完整回收路径。EnvWorker/coordinator 自身故障不在保证内 |
| P1 56 请求与 8 槽位不匹配 | 增加 CPU/RAM 预算，agent 和 verifier 一起预留，槽位上限可配置 | [不可能资源请求拒绝:157](../../../tests/env_worker/test_airs_regressions.py#L157) | 按正式节点资源配置预算、真实排队时延和吞吐 |
| P2 runner 超时后线程不退出 | 改成可终止和 join 的独立进程 | [真实子进程取消:48](../../../tests/examples/airs_bench/test_process_runner.py#L48) | 真实模型请求和 TITO 联调中的取消 |
| P2 镜像缺失长等与无效重试 | CLI 退出立即检测，启动预算默认 60 秒，删除原内部盲目重试 | [真实进程秒退:73](../../../tests/env_worker/test_airs_regressions.py#L73)，Docker 正常启动路径 | 尚未做远端真实缺失镜像故障注入。没有新增自动跨 worker 故障转移 |
| P2 双轨配置、权限和文档失真 | 删除死配置，预算归属明确，默认无网络/不继承代理，移除免密 apt，更新说明 | 配置与参数回归、实际 Docker network=none/只读挂载 | 完整 AIRS 镜像构建与权限检查。小实验使用派生的 Python 镜像 |
| 额外发现的 MLGym 适配缺口 | 补齐初始化、工具写入、评分与错误退出契约 | [真实上游 BaseAgent 正常/故障回放:74](../../../tests/examples/airs_bench/test_upstream_mlgym.py#L74) | 真实 BaseAgent、TITO、Docker、真实 AIRS 数据同时串通 |

资源数量没有被“修成 56 个槽”。默认一个 episode 为 agent 8 CPU/32 GiB，加 verifier 1 CPU/4 GiB，即 9 CPU/36 GiB。8 个并发共预留 72 CPU/288 GiB。资源不足时排队仍是预期行为，超过总等待预算仍可能失败。这是当前配置的算术，不是节点容量实测。

验收入口按以下顺序使用：

1. **本报告**：查看全局图、源码阅读路径、上表及最后的剩余限制。适合判断每项承诺是否达成。
2. **原始证据**：[CPU 汇总](after-results.json)、[远端 Docker 结果](remote-docker-results.json)、独立清理日志（本地资料，未随仓库发布）。CPU 汇总是 101 项整组中的 100 项通过，加修正后的定向回归及两个新增用例，合并后 103 个不同用例有最终通过记录。原失败日志保留，不能称为一次整组 103/103。
3. **生产使用说明**：[AIRS README](../../../examples/airs_bench/README.md)，包含环境变量、任务预检查、镜像准备、启动和 CPU 小测试。`validate_runtime` 只检查资产路径与任务契约，不会执行所有评分脚本。
4. **重跑实验**：上文“如何重跑小实验”提供 CPU 回归命令，“远端 Docker 实机补验”提供容器脚本及资源预算。容器脚本必须在 Docker 正常的目标节点执行，不能在已知 daemon 无响应的本机直接运行。
5. **源码差异**：改动位于本工作区 `psrl/`，目前尚未整理成独立 commit/PR。工作区还包含其他任务修改，不应把全部 `git diff` 算作 AIRS 交付。

从 PSRL 仓库根目录查看本次主要范围：

```bash
git status --short -- examples/airs_bench psrl/environments/mlgym_env.py \
  psrl/workers/agent_loop/loops/mlgym_agent_loop.py psrl/workers/env_worker \
  psrl/trainer/config/psrl/env_worker.yaml tests/env_worker \
  tests/environments/test_mlgym_env.py tests/examples/airs_bench docs/design/env_worker.md
git diff -- examples/airs_bench psrl/environments/mlgym_env.py \
  psrl/workers/agent_loop/loops/mlgym_agent_loop.py psrl/workers/env_worker \
  psrl/trainer/config/psrl/env_worker.yaml
```

`git diff` 不展示未跟踪的新文件正文，需结合 `git status` 中的 `??` 打开新 `grading.py`、`process_runner.py`、`prepare/validate_runtime.py` 及新增测试。env_worker 的部分文件原先就有 GPU 相关修改，因此这个命令是审查入口，不是严格归属本次工作的独立补丁。报告与证据位于工作区 `docs/`，不在 PSRL 子仓库中。

2026-10-14 答复前曾核对两份结果 JSON 的关键源码 SHA-256，均与当时文件匹配。2026-10-15 的 APPS 改造继续修改了部分源文件，新快照和回归见 [APPS 隔离报告](apps-isolation.md)。没有重跑训练或把只读核对当成新增运行结果。针对性检查通过，全库 pre-commit 仍有本次范围外的既有失败，见前文日志。

## APPS 是什么以及为什么暂时禁用

> 本节解释引入隔离之前的禁用原因。2026-10-15 已实现独立候选容器并解除禁用，当前实现与验证范围见 [APPS 隔离报告](apps-isolation.md)。

APPS 全称 **Automated Programming Progress Standard**，可以理解为“自动编程进展评测标准”。它是 2021 年提出的代码生成基准：给模型自然语言题目，让它生成 Python 程序，再用测试用例实际运行程序检查正确性。原始基准包含 10,000 道题，难度覆盖简单编程到复杂算法题。来源：[原始论文](https://arxiv.org/abs/2105.09938)、[作者仓库](https://github.com/hendrycks/apps)。

AIRS 把 APPS 包装成 `CodeGenerationAPPSPassAt5` 任务。下面是其任务要求和评分器原有执行路径，**不是当前已放行的 PSRL 路径**。改造前的运行入口会提前拒绝 APPS。

```text
[1] AIRS 给出 APPS 编程题、训练材料、提交要求
         ↓ agent 构建解题方案，生成每题五份 Python 程序
[2] submission.csv：code1, code2, code3, code4, code5
         ↓ 可信评分器读取 CSV 和私有测试数据
[3] 子进程执行每份候选程序，逐项比较实际输出与期望输出
         ↓ 每题只要有一份程序通过全部测试，就计为解出
[4] Pass@5 = 解出的题数 / 测试题总数
```

图例：图里每个部分对应的源码位置。

| 图中节点 | 本地来源与职责 |
|---|---|
| [1] 输入与任务目标 | CodeGenerationAPPSPassAt5.yaml:3（本地资料，未随仓库发布），描述代码生成任务。数据说明只给测试集的 question 和 starter_code |
| [2] CSV 提交 | CodeGenerationAPPSPassAt5.yaml:31（本地资料，未随仓库发布） 规定五列与 5000 行。evaluate.py:31（本地资料，未随仓库发布） 实际读取 code1 到 code5 |
| [3] 私有测试与执行 | evaluate.py:15（本地资料，未随仓库发布） 读取 test_with_labels。utils.py:15（本地资料，未随仓库发布） 创建子进程，testing_util.py:57（本地资料，未随仓库发布） 加载测试并执行候选代码 |
| [4] 单题判定与平均 | utils.py:51（本地资料，未随仓库发布） 逐个尝试，发现全通过的候选即返回 1。utils.py:58（本地资料，未随仓库发布） 汇总并除以题数 |
| 任务解析入口（原先在此拒绝 APPS） | [grading.py:19](../../../examples/airs_bench/grading.py#L19) 中的 task_config，其 APPS 拒绝分支已在后续隔离改造中移除 |

这些代码如果有理解上的问题，可以随时追问。

APPS 与 AIRS 的关系是“一个原始代码生成基准，被包装为 AIRS 中的一项任务”。它在当前 [训练任务列表:26](../../../examples/airs_bench/prepare/split.json#L26) 中。AIRS agent 的工作是构建解决这类任务的模型或方法，并最终产出整个测试集的预测文件，而不是把 APPS 等同于整个 AIRS。

### Pass@5 的小例子

这里的 `@5` 表示每题有五次候选机会。某一道题必须有**同一份程序**通过它的全部测试，不能把五份程序分别通过的测试拼在一起。

教学题 A：读两个整数，输出它们的和。测试为 `(2,3)→5`、`(0,0)→0`、`(-4,1)→-3`。固定输出 5 的程序只通过第一项，整题不算通过。第三份候选若正确实现加法，就能全部通过，因此 A 记 1。

| 教学题 | 候选 1 | 候选 2 | 候选 3 | 候选 4 | 候选 5 | 此题得分 |
|---|---|---|---|---|---|---|
| A：加法 | 未全通过 | 未全通过 | 全部通过 | 未全通过 | 未全通过 | 1 |
| B：取最大值 | 未全通过 | 未全通过 | 未全通过 | 未全通过 | 未全通过 | 0 |

两题的 Pass@5 为 `(1+0)/2=0.5`，即 50%。分母是题数，不是程序数，也不是测试用例数。这是当前 AIRS 每题恰好提交五份程序时的直接计分定义，不展开其他评测中从更多采样估计 pass@k 的统计公式。

可运行的 [apps-pass5-demo.py](apps-pass5-demo.py) 使用标准库和固定可信函数，2026-10-15 实际运行结果为 A 的第三份候选通过、B 全失败，Pass@5 为 0.5：

```bash
python docs/env-worker-deep-dive/airs-bench-audit/apps-pass5-demo.py
```

这是教学模拟，不加载 APPS 数据、不启动 Docker，也不执行用户提交代码。它只验证计分含义，不验证沙箱安全。

### APPS 评分源码阅读路径

这条路径从评分入口开始。agent 如何生成五份代码属于待评估方法，不是评分器固定实现。

1. [3] `evaluate.py._cli` 先调用 `load_testcases`，从私有测试集读取输入和期望输出。
2. [2] `_cli` 从 CSV 取五列，检查行数，再调用 `evaluate`。
3. [4] `evaluate_all_testcases` 按题并发调度 `_passes_any_submission`，后者依次尝试候选。
4. [3] `solves_testcases` 为候选创建子进程，调用 `run_test`。`run_test` 用 `RuntimeModule.from_string` 编译和执行程序，检查测试结果。
5. [3] `solves_testcases` 等待子进程，超时则 kill。这是执行时限控制，不能据此推导出文件访问隔离。
6. [4] `_passes_any_submission` 返回 0/1，`evaluate_all_testcases` 计算平均，`_cli` 输出 JSON。

函数的文件与行号见图例。候选代码实际执行入口为 pyext.py:7（本地资料，未随仓库发布）。

### 为什么当前普通 CSV 评分隔离还不够

普通分类任务的 CSV 单元格通常是类别或数值，可信评分代码读取这些数据与答案比较。APPS 虽然也交 CSV，但单元格是**需要执行的程序**。若让私有评分容器直接执行这些程序，它同时持有隐藏测试和期望输出，不可信代码就进入了持有答案的环境。

单独创建 Python 子进程通常仍共享该容器可见的文件系统。只读挂载限制修改，不能阻止读取。因此“独立 verifier 容器”和“候选代码无法访问 verifier 的答案”是两项不同要求。本地 `reliability_guard` 的注释也明确指出它不是安全沙箱，见 testing_util.py:453（本地资料，未随仓库发布）。这里没有通过攻击实验宣称某个具体读取技巧必定可用。

需要补齐的设计是：可信判题进程保留期望输出，将受控测试输入送入隔离的代码执行环境，只取回程序输出，在外部完成比较。代码执行环境不挂载期望输出或完整私有测试库，并限制时间、内存、进程和网络。**这段是改造前的设计说明，后续实现与证据见 [APPS 隔离报告](apps-isolation.md)。**

此前禁用的是 PSRL 中这一条 APPS 任务执行路径。APPS 基准的定义和价值不受影响，原始任务也未从 split 中静默删除。

## 验收边界与剩余限制

1. **APPS 已补齐隔离执行，但未完成全量评测。** 已解除入口禁用，真实 Docker 验证和真实题目前三个输入的参考解通过。完整 5000 题、正式镜像 Python 版本及评分逐题等价仍未验收，详见 [新报告](apps-isolation.md)。
2. **完成了受控 Docker 小实验，未完成完整 AIRS 镜像和任务验收。** 远端实测已覆盖合成任务的挂载隔离、独立评分、超时删除与清理。所有 AIRS 任务依赖、真实数据、完整镜像构建及完整 APPS 评测仍未验收。本次未重启服务或清理共享资源。
3. **未完成真实 Ray/SMG/TITO 联调或训练。** 测试没有启动 Ray，也没有测量真实训练吞吐、奖励分布或模型效果。
4. **失联回收假设 coordinator 和容器 owner 可达。** 若 EnvWorker 自身或整台节点丢失，当前会保留待删除占用，需要外部恢复，不能宣称具备持久账本与任意节点故障恢复。
5. **无网络是 AIRS 默认配置。** 有些任务需要额外模型权重或缓存，必须预装进镜像或提供明确的只读公开资源。不能仅凭 Python requirements 已安装就认定全部任务可离线执行。
6. **AIRS 仍通过 env_worker 执行。** 本次为它补齐 reservation、心跳与确认删除后归还的生命周期，没有把整条链路迁移为 Harbor/env_service。这避免混改另一套正在使用的执行接口，也保留后续统一资源入口的架构工作。

## 当前剩余问题复核

2026-10-15 再次查阅当前实现后，结论是：隔离机制与原故障已有定向证据，但 AIRS 尚不能作为完整、可比较、长期运行稳定的 benchmark 直接验收。本次复核未启动集群或训练，只运行下面的纯 Python 故障复现。此前 58 项通过与远端 Docker 成功场景属于已有记录，不是本次重新跑出的结果。

### 已复现的评分错误分类缺陷

2026-10-15 发现的问题是：APPS 函数候选返回 `{"not_an_integer": 1}` 时，比较器将字典键转为整数导致 ValueError，异常传播到 main，被统一归为 invalid_submission。2026-10-16 已修复：[comparison.integer_keys:6](../../../examples/airs_bench/apps/comparison.py#L6) 保留不可转换的键，[judge.grade:61](../../../examples/airs_bench/apps/judge.py#L61) 继续尝试后续候选，[judge.main:120](../../../examples/airs_bench/apps/judge.py#L120) 只将专门的 InvalidSubmission 当作提交错误。

教学例子：一题期望输出为数字 1，第一候选错误地返回上面的字典，第二候选可以返回正确答案。修复前在第一候选后退出，修复后继续尝试第二候选，真实 Docker 已观察到得分 1.0。私有测试损坏、输入超过协议限额与内部错误也不再归为提交格式错误。详细结果见 [评分修复与正式镜像补验](apps-isolation.md#评分修复与正式镜像补验)。

复现脚本：[review-error-classification.py](apps-isolation/review-error-classification.py)。从工作区根目录执行：

```bash
python docs/env-worker-deep-dive/airs-bench-audit/apps-isolation/review-error-classification.py
```

脚本使用真实 judge 与 comparator，仅将候选传输替换成固定 JSON，不执行任意候选代码。2026-10-15 修复前实测输出为 `exception=ValueError`，操作序列为 `begin, case, end`，`later_candidate_attempted=false`。脚本读取当前实现，因此修复后重跑会返回零分并尝试全部五个候选，因为其传输替身始终返回错误字典，不会执行字符串 correct。后续正式镜像实验才验证了第二候选的真实执行。这些源码若有理解上的问题，可以继续追问。

### 全量 APPS 的耗时与评分兼容性

[judge.grade:61](../../../examples/airs_bench/apps/judge.py#L61) 串行处理每题的最多五份程序，[evaluate_isolated:199](../../../psrl/workers/env_worker/isolated_code.py#L199) 为每份程序创建、启动并最终删除容器。5000 题最坏是 25000 次容器生命周期，且单次评分缺省只有 [3600 秒](../../../examples/airs_bench/config.py#L49)。手算可知：若走满全部候选，每份程序包括容器开销的平均预算仅有 `3600 / 25000 = 0.144` 秒。这是配置算术，不是吞吐测量，也不证明每次全量评分必定超时，但显示小样本通过不足以验收全量可用。

比较器是独立实现，候选每个输入会重新执行模块代码，总预算为 11 秒，还有输出与源码大小限制。以上均可能影响与原评分器的等价性。2026-10-16 已补四题各三个输入的新旧对照，并测得候选容器创建删除平均约 0.355 秒。完整数据集评分等价性、全量吞吐和任务声明 Python 3.10 与实际镜像 3.11 的契约仍未验收。

### 私有测试反馈与评测协议

用户已于 2026-10-16 明确要求先搁置这一项。本轮保留现有协议，以下内容仅是待讨论的机制说明。

[PSRLMLGymEnv._evaluate_with_error_handling:246](../../../psrl/environments/mlgym_env.py#L246) 对 validate 和 submit 调用同一个 grader，APPS judge 固定加载 `test_with_labels`。因此隐藏文件仍然向 agent 提供聚合分数反馈。

这不等同于直接暴露答案，但可以形成对测试集的反复试探。例如假设二分类测试有 100 个样本，每次仅修改一个预测，准确率变化 0.01 就可能暴露该修改是否正确。此例是教学推演，不是本次攻击实测。若 AIRS 协议允许这种反馈，这是需要保留并声明的 benchmark 规则。如果目标是严格独立的最终测试，须另外划分验证集和最终测试集、限制反馈次数，不能把它悄悄当作等价修复。

### 生命周期与正式运行的缺口

- **回收控制面故障尚未解决。** [coordinator:206](../../../psrl/workers/env_worker/coordinator.py#L206) 的 reservation 和 watchdog 都在内存里。agent 持有者消失可触发回收，但 coordinator 自身丢失没有持久状态恢复。EnvWorker 死亡时，[release:293](../../../psrl/workers/env_worker/coordinator.py#L293) 无法通过原 actor 确认物理删除，容量会保持占用。需要独立节点清理与状态核对机制。
- **离线任务资产未全部验收。** 无网络默认配置要求提前准备模型权重、依赖与缓存。当前只做过小镜像验证，不能推断所有真实 AIRS 任务都可运行。
- **完整训练路径未验收。** 尚未将真实 Ray 调度、TITO 数据记录和 AIRS episode 串联验证，也未测 56 路请求的实际排队与吞吐。修复准入不会凭空增加物理资源。8 个缺省 APPS episode 的预算是 80 CPU、292 GiB，而非 8 CPU。
- **交付仍在工作区。** 尚无独立 commit/PR，工作区还有其他任务的修改。全库静态检查存在范围外失败，不能写成全库检查已通过。

评分异常分类、有限样本评分对照和正式归档镜像的单 episode 回放已于 2026-10-16 完成。后续仍需按使用目标处理吞吐、完整训练链路和控制面故障恢复。已有隔离证据仍有效，但不能把它扩大为完整 benchmark 的保证。

## 第二轮接入复核

2026-10-16 用户要求继续 review 并修复。审查范围覆盖任务准备、评分到奖励的数据流、APPS 类型适配、持久 shell 与资源回收。评分反馈协议继续搁置，validate 的可用性、反馈内容与次数均未改变。

本轮主要发现：

| 优先级 | 缺陷与影响 | 修复与证据 |
|---|---|---|
| P0 | MLGym 将每次 validate 的分数累积到 AgentInfo.score。最终 CSV 缺失或无效时，没有新增分数，但 runner 仍把上一次 validate 的分数送给奖励函数 | collect_scores_from_info 根据最终失败状态清空奖励用的分数副本，保留原始记录。真实 BaseAgent 回放覆盖先验证成功再删除文件、先验证成功再写坏 CSV，两者都不再沿用旧分数 |
| P0 | 可信 verifier 退出 0 但没有合法指标时，parse_metrics 的 ValueError 被环境当成坏提交，从而把评分器故障变成零分训练样本 | grade_submission 将此类错误转为 RuntimeError，要求丢弃轨迹。NaN、缺少目标键、非 JSON 三种输出均有回归 |
| P1 | APPS 函数输入字典的键被无条件 int 转换，字符串键输入无法执行。NumPy 标量或数组返回值无法通过 JSON 编码 | 不可转换的输入键保持原样，NumPy 标量和数组转成对应的 JSON 值。真实 Python 子进程验证字符串键、数字键、np.int64 和 ndarray |
| P1 | shell stdin 使用同步 write，管道满时阻塞整个 actor。drain 没有字节预算，持续旧输出可能让循环不退出。观察截断发生在完整输出积累之后，不能防止内存膨胀 | stdin 改为非阻塞分块写入，并纳入同一命令截止时间。命令限制 1 MiB，读取与 drain 分别有 8 MiB 硬预算。溢出销毁沙箱、丢弃轨迹 |
| P1 | shell 每个字节块独立 UTF-8 解码，中文字符跨块时会变成替换字符，污染观测 | 增量解码保留未完成的字符字节。定向测试将“中文”拆在三个读取块中，结果保持原文 |
| P1 | watchdog 顺序等待每个删除 RPC，没有 RPC 截止时间。一条卡住的删除可能阻止后续所有到期占用回收 | 最多 8 个回收任务并发，每次删除 RPC 限时 30 秒。未确认删除仍保持资源计费。故障注入中一个 owner 卡住，另一个仍成功回收 |
| P2 | 未知退出状态继续进入奖励计算，删除失败又会跳过 env.close，NaN/Inf 可通过运行时正数检查 | 未知状态按 rollout error 丢弃，嵌套 finally 确保调用 env.close，时间与 CPU 参数要求有限正数 |

奖励问题的具体执行路径如下。这里描述当前实现，失败路径不再把旧验证分数交给奖励函数：

```text
[1] BaseAgent: validate -> 返回分数，保留反馈与历史
             -> 最终 submit: CSV 缺失或格式无效
                         |
[2] collect_scores_from_info: 读取最终状态 -> 奖励用 scores=[]
                         |
[3] MLGymAgentLoop: 合法结束才取训练数据，传递 airs_scores
                         |
[4] compute_score: 没有有效最终分数 -> reward=0

并行资源路径：shell 非阻塞 I/O -> 其他 episode 可运行
              heartbeat -> watchdog 有界并发删除 -> 确认后归还容量
```

图例：图里每个部分对应的源码位置。

| 节点/顺序 | 实际函数与源码 |
|---|---|
| [1] validate/submit 环境适配 | [_evaluate_with_error_handling:246](../../../psrl/environments/mlgym_env.py#L246) |
| [2] 生成奖励用的分数副本 | [collect_scores_from_info:69](../../../examples/airs_bench/runner.py#L69) |
| [3] 丢弃异常轨迹、组装奖励输入 | [MLGymAgentLoop.run:109](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L109) |
| [4] 原有无分数零奖励分支 | [compute_score:74](../../../examples/airs_bench/reward.py#L74) |
| shell 写入、清理旧输出、读取 | [ContainerShell.write:54](../../../psrl/workers/env_worker/worker.py#L54)、[drain:80](../../../psrl/workers/env_worker/worker.py#L80)、[read_until:97](../../../psrl/workers/env_worker/worker.py#L97) |
| 并行心跳与回收 | [_beat:124](../../../psrl/workers/env_worker/coordinator.py#L124)、[reap_expired:302](../../../psrl/workers/env_worker/coordinator.py#L302)、[release:284](../../../psrl/workers/env_worker/coordinator.py#L284) |

上表也是这段路径的源码阅读顺序。心跳与回收是后台并行分支，不是奖励函数调用的下一步。这些代码有理解上的问题，可以继续追问。

**有界小实验。** [真实 Bash 与回收故障脚本](../../../tests/env_worker/test_airs_regressions.py) 中，让 Bash 先 sleep 2 秒，再向它写入约 500 KB 的命令，命令预算为 0.1 秒，同时安排一个 0.02 秒的异步任务。结果写入超时、沙箱销毁，异步任务仍按预期获得执行，整个用例约 0.16 秒。输出洪泛测试把预算临时缩小到 64 KiB，真实 Bash 输出 1 MB 后触发显式拒绝。持续 drain 与卡住删除使用受控替身，未破坏真实服务。

[真实 MLGym 回放脚本](../../../tests/examples/airs_bench/test_upstream_mlgym.py#L77) 使用上游 BaseAgent、固定 ReplayModel 动作和真实 Bash，Docker 与 CSV grader 使用替身，验证完整 agent 状态累积过程。[APPS 执行脚本测试](../../../tests/examples/airs_bench/test_apps_isolation.py) 使用固定可信程序在真实 Python 子进程中运行，不在宿主机执行任意模型生成程序。

复现命令，从 PSRL 仓库根目录执行，先使用项目环境并将实际 MLGym checkout 及隔离依赖目录加入 PYTHONPATH：

```bash
timeout --signal=TERM --kill-after=10 180 python -m pytest \
  tests/examples/airs_bench tests/env_worker/test_airs_regressions.py \
  tests/env_worker/test_worker.py tests/env_worker/test_shell.py \
  tests/env_worker/test_sandbox.py tests/environments/test_mlgym_env.py -q
```

本轮整组 **145 项通过，耗时 70.00 秒**，没有跳过或失败项，见 日志（本地资料，未随仓库发布）、JUnit（本地资料，未随仓库发布）。这是一批完整运行的结果，不与前两轮 48、58 或 103 个用例相加。新增 shell 的硬预算是运行约束：长训练应将详细日志重定向到文件，不能再依赖无限 stdout。8 MiB 硬限制与面向模型的 max_observation_chars 截断是两个不同层次。

本轮涉及的 15 个 Python 文件通过 Ruff、格式检查、prose audit，相关 diff 也通过空白检查，见 [定向静态检查](review-round-two/scoped-checks.json)。全仓只读 pre-commit 仍因其他工作区文件的既有问题失败，见 全仓检查日志（本地资料，未随仓库发布），不能宣称全仓检查通过。[验收摘要与源码 SHA-256](review-round-two/summary.json) 记录了本次测试统计与文件内容指纹。

本轮没有启动真实 Docker、Ray、模型、GPU 或训练，没有改动共享依赖。项目环境激活在 20 秒预算内没有完成，终止自己的激活子进程后，使用前轮已确认的同一环境解释器与显式 PYTHONPATH 完成验证。已有正式镜像小实验是前一轮证据，不当作本轮重新验收。

全量 APPS 串行吞吐、完整评测等价性、Python 版本契约、Ray/TITO 联调和控制组件死亡后的持久恢复仍未解决。新增回收截止时间改善卡住 RPC 的影响，不等于实现了 coordinator 崩溃恢复。修改保留在工作区，尚无独立 commit/PR。
