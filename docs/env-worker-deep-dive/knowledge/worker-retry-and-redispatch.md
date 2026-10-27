# Worker 重试、整组补采与跨 worker 重派发

> **核心结论**：PSRL 原来已经支持同 worker 重试、训练失败整组补采、重新选择 worker、连续失败熔断。固定 baseline `0eeead861ffafed6bf31c962d259fd847775dcf9` 中也能找到这些实现。此前把“跨 worker 重派发尚未完成”笼统理解成“框架没有失败恢复”是不准确的。缺的是原请求在失败 worker 清理完毕后、有身份和次数约束地转到另一 worker 重跑，以及运行中进程死亡的可靠监督。本轮补监督与现有重试链路的正确性，不新增原请求自动迁移。

## 目录

- [1. 先把角色和执行链连起来](#1-先把角色和执行链连起来)
- [2. 原框架支持到哪一步](#2-原框架支持到哪一步)
- [3. 用四条轨迹解释重试和补采](#3-用四条轨迹解释重试和补采)
- [4. 为什么不能直接把原任务扔给另一台机器](#4-为什么不能直接把原任务扔给另一台机器)
- [5. 本轮修复了哪些确定问题](#5-本轮修复了哪些确定问题)
- [6. 跟着图读源码](#6-跟着图读源码)
- [7. 可运行验证与 Results](#7-可运行验证与-results)
- [8. 尚未实现的范围与后续顺序](#8-尚未实现的范围与后续顺序)

## 1. 先把角色和执行链连起来

下面是本轮修复后的实际调用关系。虚线含义用文字标明，不把后台并发画成单线程顺序。

```text
[1] DataProcessor 提供一组请求
            ↓
[2] AgentLoopManager 选择 worker，启动并监督远程调用
            ↓ 远程调用直到整批执行结束才返回
[3] AgentLoopWorker 运行每条 trajectory，并按配置在本 worker 重试
            ↓
[4] science agent → Harbor runner → EnvRouter 准入 → Docker 容器
            │                                            │
            │ 正常结束，清理环境                          │ 心跳消失
            ↓                                            ↓
[5] 成功结果进入训练数据队列                     [8] Node 按租约清理孤儿容器
            │
            └─ 可恢复失败 → [6] notify_group_failed
                              ├─ 训练：丢弃失败组，补一组新请求 → 回到 [1][2]
                              └─ 验证：缩减需要收集的组数

[3] 进程被杀或执行异常未处理
            ↓ 远程调用失败，由 [2] 持续监督到
[7] 锁存失败原因 → 唤醒当前训练/验证等待者 → 后来的等待者也立即失败

[8] 是独立清理分支。收到 [7] 的错误不等于容器已经清空。
```

图例：图里每个部分对应的源码位置。

| 编号 | 角色、函数及职责 | 源码 |
|---|---|---|
| 1 | `_retry_data` 向 DataProcessor 请求新训练样本 | [manager.py](../../../psrl/workers/agent_loop/manager.py#L498) |
| 2 | `_inner_dispatch_data` 选择执行者，`_supervise_worker_batch` 观察远程执行结果 | [派发](../../../psrl/workers/agent_loop/manager.py#L891)、[监督](../../../psrl/workers/agent_loop/manager.py#L931) |
| 3 | `add_agent_program` 执行并等待整批任务，`_run_agent_loop_inner` 做本地重试 | [worker.py](../../../psrl/workers/agent_loop/worker.py#L220)、[重试](../../../psrl/workers/agent_loop/worker.py#L313) |
| 4 | SciAccel `run` 每次进入时创建会话并执行 Harbor 任务 | [agent_loop.py](../../../examples/sciaccel_rl/agent_loop.py#L144)、[会话](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L64) |
| 5 | `postprocess_output` 处理成功轨迹并向队列和 Manager 提交结果 | [worker.py](../../../psrl/workers/agent_loop/worker.py#L461) |
| 6 | `notify_group_failed` 幂等处理组失败，`_refill_failed_group` 受熔断状态约束 | [组失败](../../../psrl/workers/agent_loop/manager.py#L764)、[补采](../../../psrl/workers/agent_loop/manager.py#L630) |
| 7 | `_trip_refill_breaker` 保存错误并唤醒等待者，`wait_for_training_batch` 检查失败状态 | [失败锁存](../../../psrl/workers/agent_loop/manager.py#L581)、[训练等待](../../../psrl/workers/agent_loop/manager.py#L1918) |
| 8 | `cleanup` 删除物理容器、查询确认后回执 | [node.py](../../../psrl/workers/env_service/node.py#L83) |

这些代码如果有不清楚的地方，可以按编号继续追问。

这里的 **AgentLoopWorker** 是运行 agent 整段任务的 Ray 进程。Ray actor 是可被远程调用的常驻进程对象。**Manager** 管派发和训练数据组，**EnvRouter** 管环境资源准入，二者职责不同。另一个组件 **EnvWorker** 面向环境和沙箱操作，不能仅凭名字相似就认为它负责整段 agent 任务的恢复。

RPC 是远程方法调用。调用返回“已放进本地队列”，只证明接收成功，不能证明任务完成。本轮把 AgentLoopWorker 的批次 RPC 保持到执行结束，Manager 仍然异步发出多批请求，不会串行等待上一批跑完才发下一批。差别在于它保留并检查执行回执。

## 2. 原框架支持到哪一步

| 能力 | 本轮之前已经支持什么 | 不能据此推出什么 |
|---|---|---|
| 单条任务本地重试 | `retry_limit` 控制同 worker 调用 agent 的总次数 | 不会因为重试自动更换 worker |
| 失败分类 | `needs_worker_retry` 筛选错误、轨迹超时等需要重跑的情况 | 模型答错、奖励为零不自动触发基础设施重试 |
| 训练组失败恢复 | `notify_group_failed` 取消同组其他子请求、清除部分结果，再补采一组新请求 | 不保证重跑同一题，不恢复中断的容器或 shell |
| 补采后的放置 | 新请求重新进入 Manager 的派发逻辑，可被分配给其他 worker | 没有“必须避开这次失败的 worker”这一同请求转移协议 |
| 补采上限 | 连续失败达到 `refill_failure_threshold`，默认 32，会锁存错误 | 原实现的锁存不等于所有调用路径都已经停止继续发补采 |
| 验证集失败 | 丢失一组后缩减本轮需要收集的组数，全失败可返回空批次 | 验证失败不会自动换一道训练题补上 |
| Env Service 派发前重试 | 暂时没有可用节点时有限重读容量、再做选择 | 不等于任务执行中崩溃后换 worker 重跑 |
| 运行中 worker 进程死亡 | 环境租约和独立清理器可回收容器 | Manager 原来没有通过执行回执可靠获知所有进程死亡 |

默认 [retry_limit](../../../psrl/trainer/config/rollout/psrl_rollout.yaml#L154) 为 1，含第一次执行，所以默认没有第二次本地尝试。设为 2 才表示第一次加一次重试。没有在本轮直接调大默认值，因为确定性的内存不足、损坏任务或错误配置会被重复执行。

[TerminateReason](../../../psrl/workers/agent_loop/loops/utils.py#L138) 是任务结束原因分类。可保留训练数据的预算截断与“没有可用数据的执行错误”不同。`ABORTED` 表示框架已经接管请求取消，worker 不再额外补采，否则会重复补偿同一个训练位置。

固定 baseline 中的证据可以在本地只读核对：

```bash
git show 0eeead861ffafed6bf31c962d259fd847775dcf9:psrl/workers/agent_loop/worker.py \
  | rg -n 'retry_limit|notify_group_failed|def add_agent_program'
git show 0eeead861ffafed6bf31c962d259fd847775dcf9:psrl/workers/agent_loop/manager.py \
  | rg -n 'def _retry_data|def notify_group_failed|refill_failure_threshold|add_agent_program.remote'
```

预期能找到两层恢复函数和默认阈值。这个检查不切换工作目录，也不修改用于性能比较的固定 baseline。

## 3. 用四条轨迹解释重试和补采

下面是教学假设，不是本轮训练测量：训练需要 2 组数据，每组对同一个题目采 4 条 trajectory。trajectory 是 agent 从题目输入到结束的一整段交互。组内相对奖励算法需要同组轨迹一起处理，所以不能随便拿其他题目的单条轨迹补缺口。

题目 P 的组号为 10，四条子请求编号是 40、41、42、43，被放到 worker A。另一组题目 Q 被放到 worker B。

1. **同 worker 重试**：40 遇到临时环境错误。如果 `retry_limit=2`，A 会再调用一次 agent，逻辑请求还是 40。Science 会重新进入任务执行和创建会话，不代表从容器内第 7 条命令接着跑。如果第二次成功，结果归到原来的 P 组。
2. **训练整组补采**：假设 40 最终失败，Manager 取消 41、42、43 并清掉 P 组的部分结果。DataProcessor 新采一组 R，组号和子请求身份重新分配。R 可能被派到 B，也可能再次落到 A，取决于派发策略和当时资源状态。训练仍需要 2 组成功数据，这不是“把 P 转移到了 B”。
3. **验证集处理**：同样是 2 组验证任务，P 组失败后目标可缩减为 1 组，以已有成功的 Q 组结束本轮。它不会把另一道 R 当成 P 的验证结果。
4. **进程直接被杀**：A 没有机会执行 `notify_group_failed`。本轮新增监督后，Manager 接收到 Ray 的 actor 死亡错误，通知等待训练数据的调用方失败。容器清理由独立节点服务继续完成。这一步没有自动把 P 放到 B。

再看本轮修复的一个具体错误：第一次尝试抛异常，变量记录了异常。第二次成功，原先该变量没有清空，于是提交成功结果之后仍可能重新抛出第一次的异常。现在每次尝试开始都重置它，第二次成功就按成功结束，避免“结果已经提交，执行却又被报告失败”。

## 4. 为什么不能直接把原任务扔给另一台机器

真正的同请求跨 worker 重派发至少要满足这些条件：

1. **确认上一轮不会继续运行**。网络暂时不通不等于旧 worker 已死。若同时在 A 和 B 跑，可能重复写文件、重复提交结果。对有 GPU 句柄的任务，还要确认设备授权和旧进程已收回。
2. **区分每一次尝试**。同一个逻辑请求需要不同的 attempt 身份。Manager 只接受当前有效尝试的结果，忽略旧尝试迟到的回包。只有一个 uid 不足以区分这两者。
3. **定义重跑还是接着跑**。从头执行需要清晰的输入、输出和幂等约束。幂等指同一个操作重复执行不会造成额外副作用，例如不能重复写同一份最终结果。接着跑则需要保存 shell、文件系统、agent 状态甚至进程状态，属于更大的状态恢复问题。
4. **保留组语义和次数上限**。一条轨迹重跑后不能破坏整组奖励、模型版本和训练数据的对应关系。必须明确同请求最多尝试几次、哪些故障值得换节点、清理超时如何退出。

现有“新采一组”已经能补训练缺口。在这些条件未闭环前，先把 worker 死亡明确暴露给训练方，比静默无期限等待更可靠，也不会偷偷引入重复执行。自动同请求转移仍是待实现能力。

## 5. 本轮修复了哪些确定问题

| 修复 | 原先可能发生什么 | 当前行为 |
|---|---|---|
| 每次尝试清空异常状态 | 第二次成功后重新抛第一次异常 | 只传播最终失败尝试的错误 |
| 执行回执持续到任务结束 | 接收入队即返回，后续异常只写日志，Manager 可能永久等数据 | Manager 保留批次监督任务，执行异常和 actor 死亡锁存到等待方 |
| 熔断后禁止补采和派发 | 失败计数达到阈值后，当前路径仍可能继续采新题 | 达到阈值不再调用补采，后续派发也检查失败状态 |
| 所有等待入口共享失败状态 | 验证等待者或晚注册的训练等待者可能漏掉错误 | 整批训练、分块训练、验证等待者都收到失败，远程创建 buffer 后再次检查 |
| 关闭时取消并等待执行清理 | 管理进程结束了轮询，worker 后台任务可能仍在跑 | 停止接收新任务，取消并等待所属任务，保留有界的 worker 收尾等待 |

补采熔断有一个教学小例子：把阈值设为 3，且每一组都失败。第一次失败补第 2 组，第二次失败补第 3 组，第三次失败停止，因此只应调用新样本采集 **2 次**，不能再补第 4 组。默认阈值仍是 32，这个较小数字只用于回归验证。

`raise_on_error=True` 下最终未处理异常会由执行监督传到训练等待方，形成明确失败。需要允许普通任务失败后补采的配方，应使用其已有的错误分类和 `raise_on_error=False` 行为。这个配置不会把进程死亡变成可安全重放的普通任务错误。

## 6. 跟着图读源码

| 执行步骤 | 图中编号 | 函数及输入输出 |
|---|---|---|
| 选择并提交一批请求 | 2 | `_inner_dispatch_data` 接收包含 uid、parent_id、version_tag 的批次，更新运行状态，创建监督任务 |
| 运行任务 | 3 | `add_agent_program` 为每条请求建立执行任务，远程方法等待整批结束 |
| 单条任务处理错误 | 3、4 | `_run_agent_loop_inner` 根据结束原因决定是否再尝试，Science 进入 Harbor 创建和清理环境 |
| 发送成功数据或处理组失败 | 5、6 | 成功数据由 `postprocess_output` 提交。可恢复组失败调用 `notify_group_failed`，训练路径调用 `_refill_failed_group` |
| 后台监督异常 | 2、7 | `_supervise_worker_batch` 等待 Ray 回执。进程死亡或未处理异常触发 `_trip_refill_breaker` |
| 唤醒训练调用方 | 7 | `wait_for_training_batch` 在远程 buffer 创建前后检查错误，并在登记等待前再次检查 |
| 清理退出 | 8 | worker 正常退出等待环境清理，异常死亡由 Node 的租约清理路径接手，Manager 关闭时取消自身监督任务 |

逐项源码链接见第 1 节图例。Manager 监督任务与 worker 执行并发运行，训练等待方也是独立远程调用。成功数据依然走原训练数据通道，不通过批次 RPC 返回大张量。

## 7. 可运行验证与 Results

在仓库根目录、项目 Python 环境中执行 CPU 回归：

```bash
python -m pytest tests/agent_loop/test_worker_supervision.py \
  tests/agent_loop/test_manager_refill_breaker.py \
  tests/agent_loop/test_empty_training_group.py \
  tests/env_service/test_manager_dispatch.py tests/trainer/test_chunk_manager.py -q
```

这些测试使用真实 worker/manager 方法和故障替身，覆盖第二次尝试成功、执行期间回执保持未完成、取消后等待清理、训练与验证等待者收到异常，以及熔断上限不再补采。没有启动 GPU 或外部服务，不能仅据这些测试宣称真实 actor 死亡链路已通过。

真实实验脚本：[real_worker_supervision.py](../env-service-verification/scripts/real_worker_supervision.py)。预检通过后，在有 Docker、已缓存 Python 镜像的目标节点运行：

```bash
read -r -p '目标节点 IP: ' WORKER_TEST_IP
read -r -p '已有 Python 镜像: ' WORKER_TEST_IMAGE
run_dir=$(mktemp -d /tmp/worker-supervision.XXXXXX)
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" nohup timeout 300 python \
  docs/env-worker-deep-dive/env-service-verification/scripts/real_worker_supervision.py \
  --node-ip "$WORKER_TEST_IP" --image "$WORKER_TEST_IMAGE" --output "$run_dir" \
  > "$run_dir/driver.log" 2>&1 < /dev/null &
```

预期依次验证正常结束、丢弃回执后杀进程、监督回执后杀进程。每阶段最多一个 0.5 核、128 MiB 的真实 Harbor 容器，容器运行 Python 计算 `6 × 7`。采用实际 Ray actor 和生产 `add_agent_program`、监督、熔断、训练等待、EnvRouter 租约清理方法。任务体是小型 CPU 夹具，PSManager 的 buffer 创建接口由最小 actor 替身提供，没有模型生成、训练数据提交和优化器更新。

丢弃回执对照是故障机制对照，不是固定历史提交的 science 性能 baseline。它有意不检查远程失败，预期训练等待超时。监督组预期收到明确 actor 死亡错误，后来的等待也立即失败。脚本在观测结束后按本次 unit 标签和 Compose project 清理资源。外层超时仍需按 `ownership.json` 核实没有遗留资源。

**Results：** 本轮在用户指定的私有节点后台执行，单次总预算 300 秒，脚本从开始到清理完成约 180.78 秒，退出码 0。阶段结果如下，时间是单次故障观测，不作为吞吐或延迟统计结论。

| 阶段 | 真实观察 | 说明 |
|---|---|---|
| 正常执行 | 容器输出 42，批次监督正常完成，没有失败锁存 | 正常结束不会误触发熔断 |
| 故意不监督回执，再杀 worker | 训练等待到 2 秒观测期限仍未结束 | 复现丢弃执行回执造成的故障不可见，不证明一定永久等待 |
| 监督回执，再杀 worker | 约 0.175 秒收到包含 `ActorDiedError` 的错误 | 实际经过生产 Manager 的监督、熔断与训练等待方法 |
| 后来的等待者 | 同样立即收到已锁存的死亡错误 | 错误不是一次性通知 |
| 所有阶段资源回收 | 每阶段租约和实际容器都回到 0 | 由生产 EnvRouter / Node 清理路径执行，再单独核验物理资源 |

[完整结果](../env-service-verification/results/worker-supervision-20261027/evidence.json)、[实际执行脚本快照](../env-service-verification/results/worker-supervision-20261027/executed-script.py)、[原始日志](../env-service-verification/results/worker-supervision-20261027/driver.log)。日志中的未领取异步异常来自故意不监督的对照组结束时终止 Manager，验收本身通过。复现脚本后续只补了该对照等待任务的显式取消和循环变量绑定，生产代码不变，未为这项脚本收尾修改重复容器实验。

本轮最终 **42 项针对性回归通过**，覆盖重试、补采、空训练组、派发、分块等待与关闭清理。更宽的初始回归有 6 项多模态测试因直接实例化抽象 `AgentLoopBase` 失败，另有 2 项旧测试夹具缺少验证等待者字段，后者已补齐并回归通过。不能把这一轮写成“全仓测试全通过”。本轮代码在主工作区，尚未提交、推送或合并，其他会话提交的 Env Service 工作保持独立记录。

收尾另行核对本次所属容器、网络、隔离 Ray 进程均已释放，见[物理释放记录](../env-service-verification/results/worker-supervision-20261027/physical-release.json)。

本轮涉及的 6 个 Python 文件定向 Ruff 和格式检查通过，339 个本地链接、章节锚点和源码行号范围检查通过。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因历史脚本和其他已有风格问题失败，见[完整检查日志](../env-service-verification/results/worker-supervision-20261027/precommit.log)，未将自动修改复制回工作区。

## 8. 尚未实现的范围与后续顺序

本轮没有实现同请求跨 worker 自动重放、容器状态迁移、失联但未确认死亡进程的强制接管，也没有证明训练速度或模型效果提升。可信进程归属仍由其他工作单独维护。它修复的是现有重试/补采链路和故障可见性。

仍存活但永久阻塞的 worker 不会产生 actor 死亡错误，不能靠本次监督自动识别。任务时间边界由 `trajectory_timeout` 等配置决定，本轮不擅自缩短长任务超时。

后续若推进跨 worker 重派发，建议先限定为“已确认旧执行结束的、可从头重跑的任务”，复用现有整组处理与有界失败策略，再增加 attempt 身份、旧结果隔离、坏 worker 排除和物理清理确认。不要先全局调高 `retry_limit`，也不要把任意 OOM 或低奖励都当成换节点后必定成功。

相关总审计与物理清理机制见[设计审视第 13 节](../experiments/router-paired-latency-and-design-review.md#13-物理资源遥测与清理交接确认)。
