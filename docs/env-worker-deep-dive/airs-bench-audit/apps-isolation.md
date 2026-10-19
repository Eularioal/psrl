# APPS 判题与候选代码的隔离实现

## TL;DR

- 2026-10-15 已实现独立的 APPS 代码执行容器，并移除原先的 APPS 入口禁用。可信判题容器保存私有测试与答案，候选容器只接收代码、单个输入和函数名。
- APPS 保持五列 CSV 提交，最终 Pass@5 只由可信判题器计算。候选输出即使伪造 JSON 分数，也只是待比较的数据。
- 58 项 CPU 回归通过。远端真实 Docker 覆盖正确代码、第五候选、函数调用、读私有文件、伪造分数、死循环、输出洪泛、语法错误和取消清理。另用真实 APPS 第 0 题的前三个测试输入运行参考解，得分 1.0。
- 2026-10-15 的实验是隔离机制和小样本兼容性验收，当时没有运行完整 5000 题、正式 AIRS 镜像或 Ray/TITO/训练链路，也没有做容器逃逸审计。
- 2026-10-16 补验已使用正式 AIRS 镜像归档：四个真实题目各三个输入的新旧判定一致，真实 MLGym 三步回放完成。评分异常分类已修复，48 项本轮回归通过。全量吞吐仍不满足验收条件，分数反馈协议按用户要求搁置。见 [后续验收](#评分修复与正式镜像补验)。

- 后续接入复核还修复了字符串字典键、NumPy 返回值以及最终无效提交沿用 validate 分数的问题，见 [第二轮复核与 145 项回归](fixes-and-results.md#第二轮接入复核)。

## 目录

- [全局路径和源码映射](#全局路径和源码映射)
- [为什么这次能隔离答案](#为什么这次能隔离答案)
- [跟着图读源码](#跟着图读源码)
- [资源与评分语义](#资源与评分语义)
- [小实验结果与重跑](#小实验结果与重跑)
- [边界与交付状态](#边界与交付状态)
- [评分修复与正式镜像补验](#评分修复与正式镜像补验)

## 全局路径和源码映射

本篇接续 [AIRS 修复报告](fixes-and-results.md) 和其中的 APPS 概念说明。原先的阻断原因是：普通 CSV 评分容器虽然与 agent 隔离，但如果它直接执行提交代码，代码仍会进入持有答案的环境。

```text
[1] MLGym agent 提交五列 CSV
             |
[2] EnvWorker 读取有界快照，创建私有判题容器
             |
[3] 可信 judge 读 Arrow 测试集与 CSV，保留期望输出
             | 仅发 code / fn_name / 单个 input
             v
[4] EnvWorker broker 创建候选容器并转发输入
             |
[5] 无数据集的候选容器：每份程序一个容器，每个输入 fork 新进程
             | 仅返回受限长度的实际输出或执行失败
             v
[3] 可信 judge 比较输出，计算 Pass@5
             |
[6] 每份候选结束就删除候选容器，评分结束删除 judge
    超时或取消也进入清理，两个容器都归属同一个 episode
```

图例：图里每个部分对应的源码位置。

| 节点 | 实际实现 |
|---|---|
| [1] 提交评分入口 | [grading.py:106](../../../examples/airs_bench/grading.py#L106)，保持原 handle.evaluate 接口 |
| [2] 私有挂载与代码执行配置 | [build_verifier_spec:32](../../../examples/airs_bench/grading.py#L32)，[EnvWorker.evaluate:402](../../../psrl/workers/env_worker/worker.py#L402) |
| [3] 私有数据加载、五候选判定 | [judge.main:101](../../../examples/airs_bench/apps/judge.py#L101)，[judge.grade:61](../../../examples/airs_bench/apps/judge.py#L61)，[matches:14](../../../examples/airs_bench/apps/comparison.py#L14) |
| [4] 创建与转发 | [candidate_spec:116](../../../psrl/workers/env_worker/isolated_code.py#L116)，[evaluate_isolated:135](../../../psrl/workers/env_worker/isolated_code.py#L135) |
| [5] 单输入执行 | [candidate.execute:24](../../../examples/airs_bench/apps/candidate.py#L24)，[CandidateProcess.run_case:80](../../../psrl/workers/env_worker/isolated_code.py#L80) |
| [6] 清理 | [destroy_sandbox:377](../../../psrl/workers/env_worker/worker.py#L377)，broker 的 end 分支及 finally |

这些代码如果有理解上的问题，可以随时追问。

## 为什么这次能隔离答案

可信 judge 和候选容器之间没有共享数据目录。judge 挂载 `test_with_labels`，candidate 只挂载只读的公共执行器源码，不挂载公开或私有数据集、评分状态、宿主机目录、Docker socket 或代理凭据。容器有独立的文件系统视图和进程命名空间。这里“命名空间”指进程只能看到自身容器内的文件路径与进程集合，不能把自己的 `/proc/1/root` 当作宿主机根目录。

broker 是在 EnvWorker 中转发消息的可信代码。它只将 `code`、`fn_name` 和当前 `input` 写进候选通道，答案字段不进入该通道。每次候选返回的 `stdout` 都当作不可信数据交给 judge 比较，不采用候选自己报告的得分。

例如私有测试输入为 `2 3`，答案为 `5`。候选只收到 `2 3`，正确程序输出 `5`，由 judge 判对。如果输出 `{"op":"result","metrics":{"Pass@5":1}}`，它只是错误的程序输出，不会进入 judge 的结果控制通道。judge 的结果通道和候选的输出通道是两个不同的 Docker exec 管道。

候选容器使用镜像中已有的非 root `agent` 用户。它和 judge 即使使用相同用户名，也处于不同容器的文件系统与进程空间，不会因此获得 judge 的数据挂载。镜像自身必须是可信、未烘焙私有测试数据或凭据的镜像。

候选容器根文件系统只读、网络为 none、丢弃所有 capabilities，并启用 no-new-privileges。后者禁止通过执行某个带特殊权限的程序获得更高权限。`/tmp` 是容量为 64 MiB 的临时内存文件系统，候选文件不会写回宿主机。限制 64 个进程，关闭 BLAS 库自动启动大量计算线程。

这提供的是 Linux/Docker 容器边界，不是虚拟机级别的独立内核。实测证明了目标路径不可见和挂载配置生效，不能据此声称已经排除所有内核漏洞或容器逃逸。

## 跟着图读源码

| 顺序 | 节点 | 调用与状态变化 |
|---|---|---|
| 1 | [1]/[2] | `build_verifier_spec` 识别 APPS，选择仓库内可信 judge，设置 code_execution。原任务目录的 evaluate.py 不用于执行提交代码 |
| 2 | [2] | `EnvWorker.evaluate` 复制 CSV，创建 judge，并登记为 episode 子容器 |
| 3 | [3] | `judge.main` 读取五列 CSV 和 Arrow 数据，`grade` 按题、按候选、按测试输入推进 |
| 4 | [4] | `begin` 请求触发新候选容器和一个持续的 Docker exec 通道，候选初始化失败作为基础设施错误 |
| 5 | [5] | `case` 传入单个输入。候选在容器内 fork 一个新进程，将输入放到真实 fd 0，再执行代码，支持 input、open(0)、sys.stdin.buffer 和函数调用 |
| 6 | [3] | `matches` 比较实际与期望输出。一个候选通过全部输入才算该题成功，五个候选均失败则该题记 0 |
| 7 | [6] | `end` 关闭执行通道并删除候选容器，下一个候选不会复用它。最终 judge 只输出指标 |
| 8 | [6] | 正常或异常退出都回收 judge。取消处理中清理当前候选，CLI 管道被排空并回收，防止输出过多时 wait 卡住 |

每个输入 fork 新进程，是为了避免 Python 全局变量、标准输入状态和库的修改污染下一个输入。这个 fork 发生在已经没有私有答案的候选容器里。fork 本身不承担答案隔离职责，答案隔离由容器挂载和消息边界完成。

## 资源与评分语义

| 约束 | 当前缺省值 |
|---|---|
| agent | 8 CPU、32 GiB |
| judge | 1 CPU、4 GiB |
| APPS 候选容器 | 1 CPU、512 MiB，串行创建 |
| 单个输入截止时间 | 4 秒 |
| 一份候选所有输入的总预算 | 11 秒，不包含候选容器启动 |
| 每份源代码 | 最多 256 KiB |
| 每个输入输出 | 最多 1 MiB |
| CSV 快照 | 最多 64 MiB |

APPS 每个 episode 的准入预算因此为 10 CPU、36.5 GiB。普通 CSV 任务仍为 9 CPU、36 GiB。它们是预算算术，不是实测平均资源使用量。`build_sandbox_spec` 从 verifier 的附加预算中计入候选容器，launcher 预检查、正式 agent loop 和 scripted episode 都传递该配置。

最终只由 judge 计算“至少一个候选全通过的题数 / 总题数”。语法错误、程序非零退出、死循环和输出超限使当前候选失败。judge 或执行器启动故障作为基础设施错误，不能给 agent 错误的零分惩罚。取消和整体评分截止时间会清理当前候选与 judge。

输出比较独立实现了本地 APPS 评分器中的 JSON 返回值、空白规范化、浮点近似以及无序 token 行的接受规则。已覆盖定向用例和一个真实题目的参考解，但**没有完成整个数据集上新旧执行器的逐题等价对照**。每个输入的新进程、真实标准输入和 Python 版本也可能影响依赖特殊执行器行为的程序。正式报告 benchmark 分数前，应固定镜像 Python 版本，并做相应兼容性验收。

## 小实验结果与重跑

现场重新确认 `192.168.1.10` 的 Docker 26.1.3 正常。实验用节点已有 Python 镜像构建一个安装 datasets 3.6.0 的测试镜像，实际使用 Python 3.13。没有启动 Ray、GPU、模型或训练。

| 实验 | 结果 |
|---|---|
| CPU 定向回归 | 58 项通过，覆盖配置、预算、调用顺序、标准输入/函数适配、协议输出限制和进程回收 |
| 正确代码、第五候选、函数调用 | 三题中前两题成功，第三题只有伪造分数代码，结果为 2/3 |
| 私有文件与只读执行器探针 | 正确加法程序先断言答案路径、`/proc/1/root` 下的答案和 Docker socket 不可见，再尝试修改公共 runner，确认写入失败后才输出正确答案 |
| 死循环、输出洪泛、语法错误、假分数 | 三题均未得到分数，结果为 0 |
| 取消 | 候选容器出现后取消评分，确认 judge 和候选消失，只保留 agent |
| 真实 APPS 小样本 | 第 0 题原有 565 个测试输入，本次只取前三个。使用数据中的参考程序，得分 1.0 |
| 最终清理 | 所有本次标签的容器为空，自建测试镜像标签删除，worker 计数归零 |

原始材料：

- CPU 最终日志（本地资料，未随仓库发布）、JUnit（本地资料，未随仓库发布）。前两批日志也保留，早期两项失败来自测试替身尚未接收新增 verifier_spec 参数。
- [首次容器失败](apps-isolation/first-remote-results.json)：专用 UID 无法访问镜像 0700 的 agent 家目录，执行器未启动，容器已清理。改用镜像已有的非 root agent 用户后继续验证。
- [核心实机结果](apps-isolation/remote-results.json)：隔离、正确/错误计分和取消全部通过，随后实验脚本因前 100 题没有满足“整题最多 12 个测试”的夹具而断言失败。此失败未被覆盖，也不算生产代码缺陷。
- [真实小样本补验](apps-isolation/real-reference-results.json)：只补跑缺失的真实题目路径，明确截取前三个输入，没有为获得整组全绿重跑已通过的容器检查。
- [最终汇总](apps-isolation/summary.json)、[独立清理检查](apps-isolation/cleanup.json)，以及 全库检查日志（本地资料，未随仓库发布）。

实验脚本为 [smoke.py](apps-isolation/smoke.py)，镜像定义为 [Dockerfile](apps-isolation/Dockerfile)。在 Docker 正常的目标节点执行，先激活项目环境，显式设置 `PSRL_PATH`、`AUDIT_DIR` 和 `APPS_PRIVATE_DATA`。后者指 prepared APPS 的 test_with_labels 目录，仅用于可信实验驱动准备小样本。

```bash
export PYTHONPATH="$PSRL_PATH${PYTHONPATH:+:$PYTHONPATH}"
timeout 180 docker build --pull=false -t airs-apps-smoke:local "$AUDIT_DIR/apps-isolation"
timeout --signal=TERM --kill-after=10 240 python "$AUDIT_DIR/apps-isolation/smoke.py" \
  --repo "$PSRL_PATH" --image airs-apps-smoke:local \
  --real-data "$APPS_PRIVATE_DATA" --run-id "apps-$(date +%Y%m%d-%H%M%S)-$$" \
  --output "$AUDIT_DIR/apps-isolation/new-results.json"
docker image rm airs-apps-smoke:local
```

脚本用专属标签清理自己的容器。若外层强制 kill 整个脚本，需根据实际 run-id 再核对并清理本次资源，不能使用全局 prune。脚本注释标出了候选执行、judge 和 cleanup 阶段。

## 边界与交付状态

APPS 已从“入口禁用”变为“使用隔离执行器判题”。这不等于完整 AIRS 可直接签收为生产训练可用：未验证全部 5000 题、完整 AIRS 镜像、任务声明 Python 3.10 与正式镜像的对应关系、长任务吞吐、Ray 故障恢复或 TITO 训练链路。

本次修改文件的 Ruff、格式和 prose 检查通过，144 个本地文档链接及行号范围检查通过。全库 pre-commit 仍因范围外文件的问题失败，不能称为全库静态检查通过。

本次没有修改共享 MLGym 或 AIRS 依赖源码，也没有修改真实 prepared 数据集。真实题目只复制到本次临时目录，实验结束删除。改动仍在工作区，包含新文件，未提交独立 commit/PR。旧报告中的 103 项回归对应旧快照，本次 58 项是 APPS 改造后的定向回归，两个数字不能相加冒充独立测试总量。

## 评分修复与正式镜像补验

本节记录 2026-10-16 的有界修复与实验。此前章节中的“未使用正式镜像”是上一批实验的边界，本节补上归档镜像的小样本验收，不把历史记录改成全量成功。用户明确要求暂不处理模型通过评分获得反馈的问题，本轮没有改动 validate/submit 的数据划分、反馈次数或反馈内容。

**评分异常的修复。** [integer_keys:6](../../../examples/airs_bench/apps/comparison.py#L6) 对无法转换为整数的字典键保留原值，候选返回错误字典时只得到比较失败。评分继续尝试后四个候选。真实 Docker 中，第一候选返回 `{"not_an_integer": 1}`、第二候选正确时，得分为 1.0。五个候选都返回错误字典时得分为 0.0，没有提前终止整个提交。

[judge:14](../../../examples/airs_bench/apps/judge.py#L14) 新增专门的 InvalidSubmission 类型，只有 CSV 行数、列数、代码长度等提交约束违反才进入格式错误分支。[load_tests:33](../../../examples/airs_bench/apps/judge.py#L33) 将损坏的私有测试数据作为基础设施错误，协议大小超限同样报 RuntimeError，其他内部 ValueError 不再被 main 误标为坏提交。

沿真实 runner 路径发现跨进程桥接原本会丢失异常种类，导致坏提交也被当作环境故障。本轮在 [RemoteHandle._call:20](../../../examples/airs_bench/process_runner.py#L20) 与 [run_in_process:90](../../../examples/airs_bench/process_runner.py#L90) 中传递受限的错误类别。只有 evaluate 的 ValueError 还原为 ValueError，其他错误保持 RuntimeError，不动态导入或反序列化任意异常类。

本轮调用路径沿用上文全局图：[1] BaseAgent 工具调用 → runner 子进程桥接 → [2] EnvWorker.evaluate → [3] judge → [4]/[5] 隔离候选 → [3] 比较与计分 → [6] 清理。模型端使用 ReplayModel 的固定动作，Docker、环境适配、runner 进程桥接与评分都是真实实现。未启动 Ray、TITO、LLM、GPU 或训练。源码理解有疑问可以继续追问。

| 验收项 | 本轮实际结果 | 边界 |
|---|---|---|
| CPU 回归 | 48 项通过 | 覆盖 APPS、跨进程错误分类、普通评分隔离、真实上游回放和 agent loop，不与旧测试数量相加 |
| 实际镜像 | 归档镜像 ID 为 `sha256:c7f5889cfb0dc50e605113bfcd1e0f021144471fdb14594866e6afdace011096` | 本轮加载共享的正式 AIRS 归档，没有重新构建当前 Dockerfile |
| Python 和依赖 | Python 3.11.15、datasets 4.0.0、NumPy 2.4.6 | 任务文字写 Python 3.10，版本契约仍未统一 |
| 新旧评分对照 | 真实数据行 0、1、2303、2365，各前三个输入，四题参考解均通过且新旧一致 | 两题 stdin、两题函数调用，只覆盖 12 个输入，不是完整题目或 5000 题等价证明 |
| 错误候选对照 | 原版拒绝错误字典和语法错误。新隔离评分在错误字典后继续接受第二候选，全错为零 | 与单测一起验证异常分类，不代表所有边界类型已枚举 |
| 单 episode 联调 | 真实 BaseAgent 生成 CSV → validate → submit，共三步，两个 Pass@5 都为 1.0，状态 submitted | 用实际 runner 子进程与真实容器，但任务描述精简，数据为真实题目子集，模型是固定动作回放 |
| 清理 | worker 剩余计数 0，本轮容器标签查询为空，导入镜像已删除 | 不清理共享缓存或其他任务资源 |

**吞吐结论。** 四次单题独立评分各耗时 2.58 至 2.91 秒，但这包括每次新建 judge，不能直接乘以 5000 来预测一次完整评分。本轮 13 个候选容器的平均创建时间为 0.252 秒，平均删除时间为 0.103 秒，生命周期开销合计约 0.355 秒。如果完整评分走满 25000 份候选，且保持相同平均开销，仅容器创建删除的线性外推就是约 8884 秒，即 2.47 小时，尚未计入 Python 启动与实际程序执行。

因此，本轮没有把串行实现签收为“完整 APPS 可在默认 3600 秒完成”。这是真实小样本耗时支持的容量风险，不是全量耗时实测。下一步性能改造应保留每份候选的独立环境，按显式 CPU/RAM 预算做题目级有界并行，或选用能证明隔离与状态重置的更低开销执行后端。复用一个可被候选污染的 Python 解释器不能作为直接优化。本轮没有扩大并行或延长评测预算。

实验脚本：[verify.py](apps-followup/verify.py)。脚本默认最多 300 秒场景预算，外层示例 360 秒，legacy 单次容器 25 秒、实际评分 60 秒、单 episode 90 秒。先激活项目环境，固定 PSRL_PATH，准备以下变量：AUDIT_DIR 为本文所在目录，APPS_PRIVATE_DATA 为私有测试 split，APPS_ASSETS 为原版 evaluate/utils/testing_util 所在目录，MLGYM_REPO 为上游 checkout，AIRS_IMAGE 为已经加载的正式镜像。将缺失的 MLGym 依赖安装到隔离目录，并通过 PYTHONPATH 显式传入，不能修改共享依赖。

```bash
export PYTHONPATH="$PSRL_PATH:$MLGYM_REPO${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=10 360 python "$AUDIT_DIR/apps-followup/verify.py" \
  --repo "$PSRL_PATH" --data "$APPS_PRIVATE_DATA" --assets "$APPS_ASSETS" \
  --mlgym "$MLGYM_REPO" --image "$AIRS_IMAGE" \
  --run-id "airs-followup-$(date +%Y%m%d-%H%M%S)-$$" \
  --output "$AUDIT_DIR/apps-followup/new-results.json"
```

证据：48 项 CPU 日志（本地资料，未随仓库发布）、JUnit（本地资料，未随仓库发布）、[正式镜像结构化结果](apps-followup/remote-b.json)、完整实机日志（本地资料，未随仓库发布）、清理日志（本地资料，未随仓库发布）。共享环境激活曾停留在 conda 阶段，本轮终止了自己的激活进程，之后使用已确认的同一环境解释器和显式 PYTHONPATH 完成检查。没有更改共享环境。

本次修改文件的 Ruff、格式与 prose 检查通过。全库只读检查（本地资料，未随仓库发布） 仍在 SciAccel、rollout 等范围外文件失败。完整 Ray/TITO 联调、全部任务离线资产、控制面故障恢复和全量 APPS 吞吐仍未完成，改动未提交独立 commit/PR。
