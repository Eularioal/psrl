# APPS判题：代码隔离与验证边界

> **核心结论**：APPS 需要执行 agent 提交的程序，同时把隐藏答案留给可信判题器。现有实现把二者放在不同容器，候选只收到代码与当前输入，最终分数由判题器计算。真实小样本、取消清理与错误处理已有证据，下一步重点是完整 5000 题的吞吐与评分兼容性。

文档整理于 2026-11-16，只读既有材料。最近源码核对为 **11-13**，基于 `6aa328d56eff746c2ae6d9fd766b2f3f76b4a670` 加当时未提交内容；11 月快检与 10 月真实隔离实验分别保留日期和协议。

## 目录

- [全局路径和源码映射](#全局路径和源码映射)
- [隔离机制与完整例子](#隔离机制与完整例子)
- [跟着图读源码](#跟着图读源码)
- [资源限制与评分语义](#资源限制与评分语义)
- [当前验证范围与未完成项](#当前验证范围与未完成项)
  - [本地 GPU 快检与代表题策略闭环](#本地-gpu-快检与代表题策略闭环)
  - [APPS 全量吞吐与评分兼容性](#apps-全量吞吐与评分兼容性)
  - [私有测试反馈与故障恢复](#私有测试反馈与故障恢复)
- [历史实机与回归证据](#历史实机与回归证据)
  - [隔离机制实验](#隔离机制实验)
  - [正式归档镜像兼容性实验](#正式归档镜像兼容性实验)
  - [CPU 回归结果的口径](#cpu-回归结果的口径)
- [复现入口与资源清理](#复现入口与资源清理)
  - [CPU 教学实验](#cpu-教学实验)
  - [隔离机制 Docker 实验](#隔离机制-docker-实验)
  - [正式归档镜像实验](#正式归档镜像实验)

## 全局路径和源码映射

本文是 [AIRS 接入审查](01-AIRS审计_修复状态与验证范围.md) 的 APPS 专题。APPS 要求 agent 根据编程题写程序。每题交五份候选，只要一份通过该题全部隐藏案例，该题就成功；**Pass@5** 是成功题数除以总题数。

判题器需要读取答案，候选只需读取输入。若两者在同一环境，提交代码可能接触答案。因此实现把“持有答案并打分”和“执行候选程序”分到不同容器，再通过受限消息连接。

```text
[1] MLGym agent 提交五列 CSV
             |
[2] EnvWorker 读取有界快照，创建私有判题容器
             |
[3] 可信 judge 读取测试集和 CSV，保留期望输出
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
    超时或取消也进入清理，所有子容器归属同一个 episode
```

图例：图里每个部分对应的源码位置。

| 节点或箭头 | 职责与实际实现 |
|---|---|
| [1] → [2] 提交评分 | [grade_submission:106](../../../examples/airs_bench/grading.py#L106) 调用 handle.evaluate；[EnvWorker.evaluate:456](../../../psrl/workers/env_worker/worker.py#L456) 复制 CSV、创建独立判题容器 |
| [2] 私有挂载 | [build_verifier_spec:32](../../../examples/airs_bench/grading.py#L32) 配置可信 judge、隐藏数据和候选执行预算；[build_sandbox_spec:60](../../../psrl/environments/mlgym_env.py#L60) 给 agent 只挂公开数据 |
| [3] 读取与判定 | [judge.main:101](../../../examples/airs_bench/apps/judge.py#L101)、[judge.grade:61](../../../examples/airs_bench/apps/judge.py#L61)、[matches:14](../../../examples/airs_bench/apps/comparison.py#L14) |
| [3] → [4] → [5] 创建与转发 | [candidate_spec:116](../../../psrl/workers/env_worker/isolated_code.py#L116)、[evaluate_isolated:135](../../../psrl/workers/env_worker/isolated_code.py#L135)；控制通道不发送期望答案 |
| [5] 单输入执行与输出返回 | [candidate.execute:24](../../../examples/airs_bench/apps/candidate.py#L24)、[CandidateProcess.run_case:80](../../../psrl/workers/env_worker/isolated_code.py#L80) |
| [6] 回收与资源返还 | [broker end/finally:212](../../../psrl/workers/env_worker/isolated_code.py#L212)、[destroy_sandbox:431](../../../psrl/workers/env_worker/worker.py#L431)、[coordinator.release:284](../../../psrl/workers/env_worker/coordinator.py#L284) |

这些代码如果有理解上的问题，可以随时追问。

## 隔离机制与完整例子

可信 judge（判题器）挂载 `test_with_labels`，保存测试输入和期望答案。candidate（候选代码容器）只挂载公开执行器源码，不挂载数据集、评分状态、Docker socket 或代理凭据。broker 是 EnvWorker 中转发受限消息的可信代码；它只把候选代码、函数名和当前输入送给 candidate。

<!-- figure:candidate-and-trusted-verifier -->
这张图沿同一个场景解释：候选程序怎样得到输入，同时不能读到隐藏答案或篡改最终分数？

[![候选执行与可信判题：输入可以交给候选，隐藏答案留在边界内（1/2）](../../_static/img/document-figures/candidate-and-trusted-verifier-screen.png)](../../_static/img/document-figures/candidate-and-trusted-verifier.png)

[![候选执行与可信判题：输入可以交给候选，隐藏答案留在边界内（2/2）](../../_static/img/document-figures/candidate-and-trusted-verifier-screen-02.png)](../../_static/img/document-figures/candidate-and-trusted-verifier.png)

同一次 APPS 检查：可信判题器拥有测试输入与答案，候选程序只负责计算输出。 隔离的是可见数据与写入权限。把候选进程放进容器，本身不足以隐藏挂载的答案。 APPS 隔离目标。已验证用例、兼容性与具体权限仍以正文为准。 [查看矢量图](../../_static/img/document-figures/candidate-and-trusted-verifier.svg)。
<!-- /figure:candidate-and-trusted-verifier -->

两个容器有不同的文件系统视图和进程命名空间。“命名空间”在这里指进程看到的路径和进程集合受到容器边界约束。candidate 访问自己的 `/proc/1/root`，不会因此进入 judge 的根目录。即使两边都用名为 `agent` 的用户，也不会共享对方的数据挂载。

以一道“输入两个整数，输出它们之和”的题为例，假设私有测试是 `2 3 → 5` 和 `-4 1 → -3`，CSV 的五列分别是五份候选程序。以下是教学例子，不是当前快检数据：

1. agent 提交五列 `code1` 至 `code5`。EnvWorker 复制提交物，新建 judge；原 agent 不能再通过改文件影响这份快照。
2. judge 请求运行第一份候选。broker 创建无数据集的 candidate，只发送代码、函数名和输入 `2 3`，不发送答案 `5`。
3. 若候选输出 `{"op":"result","metrics":{"Pass@5":1}}`，broker 只把它作为程序输出交回 judge。它不进入 judge 的结果控制通道，因此不能自行声明满分。
4. 第二份候选正确执行加法。judge 分别发送两个输入，在自己持有的答案旁比较输出；只有两个案例都通过，才把这道题计为通过。
5. 每份候选结束都删除对应容器，评分结束删除 judge。下一份候选不会复用上一份可被修改的执行环境。

judge 的结果通道和 candidate 的输出通道是两个独立的 Docker exec 管道。候选只能报告执行结果，最终分数由 judge 计算。当前比较器也允许正确的标准输入程序、函数返回值和指定的输出规范化形式。

候选容器使用镜像已有的非 root `agent` 用户，根文件系统只读、无网络、丢弃 Linux capabilities（可单独授予的特权能力），并启用 `no-new-privileges`，禁止通过执行特殊权限程序获得更高权限。`/tmp` 使用容量为 64 MiB 的 tmpfs（临时内存文件系统），限制最多 64 个进程，并关闭 BLAS 数值库自动创建大量线程。

这些是 Linux/Docker 容器边界，容器仍共享宿主机内核。已有实测覆盖目标路径不可见和挂载限制生效；没有做容器逃逸或全部内核漏洞审计。镜像本身也必须可信，不能预先烘焙私有答案或凭据。

## 跟着图读源码

沿上表 [1]→[3] 找到提交快照与私有评分容器；沿 [3]→[5] 查看消息里只有候选代码、函数名和当前输入；最后看输出怎样回到 [3] 比较，以及 [6] 正常、超时或取消时怎样清理。

判题器按题、候选与输入串行推进，初始化失败作为运行环境错误处理。候选执行把输入接到标准输入，兼容 `input()`、`open(0)`、`sys.stdin.buffer` 和函数调用。只有通过该题全部案例的候选才计成功。

每个输入 fork 新进程，是为了避免 Python 全局变量、标准输入和库修改污染下一个输入。fork 指复制出子进程；这里父进程已处于无答案容器中，所以 fork 负责执行状态重置，答案隔离由容器挂载和消息边界提供。整体流程包含异步等待、子进程和取消分支，表中步骤表示正常请求路径，不代表所有操作在一个线程顺序执行。

## 资源限制与评分语义

| 约束 | 当前缺省值与来源 |
|---|---|
| agent | 8 CPU、32 GiB；GPU recipe 显式启用 1 张 GPU，普通默认是 0 张。[runtime 配置:43](../../../examples/airs_bench/config.py#L43) |
| judge | 1 CPU、4 GiB；APPS 候选 1 CPU、512 MiB，串行创建。[配置:55](../../../examples/airs_bench/config.py#L55) |
| 单个输入 / 一份候选 | 4 秒 / 11 秒；11 秒从候选执行通道就绪后起计，不包含容器创建启动。[deadline:203](../../../psrl/workers/env_worker/isolated_code.py#L203) |
| 一次评分 | 默认 3600 秒；runner 将 per_action_timeout_s 传给 grader。[配置:52](../../../examples/airs_bench/config.py#L52)、[grader 调用:182](../../../examples/airs_bench/runner.py#L182) |
| 每份源代码 / 每个案例输出 | 最多 256 KiB / 1 MiB。[judge:69](../../../examples/airs_bench/apps/judge.py#L69)、[候选输出限制:11](../../../examples/airs_bench/apps/candidate.py#L11) |
| CSV 快照 | 最多 64 MiB。[read_file:398](../../../psrl/workers/env_worker/worker.py#L398) |

APPS 每个 episode（一次完整任务尝试）的默认准入预算为 agent + judge + 一份并发候选，即 10 CPU、36.5 GiB。普通 CSV 任务为 9 CPU、36 GiB。它们是预算算术，不能当作实测平均使用量。[build_sandbox_spec:84](../../../psrl/environments/mlgym_env.py#L84) 将 verifier 与候选的附加预算计入 episode。

语法错误、程序失败、死循环和输出超限使当前候选失败，判题器继续检查其它候选。judge、镜像或执行通道故障属于基础设施错误，单独报告并丢弃该次轨迹。

[InvalidSubmission:14](../../../examples/airs_bench/apps/judge.py#L14) 只表示 CSV 行数、列数、源码长度等提交契约错误；[load_tests:33](../../../examples/airs_bench/apps/judge.py#L33) 将损坏私有夹具归为基础设施错误。[integer_keys:6](../../../examples/airs_bench/apps/comparison.py#L6) 对非数字字典键保留原值，错误结果比较失败后仍继续其他候选。候选执行适配还支持字符串键输入、NumPy 标量和数组返回值。

跨进程时，[RemoteHandle._call:20](../../../examples/airs_bench/process_runner.py#L20) 和 [run_in_process:90](../../../examples/airs_bench/process_runner.py#L90) 传递受限错误类别；只将 evaluate 的 ValueError 还原为提交错误，其他错误维持 RuntimeError，不动态加载任意异常类。最终指标还经 [grade_submission:106](../../../examples/airs_bench/grading.py#L106) 校验；超时、非预期退出及无效数值不能成为有效奖励。

## 当前验证范围与未完成项

### 本地 GPU 快检与代表题策略闭环

当前 AIRS 验收范围以 [GPU 接入说明](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md) 和 [2026-11-11 结构化证据](../experiments/gpu-benchmarks/airs-gpu-validation-20261111.json) 为准：SICK 已验证真实 Ray/TITO/后训练公共链路，APPS 的题目级范围单独列在下表。

| 范围 | 已有结果 | 结论边界 |
|---|---|---|
| AIRS 非 Kaggle 19 个任务身份 | 19/19 真实 CUDA、独立评分、有限奖励转换及清理通过，总用时 365.68 秒 | 采用真实数据的有界子集，不是官方完整分数；Kaggle Web Traffic 暂缓 |
| 其中的 APPS | 64 条公开训练记录；2 道真实测试题，每题 5 个候选、前 2 个隐藏案例；Pass@5=0，执行完成 | GPU 使用发生在 agent 的 CUDA 向量计算/公开代码检索；候选判题仍是独立 CPU 容器。零分代表起始候选未答对，不代表执行链故障 |
| SICK 代表题策略闭环 | 真实 Qwen3.5-4B 采样、TITO 轨迹记录、一次 PSRL 优化器更新、版本 1 发布后继续 rollout 与 GPU 评分 | 有效生成 token 为 672，奖励约 0.5596；组内同奖励、梯度为 0。证明公共链路执行，不证明策略质量提高 |
| APPS 自身策略闭环 | 尚无对应逐题真实策略后训练记录 | 不能将 SICK 的公共框架验证写成 APPS 5000 题或所有 19 题各自完成后训练 |

SICK 验收使用专门的 acceptance 提示执行公开起始方案，普通研究提示仍允许 agent 改代码，二者评分器相同。达到一次更新和新版策略继续评分的目标后受控停止；版本 1 至少完成一个有效 episode，余下采样被取消。相关运行及清理条件见 GPU 接入说明和原始证据。

### APPS 全量吞吐与评分兼容性

[judge.grade:72](../../../examples/airs_bench/apps/judge.py#L72) 当前按题目、候选串行评分。5000 题最多产生 25000 份候选；[broker:195](../../../psrl/workers/env_worker/isolated_code.py#L195) 为每份候选创建、启动和删除容器。默认 3600 秒评分预算均摊到最坏候选数，仅为 `3600 / 25000 = 0.144` 秒/份。这是配置算术，不是吞吐实测。

历史正式镜像实验的 13 个候选，平均容器创建 0.252 秒、删除 0.103 秒，共约 0.355 秒。如果走满 25000 份候选且平均开销相同，仅生命周期线性外推约为 8884 秒，即 2.47 小时，还未计入 Python 启动和执行。这支持全量吞吐风险判断，不能当作完整题库实际耗时。2 道题的快检通过也不能解除该风险。

输出比较器实现了本地 APPS 的 JSON 返回值、空白规范化、浮点近似和无序 token 行接受规则。历史新旧对照只覆盖四道真实题各前三个输入，共 12 输入；没有完成整套题库的等价性证明。每个输入重执行模块、一份候选 11 秒预算和源码/输出限制也可能影响兼容性。历史正式镜像是 Python 3.11.15，任务文字写 Python 3.10，版本契约仍未统一。

若后续需要完整题库验收，先固定 Python 和评分契约，再按显式 CPU/RAM 预算设计题目级有界并行或更低开销的隔离后端。方案必须保留候选间状态隔离，不能直接复用一个可被提交代码污染的解释器。本次没有扩大并行或延长实验预算。

### 私有测试反馈与故障恢复

[PSRLMLGymEnv._evaluate_with_error_handling:259](../../../psrl/environments/mlgym_env.py#L259) 对 validate 与 submit 使用同一个 grader，并返回聚合分数；APPS [judge.main:117](../../../examples/airs_bench/apps/judge.py#L117) 读取 `test_with_labels`。因此，当前隐藏的是文件与答案，agent 仍能反复获取私有测试聚合分数。用户已明确要求暂缓更改该协议。

例如教学假设中，100 条二分类测试每次只改一条预测，准确率变动 0.01 可能暴露该条是否预测正确。这不是本次攻击实测。若研究目标要求严格独立的最终测试，需要另行约定验证/最终测试划分和反馈次数；现有隔离证据不能证明这种评测独立性。

[coordinator.release:284](../../../psrl/workers/env_worker/coordinator.py#L284) 确认物理删除后才返还预算，[reap_expired:302](../../../psrl/workers/env_worker/coordinator.py#L302) 并行限量重试过期或失败清理。owner 消失可被心跳超时发现；但 [coordinator 状态:208](../../../psrl/workers/env_worker/coordinator.py#L208) 仍只在内存里，coordinator 自身死亡没有持久恢复，worker/节点不可达时无法通过原 actor 确认删除。不能把正常取消清理等同于完整集群故障恢复。

## 历史实机与回归证据

### 隔离机制实验

以下是 **2026-10-15 的历史实验**。当时在历史实验节点确认 Docker 26.1.3，使用已有 Python 镜像构建 datasets 3.6.0 测试镜像，实际 Python 3.13；没有运行 Ray、GPU、语言模型或训练。该条件只说明这批实验，不限制上面的 11 月结果。

| 检查 | 历史结果 |
|---|---|
| 正确代码、第五候选、函数调用 | 三题中前两题成功，第三题只有伪造分数代码，结果 2/3 |
| 私有路径与只读执行器 | 加法程序确认答案路径、`/proc/1/root` 下答案和 Docker socket 不可见，且无法修改公开 runner |
| 死循环、输出洪泛、语法错误、伪造分数 | 均未得到分数，三题合计 0 |
| 取消 | 候选出现后取消，确认 judge 与候选消失，仅保留 agent |
| 真实 APPS 小样本 | 第 0 题原有 565 个输入，只取前 3 个，参考程序得分 1.0 |
| 清理 | 所属标签容器为空，自建镜像标签删除，worker 计数归零 |

[首次失败记录](apps-isolation/first-remote-results.json) 保留了专用 UID 无法访问镜像 0700 家目录的错误；改用已有非 root agent 用户后继续验证。[核心实机结果](apps-isolation/remote-results.json) 的隔离、计分和取消已通过，但随后实验脚本在夹具选择处断言失败；[真实题目补验](apps-isolation/real-reference-results.json) 只执行缺失路径，没有覆盖旧失败或重跑已通过场景。另见 [历史汇总](apps-isolation/summary.json)、[清理检查](apps-isolation/cleanup.json)。

### 正式归档镜像兼容性实验

以下是 **2026-10-16 的历史实验**，使用正式 AIRS 归档镜像和真实容器，模型端为固定动作 ReplayModel（预先写好操作的回放模型），不包含 Ray/TITO/语言模型采样/GPU/训练。原始数据见 [remote-b.json](apps-followup/remote-b.json)。

| 验收项 | 历史结果及范围 |
|---|---|
| 镜像 | `sha256:c7f5889cfb0dc50e605113bfcd1e0f021144471fdb14594866e6afdace011096`；加载归档，没有重新构建当前 Dockerfile |
| Python 与依赖 | Python 3.11.15、datasets 4.0.0、NumPy 2.4.6 |
| 新旧判题 | 数据行 0、1、2303、2365，各前 3 输入；两题 stdin、两题函数调用，四题参考解均通过且新旧一致 |
| 错误候选后继续 | 第一份返回错误字典、第二份正确时得 1.0；五份都错为 0，无异常提前退出 |
| MLGym 单 episode | 真 BaseAgent 与 runner 子进程执行生成 CSV → validate → submit 三步；两个 Pass@5 均 1.0，状态 submitted；数据为真实题目子集 |
| 清理 | worker 剩余计数 0、所属标签容器为空、导入镜像删除 |

四次单题评分各耗时 2.58–2.91 秒，包含新建 judge，不能直接乘以 5000 预测一次全量评分。前文吞吐风险使用单独记录的候选生命周期开销，二者计时边界不同。

### CPU 回归结果的口径

| 证据 | 真实范围 | 不能推出的结论 |
|---|---|---|
| 2026-10-15 的 58 项 | APPS 改造后的定向 CPU 回归，覆盖预算、调用顺序、标准输入/函数适配、协议和进程清理；汇总见 [summary.json](apps-isolation/summary.json) | 不能与旧 103 项相加作为独立总数；不是 GPU 或训练 |
| 2026-10-16 的 48 项 | 评分异常分类、跨进程桥接、隔离、真实上游回放及 agent loop 的定向回归 | 不能与前组简单相加；真实 Docker 另由 remote-b.json 证明 |
| 2026-10-16 的 145 项 | [review-round-two/summary.json](review-round-two/summary.json) 记录 145 通过、0 失败，真实 Bash 和上游 MLGym 回放，无 Docker/Ray/GPU/训练 | 是当时源码快照，不是本次当前代码重测；包含与前组重复的用例 |
| 2026-11-11 的 123 项覆盖 | 见 [AIRS GPU 证据](../experiments/gpu-benchmarks/airs-gpu-validation-20261111.json)：全套旧夹具失败经定向修复验证，另有 acceptance 配置新增用例 | 123 是全套与定向回归覆盖合计，不是一次无失败的 123 项命令，也不能和 145 相加 |

对应当前源码的主要测试入口为 [test_apps_isolation.py](../../../tests/examples/airs_bench/test_apps_isolation.py)、[test_grading_isolation.py](../../../tests/examples/airs_bench/test_grading_isolation.py)、[test_process_runner.py](../../../tests/examples/airs_bench/test_process_runner.py) 和 [test_airs_regressions.py](../../../tests/env_worker/test_airs_regressions.py)。详细问题映射见 [AIRS 接入审查](01-AIRS审计_修复状态与验证范围.md)。

历史 scoped Ruff、格式与 prose 检查通过，历史全仓检查仍有范围外失败，不能写成当前全仓检查通过。11 月接入包含未提交修改；本文不据此宣称已合并或部署。

## 复现入口与资源清理

以下 Docker 命令用于在预检通过后重做对应历史实验，本次未执行。运行前按项目 [集群预检指南](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md) 核对节点、Docker、依赖来源和资源，准备专属 run-id 与输出路径；不能直接复用旧实验结果目录。

### CPU 教学实验

[apps-pass5-demo.py](apps-pass5-demo.py) 使用标准库和固定可信函数模拟 Pass@5：两道题各五个候选，第一题第三份候选全通过，第二题没有候选全通过，故分数为 `1/2=0.5`。它对应图中 judge 的“每份候选检查全部案例，再按题汇总”逻辑；脚本内部编号属于教学模型，不是上图的容器编号。

```bash
python docs/env-worker-deep-dive/airs-bench-audit/apps-pass5-demo.py
```

预期输出第一题通过标记 `[false, false, true, false, false]`、第二题全为 false、`Pass@5: 0.5`；2026-11-13 实际运行得到相同结果。这只是 CPU 教学模拟，不执行提交代码、不创建容器，也不验证文件隔离、GPU 或 PSRL 后训练。

### 隔离机制 Docker 实验

脚本为 [smoke.py](apps-isolation/smoke.py)，镜像定义为 [Dockerfile](apps-isolation/Dockerfile)。先激活项目环境，再恢复实际 `PSRL_PATH` 为仓库路径。`AUDIT_DIR` 指本文所在目录，`APPS_PRIVATE_DATA` 指 prepared APPS 的 test_with_labels，仅由可信驱动读取并复制少量夹具。

```bash
export PYTHONPATH="$PSRL_PATH${PYTHONPATH:+:$PYTHONPATH}"
APPS_SMOKE_RUN_ID="apps-$(date +%Y%m%d-%H%M%S)-$$"
APPS_SMOKE_IMAGE="airs-apps-smoke:$APPS_SMOKE_RUN_ID"
timeout 180 docker build --pull=false -t "$APPS_SMOKE_IMAGE" "$AUDIT_DIR/apps-isolation"
timeout --signal=TERM --kill-after=10 240 python "$AUDIT_DIR/apps-isolation/smoke.py" \
  --repo "$PSRL_PATH" --image "$APPS_SMOKE_IMAGE" \
  --real-data "$APPS_PRIVATE_DATA" --run-id "$APPS_SMOKE_RUN_ID" \
  --output "/tmp/$APPS_SMOKE_RUN_ID.json"
docker image rm "$APPS_SMOKE_IMAGE"
```

脚本用专属标签清理自己的容器。若外层强制 kill，需根据实际 run-id 核实并清理所属资源，不使用全局 prune；镜像名称也应为本次实验独有，避免删掉他人的标签。

### 正式归档镜像实验

[verify.py](apps-followup/verify.py) 默认最多 300 秒场景预算，外层 360 秒；legacy 单容器 25 秒、实际评分 60 秒、单 episode 90 秒。`APPS_ASSETS` 指原版 evaluate/utils/testing_util 所在目录，`MLGYM_REPO` 指上游 checkout，`AIRS_IMAGE` 指已准备的正式镜像。缺少依赖时使用本次专属隔离目录和显式 PYTHONPATH，不能修改共享环境。

```bash
export PYTHONPATH="$PSRL_PATH:$MLGYM_REPO${PYTHONPATH:+:$PYTHONPATH}"
APPS_FOLLOWUP_RUN_ID="airs-followup-$(date +%Y%m%d-%H%M%S)-$$"
timeout --signal=TERM --kill-after=10 360 python "$AUDIT_DIR/apps-followup/verify.py" \
  --repo "$PSRL_PATH" --data "$APPS_PRIVATE_DATA" --assets "$APPS_ASSETS" \
  --mlgym "$MLGYM_REPO" --image "$AIRS_IMAGE" \
  --run-id "$APPS_FOLLOWUP_RUN_ID" \
  --output "/tmp/$APPS_FOLLOWUP_RUN_ID.json"
```

历史实验没有修改共享 MLGym/AIRS 依赖或真实 prepared 数据集；小样本复制到临时目录，实验结束删除。复现实验同样需要核对所属容器和进程实际消失，进程退出或账本归零不能单独替代物理清理证明。
