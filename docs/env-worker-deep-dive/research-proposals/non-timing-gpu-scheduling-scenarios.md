# 非计时评分的 GPU Agent 场景与调度研究候选

> **核心结论**：本 session 研究非计时评分任务中的执行与调度。SWE-Serve 正确性子集的真实 GPU 小实验已确认：新进程之间可以复用 Triton 编译缓存，同机换一张 H20 仍能命中；无关 Python 修改不必清空缓存，相关内核修改只重编译受影响部分。这也说明普通编译缓存已经很强，不能把这项已有能力包装成调度创新。显式同机命令池已在后续实现并通过真实换卡验证，见[当前实现](dynamic-command-placement.md#11-已实现的同机命令池与资源账本)；稳定端到端收益和论文新颖性尚未确证。2026-11-10 的真实 score 服务对照通过正确性检查，但准备前移的整组均值为 63.57 秒，普通路径为 63.46 秒，未见端到端收益，不据此开发复杂联合控制器。
>
> 本文的缓存实验基于 2026-11-05 PSRL 主目录 `0326d466` 加当时未提交文档及实验脚本，没有改生产调度器。后续同机池实现已提交于 `5064416`，状态与验证单列在动态放置专题，不将其倒算为本节缓存实验。SWE-Serve 固定为 `84d8fd92739a4a76ceddba56e5a70605921c4ad2`。该历史缓存实验只执行一个任务的 6 个原始算子测试，未启动 SGLang 服务或完整 verifier。2026-11-10 的后续实验已启动真实 SGLang 服务并通过原始完整 verifier，见[服务对照](#results真实-score-服务题与准备前移对照)。两者均未执行策略训练。其他 session 的评分与设备敏感调度不属于本文主线。

阅读补充：[RollArt 与 DynaRL 精读](../literature-reviews/rollart-dynarl-deep-reading.md)解释阶段解耦、动态迁移及其与本地研究的边界。该精读已计入 2026-11-05 的同机 command 池与准备前移实现，不把这些已有能力重新作为创新主张。

2026-11-06 局部研究复核将主候选收窄为“代码版本有效性约束下的准备、保留与 GPU 放置联合调度”。真实 Ray 同机池对照见[动态放置实验](dynamic-command-placement.md#14-results真实-ray-与相同工作量对照)。简单换卡、提前导入、预测返回时间都不单独作为创新主张，立项条件见本文末节。

2026-11-09 增量试验未获得支持复杂策略立项的收益证据，暂不开发生产联合控制器，详见[准备窗口结果](#results准备并发窗口的增量收益验证)。该结论限于当前短算子与共享节点条件。

2026-11-10 早一轮验证因固定镜像拉取超时结束，见[前置检查历史](#results服务题收益实验的前置检查)。用户再次要求继续后，新的有界准备完成镜像拉取，原始服务评分通过，随后运行真实 Router 的准备前移对照，见[后续实测](#results真实-score-服务题与准备前移对照)。前一轮失败证据单独保留。

## 目录

- [任务定位与进展阅读指南](#任务定位与进展阅读指南)
- [场景选择与具体任务](#场景选择与具体任务)
- [主候选：程序变更下的状态与放置联合调度](#主候选程序变更下的状态与放置联合调度)
- [已有工作与创新边界](#已有工作与创新边界)
- [在 PSRL 中怎样落地](#在-psrl-中怎样落地)
- [可运行核查与教学例子](#可运行核查与教学例子)
- [最小研究验证与停止条件](#最小研究验证与停止条件)
- [当前绑定与下一条命令换卡的区别](#当前绑定与下一条命令换卡的区别)
- [Results：跨命令上下文复用实测](#results跨命令上下文复用实测)
- [复现实验与证据边界](#复现实验与证据边界)
- [实验后调整的实现顺序与研究门槛](#实验后调整的实现顺序与研究门槛)
- [研究切入：版本有效的准备状态与资源联合调度](#研究切入版本有效的准备状态与资源联合调度)
- [Results：准备并发窗口的增量收益验证](#results准备并发窗口的增量收益验证)
- [Results：score 服务题的资产核验与运行边界](#resultsscore-服务题的资产核验与运行边界)
- [Results：服务题收益实验的前置检查](#results服务题收益实验的前置检查)
- [Results：真实 score 服务题与准备前移对照](#results真实-score-服务题与准备前移对照)
- [怎样理解本次实验与下一步探索](#怎样理解本次实验与下一步探索)

## 任务定位与进展阅读指南

**本 session 的目标**是为 PSRL 的 GPU 环境执行框架找到有实测收益、且有独立研究价值的调度问题。场景限定为按功能正确性评分的 GPU 软件开发任务。Agent 指不断编辑代码、调用工具并读取结果的智能体。它需要运行 GPU 测试，但测试奖励主要取决于结果正确与否。调度效率另用完成时间和资源持有量衡量。设备或计时评分保真研究由其他 session 探索。

2026-11-10 对主目录 `ab74120` 加本地未提交实验脚本进行服务验证。本页早期算子和资产核验结果保留原始日期，后续服务实测单列。不要把工程小实验理解为完整 science 训练收益或已经确定论文创新。

```text
[G1] 会话容器：Agent 编辑代码，提交下一条测试命令
       ↓
[G2] 可选 CPU 准备：导入库、准备输入，保留准备进程
       ↓ ready，再申请 GPU
[G3] EnvRouter → 命令调度器 → Ledger 账本
       ↓ 在显式同机候选池中选择可准入 GPU
[G4] 授权所选设备 → 原始 GPU 测试 → 确认 GPU 物理归还
       ↓ 回到 G1，容器、文件及主机资源仍在原节点
       └ 会话结束：清理准备进程、容器及租约
```

图例：图里每个部分对应的源码位置。

| 编号 | 作用与实现 |
|---|---|
| G1 | [实验 trajectory](../env-service-verification/scripts/ray_command_placement_probe.py#L178) 为每个会话顺序提交命令，真实实验没有执行 Agent 编辑或模型生成 |
| G2 | [execute_prepared](../../../psrl/workers/env_service/prepared_gpu.py#L24) 运行显式适配器的 CPU 准备，不能自动拆分任意 shell 命令 |
| G3 | [Router 入口](../../../psrl/workers/env_service/router.py#L308)、[acquire](../../../psrl/workers/env_service/command_gpu.py#L82)、[command_candidates](../../../psrl/workers/env_service/ledger.py#L410) 传递请求并筛选同机候选 |
| G4 | [_grant](../../../psrl/workers/env_service/command_gpu.py#L143) 进行节点授权，[_return](../../../psrl/workers/env_service/command_gpu.py#L259) 确认归还，[execute_prepared](../../../psrl/workers/env_service/prepared_gpu.py#L24) 的 finally 清理准备进程 |

跟着图读源码：G1 的不同会话并发，单会话命令顺序执行。选择 prepared 接口时，G2 先启动 CPU 进程，ready 后才调用 exec_gpu。G3 经 Router 申请本次命令的设备预算，G4 授权后唤醒准备进程执行。归还期间仍占用 GPU 账本，节点确认后才释放，随后会话可以发下一条命令。取消和异常也要进入清理流程。CPU 与主存的会话归属不随 GPU unit 改变，两类资源由同一个 Ledger 分别记账。源码有疑问可以按 G1 到 G4 追问。

**一个教学例子，数字不是实测。** 同一节点有 GPU0 和 GPU1，旧会话原来在 GPU0 执行过命令。新命令需要 5 秒 CPU 准备、2 秒 GPU 计算，GPU0 还要忙 12 秒，GPU1 已空闲。忽略授权开销，且准备可与等待重叠，固定 GPU0 最早第 14 秒完成，同机池可先准备，再在 GPU1 执行，第 7 秒完成。容器与文件始终没有换机器，换的是下一条命令使用的设备。若两张卡都忙，同机池不会凭空增加算力。若旧进程还持有 CUDA 状态，也不能直接把它切到另一张卡。

**工程进展与研究进展要分开。** 同机 command 池已本地提交于 `5064416`，能在命令边界重新选卡，CPU 准备接口也已有真实验证。它们是研究基础。更进一步的“代码版本有效性约束下的准备、保留与 GPU 放置联合调度”仍是候选方案，尚无生产状态目录、失效控制器或联合策略。

**为什么要考虑版本有效性。** Agent 只修改输入文本时，固定代码和部分准备状态可能继续使用。Agent 改了已导入的打分模块时，旧进程仍执行内存里的旧代码，更新磁盘文件不会自动更新它。原始权重文件可能仍相同，但已加载的服务不能默认复用。因此要先测真实变更与准备成本，再决定是否值得实现细粒度有效性协议。

### Results 与当前工作停点

| 问题 | 已有证据 | 结论边界 |
|---|---|---|
| 下一条命令能否同机换卡 | 真实容器、Ray Router 与节点 actor 验证通过 | 已验证功能，未实现跨节点或活进程迁移 |
| 换卡与准备前移能否帮助 | 小对照整组均值 37.95 → 35.05 → 32.42 秒 | 每组只有两次，短算子场景，不是稳定训练加速 |
| 再限制准备并发是否更快 | 不额外限制 28.84 秒，上限 2 为 32.86 秒，上限 1 为 51.84 秒 | 本次没有收益，且存在共享节点干扰边界 |
| 服务题是否已跑通 | 原始完整 verifier 的 18 项新行为与 8 项回归测试全部通过，奖励 1.0 | 使用参考修复，不是 Agent 自主解题或训练 |
| 服务题准备前移是否更快 | 普通路径 63.46 秒，准备前移 63.57 秒，GPU 预算持有量减少约 4.6% | 每策略两批，未见端到端收益 |
| 是否已具备论文结果 | 没有 | 尚未证明比缓存、预加载等强基线有稳定增量收益 |

这些分别对应下文各 Results 的独立实验，不将旧结果作为本次重跑。已有准备窗口试验的 324 次测试执行全部通过，表示固定测试被重复执行，不是 324 个不同 benchmark。固定 command、同机池与 prepared 的对照也不是历史 EnvRouter OFF baseline。后者始终使用用户指定的 `0eeead861ffafed6bf31c962d259fd847775dcf9`。

当前具体任务是 SWE-Serve 的多条目 score 正确性服务题。它要求一次给多个条目打分的结果与逐条打分一致，依赖真实 SGLang 模型执行。固定镜像、CUDA 和完整评分已通过，同样四项原始服务测试的准入后准备与准入前准备对照也已完成，未见端到端收益。只有出现真实准备竞争与版本失效，并能优于强基线，才进入联合调度器开发。本轮没有启动 science 训练。

理解 G1 到 G4 可以先运行已有 CPU 教学模拟，不需要模型或 GPU：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/dynamic_command_placement_demo.py
```

该脚本使用另一组显式教学参数，预期固定绑定、低换卡成本、高换卡成本下，全组完成时刻分别为 25、22、25 秒。它说明换卡是否有用取决于等待和准备成本，没有模拟真实设备授权或训练。真实实验的复现脚本与完整边界见后续 Results。

## 场景选择与具体任务

非计时评分指任务按功能、输出或任务质量打分，系统仍可用完成时间与 GPU 秒数评价调度效率。这两个层次应分开。超时仍会造成失败，不能说正确性评分完全不受负载影响，但不必再围绕毫秒测速的排序偏差定义课题。

| 场景 | 任务评分 | 真实资源形态 | 本轮建议 |
|---|---|---|---|
| [SWE-Serve](https://github.com/NVIDIA/swe-serve) 的 GPU correctness 子集 | 新功能与回归测试全部通过 | 修改推理系统代码、启动 GPU 模型服务、运行请求、调试再测试 | **首选**。Harbor 格式接近当前框架，源代码已核查 |
| [ResearchEnvBench](https://no-518.github.io/projects/researchenvbench/) | 依赖与真实 CPU/GPU/多卡执行能力检查 | 安装、编译、探测、改依赖、重新运行；不允许修改受跟踪的仓库源码 | 扩展场景，考察依赖版本变化；公开运行资产与 GPU 配额尚未本地验收 |
| [InfraDev](https://arxiv.org/abs/2610.36589) | 功能与隐藏 probe，但逐题可能含时间等约束 | Megatron-LM、vLLM、verl 的多卡功能开发 | 后续扩展，不能整体标为纯非计时；本轮未逐题筛选与执行 |
| MLGym / AIRS 的训练任务 | 准确率、损失等质量 | 数据处理与训练迭代 | 备选。固定墙钟预算会改变可训练步数；固定步数的系统实验必须另标为变体 |
| KernelBench / InferenceBench / 固定 5 分钟 autoresearch | 时间直接入分或时间预算强影响质量 | 算子测速、推理测速、限时训练 | 不作为本 session 主线 |

### SWE-Serve 不能直接拿全套

本次清单核查共 53 题，其中 41 GPU、12 CPU。GPU 中 38 题声明 `f2p_mode=correctness`，3 题为 speedup，后者排除：

- ss-sgl23856-use-torch-torch-mm-for-deepseek-v3-2-indexer
- ss-sgl25265
- ss-sgl25311-perf-mla-tma-bulk-store-set-mla-kv-buffer-up

这些是元数据筛选结果，不代表 38 题所有断言已经逐行审计。选中的两题检查了评分器、测试清单及部分真实 GPU fixture。score 题已完成离线资产核验，后续在 H20 上通过原始完整评分，详见本页末节。不能把上游测试命名中的 perf 自动当成时间评分，也不能只相信 correctness 标签。

优先具体任务：

1. **ss-sgl17780-embedding-lora**：让 embedding 请求正确选择 LoRA，即给基础模型附加的小型适配参数。任务文件包含带适配器向量、混合 batch 选择与字段传递检查。[固定版本测试](https://github.com/NVIDIA/swe-serve/blob/84d8fd92739a4a76ceddba56e5a70605921c4ad2/tasks/ss-sgl17780-embedding-lora/tests/postmerge_tests/test/registered/lora/test_embedding_lora_support.py#L124)。2026-11-09 进一步核对[执行协议](https://raw.githubusercontent.com/NVIDIA/swe-serve/84d8fd92739a4a76ceddba56e5a70605921c4ad2/tasks/ss-sgl17780-embedding-lora/tests/upstream_e2e.toml)：维护者直接评分节点是 CPU 字段检查，GPU 行为覆盖属于另列的任务测试，不能将所有节点笼统作为统一的 GPU 服务实验。后续优先选择评分链更明确的多条目 score 题。
2. **ss-sgl22544-score-multi-item-delimiters**：一次给多个条目打分，要求结果与逐条调用一致、输入格式正确、并发请求结果一致。固定资产包含 Qwen3-0.6B 与对应分类模型，适合作为初期较小模型候选。[资产与测试协议](https://github.com/NVIDIA/swe-serve/blob/84d8fd92739a4a76ceddba56e5a70605921c4ad2/tasks/ss-sgl22544-score-multi-item-delimiters/tests/upstream_e2e.toml)
3. **ss-sgl28754-spec-v2-session-commit**：修复推测解码与流式会话中的缓存所有权。更能暴露持久状态与并发问题，但初期复杂度高，先作为拓展。[配置](https://github.com/NVIDIA/swe-serve/blob/84d8fd92739a4a76ceddba56e5a70605921c4ad2/tasks/ss-sgl28754-spec-v2-session-commit/task.toml)

F2P 是原来失败、修复后应通过的新行为测试；P2P 是原来通过、修改后仍应通过的回归测试。抽查评分器明确以两类测试全通过产生 1，否则 0，没有把耗时直接变成奖励。[评分源码](https://github.com/NVIDIA/swe-serve/blob/84d8fd92739a4a76ceddba56e5a70605921c4ad2/tasks/ss-sgl17780-embedding-lora/tests/score.py#L79)

上述服务题的上游目标硬件为 H100，本地使用 H20。score 题在固定官方镜像下的 CUDA 与原始评分已验证通过，不推广为所有题目兼容。本文早期缓存实验运行的是 `ss-sgl-cluster-gpu-fused-kernels` 的 GDN 算子正确性子集。该任务未声明固定 H100 型号；其 CUDA 13 官方镜像未运行，本次使用已有 CUDA 12.8 环境。同属 Hopper 不等于所有内核与超时协议等价。部分服务测试预留显存比例较高，模型小也不能直接承诺多路并发。上游明确禁止 benchmark 数据进入训练语料，本轮按评测研究资产使用，不默认将题目投入 PSRL 策略训练。上游本身已有 fixture 分组和服务复用，比较基线必须保留，不能故意每条断言重启服务制造优势。

## 主候选：程序变更下的状态与放置联合调度

**研究问题**：当多个 Agent 不断修改 GPU 应用程序时，调度器应当何时准备哪个版本、保留哪些仍有效的准备结果、何时分配 GPU，才能在 CPU、主存与显存预算内缩短完整任务完成时间？同机文件缓存可以跨卡保留，因此不能预先把换卡视作昂贵重建，先测准备与关键路径。

这里研究的是执行策略，不改变评分器、候选优劣标准或测量精度。暂定题目可表述为“面向迭代 GPU 软件开发 Agent 的变更感知执行调度”，不是宣称已有定论或首创。

```text
[P1] Agent 的 CPU 工作区：编辑 → 生成代码/依赖版本 v
          ↓ 提交公开测试或工具调用，隐藏 verifier 保持隔离
[P2] 状态目录：哪些权重文件、构建缓存、启动状态仍与 v 兼容
          ↓ 形成每个可选节点的安全准备方案和准备成本
[P3] EnvRouter：联合决定排队、保留/驱逐、节点与 GPU 放置
          ↓ 主机预算与本次 GPU 执行预算分别承诺
[P4] 受管测试阶段：准备 → 启服务 → 发请求 → 停服务
          ↓ 物理归还、产物绑定 v、结果返回
[P1] 下一轮编辑；任务结束后清理会话与所属资源
```

图例：图里每个部分对应的源码位置。

| 节点 | 当前实现与缺口 |
|---|---|
| P1 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275) 编排会话，代码版本与执行计划绑定待新增 |
| P2 | 新提案，尚无依赖/代码变更驱动的有效状态目录。不能把镜像缓存说成已实现完整能力 |
| P3 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L82) 未配置池时按租约 unit 排队，显式同机池已可重新选卡，仍无上述状态与准备成本联合策略 |
| P4 | [exec_gpu](../../../psrl/workers/env_service/managed_execution.py#L53)、[start_gpu_job](../../../psrl/workers/env_service/managed_execution.py#L137) 可复用执行接口；[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L233) 负责物理归还 |

跟着图读源码：runner 建立 P1，当前命令经 P3 原有准入进入 P4。提案在申请前加入 P2，使调度器拿到可比较的准备方案。P4 中服务与测试请求有父子依赖，不能把 `server &` 的 shell 返回当作 GPU 已归还；初期把完整“启动—测试—停止”作为一条受管执行，不改变内部测试顺序。并行来自不同会话的准备与执行。P4 结束确认归还后，P1 可以继续编辑，最终由 runner 清理。源码有理解上的问题可以随时追问。

### 为什么有别于只找空闲卡

Agent 第一次测试创建了一个模型服务，第二次修改可能只是公开 API 参数转换，也可能改了模型层、量化加载、CUDA 算子或依赖包。两次都叫“同一任务”，但可复用状态完全不同。

- 权重原始文件没变，文件缓存可能仍可用。
- 改了已经导入的 Python 模块，旧进程不能因工作目录文件已更新就自动变成新版本。
- 改了算子或模型布局，旧编译产物、CUDA graph 或 GPU 运行状态可能失效。
- 依赖包变化，连原来的启动快照都可能不可用。

调度器需要选择“等有有效状态的卡”“使用空闲卡重新准备”“只保留主机缓存并归还 GPU”等方案。适配器必须声明安全复用边界，信息不全时回退新进程；不能靠 LLM 猜测改动无关，不能通过复用旧结果代替实际执行新的候选。

**可能的核心技术问题是状态有效性与放置的相互作用。** 只按任务 ID 做亲和会错误高估缓存收益；按整个仓库 hash 一变就全量清空则可能损失不受修改影响的复用机会。中间方案需要受控依赖边界、可验证的失效机制及资源成本模型。但普通增量构建缓存和精确版本快照已经能做到很多事情，必须拿它们做强基线，不能只击败错误缓存键。

### 一个可手算的例子

全部为教学假设，且要求两处的有效准备状态确实不同。节点 A 的 GPU 要等 20 秒，有效缓存使准备只需 2 秒；节点 B 的 GPU 立即空闲，但缺少这些资产，需要 30 秒准备；实际测试均为 3 秒，评分只看功能通过。

- 没有相关修改：等 A 共 25 秒，立刻去 B 共 33 秒，等待有利。
- 修改使 A 的优势状态失效，A 也需 30 秒准备：A 共 53 秒，B 共 33 秒，此时应换位置。

当前代码版本决定可复用准备工作的价值。两种方案最终都应执行新代码并得到同样功能结果。这个例子只说明决策机制，不能证明论文创新。**不能把同一节点的两张卡默认设为一张热、一张冷**：本轮实测的编译缓存同机跨卡仍然命中。现有 [CPU 教学脚本](../env-service-verification/scripts/versioned_placement_demo.py) 的 gpu0/gpu1 两个位置也只有在状态局部性确实不同的前提下才适用，不是当前同机 command 的测量模型。

### 第二候选只作为备选

ResearchEnvBench 的多阶段环境构建可以研究“准备就绪与资源准入联合调度”：安装与编译阶段主要用 CPU，只有真实 GPU 探测时取得 GPU，多卡验证需完整设备组就绪。这能避免提前持卡等待构建、以及部分资源到位却无法推进。

但 DAG 调度、阶段解耦、gang scheduling（将一个并行作业所需资源同时分配）和 backfilling（在不破坏已有预约时填补空档）均有大量工作。它更适合作为第一候选的外部场景，暂不作为独立创新主张。只有观测到未知环境准备、频繁失效与多卡准入造成现有方法难处理的组合，才进一步研究。

## 已有工作与创新边界

| 近邻 | 已有覆盖 | 本研究必须证明的差异 |
|---|---|---|
| [ServerlessLLM，OSDI 2024](https://arxiv.org/abs/2401.14351) | 多层模型加载、局部性感知放置 | 不能只称“考虑冷启动”；要处理候选程序变化使复用方案失效 |
| [gCROP，SoCC 2024](https://ipads.se.sjtu.edu.cn/_media/publications/yang-socc24.pdf) | GPU 应用 checkpoint/restore | 快照本身不是贡献，需要比较完整版本快照与保守重启 |
| [gKSM，2026](https://arxiv.org/abs/2610.25219) | GPU 内存去重与复用感知放置 | 不把共享权重或省显存写成独有贡献；首期不改驱动 |
| [DSec §2.2、§7](https://arxiv.org/html/2609.22978v1) | 预创建 GPU 容器、共享/独占执行、预热进程和环境分层 | “建立 GPU 工具池”“预热 Python”已经有直接近邻 |
| [SAGA](https://arxiv.org/abs/2605.00528) | 工作流原子抽象、会话亲和与 work stealing | 完整会话、亲和加动态换卡并非新的组合 |
| [InfraDev](https://arxiv.org/abs/2610.36589) | 跨任务复用依赖和产物的执行环境 | 普通构建和依赖复用也不够，需要展示在线状态失效、驻留与调度的额外价值 |

本轮核对了上述摘要或指定章节，未逐项复现或审计所有实现。当前结论是**场景有依据、选题有具体可验证机制，但创新仍有风险**。若简单全局队列加内容寻址缓存已经达到同样效果，应停止复杂化，不能靠新命名包装已有方法。

## 在 PSRL 中怎样落地

保留 EnvRouter、现有容器提供者和物理归还机制，先做可观察性，不先进行通用 GPU 进程迁移。

1. 每次测试绑定不可变代码版本，记录依赖、模型资产和启动参数身份；测试结束后不能把旧版本结果写给新版本。
2. 记录完整测试阶段的排队、构建、加载、GPU 执行、清理时间，区分后台服务和 shell 主进程。
3. 首期仅复用明确安全的只读文件/主机缓存与精确版本构建产物，每次新版本使用新进程。实际启动快照或 GPU 状态复用是后续受控扩展。
4. 给调度器提供多个可行准备方案及其成本，而不是只有一个显存数字。先用简单全局 FIFO、最早预计完成时间及有限亲和作为基线。
5. 有证据后才增加依赖粒度失效与驻留/驱逐联合选择。禁止默认保留有活 CUDA 状态却把 GPU 预算归零；这与现有 direct command 归还契约冲突。

模型推理卡与环境测试卡仍分开优先使用；本 session 不研究因设备不同导致评分改变，也不研究测量误差校正。异构硬件兼容仅是执行前提，不是论文的评分研究对象。

## 可运行核查与教学例子

在仓库根目录，令 SWE_SERVE_CHECKOUT 指向上游固定版本 checkout：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_swe_serve_candidates.py "$SWE_SERVE_CHECKOUT"
python docs/env-worker-deep-dive/env-service-verification/scripts/versioned_placement_demo.py
```

第一条是实际元数据核查，已运行，结果为 53 总任务、38 个 GPU correctness 候选、3 个 GPU speedup 排除项；其余为 CPU 任务。它不是自动证明所有测试非计时的工具。第二条已运行，仅用 Python 算术重现前面的 25/33/53 秒决策，未调用 EnvRouter、GPU 或恢复真实状态。正确版本缓存与原始文件缓存都属于必要强基线。

## 最小研究验证与停止条件

研究验证不跑上游长时间 Agent 任务或完整 verifier，也不启动 RL 训练。本文末尾的机制实验已经完成，性能与真实轨迹层面的验收仍未完成。

下一阶段若做策略对照，先选两题各两条固定测试轨迹，限制候选代码与输入、测试集合及模型请求预算，现场核验依赖和硬件后设置明确外层期限。小规模控制实验可限定 2 GPU、总计 20 分钟，预算不足时保留日志并说明未验证，不删减正式测试后宣称官方通过。

先回答三个量化问题：准备/加载占完整测试多少；相邻代码版本之间哪些成本可安全复用；原卡等待是否经常超过异地重新准备。若三者没有可观空间，不开发复杂策略。

最少比较：原生 fixture 生命周期 + 静态放置；同样 fixture + 全局队列；全局队列 + 内容寻址文件/构建缓存；精确版本快照/有限亲和；拟议联合策略。只对静态绑卡赢，不构成算法贡献。测量固定任务集完成时间、有效测试完成量、总 GPU 秒和 CPU/主存开销，以及新进程执行与复用执行的一致性。所有测试均保留，基础设施故障与真正功能失败分开。

正式结果还需要真实 Agent 多轮轨迹，而非只有预先规定的重复测试。原始 benchmark 用于评测；若未来需要策略训练，先使用独立、允许训练的数据任务。H20 运行只能报告硬件适配结果，不能冒充官方 H100 设置。

**停止条件**：实际只有一次 GPU 测试；GPU 计算支配总时间；代码改动几乎使全部状态失效；收益被现有缓存/快照完全解释；或一致性只能靠每次完全重启保证且无剩余优化空间。此时调整场景，而非回到测速评分来制造研究问题。

## 当前绑定与下一条命令换卡的区别

以下描述未配置 `command_gpu_pool` 的固定绑定路径：command 归还的是本次 GPU 使用权，**不是取消会话与 unit 的绑定**。unit 是账本管理的资源单元，这里讨论每个 unit 对应一张 GPU 的配置。另一张卡空闲，并不会让旧会话的下一条命令自动过去执行。

```text
[C1] 会话租约绑定 unit0、节点和容器设备
        ↓
[C2] 命令 1 → unit0 准入 → 新 GPU 程序执行
        ↓
[C3] 程序结束、设备撤权、确认没有所属 GPU 句柄/计算进程
        ↓ GPU 活动预算归还，主机工作区和原绑定保留
[C4] 命令 2 → 仍然申请 unit0 → 忙则等待

本轮实验（独立探针，不经过 EnvRouter）
[E1] 完整原始算子模块 + 6 个未改测试 + 实验专用缓存目录
        ↓
[E2] 新容器/进程在 GPU0 执行并退出 → 确认物理显存释放
        ↓ 保留文件缓存
[E3] 下一新容器/进程选 GPU0 或 GPU1 → 记录编译命中与正确性
        ↓
[E4] 每组核实归还，结束核实无实验容器与 GPU 进程
```

图例：图里每个部分对应的源码位置。

| 编号 | 源码与含义 |
|---|---|
| C1 | [EnvLease](../../../psrl/workers/env_service/types.py#L240) 保存会话的 unit 和 GPU 集合 |
| C2 / C4 | [acquire](../../../psrl/workers/env_service/command_gpu.py#L82) 取 `lease.unit_id` 排队；[exec_gpu](../../../psrl/workers/env_service/managed_execution.py#L53) 执行受管命令 |
| C3 | [revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L233) 撤权并检查句柄/进程；[_return](../../../psrl/workers/env_service/command_gpu.py#L259) 完成记账归还 |
| E1 | [prepare](../env-service-verification/scripts/run_command_context_probe.py#L48) 校验上游文件 Git 对象、应用官方补丁并产生受控变体 |
| E2 / E3 | [main](../env-service-verification/scripts/run_command_context_probe.py#L82) 启动独立 Docker 进程；[测试与编译观察](../env-service-verification/scripts/command_context_probe.py#L12) 原样调用 6 个测试 |
| E4 | [require_idle](../env-service-verification/scripts/run_command_context_probe.py#L39) 检查 UUID 对应进程与显存，main 的 finally 按实验标签清理 |

跟着图读源码：C1 的租约传入 C2，命令等待原 unit；完成后 C3 异步归还并等待物理确认，下一次仍回 C4。不同会话存在并发，不能把这个单会话流程理解为全局串行。实验 E1 生成固定版本文件，E2/E3 则有意串行，每组输入相同测试集合，输出正确性、缓存事件、计时和物理归还记录，最后 E4 清理。实验没有调用 C2 的 Router，因此不证明迁移协议已经实现。源码有理解上的问题可以随时追问。

这里“上下文”要分成几种东西。CUDA context 是进程与驱动维护的设备执行状态，包括已加载的 GPU 程序、显存对象等；磁盘上的编译缓存则是以后可以重新加载的文件。**复用编译文件不等于保留活 CUDA context。** 本次每组都退出旧进程，新进程仍执行 CUDA 初始化，测到约 0.42–0.99 秒。不能把上一条命令的 GPU 指针直接给新进程使用。

| 状态 | 本次观察或机制边界 |
|---|---|
| Triton 编译产物 | 已实测跨进程、同机同型卡复用，且能按相关内核变更失效 |
| 模型权重文件、数据文件 | 可以挂载同一份只读文件，但本次没有加载模型，未测收益 |
| 操作系统文件页缓存 | 没有全局清空，可能影响后续运行；未单独量化 |
| Python 已导入模块、普通进程内对象 | 新进程没有继承，仍需重新导入；已有 cooperative prepared 接口只解决当前命令的准备阶段，不是跨命令常驻池 |
| GPU 模型权重、KV 缓存、CUDA graph | 本次不保留、不迁移，不能把这类状态收益算进结果 |

Triton 是将 Python 风格内核程序编译为 GPU 可执行代码的工具。JIT 即即时编译：首次遇到一组内核源码和参数时生成对应程序，后续相同请求可读取编译产物。6 个测试对应 6 次编译查询，不等于 6 个完全不同的 Python 函数；输入布局和常量参数也能产生不同的编译版本。

## Results：跨命令上下文复用实测

**验收问题**只有三个：进程退出后是否还有可复用状态；换到同机另一张卡是否仍可用；代码变化是否必须全量失效。实验预算 900 秒、单命令最多 120 秒，不训练模型。11 个新进程顺序执行，总耗时 244.65 秒，66 次正确性测试全部通过。每组退出后两张卡均无计算进程、显存为 0 MiB，结束无实验容器。证据见 [evidence.json](../env-service-verification/results/command-context-20261105/evidence.json)。

任务是 SWE-Serve `ss-sgl-cluster-gpu-fused-kernels` 中 GDN 模块的 4 个 F2P 和 2 个 P2P 测试。QKV 是注意力计算的 Query、Key、Value 三类张量，这里测试把打包张量拆成正确形状与内存布局。测试将 GPU 内核输出与 PyTorch 参考实现比较，要求精确相等，不用速度计算分数。选的是官方 oracle，即已知参考修复后的源码，不是本轮 Agent 自主求解结果。

环境是同节点两张 H20、PyTorch `2.11.0+cu128`、Triton `3.6.0`，按 UUID 选卡并使用现场 minor 映射。SGLang 基础源码为 `6ac30192fa2fb48ede99b29bac63c0448e8f964d`，应用任务官方补丁后的完整 GDN 模块 Git 对象为 `076a1cc3f7be13ee8e79691a77c25218f92f7da6`，与补丁声明一致。实验没有安装或修改共享依赖。

为避免把顺序影响全部归因于选卡，A 轮从 GPU0 开始，B 轮从 GPU1 开始，每轮使用独立的新缓存目录。这里“冷”只指**实验目录内的 Triton/驱动文件缓存为空**，并非机器、Python 库或系统页缓存全冷。没有清空任何共享缓存。

| 组别 | 完整 Docker 命令 / 秒 | pytest 阶段 / 秒 | 编译缓存命中 | 新增缓存文件 | 正确性 |
|---|---:|---:|---:|---:|---:|
| A：GPU0 空缓存 | 15.292 | 3.806 | 0/6 | 51 | 6/6 |
| A：GPU0 复用缓存 | 9.283 | 1.216 | 6/6 | 0 | 6/6 |
| A：GPU1 复用同一缓存 | 10.192 | 1.138 | 6/6 | 0 | 6/6 |
| A：GPU1 另开空缓存 | 14.994 | 5.563 | 0/6 | 51 | 6/6 |
| B：GPU1 空缓存 | 32.735 | 12.523 | 0/6 | 51 | 6/6 |
| B：GPU1 复用缓存 | 18.351 | 1.236 | 6/6 | 0 | 6/6 |
| B：GPU0 复用同一缓存 | 27.609 | 5.589 | 6/6 | 0 | 6/6 |
| B：GPU0 另开空缓存 | 20.128 | 5.108 | 0/6 | 51 | 6/6 |
| 添加无关 Python 函数 | 13.902 | 1.400 | 6/6 | 0 | 6/6 |
| 修改 QKV 内核表达式 | 17.946 | 5.528 | 2/6 | 32 | 6/6 |
| 修改后的内核换卡重跑 | 19.000 | 1.727 | 6/6 | 0 | 6/6 |

完整 Docker 命令计时从 `docker run` 前到客户端返回，包含启动、Python 导入、CUDA 初始化、pytest、退出与容器收尾；不含外层 GPU 归还遥测和缓存文件清单采集。pytest 阶段包含收集、加载被测模块、测试、编译和同步，**不是纯 GPU 内核耗时**。因此两列不能混作同一计时口径。

变更对照也有明确含义：

- 无关改动是在同一个模块末尾添加一个未调用的 CPU 函数，整个文件 hash 已变，但所有已测内核缓存仍有效。
- 相关改动仅把 QKV 内核的索引表达式改为等价的 `tl.arange(0, BLOCK_SIZE) + 0`。4 个相关编译版本失效，2 个未改兄弟内核仍命中，测试输出不变。
- 随后换卡再跑修改版，6 次全部命中，说明新的有效产物也不是只能供生成它的 GPU 使用。

该受控改动只用于验证缓存键和失效粒度，不代表真实 Agent 修改分布，也不是更快的内核优化。缓存事件来自实际 Triton `knobs.compilation.listener` 的 `cache_hit`，文件清单和哈希提供辅助证据。未改计算或测试断言。

**可以确认的结论**：编译文件跨 command 有复用价值，同机同型 GPU 换卡不必丢掉这种价值，Triton 本身已具备细于整个文件的失效能力。

**不能确认的结论**：稳定端到端加速倍数、跨节点效果、完整 SGLang 服务收益、动态调度收益。Python 导入在 5.94–18.64 秒间波动，B 轮换卡热缓存的完整命令甚至比随后同卡空缓存更慢；热缓存的编译查询也有明显时间波动。当前证据不足以定位噪声来源，不能简单归因于网络盘、CPU 竞争或选卡。缓存命中可靠，不代表命中后的加载成本恒定。已达到机制验收目标，因此停止，不靠追加大量重复获得更漂亮的比例。

## 复现实验与证据边界

两个脚本都位于 [验证 scripts](../env-service-verification/scripts)：外层 `run_command_context_probe.py` 只用 Python 标准库，负责有界顺序执行、设备身份检查、缓存目录、日志和按标签清理；内层 `command_context_probe.py` 需要 PyTorch、pytest、支持编译 listener 的 Triton。本次脚本已真实运行，Ruff 检查通过。它们是实验探针，未接入生产 API。

先激活项目环境并在已通过 Docker/CUDA 预检的执行节点运行。以下三个变量由执行者填写：`SWE_SERVE_CHECKOUT` 为固定版本 checkout，`PROBE_ROOT` 为新的可写结果目录，`PYTHON_ENV` 为已有可挂载的 Python 环境目录。`PROBE_IMAGE` 是已有镜像名，须提供 gcc 和与环境兼容的系统库。外层脚本当前要求 Linux Docker、主机驱动库位于 `/usr/lib64`、设备可直接映射，不是任意平台通用 launcher。

```bash
# 在 PSRL 仓库根目录执行，先设置上述四个变量。
test "$(git -C "$SWE_SERVE_CHECKOUT" rev-parse HEAD)" = \
  84d8fd92739a4a76ceddba56e5a70605921c4ad2
mkdir -p "$PROBE_ROOT/assets"
cp -a "$SWE_SERVE_CHECKOUT/tasks/ss-sgl-cluster-gpu-fused-kernels" "$PROBE_ROOT/assets/task"
curl -fL --connect-timeout 10 --max-time 45 \
  https://raw.githubusercontent.com/sgl-project/sglang/6ac30192fa2fb48ede99b29bac63c0448e8f964d/python/sglang/jit_kernel/triton/gdn_fused_proj.py \
  -o "$PROBE_ROOT/assets/gdn_fused_proj.base.py"
cp docs/env-worker-deep-dive/env-service-verification/scripts/*command_context_probe.py "$PROBE_ROOT/"
nohup python "$PROBE_ROOT/run_command_context_probe.py" \
  --root "$PROBE_ROOT" --python-env "$PYTHON_ENV" --image "$PROBE_IMAGE" \
  --gpus 0 1 --budget-s 900 > "$PROBE_ROOT/run.log" 2>&1 < /dev/null &
```

这些命令对应已经执行的流程，变量示例未再次启动 GPU。预期 `summary.json` 最终为 `passed`，11 组各 6 个测试通过，空缓存为 0/6、同卡和换卡复用为 6/6、相关改动为 2/6 命中。耗时不应作为通过断言。启动器每组都检查两张选定卡空闲，出现占用或探测失败便停止；检查不等于平台独占，不能与其他任务抢卡。

本次有意从 SGLang 抽取**完整单模块**，用 namespace package 提供原模块路径，跳过了 SGLang 顶层初始化；两个测试文件未修改。这样能回答算子编译状态能否复用，却不能回答完整服务的导入、模型加载、进程树、请求缓存或 Harbor 生命周期。没有启动 Ray，也没有走 PSRL 调度，因此本轮不要求把 Ray 身份冒记为已验证。

## 实验后调整的实现顺序与研究门槛

要实现用户设想的自动策略，最终确实需要可换卡执行路径（现已完成显式同机池，见动态放置专题）；但**不需要先完成整个迁移系统才知道哪些状态可复用**。本轮先用独立执行器测清状态边界，随后据此实现了同机接口，完整结果见[真实换卡验证](dynamic-command-placement.md#12-results真实换卡验证与复现)。

建议保留 EnvRouter，按以下顺序推进：

1. **先声明安全执行边界。** 只接收无存活 GPU 状态、输入和代码版本明确、产物已落盘的下一条命令。已有后台服务尚未结束的场景不能假装成这个边界。
2. **同机重新放置已实现。** 会话保留 CPU/主存/文件，命令单独领取短期 GPU 执行租约。沿用现有设备授权、显存准入和物理归还，不能只改 `CUDA_VISIBLE_DEVICES` 或直接改旧租约的 unit。
3. **缓存按适当范围复用。** 本轮证据支持同节点兼容 GPU 共用可信来源的编译缓存，身份至少涉及实际内核/编译环境/目标架构。跨用户可写缓存需要隔离，不能无条件让互不信任任务共享可执行文件。
4. **先做简单预计完成时间策略，再判断研究空间。** CPU 准备已完成时，对候选卡比较 `预计等待 W + 剩余准备 P + 执行 E + 归还 R`。如果未来把 CPU 准备与排队并行，应按依赖关键路径计时，不能把可重叠时间机械相加。现有 [execute_prepared](../../../psrl/workers/env_service/prepared_gpu.py#L24) 已能在 GPU 准入前准备当前命令，不应将这一点重新当成未实现创新。

例如，**教学假设**：GPU0 还需等 12 秒，GPU1 只等 2 秒，两者都能读取同机编译缓存；后续准备、执行、归还合计均为 4 秒。原卡完成需 16 秒，换卡需 6 秒。此时换卡收益来自少等 10 秒，而非搬运了 CUDA context。只有换去另一节点、需要额外下载模型或构建时，才加入那里的实际准备成本。两张同型卡的共享缓存不构成等待原卡的充分理由。

对论文方向也应收紧：**本次证据削弱了“按整个源码版本做缓存失效就是创新”的主张**，因为现有 Triton 已保留无关修改下的缓存。值得继续找的是多任务竞争时的节点文件/主存驻留、准备与 GPU 队列协同、版本变化后的安全执行边界，以及它们能否超过“全局队列 + 原生内容缓存 + 准备前移”的强基线。完整模型服务是否出现足够大的、不能被这些已有机制消除的准备差异，仍是下一阶段选题的关键证据缺口。

## 研究切入：版本有效的准备状态与资源联合调度

**建议有条件立项，尚未达到论文结论。** 工作题目可以是“面向迭代 GPU 软件开发 Agent 的版本有效准备调度”。面向的场景是多个 Agent 在不同隔离会话中编辑代码，反复执行正确性测试或启动服务验证。评价调度效率可以用固定工作量的完成时间，任务评分仍然只用原始正确性标准，不参与另一个 session 的测速评分研究。

研究对象是一个完整决策：何时消耗 CPU/主存准备任务、准备状态何时仍有效、何时保留或释放、最后在哪张兼容 GPU 执行。准备不是免费过程，过早准备会占用主存并争用 CPU/I/O，过晚准备会让 GPU 等待。Agent 修改代码或依赖后，一些旧状态可能失效，而无关的库、权重文件、构建产物可能仍可复用。最终目标是完成整组有效任务，不是把 GPU 利用率或缓存命中率各自最大化。

```text
[R1] 会话提交不可变版本 v + 测试输入 + 环境身份
       ↓
[R2] 适配器列出安全准备方案与状态有效条件（研究提案）
       ├─ 只复用文件缓存，新进程执行
       └─ 复用与 v 兼容的 CPU 准备状态，候选代码在明确边界后加载
       ↓
[R3] 研究调度器：准备 / 延后准备 / 保留 / 释放，受 CPU 与主存预算约束
       ↓ 准备完成，仍未持有 GPU，激活前核对版本
[R4] EnvRouter 同机池选择 GPU → 授权 → 执行原始测试
       ↓
[R5] 结果绑定 v → 撤权及物理归还 → 会话继续编辑或清理
```

图例：图里每个部分对应的源码位置。

| 编号 | 已有实现或明确缺口 |
|---|---|
| R1 | 实验使用只读固定资产，真实 Agent 的不可变版本提交协议尚未实现 |
| R2 | 当前实验适配器 [prepare](../env-service-verification/scripts/prepared_correctness_probe.py#L13) 只前移库导入，不保存跨命令活状态，也不是完整依赖有效性协议 |
| R3 | [execute_prepared](../../../psrl/workers/env_service/prepared_gpu.py#L24) 已有先 CPU 准备、后 GPU 申请的单命令接口。跨会话预算控制与保留策略尚未实现 |
| R4 | [acquire](../../../psrl/workers/env_service/command_gpu.py#L82) 已有同机候选池，[exec_gpu](../../../psrl/workers/env_service/managed_execution.py#L53) 执行获准命令 |
| R5 | [_return](../../../psrl/workers/env_service/command_gpu.py#L259) 确认物理归还，跨版本结果提交协议尚未实现 |

跟着图读源码：现有 prepared 接口先启动 R2 的 CPU 进程，准备完成后 R3 才调用 exec_gpu。R4 选择设备并把 UUID 传入准备进程，随后执行测试。R5 确认无 GPU 活动才释放 GPU 预算，最终清理准备进程与临时文件。不同会话的准备和执行可以并发。新研究控制器应接在 R2/R3 之间，不能把 GPU 授权提前到准备开始，也不能在 GPU 句柄尚存时声称已归还。上述源码有疑问可以按编号追问。

**一个教学例子。** 假设两张卡各有 5 秒后可用，A、B、C 每个准备需 4 秒并持有 4 GiB 主存，可用于准备的主存只有 8 GiB，执行均需 1 秒。若所有会话立即准备，第三项就越过主存预算。若等到 GPU 空闲再准备，第一批要到第 9 秒才能运行。一个有界策略可先准备 A、B，并在其完成释放状态后准备 C。若 A 已准备的候选代码从 v1 改成 v2，旧进程已导入该模块，则不能把旧状态当作 v2 使用。系统必须执行绑定的 v1，或取消后重新准备 v2；只有被证明不依赖该修改的部分可继续复用。以上不是实测，也不证明该简单策略优于已有预加载方法。

**真正值得研究的增量在哪里。** 就绪状态应有依赖身份、物理位置、准备成本、主存占用及明确失效条件。调度器需要联合决定状态有效期内的准备时机与设备放置，而不是给会话固定加一个亲和分数。初期只支持适配器声明的边界，不能自动猜测任意 Python 程序的状态等价性。候选源码通过只读快照绑定，设备授权前重新核对版本，结果携带执行版本。准备时长、可复用部分和返回间隔的估计必须来自先前观察，不给在线策略未来轨迹信息。

一个可实现的控制器是滚动短窗口规划：枚举已经提交任务的合法准备方案，利用已完成的阶段计时估计各方案的可执行时刻，联合选择有限数量的 CPU 准备与已就绪 GPU 命令。窗口内保留的主存状态不能超过预算。以预计整组完成时间为主目标，内存驻留成本作为约束或次级代价。预测误差较大时采用保守窗口；空闲设备仍可执行就绪任务，不无限等待未知的未来返回。该控制器目前只是设计，尚未写入生产调度器。

**近邻工作比“动态选卡”更强，必须逐项对比。** 以下为 2026-11-06 对原论文的定向核查，未宣称穷尽检索。

| 原始来源 | 已经覆盖的内容 | 本课题不能重复宣称的内容与待证差异 |
|---|---|---|
| [VenusRL](https://arxiv.org/html/2610.03286v1) | 关键 rollout 组优先、环境资源准入及模板页共享 | 组优先和更密集环境都不是新意。这里须证明可变 GPU 工具程序的准备与激活决策需要额外机制 |
| [TAO](https://arxiv.org/html/2610.37230v1) | 工具返回就绪预测、节点 KV 价值与联合放置 | 不能只把“工具返回”改名为“命令就绪”。其工具 CPU/RAM 位于模型 GPU 准入预算外，本课题须显式处理工具自身的准备资源 |
| [ServerlessLLM，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/fu) | 按检查点位置与启动成本选择推理节点 | 预计等待加加载成本不足以成为创新，需对比这种局部性策略 |
| [InstaInfer，SoCC 2024](https://faculty.washington.edu/wlloyd/courses/tcss562_f2024/papers/Pre-WarmingisNotEnough_AcceleratingServerlessInferenceWithOpportunisticPre-Loading.pdf) | 机会式预加载，考虑到达概率、加载收益及 CPU/GPU 内存成本 | “准备提前、内存受限”本身已有直接先例，是必须实现的强对照 |
| [SEUSS](https://www.cs.bu.edu/~jappavoo/Resources/Papers/seuss.pdf)、[HotSwap](https://arxiv.org/abs/2409.09202) | 初始化快照与已初始化依赖共享 | 提前导入库、缓存 CPU 进程状态本身不是创新，不能只与每次全冷启动比较 |
| [SWE-MiniSandbox](https://arxiv.org/html/2602.11210v2) | 环境缓存、准备阶段 I/O 并发窗口 | 固定准备并发窗口必须作为简单强基线 |
| [LOCAL](https://arxiv.org/html/2608.15241v1) | 模型适配器版本、KV 有效性与调度联合维护 | 版本有效性加调度也不是空白。需要证明代码/依赖变化的失效结构与其不同，并确有额外收益 |
| [多版本多阶段 DAG 调度，MCSoC 2021](https://pure.uva.nl/ws/files/180498376/2021_MCSoC21_multi_phase_scheduling.pdf) | 异构任务实现的 CPU/GPU 阶段协同调度 | 阶段拆分本身已有研究。这里的版本是 Agent 修改程序形成的状态有效性，不是仅选 CPU 或 GPU 实现 |

因此，论文价值应来自三个都能验真的贡献：真实多轮 GPU 软件任务的准备/变更特征，带明确版本有效性边界的可复用执行状态抽象，以及在相同内存与 CPU 预算下优于强基线的联合调度。单独提供接口、缓存或几个手造例子，只能算工程原型。

**首选场景仍是 SWE-Serve 的 correctness 服务题。** 先沿用已有 GDN 子集作为执行回归，再评估 embedding LoRA 与多条目 score 两类服务任务，要求完整保留 fixture 生命周期和测试集合。它们有模型加载、CPU 库初始化与服务启动阶段，更适合检验准备状态的成本，该候选提出时尚未完成 H20 适配，后续 score 题已通过本页末节所列的固定配置原始评分，embedding LoRA 未运行。模型小不意味着显存配置小，上游标准硬件为 H100，不能推广为所有服务任务兼容。[上游任务与硬件说明](https://github.com/NVIDIA/swe-serve)

**立项门槛与停止条件。** 先在至少两类任务中量化准备成本、版本变化及状态失效，再比较以下同工作量方案：固定 command，简单池化加原生文件缓存，已有 prepared 接口，固定窗口准备，按历史到达与成本预加载，以及新联合策略。前两项是机制对照，后四项才逐步检验研究增量。未来轨迹全知的调度只作为上界。报告整组完成时间、GPU 持有秒数、CPU 开销、主存时间积分与正确性一致性，不以降低测试数量、改变评分或增加硬件换优势。

若收益被普通文件缓存或固定窗口 prepared 完全解释，或者跨版本准备状态只能全部丢弃，就停止这条复杂策略路线。当前结论是“值得开展有界场景验证”，不是“已证明能发论文”。本轮真实实验与下面的离线分析足以复核观察，尚不足以跨过这些立项门槛：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/summarize_command_placement.py \
  --runs "$PLACEMENT_RUNS" "$PREPARED_RUNS" --output "$SUMMARY_JSON"
```

三个变量分别指真实 Ray 对照结果目录、prepared 对照结果目录及新摘要文件。该分析只读取已结束的实验，不申请 GPU。原始运行条件及实测结果统一维护在[动态放置实验](dynamic-command-placement.md#14-results真实-ray-与相同工作量对照)，避免把研究设想写成已验证能力。

## Results：准备并发窗口的增量收益验证

2026-11-09 对照 `0326d466` 加主目录未提交同机池代码。验收问题是“在动态池和已有 prepared 接口之上，限制准备并发是否还有完成时间收益”。预先限定两张 H20、三个会话、每会话两条相同正确性命令，三种准备窗口各跑两次，顺序为 0、1、2、2、1、0，外层预算 900 秒。0 表示不额外限制，实际最多三个会话同时准备。1、2 表示最多同时准备一个、两个命令，不限制准备完成后的 GPU 排队或执行数量。不是全会话并发上限，也不是主存硬限制。

```text
[W1] 三个会话各自提交下一条测试命令
       ↓
[W2] 实验控制器申请 CPU 准备窗口
       ↓ 导入 pytest / torch / triton，CUDA 尚未初始化
[W3] 准备完成，立即归还窗口 → 其他会话可以开始准备
       ↓ 当前进程保留 CPU 状态，进入已有 GPU 准入
[W4] 同机池选卡 → 授权 → 原始正确性测试 → 归还 → 下一轮
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与实现 |
|---|---|
| W1、W2 | [实验脚本](../env-service-verification/scripts/ray_command_placement_probe.py) 的 trajectory/execute 为每个会话顺序提交两条命令，Semaphore 控制准备窗口 |
| W3 | [准备适配器](../env-service-verification/scripts/prepared_correctness_probe.py#L13) 写 ready 证据，实验控制器随即释放窗口，已有 [execute_prepared](../../../psrl/workers/env_service/prepared_gpu.py#L24) 继续等待 GPU |
| W4 | [acquire](../../../psrl/workers/env_service/command_gpu.py#L82) 实际分卡，[_return](../../../psrl/workers/env_service/command_gpu.py#L259) 物理确认后释放预算 |

跟着图读源码：W1 的三条 trajectory 并发，单个 trajectory 的两轮串行。W2 仅包围准备阶段，W3 后窗口可复用，不能把它理解为 GPU 执行串行化。W4 结束后原会话继续下一轮，最后按所属容器、租约、节点 actor 顺序清理，再确认物理资源。源码有问题可以按编号继续追问。控制器仅存在于实验脚本，没有加入生产调度器。

每组先给三会话各预热一次，随后同时提交计时工作，不人为占显存、不增加等待睡眠、不改变测试断言。各组均用同一份官方参考修复模块和六个原始 GDN 测试。该同版本机制对照不替代固定历史 OFF baseline。准备状态的最大 RSS 乘以等待时间只作为主存驻留代理指标，不能冒充实际全节点物理内存采样，更不能证明主存压力收益。

复现使用前节真实 Ray 探针的环境变量与现场预检，增加以下参数：

```bash
nohup python docs/env-worker-deep-dive/env-service-verification/scripts/ray_command_placement_probe.py \
  --assets "$ASSETS" --python-env "$PYTHON_ENV" --image "$IMAGE" \
  --node-ip "$NODE_IP" --gpus 0 1 --output "$OUT" \
  --preparation-windows 0 1 2 > preparation-windows.log 2>&1 < /dev/null &
```

离线分析无需 GPU，只读取已完成的证据：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/summarize_preparation_windows.py \
  --runs "$OUT" --output "$SUMMARY_JSON"
```

**该轮服务任务预检。** 多条目 score 的[固定 Dockerfile](https://raw.githubusercontent.com/NVIDIA/swe-serve/84d8fd92739a4a76ceddba56e5a70605921c4ad2/tasks/ss-sgl22544-score-multi-item-delimiters/environment/Dockerfile) 绑定 SGLang `v0.5.10.post1-cu130-runtime` 的镜像摘要与具体源码提交。准备窗口实验当时没有该镜像，也没有安装共享依赖或运行服务题。后续镜像拉取与资产核验见下一节。H20 与镜像 CUDA 兼容性仍未做真实核验，不仅凭驱动版本判断能否运行。embedding LoRA 的维护者直接节点与 GPU 行为覆盖也需分别核对。这些缺口意味着准备窗口结果只适用于短算子测试场景。

**本次结果与决定。** 六组全部完成，包含预热共 324 次原始测试执行通过，观测到的同时准备进程峰值分别为 3、1、2，与窗口控制一致。所有本次所属容器与 GPU 使用已经清理，后台启动进程已退出。摘要见 [evidence.json](../env-service-verification/results/preparation-window-20261109/evidence.json)。

| 同时准备上限 | 两次整组完成时间 / 秒 | 均值 / 秒 | 相对不额外限流 |
|---|---|---|---|
| 不额外限流，实际最多 3 | 28.94、28.75 | 28.84 | 参照 |
| 1 | 45.69、57.98 | 51.84 | 更慢约 79.7% |
| 2 | 29.52、36.20 | 32.86 | 更慢约 13.9% |

这些是本次有限样本的描述值，不能作为稳定性能估计。三组计时工作中的 GPU 预算持有总和均值分别为 25.21、22.14、25.29 秒。窗口 1 减少少量 GPU 持有，却显著拖长完成时间，说明仅压低资源持有量不是正确优化目标。准备完成后等待的主存代理指标均值分别为约 5163、1339、1134 MiB·秒，受限窗口降低了这一代理值，但没有换来更快完成。该指标仅计算每进程最大 RSS 乘就绪等待时间，不包含全部准备和执行主存，也没有扣除共享页，不能解释为物理内存节省率。

**共享节点干扰边界。** 收尾时发现另一会话的 FlashInfer 容器，创建时间与后半程重叠，归属标签不在本实验 token 集合内，因此没有删除。最终目标 GPU 没有计算进程且显存整数读数为 0 MiB，不代表此前完全没有其他 CPU、I/O 或 GPU 干扰。本轮没有完整的外部负载时间线，不能将第二轮波动全部归因于窗口策略，也不为取得更好结果继续扩大重跑。结论限定为“本次没有获得足以支持新增调度策略的增量收益证据”，不是证明所有资源压力和服务任务下都无收益。

**立项判断：暂不开始复杂联合调度器。** 上次动态池与准备前移的工程收益仍成立，本次没有证明固定准备窗口之上的新研究空间，也没有运行版本失效策略。若继续这个方向，下一个必要条件是先让多条目 score 服务题在隔离依赖中通过原始正确性验证，再采集模型加载、主存驻留和实际代码变更轨迹。只有确实出现准备状态容量竞争及版本失效，才比较预算感知控制与简单窗口，并评估论文价值。本轮没有实现生产控制器、没有启动训练，所有修改均为实验脚本、分析和文档。

本次脚本 Ruff、格式与 prose 检查通过。全仓 pre-commit 在隔离副本仍因范围外既有 Ruff/prose 问题失败，未将其写为全仓通过。

## Results：score 服务题的资产核验与运行边界

2026-11-09 继续核验 `ss-sgl22544-score-multi-item-delimiters`，PSRL 基准为 `bc5665f`，加本次未提交的离线核验脚本与文档。没有修改生产调度器，没有提交或推送本次改动。SWE-Serve 使用本文固定提交，SGLang 使用任务指定的 `cfd49e233ceb94de3f702f70aef06e299836aede`，参考修复的 `changes.patch` 已检查并应用于隔离源码。它只用于后续执行参考解，不作为 Agent 解题或训练收益。

**本轮通过的是资产核验，服务正确性与收益仍未验证。** 镜像准备上限 300 秒，独立资产下载上限 480 秒，下载并发最多 3。后台资产下载完成，镜像拉取超时退出 124，随后确认目标镜像仍不存在，没有自动延长预算。没有创建本实验 GPU 容器或启动训练，也没有改动共享 Python 依赖。结束检查时 8 张 GPU 均为 0 MiB 读数，另一会话的容器保留未动。磁盘下载缓存保留供后续复用，不执行共享镜像清理。

| 核验项 | 结果 | 能说明什么 |
|---|---|---|
| 两个模型的固定版本文件 | 14 个 SHA-256 全部一致 | 权重、配置和分词器与题目清单一致 |
| 上游 vendor 校验 | 通过，1 个离线 wheel | 后续容器准备使用的依赖文件未漂移 |
| 上游测试包校验 | 通过，18 F2P、8 P2P、8 组 | 测试源、分组与评分清单一致，不代表测试执行通过 |
| 官方 CUDA 13 镜像 | 300 秒内未拉取完成 | 尚不能判断当前 H20 上的 CUDA 或服务兼容性 |
| 服务启动、26 项测试、调度对照 | 未运行 | 没有新增启动成本或收益结论 |

可复查证据为 [assets.json](../env-service-verification/results/score-service-20261109/assets.json) 和 [运行边界](../env-service-verification/results/score-service-20261109/run-status.json)。这里只将本地文件布局映射到上游校验器支持的路径变量，没有更改原始测试、评分代码或模型清单。第一次路径探测遗漏了测试源码根变量，校验器正确拒绝，补齐两个根变量后通过，不将那次失败解释为上游文件损坏。

```text
[S1] 固定任务目录与本地模型缓存
       ↓
[S2] 调用原始 vendor 与测试包校验器
       ↓
[S3] 流式计算模型文件哈希，静态读取 fixture 与参考修复文件列表
       ↓
[S4] 写证据并退出，不导入候选 SGLang，不申请 GPU
```

图例：图里每个部分对应的源码位置。

| 编号 | 实现与职责 |
|---|---|
| S1 | [main 参数入口](../env-service-verification/scripts/audit_score_service_assets.py#L24) 接收任务目录、缓存目录和输出路径 |
| S2 | [原始校验器调用](../env-service-verification/scripts/audit_score_service_assets.py#L34) 设置两个路径变量，分别保留退出码与输出 |
| S3 | [sha256](../env-service-verification/scripts/audit_score_service_assets.py#L15) 分块读取权重，[静态源码核查](../env-service-verification/scripts/audit_score_service_assets.py#L58) 只解析代码结构，不执行候选模块 |
| S4 | [证据与退出状态](../env-service-verification/scripts/audit_score_service_assets.py#L73) 汇总是否通过，明确没有验证 CUDA、任务奖励或收益 |

跟着图读源码：S1 完成路径解析后，S2 顺序执行两个上游校验器，每个最多 30 秒。S3 逐文件读取并关闭文件，再收集测试准备、清理函数的位置。S4 写出结果并退出，失败返回非零状态。没有 GPU 分配、服务复用或并发调度分支。源码有疑问可以按 S1 到 S4 追问。

复现只需 Python 3.11 及以上和已下载的固定资产，不需要 GPU。`TASK_DIR` 指该任务完整目录，`HF_CACHE` 指包含 `hub/` 的本地缓存目录，`AUDIT_JSON` 指输出文件。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_score_service_assets.py \
  --task "$TASK_DIR" --hf-cache "$HF_CACHE" --output "$AUDIT_JSON"
```

本次实际执行得到 `passed: true`，包计数为 26，所有 14 个模型文件的 `matches` 为 true。哈希核验是 CPU 与文件读取操作，不是服务运行模拟，也不会产生任务 reward。

**场景核查修正。** 原始 `test.sh` 按类启动独立测试进程，但不是每个类都共享一次 Engine。`TestMultiItemScoringClassificationAdapter` 使用每方法准备与清理，其他部分适配类使用类级准备与清理，还有测试在方法内部创建比较引擎。fixture 指测试框架的准备和清理流程。后续必须保留这些边界，也不能把每次原有服务启动都算作待消除的调度浪费。

**具体例子与研究含义，以下是机制推断。** 同一类先测“一次给多个条目打分”，再测“与逐条打分一致”，原有类级引擎已经能复用，调度器不能重复认领该收益。若 Agent 接着修改注意力或打分代码，上一进程已导入的代码不能直接充当新版本。参考补丁实际修改 11 个 SGLang Python 文件，包括注意力后端、请求调度和打分管理。模型文件的字节身份仍可能不变，因此文件缓存与某些不依赖候选代码的 CPU 准备是可评估对象，但 GPU 上的活跃引擎不能默认安全复用。相同文件是否命中页缓存、CPU 权重能否拆出、准备能否前移以及能节约多久，都还需要真实计时。

**下一步验收边界。** 先在另一次明确预算的准备任务中补齐官方镜像，做小型 CUDA 核验，再保留原始分组执行服务测试。通过后才记录导入、模型加载、首请求、测试与清理时间，判断真实等待期间有哪些准备可提前。当前既没有证据支持复杂联合调度器立项，也没有证据否定服务场景。它仍是待验证候选，不是已实现的论文方案。固定历史 OFF baseline 仍为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，本次资产核验没有形成 ON/OFF 对照。

本次核验脚本的 Ruff、格式与 prose 检查通过，所改文档和总索引的本地链接检查通过。全仓 pre-commit 在 `bc5665f` 加本次文件的隔离副本中失败，问题位于范围外的既有 Ruff、格式与 prose 检查，未修改这些文件。已检查当前设计入口，本轮没有改变调度接口、能力或资源生命周期，因此保留其架构说明，仅维护本研究专题及两个导航入口。

## Results：服务题收益实验的前置检查

本节保留早一轮镜像超时的历史证据。后续已运行成功，现状见[服务实测](#results真实-score-服务题与准备前移对照)，不回写本节的失败状态。

2026-11-10 对 `ab74120` 加本地实验脚本与文档启动后台任务。目标是先执行原始正确性 verifier，即任务评分程序，再测 CPU 准备能否与 GPU 等待重叠。参考源码、修复、模型与测试沿用上一节固定版本，保留原始类级和方法级服务生命周期。没有降低测试数量或通过标准，没有形成历史 OFF baseline 对照。

本轮先现场确认目标节点 Docker 26.1.3 可响应、8 张 H20 显存均为 0 MiB、没有运行容器，并核对 GPU UUID 与设备 minor number。独立后台镜像准备最多 600 秒，后续脚本仅在镜像拉取成功后启动。预先限定 CPU 导入 90 秒、小型 CUDA 内核 90 秒、原始 verifier 900 秒，条件满足后的性能对照另限 600 秒。镜像失败时后续预算不会使用。

**实际结果：镜像拉取 600 秒后退出 124，目标镜像未就绪。** 控制脚本检查到失败后退出 1。CPU 导入、CUDA、26 项正确性测试及性能对照均未执行。因此无法判断该服务题是否能在此配置运行，也没有新调度收益、无收益或兼容性结论。下载完成的若干层不是完整镜像可用的证明。本轮没有自动增加拉取时间、改用其他镜像或安装共享依赖。

[本轮状态证据](../env-service-verification/results/score-service-20261110/run-status.json) 保留预算、退出状态及未执行范围。收尾核对两个后台进程均已退出，没有本次实验容器，8 张 GPU 的显存整数读数均为 0 MiB。缓存未做全局清理，生产调度器没有修改。当前设计入口的架构与能力边界不受本次前置探针影响。

新增的 [score_service_probe.py](../env-service-verification/scripts/score_service_probe.py) 接收本地固定资产，在独立容器中顺序运行导入、CUDA 和原始 test.sh。它映射实际 GPU 设备，设置 CPU、主存及共享内存上限，为所属容器加唯一实验标签，保存各阶段日志并定向清理。该脚本的真实容器路径本轮尚未进入，不能将静态检查通过写成运行验收。它是服务任务前置探针，不经过 Router，后续要证明 Router 调度收益仍须走生产准入接口。

以下命令只适用于固定镜像已在执行节点就绪、且节点预检通过时。`SCORE_ASSETS` 指含 `code/`、`hf-cache/` 和原始任务 `upstream/tasks/` 的资产目录，`SCORE_OUT` 指尚不存在的绝对输出目录，`SCORE_IMAGE` 指上一节固定摘要的官方镜像。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/score_service_probe.py \
  --assets "$SCORE_ASSETS" --output "$SCORE_OUT" --image "$SCORE_IMAGE" \
  --gpu 0 --task-budget 900
```

真实调度对照的计时从相同工作量提交时开始，镜像下载和资产准备单独记录。应保持相同测试集合、服务复用和重建边界，并检查最终评分一致。脚本尚未测得这些成本，不能根据 0.6B 模型名称预估加载足够重、准备一定能前移或策略必有收益。

脚本的 Ruff、格式和 prose 检查及启动 shell 语法检查通过，本次编辑文档的本地链接检查通过。全仓 pre-commit 在隔离副本中仍有范围外的既有 Ruff、格式及 prose 失败，没有改动共享工作区中的这些文件。新脚本与文档尚未提交或推送，脚本的容器路径尚未运行验收。


## Results：真实 score 服务题与准备前移对照

**核心结论。** 2026-11-10 后续实验已在用户指定的私有节点完成真实 GPU 服务评分，并经 Ray 的 EnvRouter 与 EnvWorker 执行相同工作量对照。完整原始评分通过，但把 CPU 导入放到 GPU 准入前，没有缩短两会话的整组完成时间。本轮到此停止，不扩大训练、模型或采样规模，也不进入复杂联合控制器开发。这里的 baseline 是同版本机制对照，**不是**用户指定提交 `0eeead861ffafed6bf31c962d259fd847775dcf9` 的历史 EnvRouter OFF。

**版本与运行边界。** PSRL 为主目录 `ab74120` 加未提交实验脚本，没有改生产调度器。SWE-Serve 固定 `84d8fd92739a4a76ceddba56e5a70605921c4ad2`，SGLang 固定 `cfd49e233ceb94de3f702f70aef06e299836aede`。隔离源码应用任务自带参考修复，并只读挂载，原始测试与模型版本不变。这是参考修复的评测与调度实验，不是 Agent 自主解决任务、完整 episode 或 PSRL 训练。归档的[汇总](../env-service-verification/results/score-service-preparation-20261110/summary.json)可通过下方 CPU 脚本从私有原始日志重新计算，[版本及适配记录](../env-service-verification/results/score-service-preparation-20261110/provenance.json)保留实际运行脚本哈希。

官方镜像固定为 `lmsysorg/sglang:v0.5.10.post1-cu130-runtime@sha256:4d9dbdd132d68f0484f469c178669a62d5bdba976a017483b4c9ddc1770848b2`，压缩层合计约 12.07 GiB。用户再次要求继续后，本次另设 1800 秒镜像准备上限，实际 302 秒完成。使用 H20、535.247.01 驱动及镜像内 CUDA 兼容库，实际 PyTorch 为 `2.9.1+cu130`。小型 CUDA 内核成功，实际设备 UUID 与选定设备一致。不能仅凭驱动显示的版本推断此运行结果。

原始完整 `test.sh` 用时 419.65 秒，18/18 项 F2P 新行为测试、8/8 项 P2P 回归测试全部通过，奖励 1.0。保留原始 8 个评分分组以及原本排除的测试，没有更改断言。它证明这个固定任务的参考解可运行，不证明全部 SWE-Serve 任务兼容。

**真实调用与适配。** 下图是实验执行路径，不是新生产架构。

```text
[S1] 两个持久会话，同一显式 command 池，仅一张空闲 GPU
       ├ 普通路径：申请 GPU → [S2] CPU 导入
       └ 准备前移：[S2] CPU 导入完成 → 申请 GPU
                              ↓ EnvRouter 授权
[S3] 检查授权 UUID → 转数字设备序号 → 再核对实际 CUDA UUID
       ↓
[S4] 原始完整测试类：启动一个 Engine，运行 4 项测试，关闭
       ↓
[S5] 命令归还 GPU → 两会话都结束后记录整组时间
       └ 全部批次结束：停止本次容器、释放租约、检查物理资源
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与源码 |
|---|---|
| S1 | [scenario](../env-service-verification/scripts/ray_score_preparation_probe.py#L35) 建立一张 GPU 的显式池和两会话，各申报 49152 MiB，池预算 65536 MiB，因此命令串行准入 |
| S2 | [execute](../env-service-verification/scripts/ray_score_preparation_probe.py#L61) 选择现有接口，[prepare](../env-service-verification/scripts/prepared_score_probe.py#L14) 导入 pytest、torch、transformers，检查尚未初始化 CUDA |
| S3 | [execute 回调](../env-service-verification/scripts/prepared_score_probe.py#L49) 检查授权、转换序号并核验实际设备 |
| S4 | [原始 runner 入口](../env-service-verification/scripts/prepared_score_probe.py#L64) 传递原始清单，执行未经修改的分组脚本 |
| S5 | [批次与清理](../env-service-verification/scripts/ray_score_preparation_probe.py#L143) 保存时间，取消未完成命令、关闭所属容器并检查 GPU |

跟着图读源码：先在 S1 建立会话并各预热一次。随后按普通、前移、前移、普通顺序执行四批，每批并发提交两条命令。S2 的两条 CPU 准备可以并行，但单 GPU 的预算只准入一条 GPU 命令。S3 回调仅在授权后初始化 CUDA，S4 每条命令保留原始类级 Engine 复用。S5 等两条命令都返回才结束批次，最后进入异常也会执行的清理。上述源码有疑问可以按 S1 到 S5 追问。

上游测试工具用 `CUDA_VISIBLE_DEVICES` 的首字符解析端口，直接传 `GPU-...` UUID 会报整数解析错误。因此实验适配器在授权后将 UUID 转为数字序号，并用 PyTorch 实际 UUID 再核对，没有改测试。首次非 pool 配置未提供适配器要求的授权 UUID，在约 11.19 秒内失败并清理。本轮仅更正实验为显式单卡 pool 后继续性能阶段，没有重跑已通过的完整评分。失败日志在私有原始输出中保留，不计作性能样本。

两组均使用相同只读资产、无网络容器、每会话 8 CPU 与 64 GiB 主存。准备前移只提前导入三个依赖，**没有提前加载模型权重、保留 GPU Engine 或实现代码版本失效检测**。实验使用 host IPC，即共享宿主机进程间通信空间，因为当前 native EnvWorker 未提供 `shm_size` 覆盖，两组一致。不能据此声称已验证私有共享内存隔离。准备阶段进程峰值常驻内存约 507 MiB，也需要算入会话主存预算。

**等量结果。** 每条命令执行 `TestMultiItemScoringClassificationAdvancedAdapter` 的完整四项正确性测试。两次预热加四批两会话，共 40 次测试执行全部通过，其中正式计时 32 次。不是 40 个不同任务。每策略两批，不做统计显著性或稳定加速声明。

| 策略 | 两批整组完成时间 | 整组均值 | 每批 GPU 预算持有时间之和的均值 |
|---|---|---|---|
| 准入后导入，普通路径 | 64.107、62.805 秒 | 63.456 秒 | 63.338 秒 |
| 准入前导入，准备前移 | 64.153、62.981 秒 | 63.567 秒 | 60.398 秒 |

准备前移整组慢约 0.17%，在这组小样本中应解释为持平，不能包装为加速。GPU 预算持有量减少约 4.64%，它是从准入到确认归还的时间，不是 GPU 真正计算时间、SM 利用率或能耗。SM 指 GPU 内执行线程块的计算单元，本轮没有测量它的利用率。

**用实测理解为什么“少占预算”不等于“更快”。** 每条 CPU 导入约 2.25 至 2.30 秒。前移后两条命令几乎同时准备，第二条准备完成后仍等待第一条归还 GPU，说明预期的重叠机制确实发生。但第一条的准备仍位于整组关键路径，准备进程启动和控制等待也在总时间内。普通路径的 GPU 持有量约 63.34 秒，前移约 60.40 秒，而前移整组仍为 63.57 秒。持有区间以外的约 3.17 秒不能从完成时间中扣掉。这一差值包含路径上的准备及控制成本，现有采样不能进一步将每一部分精确归因。不能把减少的 2.94 个 GPU 预算秒直接当成节省的墙钟时间。

**复现与停止条件。** 镜像准备、完整评分和性能阶段分别有界。性能脚本原定活动预算 540 秒，配置失败后仅给剩余阶段 520 秒，实际 368.87 秒完成，包含容器建立与两次预热。正式计时不含镜像下载和会话建立，两策略复用相同会话。完整实验已结束，容器清理无错误，目标 GPU 无本次计算 PID，显存回落到约 0.19 MiB，最终 `nvidia-smi` 整数显示 0 MiB。未清理共享缓存或停止他人的服务。

仅在重新完成节点预检、固定资产和环境路径核验后运行 GPU 复现。`SCORE_ASSETS` 为固定资产目录，`SCORE_PERF_OUT` 为新的绝对输出目录，`SCORE_IMAGE` 为上面的固定镜像，`PRIVATE_NODE_IP` 为核实过的私有节点地址。脚本会使用 GPU0，必须先确认其空闲。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/ray_score_preparation_probe.py \
  --assets "$SCORE_ASSETS" --output "$SCORE_PERF_OUT" --image "$SCORE_IMAGE" \
  --node-ip "$PRIVATE_NODE_IP" --gpu 0 --budget-seconds 540
```

已有原始日志的汇总不需要 GPU。`SCORE_RUN` 为同时包含 `probe/` 与 `performance/` 的本轮原始输出根目录，`SCORE_SUMMARY` 为新汇总文件路径。脚本先检查完整评分、四批顺序、设备一致性、准备时序和物理释放，再输出上表数据。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/summarize_score_preparation.py \
  --run "$SCORE_RUN" --output "$SCORE_SUMMARY"
```

本次实际运行了 CPU 汇总脚本，输出为上面链接的 `summary.json`。没有新增版本目录、状态有效性协议或联合优化策略。现有证据不足以为“版本有效的准备与放置联合调度”立项。下一项研究验证应先从真实编辑轨迹寻找**依赖导入之外**的昂贵、可重复、部分失效的准备状态，量化强缓存基线后的可避免成本。未观测到这类成本前，不能把候选想法写成论文贡献，也不自动启动更多 GPU 实验。

本轮维护了本专题的核心结论、进展表、服务兼容性描述、复现路径和两个导航索引。已复核[当前设计入口](../knowledge/current-design.md)的 command 池与 prepared 能力说明，本轮没有改变接口、默认值或资源生命周期，因此未改其架构结论。实验不能验证跨节点、GPU 活状态迁移、真实训推共存或完整 Agent 轨迹收益。

四个实验与汇总脚本的 Ruff、格式和 prose 检查通过，CPU 汇总实际通过，所改三篇文档共 231 个本地链接与章节、行号目标检查通过，新增源码链接已人工对应上述调用点。全仓 `pre-commit run --all-files` 在 `ab74120` 加本次交付文件的临时隔离副本执行，因范围外既有 Ruff、格式和 prose 问题失败，例如 `tests/env_service/_light_stubs.py` 与 `psrl/workers/env_service/client.py`。未将 hook 自动改写带回主目录。脚本、证据和文档为主目录本地未提交内容，本轮未提交、推送或合并。


## 怎样理解本次实验与下一步探索

本节于 2026-11-11 对照 `ab74120` 主目录的本地实验脚本和 11-10 原始汇总补充解释，没有重跑 GPU、训练或更新生产架构。现状入口与导航已检查，本次解释不扩大已实现能力。

**先明确实验问题。** 本次只问：多个环境会话争用一张 GPU 时，能否在等卡期间先做 CPU 工作，从而更早完成相同测试？这里的会话就是一个持续存在的容器工作区，command 是其中执行的一条命令，GPU 准入是调度器允许这条命令占用设备预算。CPU 准备具体只有导入 pytest 测试工具、PyTorch 张量计算库和 Transformers 模型工具库，没有提前加载模型权重。

```text
[S1] 实验驱动并发提交会话 A、B 的测试命令
       ├ 普通：EnvRouter 准入 → [S2] CPU 导入
       └ 前移：[S2] CPU 导入 → EnvRouter 准入
                                ↓
[S3] 授权 GPU，核对真实设备
       ↓
[S4] 启动 SGLang 模型执行引擎 → 原始四项正确性测试 → 关闭引擎
       ↓
[S5] 确认归还 GPU → 两命令全部完成 → 记录整组耗时
       └ 实验结束：清理本次容器、租约，核验物理释放
```

图例：图里每个部分对应的源码位置。

| 部分 | 对应函数与源码 | 输入和状态变化 |
|---|---|---|
| S1 | [scenario](../env-service-verification/scripts/ray_score_preparation_probe.py#L143) | 两个会话并发提交，单卡预算只允许一条 GPU 命令 |
| S2 | [execute](../env-service-verification/scripts/ray_score_preparation_probe.py#L61)、[prepare](../env-service-verification/scripts/prepared_score_probe.py#L14) | 两策略调用相同准备逻辑，仅改变准入前后位置，准备结束时 CUDA 未初始化 |
| S3 | [execute 回调](../env-service-verification/scripts/prepared_score_probe.py#L49) | 收到 GPU 授权后核对实际设备，再运行 GPU 路径 |
| S4 | [原始 runner 调用](../env-service-verification/scripts/prepared_score_probe.py#L64) | 固定清单和测试类，保留原始引擎生命周期 |
| S5 | [scenario 批次与 finally](../env-service-verification/scripts/ray_score_preparation_probe.py#L143) | 等待两命令返回后记录完成时间，结束时定向回收并检查 GPU |

跟着图读源码：S1 中 A、B 并发，S2 的前移准备也可以并行，S3、S4 的单卡执行按准入串行。普通路径 B 的 CPU 导入只能在它拿到卡后开始，前移路径 B 可以在 A 用卡时准备。S5 记录的整组时间包括从提交到最后一条命令返回的等待及控制成本，清理整个实验的容器发生在所有批次之后。异常路径也进入 finally，不把并发调用解释成一个顺序线程。这些源码如果有理解上的问题，可以按 S1 到 S5 继续追问。

**任务本身在做什么。** SWE-Serve 这道题验证 SGLang 的多条目打分功能。例如同一个问题分别与条目甲、乙打分，再一次提交甲、乙，结果应符合原始测试对一致性和正确性的要求。这个例子描述功能，不代表本轮实际输入文本。SGLang 是执行模型推理的系统。本轮使用题目自带参考修复，先跑完整评分的 26 项测试，得到 1.0，再选择一个完整四项测试类做调度对照。评分依据测试是否通过，耗时是另外记录的系统指标。

**对照怎样保证含义清楚。** 两容器始终在同一节点、使用同一张 GPU。每个会话都先预热，再按普通、前移、前移、普通执行四批，每批两命令。这种顺序是减少固定先后顺序影响的简单安排，不代表消除了噪声。每策略只有两批。没有进行换卡、活进程迁移、CPU 模型状态跨命令保留，也没有运行 Agent 编辑或策略训练。本次不能回答这些机制是否有收益。

**为什么少占 GPU 预算却没有更快。** GPU 预算持有时间是允许命令占用设备直到确认归还的时间，其中可以包含 CPU 工作，并不等于 GPU 真正在计算。整组完成时间则从两命令提交开始，直到它们都返回。把工作挪出持有区间，只是改变它记在哪个阶段，只有真正与别的工作重叠、且没有被新增开销抵消，才会更快。

用一组可手算的教学假设说明，数字不是实测。A、B 同时提交，两份 CPU 准备各 2 秒、GPU 工作各 30 秒，有两份足够的 CPU 资源。普通路径 A 在 0 至 2 秒准备、2 至 32 秒用卡，B 在 32 至 34 秒准备、34 至 64 秒用卡，整组 64 秒。前移且无额外开销时，两者在 0 至 2 秒准备，GPU 工作在 2 至 32、32 至 62 秒执行，整组 62 秒。若前移路径还增加 2 秒启动或控制成本，整组又是 64 秒，但 GPU 持有时间仍从 64 降为 60 秒。原本值得隐藏的只有 B 的 2 秒，A 的准备仍必须先完成。这解释可能的机制，不是把实测差值全部归因为某个确定的控制开销。

真实结果与这一现象相符：整组 63.456 → 63.567 秒，GPU 预算持有时间 63.338 → 60.398 秒，未见端到端收益。每条实际导入约 2.25 至 2.30 秒。它支持“这项简单前移在本配置没有可见加速”，不支持“EnvRouter 没用”或“所有阶段拆分都没用”。

以下教学脚本只用标准 Python 和虚拟时间，不启动进程、容器或 GPU。注释中的 S1 至 S5 对应上图，但不模拟真实设备授权、IPC 控制、缓存、资源竞争和测试行为。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/preparation_critical_path_demo.py
```

预期三种假设分别输出整组 `64 → 62`、`64 → 64`、`100 → 82` 秒。第三种把每份准备改为 20 秒，说明昂贵准备可能提供更大的优化空间，不能据此声称当前任务有 18% 收益。本轮 CPU 实际运行输出与上述手算一致。

**下一步优先做成本与变更轨迹画像。** 画像就是把真实执行过程拆成能测量的阶段，并记录每轮代码改了什么。先选当前已跑通的 score 题及一个确实需要不同准备状态的正确性任务，在不改原始测试生命周期的前提下记录 CPU 导入、权重读取和解析、GPU 初始化、模型加载、测试、退出与归还。再看少量真实代码编辑后哪些状态仍有效。不能预先假定权重读取可以从 SGLang 启动流程中独立拆出，也不能把操作系统已缓存的数据再次当作新增收益。

画像先回答两个问题：主要时间是否处在可优化阶段？普通文件缓存和现有类级引擎复用之后，还剩多少可避免的重复工作？即便理想情况下全部避免也只有 2% 的整组时间，就不值得先做复杂策略。这里的 2% 是决策示例，不是实测阈值或硬性标准。

| 候选探索 | 核心问题 | 成立前需要的证据 | 何时停止 |
|---|---|---|---|
| 优先 A：代码变更约束下的状态保留与放置 | 哪些昂贵状态仍对应新代码，保留哪个状态最能减少后续准备？ | 多轮真实编辑中存在成本明显、可部分复用的状态，强缓存后仍有增量空间 | 只有导入成本，或每轮必须全重建，或仅命中文件缓存即可解释收益 |
| 备选 B：面向下一次有效反馈的命令排序 | 同样资源下先完成哪条正确性命令，能更早解除 Agent 或整组的等待？ | 真实命令依赖与 CPU 空窗造成可避免的 GPU 空闲，现有队列与简单短任务优先仍有缺口 | 只是把排队顺序改名，或单 GPU 已持续计算、总工作量不变而无可减少空闲 |

A 的教学例子：一次加载产生权重文件缓存、CPU 解析张量、已导入的打分代码、GPU 引擎四种状态。Agent 只改输入构造，某些权重状态可能仍有效。若改打分代码，旧进程内已导入函数可能已失效，即使权重文件没变。调度器要知道“保留什么、丢弃什么、重新准备何时做”，并将容器仍在原节点和设备可选范围纳入成本。这些是待验证设计，不是现成能力，更不能默认复用完整 GPU 引擎。原始测试要求新进程或新引擎时必须保留，不能跨命令偷改评测隔离。

B 的教学例子：Agent A 的 GPU 命令返回后还要做 20 秒 CPU 分析，而 B 返回后还会立即提交第二条 GPU 命令。排序可能改变下一批 GPU 工作何时就绪。不过这依赖真实后续行为，不能向策略提供未来完整轨迹作为正常输入，最多把全知轨迹用于估计上界。评价仍用原始正确性及固定工作量完成时间，不进入另一个 session 的计时评分保真研究。

这两者都只是研究候选。缓存、预加载、依赖调度和短任务优先本身已有大量工作，不能把组合名称当作创新证明。本轮未新增论文检索，不宣称候选学术空白。若画像显示明确成本，才围绕实际问题定向复核近邻，并比较相同资源、相同测试和强缓存基线。下一轮建议先限制为两个任务、每题少量编辑轮次和单卡十分钟以内 GPU 预算，证据不足也按预算停止。本次只完成讲解及 CPU 教学实验，没有启动该 GPU 计划。

本节新增的 CPU 教学脚本已实际运行，Ruff、格式与 prose 检查通过。本页及两个既有索引共 250 个本地链接、章节或行号目标检查通过，新增链接已对照源码调用点核实。全仓 pre-commit 在 `ab74120` 加本节文件的隔离副本运行，仍因范围外既有 Ruff、格式和 prose 问题失败，没有将自动修改带回主目录。文档与教学脚本为本地未提交内容，未推送。
