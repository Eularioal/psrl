# 固定基线的状态保留、版本管理与复用验证

> **核心结论**：固定基线已经保留容器、文件和应用自己维持的进程。真正需要额外管理的是：改代码后哪些状态仍有效，当前命令执行哪一版，返回的结果能否交给这次尝试。

例如模型权重没变、只改了打分脚本，可以保留模型准备，并重新读取打分代码；如果改的是推理运行时，旧服务内已导入的代码也需要更新。本文沿一个代码修改例子说明依赖、执行快照和结果交付怎样配合。

真实固定基线 `0eeead861ffafed6bf31c962d259fd847775dcf9` 的 CPU 对照中，人工拆分复用耗时 22.326 秒，版本管理原型为 22.374 秒，基本持平。原型的已验证价值是明确版本与拒绝过期结果，生产 Env 尚待接入这套通用管理。当前三个 RE 源码直接入口的旧字节码问题已定向修复，详见[真实任务核查](../research-proposals/benchmark-state-reuse-audit.md)。

局部核查截至 2026-11-13；Docker 对照、交付实验和主存 GPU 数据的版本与条件保存在各自证据文件中。原型的主存缓存条件和真实任务适用性分别见[主存研究](../research-proposals/memory-bounded-state-scheduling.md)与[benchmark 核查](../research-proposals/benchmark-state-reuse-audit.md)。


## 目录

- [范围与全局执行图](#范围与全局执行图)
- [基线已有的保留能力](#基线已有的保留能力)
- [改代码不等于更新常驻进程](#改代码不等于更新常驻进程)
- [与当前 Env 能力的比较](#与当前-env-能力的比较)
- [基线调度缺口与研究门槛](#基线调度缺口与研究门槛)
- [状态版本契约与受控验证](#状态版本契约与受控验证)
- [一次代码修改的完整例子](#一次代码修改的完整例子)
- [Results：真实固定基线与强对照](#results真实固定基线与强对照)
- [故障注入与正确性边界](#故障注入与正确性边界)
- [真实任务成本与研究验收](#真实任务成本与研究验收)
- [复现方法与验证边界](#复现方法与验证边界)

## 范围与全局执行图

固定基线通过 `git show` 读取，源码哈希与当时核对环境保存在证据文件，不切换主目录，不执行历史清理逻辑。当前生产接口核对基于主目录 `6aa328d` 加既有未提交修改，研究原型独立于生产调度器。science 基线调用 Harbor；另一条 EnvWorker 后端独立保留，下面分别画出两条执行链。

```text
[B1] science AgentLoop：开始一次完整 episode，调用 run_harbor_episode
       ↓
[B2] 一个 Harbor Job：同一任务环境中的 Agent 多轮执行命令
       ├ 编辑磁盘文件 → 下一条命令可读到新文件
       ├ 新建 Python 进程 → 重新导入代码，退出后其内存不再保留
       └ 任务自行保留后台进程 → 内存继续存在，但不自动更新已导入代码
       ↓ Agent 与评分结束，或 episode 失败/取消
[B3] runner finally：回收本 episode 的容器与相关镜像

[B4] 基线另一个后端：EnvWorker → 持久 bash → 连续写入命令
       └ 保存 shell 状态；独立于上面的 science / Harbor 路径
```

图例：图里每个部分对应的源码位置。为避免当前文件改变造成历史引用失真，下表基线链接指向从固定提交逐字提取的只读证据快照，哈希记在 [audit.json](../env-service-verification/results/baseline-state-20261111/audit.json)。

| 编号 | 对应函数和源码 | 职责 |
|---|---|---|
| B1 | 固定提交 [SciAccelAgentLoop.run](../env-service-verification/results/baseline-state-20261111/baseline-agent-loop.py.txt#L113)，调用 [run_harbor_episode](../env-service-verification/results/baseline-state-20261111/baseline-agent-loop.py.txt#L151) | 一次 episode 传入一个 task、会话 ID 和模型端点 |
| B2 | [JobConfig](../env-service-verification/results/baseline-state-20261111/baseline-runner.py.txt#L212)、[Job.create 与 run](../env-service-verification/results/baseline-state-20261111/baseline-runner.py.txt#L272) | 创建并运行整个 Job，PSRL 不在每条工具命令后重建 Job |
| B3 | [finally](../env-service-verification/results/baseline-state-20261111/baseline-runner.py.txt#L317) | episode 返回、异常或取消后进入所属资源回收 |
| B4 | [ContainerShell](../env-service-verification/results/baseline-state-20261111/baseline-worker.py.txt#L28)、[exec](../env-service-verification/results/baseline-state-20261111/baseline-worker.py.txt#L257)、[wrap_command](../env-service-verification/results/baseline-state-20261111/baseline-shell.py.txt#L23) | 保持容器 shell，写入命令并读取完成标记，不在每次 exec 后重建 shell |

跟着图读源码：science 的 B1 将一次完整任务交给 B2，B2 内部 Agent 可以多轮与模型和终端交互。整个 Job 完成或失败后，外层 finally 进入 B3 清理。B4 是另一条执行链：保存 shell 对象，后续命令继续写到同一输入流，结束 sandbox 时关闭。两条路径各自具有状态保留边界。

**依赖版本边界。** science 的命令级实现属于外部 Harbor。已查看当前安装的 Harbor 0.22.0：[Terminus2 摘录](../env-service-verification/results/baseline-state-20261111/installed-harbor-terminus.txt)显示 setup 创建一个 tmux 终端会话，后续命令送到该会话；tmux 是能持续保存终端与其中进程的工具。[环境摘录](../env-service-verification/results/baseline-state-20261111/installed-harbor-docker.txt)与 [trial 摘录](../env-service-verification/results/baseline-state-20261111/installed-harbor-trial.txt)显示环境启停边界。版本及文件哈希见[依赖记录](../env-service-verification/results/baseline-state-20261111/installed-dependency.json)。基线当时实际安装的 Harbor 精确版本尚未恢复，不能将当前安装依赖的逐行行为冒称历史实机复现。

## 基线已有的保留能力

这里的 episode 指 Agent 从进入任务、反复修改与测试，到最后评分的一整次尝试，不是一次 shell 命令。

| 状态 | 同一 episode 内 | episode 结束后 | 谁决定它有效 |
|---|---|---|---|
| 修改后的代码、生成文件 | 在同一环境中继续存在，除非任务自己删除或重置 | 容器私有写层通常随容器删除，导出的 artifacts 与挂载目录另论 | 文件系统与任务逻辑 |
| 磁盘编译缓存、下载权重 | 缓存路径仍在就可能复用，任务可以主动清除 | 取决于镜像、卷、宿主目录与清理范围，不保证跨 episode 保留 | 编译器、加载器、任务配置 |
| shell 当前目录、export 环境变量 | 基线持久 shell 路径已有保留，science 需看 Harbor 终端后端 | shell 结束后不再保留 | shell 自身 |
| Python 已导入模块、CPU 张量 | 仅当对应 Python 进程继续活着才保留 | 进程退出或容器删除后消失 | 应用进程 |
| 模型服务与 GPU 状态 | 若任务保持服务运行，命令返回本身不等于服务退出或显存释放 | 服务退出、容器销毁时才结束相应状态 | 应用与 CUDA 运行时，尚未做 GPU 状态复现 |
| 已导出产物用于重评分 | 基线 `_regrade_from_artifacts` 已能尝试复用产物评分 | 产物可独立于原容器存在 | [regrade 路径](../env-service-verification/results/baseline-state-20261111/baseline-runner.py.txt#L64)和任务评分协议 |

所以“基线每步清空、我们才保留文件或缓存”是错误比较。反过来，“容器一直在，所以所有 Python 内存都会保留”也错误。反复执行 `python test.py` 会反复创建进程，哪怕容器没换，旧进程的内存对象也不会自动进入新进程。

基线确实在 episode 结束时清容器，runner 还调用相关镜像回收和 dangling image 清理。dangling image 是已没有名称标签引用的镜像。这个操作不等于逐命令清理所有权重、卷、编译缓存和宿主机页缓存，不能据此推断下次必然完全冷启动。历史全局清理函数仅做源码审计，未执行。

## 改代码不等于更新常驻进程

已执行的 CPU 实验使用真实 Python 进程，结果如下：

1. 写入 `candidate.py`，其中 `VALUE = 1`。启动一个常驻 Python，导入它并读取，得到 1。
2. 修改同一个磁盘文件为 `VALUE = 2`，不重启进程。常驻进程再次读取已导入模块，仍得到 1。
3. 在该例没有已有旧字节码的临时目录中，启动一个新 Python 进程导入该文件，得到 2。该观察只区分进程内已导入对象的寿命。

**新进程也不自动保证读取当前源码字节。** 真实 RE 入口的故障注入显示，同时间戳、同大小的旧 `.pyc` 可能继续生效，`python -B` 仅禁止写入缓存。当前三个候选入口已定向改为读取并编译源码，10 项 CPU 回归通过。它不解决间接导入、后台服务或提交权重的一致性，详见[复现、修复与边界](../research-proposals/benchmark-state-reuse-audit.md#results真实入口的旧字节码问题与修复)。

原因是导入后，模块和其中对象已经在进程内存里。改磁盘文件不会自动重新执行旧进程中的导入过程。本例展示普通 Python 机制，不是在声称基线对所有任务都会读旧代码。如果原任务每次启新进程，它本来就可以看到新代码。应用若自带重载协议，也需要单独评估。

“保留状态”因此有两层含义：**被动保留**是旧进程自然还活着，**有效性管理**是系统明确知道哪部分仍对应当前代码、哪些必须重建。后者还涉及依赖、线程、对象引用和测试隔离，不能简单对任意 Python 模块执行 reload 就当作解决。

## 与当前 Env 能力的比较

| 能力 | 固定基线 | 当前本地 Env |
|---|---|---|
| 同任务文件与任务自行缓存延续 | 已有 | 沿用，不能算新增研究贡献 |
| 任务自行启动并保留服务 | 应用可以这样做，框架没有自动理解其有效性 | 同样不能仅凭进程存活认定结果有效 |
| 受管命令 GPU 准入与确认归还 | science 走静态 GPU compose，未见当前 command 协议 | 已有 EnvRouter command 协议 |
| 命令 GPU 排队前先做显式 CPU 准备 | 未见当前受管接口，但任务也可能自行做 CPU 工作 | [execute_prepared](../../../psrl/workers/env_service/prepared_gpu.py#L24) 已有显式接口 |
| 下一条安全命令自动同机选卡 | 未见当前显式池接口 | 同机 command 池已有，容器仍在原节点 |
| 自动跨多条命令维护准备状态、识别代码失效并联合调度 | 在所核查入口未发现 | 仍未实现，属于候选设计 |

当前 prepared 接口保留的是**一条命令内部**从准备到获得 GPU 的准备进程，finally 随后清理辅助进程与目录。跨命令复用还需要本文讨论的依赖版本、状态目录和交付检查，这部分属于研究原型。

## 基线调度缺口与研究门槛

源码支持以下能力缺口，但缺口不自动等于论文贡献，也不自动证明基线在真实任务中更慢。

| 位置 | 已核实机制 | 可能影响与待验证条件 |
|---|---|---|
| science GPU 放置 | [runner](../env-service-verification/results/baseline-state-20261111/baseline-runner.py.txt#L188) 传入静态 compose GPU 配置 | 没有当前池的下一命令候选设备选择，固定候选忙、其他卡空闲时可能有改进空间。静态可见设备不等于物理独占，不能假设它阻止所有其他任务用卡 |
| science 准入 | [episode semaphore](../env-service-verification/results/baseline-state-20261111/baseline-agent-loop.py.txt#L64) 只控制当前 worker 的 episode 数量 | 计数不描述每条命令是否就绪、显存需求、准备状态及其他 worker 的请求，不能提供完整跨 worker 命令资源共识 |
| 旧 EnvWorker 分配粒度 | [分配](../env-service-verification/results/baseline-state-20261111/baseline-worker.py.txt#L171)在创建 sandbox 时标记本地 GPU busy，[destroy](../env-service-verification/results/baseline-state-20261111/baseline-worker.py.txt#L344) 才移除标记 | CPU 工作和等待期间也保留该 worker 的逻辑分配，可能造成已分配但无计算的空窗。这不等于 GPU 已被硬隔离或 science 采用了同一个机制 |
| 状态信息 | 原执行接口不提供完整代码依赖和准备状态目录 | 调度器无法区分有效与失效的准备状态，也无法可靠估计换卡、重建和等待代价 |

当前 Env 已解决部分工程能力，例如命令准入、确认归还与显式同机候选池。因此研究不能只重新实现这几项。尚需验证的问题是：多个任务连续修改代码时，能否在相同 CPU、主存和 GPU 预算下，联合选择有效状态保留、准备时机与命令放置，并优于强缓存和现有池化路径。

**后端会不会也返回旧代码结果？** 如果任务修改代码后继续请求一个已经导入旧代码的服务，旧 Harbor、旧 EnvWorker、当前 Env 都可能观察到该应用的旧行为。当前 [persistent exec](../../../psrl/workers/env_worker/worker.py#L278) 保留 shell，[exec_process](../../../psrl/workers/env_worker/worker.py#L352) 创建新的命令进程，但两者都没有自动升级其他已存活的服务。若每条命令新启动 Python，并正确加载新文件，就没有上述常驻模块问题。现有证据是条件性风险和能力缺口，不是某个真实 benchmark 已被框架错误评分的复现。

## 状态版本契约与受控验证

本节解释独立研究适配器的现有实现，不是生产 EnvRouter 的新功能。适配器是任务和执行器之间的一小层代码，明确告诉系统哪些文件影响哪部分常驻状态。本实验只管理固定模型、两个打分文件和一份服务配置，不尝试理解任意 Python 程序。

先把所有角色连起来。下图每条执行命令都经过从固定提交提取的原始 EnvWorker，而不是用当前版本关掉 Env Service 冒充基线。

```text
[V1] 实验调用方提交代码修改
       ↓ commit：与结果交付使用同一把协作锁
[V2] 工作区管理器 capture：复制声明文件，计算内容指纹
       ├ 模型配置没变：保留原模型进程
       └ 配置变了或依赖不确定：关闭并重建模型进程
       ↓ 通过基线 EnvWorker 的持久 shell 发送请求
[V3] Docker 内常驻 CPU 模型：每条请求都做一次真实推理
       ↓ 推理数值，不传递 Python 模型对象
[V4] 新的轻量 Python 子进程：从本次固定快照读取打分代码
       ↓ 数值结果 + 实际代码指纹 + 已加载配置指纹
[V5] 工作区管理器 publish：持锁核验版本并写入接受记录
       ├ 请求版本与执行版本不符：拒绝
       ├ 要求当前版本，但相关文件已变：拒绝后重新提交
       └ 满足要求：记录结果及其版本，再放开锁
       ↓ 所有实验组结束
[V6] 停止模型进程 → 基线 destroy_sandbox → 核实容器已消失
```

图例：图里每个部分对应的源码位置。

| 编号或箭头 | 真实函数与源码 | 输入、输出与状态变化 |
|---|---|---|
| V1 → V2 | [Workspace.commit](../env-service-verification/scripts/state_reuse_study/workspace.py#L39)、[locked](../env-service-verification/scripts/state_reuse_study/workspace.py#L30) | 正常参与协议的编辑先取得文件锁，修改结束再解锁 |
| V2 | [Workspace.capture](../env-service-verification/scripts/state_reuse_study/workspace.py#L51) | 读取完整声明集合，产生只读快照、代码指纹和配置指纹 |
| V2 → V3 | [request](../env-service-verification/scripts/state_reuse_study/baseline_compare.py#L121) | 选择复用或重建，再由原始 worker 执行服务请求命令 |
| 原始后端入口 | [load_baseline](../env-service-verification/scripts/state_reuse_study/baseline_compare.py#L42)、[execute](../env-service-verification/scripts/state_reuse_study/baseline_compare.py#L93) | 从指定提交逐字提取后端，核对实际导入来源，再调用原始 `exec` |
| V3 | [CpuModel](../env-service-verification/scripts/state_reuse_study/model_process.py#L41)、[serve](../env-service-verification/scripts/state_reuse_study/model_process.py#L102) | 模型权重留在 CPU 主存，重复处理请求，不初始化 CUDA |
| V3 → V4 | [answer](../env-service-verification/scripts/state_reuse_study/model_process.py#L72)、[postprocess](../env-service-verification/scripts/state_reuse_study/model_process.py#L31) | 模型先推理，把数值交给新的打分进程，后者重新读取快照代码 |
| V5 | [Workspace.publish](../env-service-verification/scripts/state_reuse_study/workspace.py#L72) | 检查实际执行身份和当前内容，在同一锁内写入接受记录 |
| V6 | [stop_service](../env-service-verification/scripts/state_reuse_study/baseline_compare.py#L101)、[finally](../env-service-verification/scripts/state_reuse_study/baseline_compare.py#L314) | 等待模型退出，销毁本实验 sandbox，检查所属标签下没有残余容器 |

跟着图读源码：先看 V1 如何编辑，再看 V2 如何捕获版本。`request` 把模型配置指纹与已有服务比较，决定是否重建，然后调用原始 worker 的命令通道。V3 的模型服务一直活着，V4 则每条命令新建，二者的生命周期不同。V5 位于异步请求返回之后，因此 V3 计算期间 V1 可以发生新的编辑。V5 自身的检查与写入受同一把锁保护，其他参与者的编辑要等待它完成。最后无论正常或异常，都进入 V6 的所属资源清理。代码如果有理解上的问题，可以按 V1 至 V6 随时追问。

**模型对象与可修改代码分别怎样处理。** 模型对象包含已经导入的计算库与读入主存的权重，建立它较贵。打分代码在本实验中只处理模型输出的一个数值，计算很轻。模型服务通过本机 Unix socket（本机进程之间的通信通道）接收请求，完成推理后，再启动一个轻量 Python 子进程，从本次快照读取打分代码。这样，昂贵的模型准备可以继续使用，轻量代码每次都读取明确的版本。

这个拆分适用于“修改只影响模型输出后的处理”这一条件。若 Agent 改动模型实现、加载方式或模型依赖，适配器应使模型状态失效并重建。该实验主动提供了可拆分接口；新任务接入时也需先核对依赖关系。

**文件指纹怎样计算。** `digest` 接收形如 `{文件名: 文件内容}` 的字典，按固定键顺序编码为 JSON，再计算 SHA-256 摘要。代码内容变化会改变这串摘要，包括尚未提交到 Git 的修改。实际摘要很长，下面用 `C1`、`C2`、`M1` 作为教学简称：

| 变化 | 代码指纹 | 模型配置指纹 | 本适配器的处理 |
|---|---|---|---|
| 只换输入 | C1 | M1 | 保留模型，按新输入重新推理 |
| 只修改 README | C1 | M1 | README 在本受控程序的依赖集合之外，继续使用模型 |
| helper 的 OFFSET 从 0 改为 1 | C2 | M1 | 保留模型，新子进程读取新 helper |
| 改服务配置 | C2 | M2 | 按本适配器的保守规则重建服务 |
| 无法确认依赖完整 | 标记依赖不确定 | 标记依赖不确定 | 停止旧服务并重新建立，同时保留依赖未查明的状态 |

**实验身份覆盖什么。** 本适配器声明 `helper.py`、`scorer.py`、`loader.json` 三份依赖；代码身份覆盖前两个文件，`model_identity` 覆盖解析后的服务配置。模型权重另外按固定 revision 只读挂载，运行镜像也固定。执行进程可信，并按这份声明报告身份，因此实验能核对“这次请求实际用了哪份代码与配置”。新任务若有动态导入、外部服务或更多配置，应先补齐自己的依赖声明。

**快照怎样固定本次读取。** 管理器把声明文件复制到 `versions` 下的独立目录，并让执行代码只读取这份内容。Agent 后续编辑的是 `live` 目录，本次计算继续读取捕获的快照。只读权限用于减少合作实验进程的误改；这一原型采用可信进程模型。

**为什么计算结束后还要检查。** 快照能保证“这次执行哪一版”，却不能保证“交付时它仍是最新一版”。管理器因此问两个不同的问题：实际执行的代码与服务配置是否等于请求版本？如果调用方要求当前版本，当前声明内容是否仍等于请求快照？前者防止错用状态，后者防止把计算期间已经过期的结果当作当前反馈。

**为什么需要同一把锁。** 若先检查文件、放开控制，再写结果，编辑可能恰好夹在这两步中间。`commit` 与 `publish` 都用 `flock` 文件锁，含义是所有合作进程在编辑或接受结果时轮流进入这段操作。`publish` 持锁完成检查及接受记录写入，因此正常参与者不能插进两者之间。模型计算本身不持锁，避免让一次耗时推理阻止所有编辑。

这里“有效”指**结果被接受的那个时刻符合约定**。结果接受之后发生新编辑，并不会回溯地让之前合法的记录变成错误。记录必须带版本，使用者不能把任何历史记录永远当作最新结果。调用方若明确请求某个历史快照，允许返回它是合法的，原型通过 `require_current=False` 表达这种需求。

## 一次代码修改的完整例子

下面使用真实模型对照中的数值，不是原始 SWE-Serve 评分。模型为固定版本的 `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`，用短 token 输入在 CPU 上运行。实验的打分函数是人为编写的简单后处理，用来清楚区分模型与代码两种状态。

1. 初始 `helper.py` 写着 `OFFSET = 0`，`scorer.py` 计算 `模型输出 + OFFSET`。模型输出为 `-2.354431`，结果也是 `-2.354431`。
2. 下一条命令只改输入，模型再次推理，输出变为 `-2.483530`。原模型进程可以继续使用，但不能把上一条答案直接返回。
3. 修改 README。它不在这个程序的依赖中，不必重新加载模型。
4. 将 helper 改为 `OFFSET = 1`。新方案保持同一个模型进程，新建打分进程读取新 helper，结果为 `-2.354431 + 1 = -1.354431`。首次 versioned 组中的模型 PID 在这四次请求间始终为 445。
5. 再把 scorer 改成 `2 × 模型输出 + OFFSET`。同一模型仍可用，新打分结果为 `2 × -2.354431 + 1 = -3.708862`。

始终保留第一次导入的 scorer 与 helper 的诊断组，在第 4 步仍返回 `-2.354431`。磁盘上的 1 没有自动替换旧进程内存中的 0。**这是该诊断应用盲目复用的后果，不是 EnvWorker 每次都必然执行旧文件。** 基线每条命令新启 Python 本来就正确，基线人工拆分模型与新打分进程也正确。

如果改动的是模型计算代码而不是这个独立后处理，以上保留规则就不能直接沿用。真实 benchmark 的原始测试若要求每次创建全新引擎、重置随机状态或隔离测试，研究适配器也必须遵守，不能靠跳过这些步骤取得所谓收益。

## Results：真实固定基线与强对照

实测为 2026-11-12 私有实验节点上的真实 Docker 任务。后端使用固定基线的 `worker.py`、`shell.py`、`sandbox.py` 原始字节，仅用空 `__init__.py` 隔离导入，不改后端执行和调度源码。使用普通 `EnvWorker` 对象，不启动其 Ray 包装层。它验证基线 EnvWorker 的真实容器命令能力，**不等于恢复当时 science 的 Harbor、模型策略或完整训练环境**。

容器限制为 8 CPU、16 GiB 主存、网络关闭、没有 GPU device 或 DeviceRequest。模型权重固定在 revision `6a5829f5079c66e78d911e06fe21931cc00232f7`，运行镜像按 digest 固定。所有组都执行真实模型前向与相同的 8 条命令：初始、仅换输入、无关编辑、helper 编辑、scorer 编辑、服务配置编辑、依赖不确定标记、代码回退。

| 策略 | 每组模型建立次数 | 结果不符 | 整组耗时 |
|---|---:|---:|---:|
| fresh：基线每条命令新建模型进程 | 8 | 0/8 | 54.500 秒，1 组 |
| restart：相关代码或配置变了就重启整个服务 | 6 | 0/8 | 42.511 秒，1 组 |
| split：基线人工拆分常驻模型与新打分进程 | 3 | 0/16 | 22.326 秒均值，2 组 |
| versioned：相同拆分加快照、版本核验和交付锁 | 3 | 0/16 | 22.374 秒均值，2 组 |
| blind：保留第一次模型及已导入打分代码 | 1 | 5/8 | 8.602 秒，诊断组，不能作为合法加速 |

五组应用生命周期策略由实验调用方实现，并共同使用原始基线后端。这样的对照将变化集中在状态管理，便于判断复用与版本检查各自的作用。

**计时边界。** 每组从第一条请求前开始，到全部 8 条命令执行完且本组模型服务退出为止，包含模型首次准备、期间重建和服务停止。Docker 容器创建、全局一次预热及实验最后销毁容器在各组之外。不能把此表称为包含完整容器初始化与清理的 science 端到端耗时。故障注入等待不在性能组内。

强对照按 split、versioned、versioned、split 顺序执行，分别为 `22.192、22.143、22.604、22.460` 秒。这种 ABBA 顺序用于降低固定先后影响，不消除噪声。新方案均值比人工拆分多 0.047 秒，约 0.21%，而组内波动本身更大。n=2 不能支持显著性结论，更不能把这个差值精确解释为版本检查的开销。

相对本工作负载的逐次重建，versioned 时间减少约 58.9%。但是相同复用收益已被基线人工拆分覆盖。**新方案在这个实验中的增量价值是版本约束和结果追溯，没有观察到额外性能优势。** 不存在“靠这些数字就能立项发表”的结论。

**保守重建规则也有局限。** 配置编辑在本实验中是长度 8 改为 12，服务把它保存在初始化配置里，因此两条拆分路径都按相同规则重启。模型本身通常能支持不同长度，若把长度作为请求参数传入，这次完整加载本来可以避免。依赖不确定也由脚本主动标记，并非发现了真实隐藏依赖。这些重建数是该适配器的结果，不是所有合法实现的最低代价。

完整数值、各次 PID 与代码身份见 [model-comparison.json](../env-service-verification/results/deep-state-20261112/model-comparison.json)，计算汇总见 [summary.json](../env-service-verification/results/deep-state-20261112/summary.json)，后端和冻结脚本哈希见 [provenance.json](../env-service-verification/results/deep-state-20261112/provenance.json)。实验主体约 211.9 秒，控制器正常退出，所属容器检查为空，CPU 模型没有初始化 CUDA。

## 故障注入与正确性边界

真实模型的交付实验先固定旧代码完成推理，在返回前暂停。随后把当前 helper 改为 `OFFSET = 10`，再让旧请求返回。版本管理器拒绝把旧结果 `-2.354431` 接受为当前结果。重新提交后得到 `7.645569`，模型 PID 前后均为 753，说明拒绝旧反馈并不要求把仍有效的模型权重也重建。

另一个仅用标准库的脚本进行了 10 项受控检查，结果见 [publication-checks.json](../env-service-verification/results/deep-state-20261112/publication-checks.json)：

| 检查类别 | 实际观察 |
|---|---|
| 错配状态 | 报告的代码或模型配置身份不符时拒绝，拒绝结果没有写入接受文件 |
| 修改范围 | 无关 README 编辑仍接受，相关 helper 改动后拒绝旧的“当前版本”结果 |
| 固定历史版本 | 显式请求旧快照时允许接受，原快照内容不随 live 编辑改变 |
| 内容回退 | 文件内容完整回到旧值时允许复用，协议比较内容，不把编辑次数本身当语义变化 |
| 交付和编辑并发 | 在真实 `publish` 写记录的位置暂停，合作编辑等待，接受记录写好后编辑才能完成 |

交付实验使用真实模型，核对结果与版本的对应；并发锁实验使用合成执行身份，核对合作编辑与接受记录的先后顺序。两者分别覆盖结果有效性和接受时的竞争窗口。

生产接入还需要补齐以下条件：

| 条件 | 目前核对的范围与待补项 |
|---|---|
| 完整依赖 | 实验依赖来自可信适配器。新任务需识别动态导入、环境变量、外部服务与原生库状态等输入；依赖不确定时采用重建，并保留不确定标记 |
| 统一编辑入口 | 合作进程通过 `commit` / `publish` 使用同一文件锁。生产 Agent 需要统一版本提交入口，或先在隔离工作区编辑，再提交完整版本 |
| 故障与持久化恢复 | 文件锁已检查合作进程的排他顺序；多文件编辑中途崩溃、磁盘写失败、管理器重启、重复请求及接受记录恢复仍待验收 |
| 可信执行身份 | 实验由可信执行进程报告代码和配置身份。对抗恶意代码时，需要额外的认证和隔离机制 |
| 结果使用时刻 | 接受记录保证接受点的版本关系；后续使用者应携带并核对记录中的版本 |
| benchmark 测试隔离 | 原始 benchmark 的随机状态、测试 fixture 和引擎重建要求需要逐任务审查，复用策略必须遵守其评分协议 |

早期 [versioned_state_probe.py](../env-service-verification/scripts/versioned_state_probe.py) 使用小型 JSON 权重，六命令的 `6/1/5/4` 次建立、`0/3/0/0` 次不符保留为[教学证据](../env-service-verification/results/state-validity-20261112/evidence.json)。该脚本没有交付锁，其限制仍然成立。它不再代表本节真实模型适配器的能力，也不能替代真实基线后端对照。

## 真实任务成本与研究验收

**已有服务任务给出的限制。** 原始 score 服务实验的完整 verifier 已通过，但简单把 CPU 导入移到 GPU 准入前，没有整组加速。导入均值约 2.270 秒，pytest 分组约 25.08 至 26.60 秒，其中包含测试前后的引擎建立和销毁。权重分片加载进度约 1.78 至 2.06 秒只是局部阶段，不能把剩余二十多秒都算作可复用准备。历史条件见 [audit.json](../env-service-verification/results/baseline-state-20261111/audit.json)与[原始服务对照](../research-proposals/non-timing-gpu-scheduling-scenarios.md#results真实-score-服务题与准备前移对照)。

本页真实 CPU 模型实验补上了“保留非平凡模型对象时，能否同时更新独立代码”的机制证据，但没有保留原始服务任务生命周期，也没有真实 Agent 编辑轨迹。因此不能将表中的 58.9% 移植为该 benchmark 或 science 的收益。[主存状态池研究](../research-proposals/memory-bounded-state-scheduling.md)另外运行了 GPU 机制试验，仍未建立原始 benchmark 或生产 Env 的端到端收益。

**哪些部分不能再算新颖。** 内容指纹、依赖变化后重建，以及根据模型位置减少加载，已有成熟研究和系统：

| 一手资料 | 已有内容与本研究边界 |
|---|---|
| [Build systems à la carte，ICFP 2018](https://www.microsoft.com/en-us/research/publication/build-systems-la-carte/) | 系统梳理并组合构建系统机制。依赖检查、选择是否重建不是新的 Agent 调度发明 |
| [Bazel 的 hermeticity 说明](https://bazel.build/basics/hermeticity) | hermeticity 指通过固定工具、隔离未声明影响等方式约束输入来源。声明完整和输入一致性是安全缓存的基础，本原型尚未做到通用隔离 |
| [ServerlessLLM，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/fu) | 已结合多层模型加载、迁移和位置感知调度。只说“模型在哪里就优先去哪运行”不足以形成新贡献 |

这次定向查阅用于约束贡献表述，没有完成所有 Agent 状态管理论文的穷尽检索，也不宣称无人研究。更可检验的候选是：**多个 Agent 不断改变代码、主存不能保存所有准备状态时，怎样联合决定保留哪份仍有效的状态、何时准备，以及命令在哪里执行。** 版本管理是合法性前提，资源受限下的联合策略才可能是独立研究对象。

例如，教学假设是主存只容得下一份 6 GiB 的准备状态，A、B 都希望保留。普通最近最少使用策略只看最近访问时间，却不知道 A 的下一次改动是否会让当前状态失效，也不知道 B 的 GPU 命令还会排多久。研究策略需要利用当下可见的代码差异、已就绪命令、准备耗时和排队状态，决定保留谁或是否延迟准备。未来修改只能预测或作为离线上界，不能把完整未来轨迹偷偷交给正式策略。这个例子的数字不是实测，独立受限状态池的计时和反例见[GPU Results](../research-proposals/memory-bounded-state-scheduling.md#results成本画像与真实-gpu-命令)。

必须把 CPU 状态与 GPU 状态分开：本页固定基线模型对照只用主存，未换卡。独立主存状态池研究使用 GPU，但同样没有换卡。同机下一命令改用另一张 GPU，不必搬走容器和主存。若状态本来可以被两张卡共同读取，不能凭空算一次完整冷启动。真正留在某张 GPU 的模型、CUDA 上下文等是否可迁移、迁移多久，需要另外实测，当前实现没有通用迁移能力。

继续投入前的验收门槛是：两个非计时正确性任务中，存在真实多轮修改、明显准备成本和主存竞争。保持原始测试与相同工作量，至少比较合理的人工常驻拆分、文件或构建缓存、版本正确的 LRU 保留策略和现有命令池。只有新策略在这些强对照后仍有增量，才开发生产联合控制器。LRU 即最近最少使用，通常先淘汰最长时间未被访问的缓存。CPU 对照已经足够否定“仅增加版本指纹就能超过合理基线”的性能主张。独立 GPU 状态池试验中，复杂反馈策略也只在两个受限场景显示单次约 1% 至 3% 的总时长差异，完整成本回放没有普遍优势。当前应先建立真实状态生命周期与变更轨迹，不开发生产复杂策略，详见[研究门槛](../research-proposals/memory-bounded-state-scheduling.md#值得继续的问题与立项门槛)。

## 复现方法与验证边界

先运行不依赖模型、Docker 或 GPU 的故障实验。在仓库根目录使用已激活的项目 Python：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/state_reuse_study/contract_checks.py \
  --output /tmp/psrl-state-publication-evidence.json
```

预期 `status` 为 `passed`，10 个检查值都为 `true`。它会建立临时工作区，注入代码错配、过期和并发编辑，在结束时清理临时文件。它只验证管理协议，不把数值 1 的合成结果称为模型推理。

真实模型对照需要 Docker、已缓存的固定镜像和只读模型缓存，启动前按[集群预检指南](../../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)检查当前节点。以下变量由执行者设置，不自动下载模型：

| 变量 | 内容 |
|---|---|
| `HF_CACHE` | 包含 `hub/models--tomaarsen--Qwen3-Reranker-0.6B-seq-cls/` 的 Hugging Face 缓存根目录，须包含上面的固定 revision |
| `STUDY_OUT` | 本次独立且尚不存在的证据目录，不能使用别人的输出或自动续跑目录 |
| `STUDY_IMAGE` | 下方按 digest 固定的镜像名称，使用前需已在本节点缓存 |

```bash
: "${HF_CACHE:?请先设置本地模型缓存目录}"
: "${STUDY_OUT:?请先设置本次独立输出目录}"
STUDY_IMAGE='lmsysorg/sglang:v0.5.10.post1-cu130-runtime@sha256:4d9dbdd132d68f0484f469c178669a62d5bdba976a017483b4c9ddc1770848b2'
python docs/env-worker-deep-dive/env-service-verification/scripts/state_reuse_study/baseline_compare.py \
  --repo "$PWD" --output "$STUDY_OUT" --hf-cache "$HF_CACHE" \
  --image "$STUDY_IMAGE" --budget-seconds 480
```

脚本主动预算为 480 秒，命令失败或超时就终止整组，不在可能仍有旧命令运行的 shell 上继续比较。它只清理自己创建的 sandbox 和临时文件，不重启 Docker、不全局停止 Ray、不 prune 镜像。本轮实际在脱离连接的后台控制器中运行，另有 560 秒外部上限与按本实验标签回收的兜底，正常完成后控制器和实验均退出。没有因为文档复现命令再次运行昂贵实验。

固定基线生命周期的轻量审计仍可用：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_baseline_state.py \
  --output /tmp/psrl-baseline-state-audit
```

它只用标准库、Git 和 bash，预期环境及文件延续，Python 常驻/新进程依次得到 `1、1、2`。它不是 Harbor 或基线 Docker 端到端复现。真实模型实验及额外故障检查均已实际执行，原始性能脚本已冻结在[证据目录](../env-service-verification/results/deep-state-20261112)。

研究目录的 Ruff、格式和 prose 检查通过，48 条正确对照命令另外按打分公式核验。文档链接与新增源码定位已核对，详情见 [validation.json](../env-service-verification/results/deep-state-20261112/validation.json)。全仓 pre-commit 在隔离副本执行，仍有范围外既有 Ruff、格式和 prose 失败，自动改写没有带回主目录。研究代码与本文为本地未提交内容，未新增生产接口、未推送、未部署。
