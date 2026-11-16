# 从真实 benchmark 判断状态复用是否值得做

> **核心结论**：真实任务中有昂贵状态，但能否复用取决于 Agent 实际改了什么。SWE 的 41 个 GPU 题参考补丁都修改推理运行时，训练任务会改变权重和训练状态，AIRS 最终交付的是预测 CSV。研究应先沿这些真实变化检查版本，再选择缓存策略。

例如 Agent 改了一个推理函数，已经启动的 Python 服务仍可能运行旧代码；模型权重文件相同，也不足以证明整个服务仍有效。反过来，原测试若已经在一组请求之间保留模型，就应把它作为现成的复用对照。

本次核查有两个直接结果：三个 RE 候选源码入口已定向修复旧字节码读取；PostTrain 模型加载后的请求约从 9.5 秒降至 4.4 秒，但完整短批次 113.84 秒仍长于普通进程复用的 91.96 秒。前者解决版本正确性，后者说明要同时看局部耗时和整批成本。

核查截至 2026-11-13，使用 `6aa328d` 加工作区内容、固定 SWE-Serve 41 题参考修复和已有交互记录。证据包括源码审计、入口 CPU 回归与九次本地 PostTrain GPU 评分；本轮整理保留原数据。目录按“状态在哪里→能否复用→真实计时→版本故障→研究取舍”展开。


## 目录

- [从真实交付流程看状态在哪里](#从真实交付流程看状态在哪里)
- [先回答什么才算可复用](#先回答什么才算可复用)
- [SWE 的真实修复为什么不符合轻量打分假设](#swe-的真实修复为什么不符合轻量打分假设)
- [训练任务真实改变的是哪些状态](#训练任务真实改变的是哪些状态)
- [PostTrain 评分准备的真实计时](#posttrain-评分准备的真实计时)
- [现有交互记录能证明多少](#现有交互记录能证明多少)
- [Results：真实入口的旧字节码问题与修复](#results真实入口的旧字节码问题与修复)
- [为什么文件路径不是数据版本](#为什么文件路径不是数据版本)
- [哪些研究方向保留哪些应停止](#哪些研究方向保留哪些应停止)
- [相邻研究已经覆盖什么](#相邻研究已经覆盖什么)
- [跟着图读源码与复现](#跟着图读源码与复现)
- [验证边界和停止条件](#验证边界和停止条件)

## 从真实交付流程看状态在哪里

这里研究的是 Agent 解题时运行的模型、程序和数据，不是 PSRL 负责生成下一条操作的策略模型。先沿已有任务走一遍，再讨论可以保留什么。下图的 SWE 分支特指已接入的本地补丁提交协议，不能扩展成所有上游题都采用同一个交付实现。

```text
[1] Agent 在自己的任务环境里修改程序，或训练一个模型
                     ↓
[2] 明确交付文件
    SWE：submission.patch   训练类：checkpoint   AIRS：submission.csv
                     ↓
[3] 独立 verifier 接收提交物
    SWE：在干净源码上应用补丁
    训练类：读取提交参数和隐藏评估数据
    AIRS：复制本次 CSV 字节，独立读取隐藏标签
                     ↓
[4] 执行对应检查
    SWE：启动新测试进程，同一测试组内部可以保留 Engine
    训练类：新建评分模型，加载本次提交权重
    AIRS：对 CSV 计算指标，通常不需要重新加载 Agent 的训练模型
                     ↓
[5] 指标返回 Agent / PSRL，所属评分环境清理
```

图例：图里每个部分对应的源码位置。

| 编号 | 对应实现 | 输入、输出及真实状态变化 |
|---|---|---|
| 1 → 2，SWE | [SUBMIT](../../../examples/swe_serve/patch_submission.py#L5) | 可修改 `/code` → Git 补丁文件，不交付原进程 |
| 2 → 3，SWE | [REPLAY](../../../examples/swe_serve/patch_submission.py#L9) | 独立环境检查并应用补丁，随后运行可信测试 |
| 3 → 4，SWE | [attested_verify.main](../../../examples/swe_serve/attested_verify.py#L10) | 逐组建立新 Python 子进程，收集测试身份与通过数 |
| 2 → 4，PostTrain 本地 | [execute](../../../examples/posttrain_bench/local_model.py#L9) | 只读基础模型与提交参数 → 新评分模型 → 隐藏答案似然 |
| 2 → 3，AIRS | [EnvWorker.evaluate](../../../psrl/workers/env_worker/worker.py#L456) | 读取 CSV 字节，写入独立子 sandbox，执行后销毁 |
| 4，AIRS | [build_verifier_spec](../../../examples/airs_bench/grading.py#L32) | 可信评分脚本、隐藏标签与提交 CSV → CPU 指标 |
| 5 | [runner 的 finally](../../../examples/sciaccel_rl/runner.py#L610)、[EnvWorker.evaluate 的 finally](../../../psrl/workers/env_worker/worker.py#L476) | 所属资源清理，不清除其他任务或共享模型缓存 |

以上代码如果有理解上的问题，可以按图中编号继续追问。详细执行顺序和 CPU 复现入口见文末。

## 先回答什么才算可复用

复用的前提是后续工作仍需要同一个对象。名字相同不够，文件路径相同也不够。例如先训练出参数 A，再把参数 B 写到同一个 `final_model` 路径，后续评估必须分清自己要评的是 A 还是 B。

| 状态 | 实际含义 | 何时可能继续使用 | 何时必须重新处理 |
|---|---|---|---|
| 只读文件 | 基础权重、公开数据、固定依赖、镜像 | 内容和所需环境未变，且隔离边界允许 | 文件内容或依赖身份改变 |
| 原生编译缓存 | Triton、JAX 等已经编译出的程序 | 原生缓存键仍匹配代码、形状、精度和设备等条件 | 缓存键不匹配，交给原生机制重新编译 |
| 活的推理服务 | 已导入的 Python 方法、请求对象、GPU 模型及子进程 | 执行实现和配置仍匹配，并满足资源生命周期契约 | 修改了运行路径中的实现或影响状态的配置 |
| 正在训练的模型 | 当前参数及后续更新所需的状态 | 同一训练运行继续执行 | 开始另一独立实验，或需恢复到不同训练点 |
| 评估结果 | 某个确定提交、数据与评分协议下的分数 | 比较该提交，或用户明确接受其历史结果 | 不能直接标成另一个新提交的结果 |

CPU 主存缓存和 GPU 显存驻留也不是一回事。只保留只读模型文件，不等于保留完整模型服务。训练模型除了参数，还可能需要优化器状态、随机数状态和数据读取进度。上一项训练的参数没有恢复干净，就不是一次独立的新实验。

## SWE 的真实修复为什么不符合轻量打分假设

固定源版本是 `84d8fd92739a4a76ceddba56e5a70605921c4ad2`。标准库审计从 `task_list.csv` 选择声明 GPU 数量大于零的任务，读取每题参考补丁、测试入口和 Python 测试语法树，不导入任务代码，也不执行模型。

| 审计结果 | 数量 | 能推出什么 |
|---|---:|---|
| 清单全部任务 | 53 | 上游清单范围，不是本地全量实测数量 |
| 声明需要 GPU 的任务 | 41 | 有设备需求声明，不保证每个直接计分节点都使用 GPU |
| 参考补丁修改 `python/sglang/srt/` | 41/41 | 真实修复涉及推理运行时，不能一概视为“固定模型之外的轻量 scorer 改动” |
| 原 `tests/test.sh` 明确调用分组进程运行器 | 12/41 | 这些入口本来就将不同组隔在新进程中 |
| 测试源码出现类级初始化 | 17/41 | 已有组内状态保留的语法结构，不是自动统计出的运行复用次数 |

这是参考答案的静态审计，不代表 Agent 每次编辑都改全部运行时，也不代表统计出了真实编辑频率。工厂函数转发、未直接计分的附加测试和动态构造均会影响语法统计，不能把直接构造语句数当成完整复用覆盖率。逐题路径、内容哈希及限制见 [swe-static.json](../env-service-verification/results/benchmark-state-audit-20261113/swe-static.json)。

**具体看一直使用的 score 题。** `ss-sgl22544-score-multi-item-delimiters` 要让一次多条目评分与逐条评分一致，分数由正确性决定。它的参考补丁修改了 11 个运行时文件，包含注意力后端、输出处理、请求结构、调度器和服务器配置。

最直接的例子是 [changes.patch](../../../../benchmark-sources/swe-serve/tasks/ss-sgl22544-score-multi-item-delimiters/solution/changes.patch#L770)：组织输入的函数开始返回额外的条目分界位置，调度和模型执行侧也要接收这些信息。仅让一个新的轻量进程处理最终输出，不能使旧 Engine 的方法、旧请求结构及旧子进程自动获得这些变化。

与此同时，原始 [setUpClass](../../../../benchmark-sources/swe-serve/tasks/ss-sgl22544-score-multi-item-delimiters/tests/postmerge_tests/test/registered/prefill_only/test_multi_item_scoring.py#L67) 已在一个测试类内建立并复用 Engine，[tearDownClass](../../../../benchmark-sources/swe-serve/tasks/ss-sgl22544-score-multi-item-delimiters/tests/postmerge_tests/test/registered/prefill_only/test_multi_item_scoring.py#L84) 才关闭。原 [test.sh](../../../../benchmark-sources/swe-serve/tasks/ss-sgl22544-score-multi-item-delimiters/tests/test.sh#L40) 明确说明类内保留温热 Engine、类间释放 GPU 和主存。不能拿逐断言冷启动作为它的基线，也不能随意拆开包含并发或会话状态的测试组。

**合理的安全边界**是：修改影响运行时的源码后，重新建立相关服务进程，继续利用内容没变的模型文件和兼容编译缓存。更细的重建粒度需要逐个证明依赖关系，不能先假设模型服务内部全部与修改无关。

另外，embedding LoRA 题的 [upstream_e2e.toml](../../../../benchmark-sources/swe-serve/tasks/ss-sgl17780-embedding-lora/tests/upstream_e2e.toml#L6) 中，默认直接计分节点是 CPU 字段检查，额外 GPU 行为测试有另列范围。会话提交、推测解码等题测试的是服务内部状态正确性，保留错误的上一版本服务反而可能使验证失去意义。它们可以用来检验状态隔离，不能直接证明 Env 应跨版本复用它们的活进程。

## 训练任务真实改变的是哪些状态

**PostTrainBench。** 原始 [任务说明](../../../../benchmark-sources/PostTrainBench/README.md#L147) 允许多次训练、查询评估器，并把最佳模型存为 `final_model`。原 [GSM8K 评估入口](../../../../benchmark-sources/PostTrainBench/src/eval/tasks/gsm8k/evaluate.py#L74) 把模型路径传给 vLLM 评估。这是自然存在的“候选模型产生 → 评估 → 继续改进”循环。因此版本关系应围绕候选权重和评估请求建立。

本地已经接通的 9 个组合使用受限协议：[local_model.execute](../../../examples/posttrain_bench/local_model.py#L26) 仅训练最后的归一化参数向量，独立 verifier 重新加载基础模型和该向量，评分是隐藏答案似然。这里有固定基座与小提交物的结构，但它是本地限制，不是原版完整微调的共同保证。更不能把这个小向量等同于 LoRA，二者参数化不同。

例如 GSM8K 和 AIME 使用同一基础模型时，只读基础权重有共享机会。两个候选的已训练向量仍必须分别加载干净。把上一候选留下的可变参数当成共同基座，会污染下一候选。若要建立常驻 verifier，还须保持隐藏数据和候选隔离，并拿“固定基座缓存加完整候选重置”作强对照。

**MLGym。** 本地 [MNLI](../../../examples/mlgym_bench/nlp.py#L36) 冻结 BERT 编码器、只更新分类头，类似固定基座和可变头。其他本地视觉、FineWeb、RL 任务提交更完整的模型参数，不能都按这个结构处理。每个新候选的参数、优化器和随机数状态都属于自己的实验，不能混用上一候选的可变对象。数据文件、依赖及兼容编译结果可以另行保留。

**AIRS。** 19 个非 Kaggle 任务交付 CSV。当前 [独立评分](../../../examples/airs_bench/grading.py#L64) 读取预测和隐藏标签，并不接管 Agent 训练模型。值得检查的 GPU 工作在 Agent 的数据处理、训练或预测阶段，不能把 19 个任务说成 19 种需要驻留的 GPU 评分模型。现接入的短基线也不能代表真实研究型 Agent 的训练规模和迭代分布。

**RE 聊天训练。** 原始 [rl_finetune.py](../../../../benchmark-sources/RE-Bench/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L79) 确实在同一进程中生成回答、等待偏好裁判，再更新模型。等待时保留 GPU 状态是真实机制。但如果要让卡，必须处理活跃张量、优化器、梯度缩放器和随机数状态，只有最终模型文件不够。原起始程序末尾的 [model.save](../../../../benchmark-sources/RE-Bench/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L151) 并不是通用的中途恢复协议。

历史小样本 [本地裁判结果](../experiments/gpu-benchmarks/re-bench-local-qwen.json) 中，一次等待约 0.824 秒，前后仍分配约 6.01 GiB GPU 张量。这证明有保留状态的等待，却不支持假定存在足够长、值得卸载恢复的空窗。它也不是完整 Adam 状态达到稳定后的测量。本次不因想做 offload 而人为延长等待。

## PostTrain 评分准备的真实计时

沿原任务取得两份已训练候选，不新增训练，各使用原有四条隐藏输入，按 A/B/A 比较每次新进程、同进程重载、同进程保留基座。三组全部执行相同评分实现与内容核对。九次评分的实际候选参数和对应分数一致，最终容器归零、显存回到 0 MiB。

三请求容器完整耗时分别为 274.27、91.96、113.84 秒。第一次新进程启动明显更慢，单轮固定顺序不能当成稳定倍率。保留基座的后两次完整请求分别为 4.36、4.53 秒，对照重载为 9.58、9.42 秒，说明重复加载和传卡可以省掉。每次完整内容核对仍需约四秒，已计入时间。整批没有胜过普通进程复用，因此还不能把模型缓存池接入生产后宣称收益。

逐请求扫描约 3.45 GB 是实验审计方式，不是正式请求必须承担的版本管理方案，未加入生产 EnvRouter。两组内容身份阶段三次合计仅差约 0.314 秒，不能解释完整时长的 21.88 秒差距。应保留小候选的合法性检查和完整替换，并在内容固定的版本发布后减少大基座重复扫描，详见[检查的价值与代价](../experiments/gpu-benchmarks/posttrain-verifier-reuse.md#检查的价值与代价)。

原理、三种执行方式、逐阶段计时、CPU 教学实验及完整证据见 [PostTrain 评分准备复用](../experiments/gpu-benchmarks/posttrain-verifier-reuse.md)。本次是受限本地协议的控制重放，不是自主 Agent 多候选轨迹，也不是原版完整微调或 EnvRouter 排队实验。

## 现有交互记录能证明多少

静态代码回答“机制是否允许”，真实轨迹才回答“实际发生多频繁”。二者不能互相替代。

指定三组历史训练验收目录中的 74 条 MLGym / PostTrain 交互记录已做只读审计：60 条含功能验收运行说明，59 条只有一次受管程序启动。4 条观察到多次成功作业，但只是同一起始训练程序重复运行，或同一程序的训练与公开评估，未观察到成功修改算法后再训练新候选。编辑命令检测器唯一命中的样本，是向只读 `/opt/task` 写入失败，不能算一次有效研究循环。该检测器是启发式，不是任意写文件的完备分析。

因此这些记录支持任务接线和执行有效，**不支持估计自然研究 Agent 的模型复用率、编辑失效概率或缓存容量竞争分布**。SWE 的有界搜索找到 3 条 score 策略轨迹，实际提示明确要求运行一次公开参考补丁，不继续做实验，也没有提供自然多次编辑样本。这不等于真实 SWE Agent 不会多轮调试。

详细目录范围、轨迹身份、统计方法及限制见 [training-state-audit.json](../env-service-verification/results/benchmark-state-audit-20261113/training-state-audit.json) 和 [trajectory-search.json](../env-service-verification/results/benchmark-state-audit-20261113/trajectory-search.json)。原先脚本中的 24 项命令和 504 项总检查，均不能补足这个证据缺口。

## Results：真实入口的旧字节码问题与修复

**可复现的错误不只存在于常驻进程。** Python 的 `.pyc` 是源码编译成的字节码缓存。普通时间戳模式可以用修改时间和源文件大小判断是否继续采用缓存。两版源码若恰好同大小、修改时间落在相同时间戳粒度内，旧缓存可能仍被判为有效。[PEP 552](https://peps.python.org/pep-0552/) 说明了时间戳和内容哈希两类失效机制。

本次使用真实 `restricted_mlm_verify.load` 做 CPU 故障注入，未执行 GPU 模型。

1. 创建一个候选模块，内容含 `VALUE = 1`，生成对应字节码缓存。
2. 把源码改为同样大小的 `VALUE = 2`，保留相同修改时间。
3. 用真实 verifier 加载函数重新建立 module 对象，读到的仍是 1。
4. 换成全新 `python -B` 子进程，仍读到 1。`-B` 禁止写入字节码，不等于拒绝读取已有缓存，见 [Python 官方说明](https://docs.python.org/3.12/using/cmdline.html#cmdoption-B)。

这比“多新开几个进程”更具体。新 module 对象、新 Python 进程和直接读取当前源码字节，是三种不同保证。

修复采用一次读取源码字节，直接编译并在带有正常模块元数据的 namespace 中执行，绕过候选入口的 `.pyc`。覆盖三个真实 RE 候选入口：`restricted_mlm_verify.py`、`cumsum_verify.py` 和 `rust_verify.py`。它们仍被原准备器单独复制到评分容器，没有新增找不到的仓库导入依赖。

原始复现见 [candidate-source-before.json](../env-service-verification/results/benchmark-state-audit-20261113/candidate-source-before.json)。[修复前后对照](../env-service-verification/results/benchmark-state-audit-20261113/candidate-source-comparison.json) 使用两种缓存，各经同进程、新进程和新进程 `-B` 加载，旧实现 6/6 读取旧值 1，修复后 6/6 读取当前源码值 2。[10 项回归](../../../tests/examples/gpu_benchmarks/test_candidate_source_versions.py) 全部通过，另覆盖编码、模块路径及新语法错误不被旧缓存掩盖，见 [validation](../env-service-verification/results/benchmark-state-audit-20261113/candidate-source-validation.json)。cumsum 和 Rust 回归在执行真实加载语句后用候选异常终止，未执行 GPU 评分主体。

**修复边界**：这是受控故障复现和定向修复，尚未从自然 Agent 轨迹中观测到该错评分事件。它不负责候选模块内部普通 import 的所有传递依赖，也不会使一个已有 GPU 服务热更新。若另一个进程边写文件边让 verifier 读取，仍需要生产者使用完整提交协议。本次不把这几处修复宣传成“所有代码和数据版本问题已经解决”。

## 为什么文件路径不是数据版本

原始 PostTrain 多次评估允许出现下面的机制问题。以下是根据真实接口展开的说明例子，不是已经在自然日志中观测到的故障。

```text
提交评估请求：“请评 final_model 里的模型 A”
    ↓ 请求排队
训练程序把同一个 final_model 路径更新为模型 B
    ↓ 评估器开始读取
若请求只记了路径，实际读到 B，却可能仍把结果解释成 A
```

正确的做法是让评估请求绑定一份确定的提交物，记录源码或补丁、权重、配置、评估输入与协议身份。内容指纹指依据实际字节计算出的标识，不是文件名或修改时间。大模型可用完整文件清单与已验证的分片摘要，但产生快照时仍需保证各分片属于同一次完整提交。

这里还有一个容易误解的地方：**A 的结果晚于 B 返回，不自动表示 A 的结果没用。** 如果评估确实运行在 A 上，就应保留为 A 的有效历史结果，Agent 可能正要比较 A 和 B。只有把 A 的结果标成 B，或调用方明确只接受当前最新版本，却收到 A，才是版本契约不满足。不能为了“新鲜度”随意丢弃所有旧实验分数。

状态可以分别处理：只读基座按自身身份共享，每份候选权重固定其版本，每次评估记录其输入，服务进程则在实现或配置变化时重建。不需要因此让所有文件一起清空，也不能因此保留所有旧对象。

现有独立 [Workspace 原型](../knowledge/baseline-state-lifetimes.md#状态版本契约与受控验证) 已演示合作写入、快照及按需拒绝过期结果，但只管理声明过的少量文件。生产 Env 尚无任意目录的自动一致快照，也没有通用 checkpoint 身份和后台服务代次控制。对大分片权重，仅在读完后算一个 hash，不能修复读取期间多个版本混合的问题。

## 哪些研究方向保留哪些应停止

| 方向 | 当前判断 | 依据和下一步证据 |
|---|---|---|
| 四会话、轻量 scorer 编辑、3/5 GiB 转换缓存竞争 | 停止作为真实场景和论文主证据 | 原型刻意固定了昂贵模型实现，SWE 实际修复不满足这个保证，训练类变化对象也不同 |
| 按整个源码目录变化清空全部缓存 | 不做通用策略 | 只读模型与原生内容缓存可能继续有效，已有 Triton 受控实验已表明无关编辑可继续命中 |
| 跨版本保留任意 SGLang Engine | 不实施 | 修改涉及已导入方法、请求结构和子进程，缺少完整依赖与重建协议 |
| 候选源码直接入口读取正确性 | 已做定向修复 | 有真实入口的确定故障机制及 CPU 回归，不依赖性能收益才能成立 |
| checkpoint 版本绑定的评估工作 | 保留为最接近真实训练循环的候选 | 原版 PostTrain 已允许多轮训练评估，但当前本地轨迹尚无足够自然循环，先核对真实发生次数和耗时组成 |
| 固定基座、可变小参数的评估器复用 | 本地 PostTrain 已实测局部收益，短批次未胜过普通进程复用 | 后两次请求约 9.5 → 4.4 秒，完整批次 91.96 → 113.84 秒。保留强简单对照，不宣布论文创新 |
| RE 裁判等待时卸载状态、交还 GPU | 真实安全点候选，暂不扩大实现 | 已有等待实测很短，完整状态恢复尚未实现，先证明等待长于搬运和交接成本 |

若继续寻找论文场景，优先选择原版 PostTrain 的真实候选生成与评估循环，而不是给原型再添加几种人为状态。先让 Agent 按题目原接口产生数个真实候选，固定版本和资源预算，记录训练、评估、准备及等待之间的依赖关系。只有看到足够多不能被普通文件缓存、合法模型常驻或全局队列消除的重复工作，再设计联合调度。

已有状态复用与预测换卡的收益应分别判断。后者的[逐 benchmark 适用性核查](../knowledge/command-predictability.md#11-当前-benchmark-中哪些值得使用这套机制)区分 PostTrain 衍生固定步数长等待的历史 GPU 结果、MLGym / RE 等尚待验证的候选，以及短命令提前返回与整组完成时间的区别。新增预测门控尚未实机复测，不能据旧实验宣称自然 Agent 轨迹已有同等收益。

这不是已经证明该方向可以发论文。当前交付是正确性修复、场景否证和本地候选评分的准备成本证据。局部复用已经可行，完整短批次仍未胜过普通进程复用。若真实循环中每份 checkpoint 只评一次、计算占主导且准备成本小，就结束这条复用方向，不把“没有收益”解释成还需要更复杂算法。

## 相邻研究已经覆盖什么

本次定向查阅的目的，是排除明显重合，不能据此保证不存在其他近邻工作。

| 已有工作 | 已覆盖的机制 | 对本项目的要求 |
|---|---|---|
| [FaasCache，ASPLOS 2021](https://afuerst.github.io/assets/FaasCache.pdf) | 将保温执行环境视作有成本、有大小的缓存对象，选择保留和淘汰 | 把准备时间除以大小、按频率保留进程，不能直接当新贡献 |
| [SEUSS，EuroSys 2020](https://www.cs.bu.edu/~jappavoo/Resources/Papers/seuss.pdf) | 用执行状态快照避开重复初始化，并维护隔离 | 拆分固定运行时与后续函数执行不是新颖性证明，GPU 活状态仍另有难点 |
| [S-LoRA，MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html) | 固定基础模型、多适配器、CPU 与 GPU 间按需放置 | 共享基座和切换少量参数已有成熟近邻，不能把本地 norm 向量换个名字当算法创新 |
| [Tune，2018](https://arxiv.org/abs/1807.05118) | 为训练脚本与搜索算法提供分布式模型选择接口 | 发现多次训练、保存恢复和候选评估，并不足以构成新问题 |
| [Bazel 内容缓存](https://bazel.build/remote/caching) | 用输入和动作身份复用输出 | 版本号和内容指纹是正确性基础，不是单独的系统论文贡献 |

可能的剩余差异需要由真实工作负载证明：Agent 的代码与训练状态不断变化，执行依赖在线产生，准备的有效期和恢复成本不固定，同时要保持每次反馈对应正确提交。在这些条件真实出现并超过上述强对照前，不应把它写成已成立的新方向。

质量评分也不意味着调度可以任意改变实验预算。原版 PostTrain 给定墙钟预算，排队可能减少训练或研究次数。研究纯执行效率时先固定候选、输入和训练工作量，比较全部完成时间及输出一致性。研究真实 Agent 效益时再固定总预算，报告完成的候选数和最终质量，明确二者不是同一个对照。

## 跟着图读源码与复现

1. 图 1：读 [initialize / run 的任务入口](../../../examples/gpu_benchmarks/agent_loop.py#L32)，它把完整任务委托给 Harbor，不会把每个断言变成一次新 rollout。
2. 图 2：读 [SWE 提交与重放](../../../examples/swe_serve/patch_submission.py#L5)，区分 Agent 当前可写源码与 verifier 收到的补丁。
3. 图 3：读 [EnvWorker.evaluate](../../../psrl/workers/env_worker/worker.py#L456)，AIRS 在启动评分子环境前复制 CSV 字节，之后 Agent 文件变化不会自动改写这一份已复制字节。
4. 图 4：读 [PostTrain.execute](../../../examples/posttrain_bench/local_model.py#L9) 的新建模型和加载提交分支。读 [score 原测试](../../../../benchmark-sources/swe-serve/tasks/ss-sgl22544-score-multi-item-delimiters/tests/postmerge_tests/test/registered/prefill_only/test_multi_item_scoring.py#L67)，确认组内已有复用。
5. 图 4 的候选源码边界：读 [restricted_mlm_verify.load](../../../examples/re_bench/restricted_mlm_verify.py#L10)，本次改变的是如何取得候选程序的执行代码，不是 GPU 运算或分数公式。
6. 图 5：读 [runner finally](../../../examples/sciaccel_rl/runner.py#L610)。Agent 与测试子进程有并发或分组关系，上述步骤表示依赖，不表示所有角色在一个线程中顺序执行。

标准库静态审计可以从仓库根目录运行，不需要 Torch、Docker 或 GPU。`--swe-root` 指向已有固定版本的 SWE-Serve checkout，输出使用自己的文件路径，不覆盖历史 GPU 证据。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/swe_state_audit.py \
  --swe-root ../benchmark-sources/swe-serve --output /tmp/psrl-swe-state-audit.json
```

这个脚本是实际源码审计，不是模拟 benchmark 执行。预期在上述固定提交得到 53 题、41 个 GPU 声明和 41 份运行时修改参考补丁。语法扫描不等于运行时行为，其他提交的数字不保证相同。

实际候选源码入口的标准库 CPU 故障实验也可独立运行。`--before` 是本次保存的修改前 verifier 原文，输出同时记录旧实现与当前实现读取到的值。

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/candidate_source_version_probe.py \
  --before docs/env-worker-deep-dive/env-service-verification/results/benchmark-state-audit-20261113/restricted-mlm-before.py.txt \
  --output /tmp/psrl-candidate-source-comparison.json
python -m pytest -q tests/examples/gpu_benchmarks/test_candidate_source_versions.py
```

故障脚本确实执行实际加载函数和真实 Python 子进程，只用临时小文件，不模拟大型模型或性能。它应显示旧实现读取 1、修复后读取 2。pytest 则另外检查三个入口，已实际执行通过，不是完整 benchmark 或训练验收。

## 验证边界和停止条件

证据必须分别解释：本次源码审计、历史轨迹审计、实际 Python 加载入口故障注入、CPU 回归和历史 GPU 结果不是同一层验证。没有给上述检查补上未经运行的任务分数，也没有将固定旧 baseline 换成当前代码关闭 EnvRouter。

固定 baseline 仍为 `0eeead861ffafed6bf31c962d259fd847775dcf9`。这三个本地 RE recipe 是当前工作区的新增接入，不属于该固定 baseline。本次故障实验不证明该提交的原始 science 任务曾错评分。需用现有准备器重新生成相应任务资产，已经存在的 `/tests` 副本不会自动更新。当前改动在主目录，未提交、未推送、未合并或部署。

验证已覆盖原假设的任务适用性、候选源码读取故障、CPU 回归，以及本地 PostTrain 准备复用。九次 GPU 评分和物理回收完成后停止，没有追加训练、延迟注入或缩小缓存配额。自然研究轨迹及调度策略增量仍未验证。

三个入口、对应测试和两个审计脚本的定向 Ruff、格式及适用的 prose 检查通过。全库 `pre-commit run --all-files` 在一次性隔离快照运行，仍因 `docs/conf.py`、历史实验脚本及范围外注释等问题失败，自动修改已随快照丢弃，没有写回主目录。文档链接检查和验证范围见 [validation](../env-service-verification/results/benchmark-state-audit-20261113/benchmark-state-validation.json)。
