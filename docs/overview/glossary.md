# PSRL 系统与实验术语速查

> **核心结论**：这里按需查询系统机制、项目协议和实验口径，默认已有机器学习与 agent 基础。想先串起组件和数据流，读[从一道题理解 PSRL](reading-guide.md)。

## 目录

- [项目中的训练约定](#项目中的训练约定)
- [数据与版本](#数据与版本)
- [运行环境与资源](#运行环境与资源)
- [实验结果怎样读](#实验结果怎样读)

## 项目中的训练约定

| 术语 | 通俗解释与例子 | 想深入时读 |
|---|---|---|
| 任务内训练与策略训练 | GPU 题目可能要求 agent 训练一个分类器；PSRL 另外训练负责操作的语言模型。两者使用的模型、奖励和参数更新分别记录 | [接入与验收](../env-worker-deep-dive/milestones/science-gpu-full-onboarding.md#用数字分类题串起两种训练) |
| 优化器步骤与版本发布 | 一次优化器调用完成后可发布策略版本；参数是否有非零变化、任务表现是否提升，需要各自检查 | [策略闭环结果](../env-worker-deep-dive/milestones/science-gpu-full-onboarding.md#真实-gpu-与策略闭环结果) |

## 数据与版本

| 术语 | 通俗解释与例子 | 想深入时读 |
|---|---|---|
| Batch，批次 | 一次训练更新集中处理的一组样本。例如两道题各生成两次，得到四条轨迹 | [批次与版本](../design/staleness_control.md) |
| Group，回答组 | 同一道题生成的多条回答。GRPO 用这个分组比较表现，不能与整批题目混为一谈 | [组采样](../design/staleness_control.md#group-sampling) |
| 模型版本 | 参数更新后的编号，用来判断样本和实例使用了哪版参数 | [参数服务器](../design/parameter_server.md) |
| Staleness，陈旧度 | 使用样本时，训练版本比生成起始版本领先多少步。例如 `13 - 10 = 3` | [版本约束](../design/staleness_control.md#what-is-staleness) |
| PS，参数服务器 | 接收训练端的新参数，协调生成端获取。控制信息与参数大数组可以走不同路径 | [参数服务器](../design/parameter_server.md) |
| TransferQueue | 按样本键保存训练字段的存储层。各组件传样本键，再按需读取回答、分数等数组 | [样本数据流](../design/transfer_queue.md) |
| TITO | 路由层记录真实 token 和每轮生成信息的机制，便于把外部 Agent 对话转换成训练数据 | [TITO 工作过程](../design/router_tito.md#tito) |
| KV cache，注意力中间结果缓存 | 保存模型处理已有前缀时计算出的键和值，后续相同前缀可复用部分计算。它不是模型权重缓存 | [KV 缓存管理](../design/kv_cache.md) |

## 运行环境与资源

| 术语 | 通俗解释与例子 | 想深入时读 |
|---|---|---|
| 进程 | 程序的一次运行实例，有自己的内存地址空间，也可以启动子进程。前台命令结束后，子进程或后台服务可能继续占用 GPU | [命令与常驻服务](../env-worker-deep-dive/knowledge/current-design.md) |
| 并发与并行 | 并发表示多项工作能交替或重叠推进，并行表示同一时刻实际执行多项工作。等待一个工具回复时处理另一任务，可以提高并发度；能否同时计算还取决于硬件与调度 | [CPU 与 GPU 的并行差异](../env-worker-deep-dive/knowledge/cpu-gpu-parallelism-architecture-comparison.md) |
| Worker，工作进程或 Actor | 承担某一类工作的执行单元。名字含 Worker 不表示它必然占用一整张 GPU | [系统架构](../design/architecture.md) |
| Router，路由器 | 收到请求后选择执行位置。例如把生成请求送到可服务正确参数版本的实例 | [路由入口](../design/router_tito.md) |
| Ray Actor | 由 Ray 把 Python 对象放在工作进程中运行。远程方法调用把参数交给该进程，对象里的队列、计数或模型可跨调用保留 | [源码阅读入口](reading-guide.md#跟着图读源码) |
| Sandbox，沙箱 | 给任务执行命令、读写文件的隔离环境。本项目常用容器实现，跨轮次可保留文件和进程状态 | [环境当前设计](../env-worker-deep-dive/knowledge/current-design.md) |
| Verifier，评分器或验证器 | 检查 Agent 产物是否满足题目要求的程序；可能独立启动容器或使用 GPU | [接入与验收说明](../env-worker-deep-dive/milestones/science-gpu-full-onboarding.md) |
| 镜像、容器、挂载 | 镜像提供初始文件，容器是运行实例，挂载把指定目录接进容器。只读挂载让评分输入保持稳定 | [任务接入与环境隔离](../env-worker-deep-dive/milestones/science-gpu-full-onboarding.md) |
| CPU 内存与 GPU 显存 | 分别是主机和 GPU 的工作存储。模型从磁盘加载后成为内存中的张量，迁到 GPU 又占用显存；文件只读仍允许更新已加载张量，保存结果要写到可写路径 | [只读模型怎样训练](../env-worker-deep-dive/milestones/science-gpu-full-onboarding.md#例子一posttrain-怎样训练只读模型) |
| Admission，准入 | 启动前检查资源预算、版本和并发条件是否满足。准入账本与实际硬件隔离的作用不同 | [命令级资源申请](../env-worker-deep-dive/knowledge/command-gpu-reservations.md) |
| Lease，租约 | 带所有者及有效期的临时资源使用凭据。到期后仍需执行回收并确认实际资源释放 | [环境回收机制](../env-worker-deep-dive/knowledge/router-configuration-and-recovery.md) |
| MPS | NVIDIA 的多进程 GPU 执行服务，用于让多个进程并行使用 GPU。性能取决于工作负载的竞争情况 | [MPS 并行执行](../env-worker-deep-dive/knowledge/mps-parallel-execution.md) |
| Warm / cold，热 / 冷状态 | 热状态已保留可复用内容，冷状态需要重新准备。例如 A 卡已有模型，B 卡需要加载 | [等待还是换卡](../env-worker-deep-dive/experiments/gpu-benchmarks/warm-placement-matrix.md) |
| Offload，卸载 | 把暂不用的数据或状态移出当前设备或释放可重建空间，以减少驻留占用 | [资源生命周期](../design/resource_elasticity.md) |

## 实验结果怎样读

| 术语 | 它在报告里回答什么 | 阅读时看哪里 |
|---|---|---|
| Baseline，基线 | 新方案要与哪个旧方案或简单方案比较 | 实际提交、补丁、配置和资源约束 |
| GPU quick check，GPU 快检 | 指定路径能否在 GPU 上完成一次受控计算 | 运行的是任务、评分器，还是仅 CUDA 小算子 |
| Oracle，参考解 | 按已知方法能否得到合格产物 | 证明评分链可达的程度，与模型自行解题能力分开看 |
| 闭环 | 哪些环节已经接起来 | 是产物到评分，还是 Agent 生成、执行、奖励、参数更新整条链路 |
| 吞吐与延迟 | 一段时间完成多少工作、单个请求花多久 | 相同工作量、相同计时起止点、是否包含初始化和清理 |
| Paired，对照配对 | 同一任务或工作量在两个方案下各执行一次 | 配对是否齐全，完成的配对结果如何，未完成部分有什么原因 |

不熟悉某个术语时，可以继续问“它在这次任务里具体做了什么”或“用一个能手算的例子解释”。深入链接用于继续探索，阅读一篇报告的主要结论应当不依赖把所有链接全部读完。
