# GPU 准入 / 归属 / 恢复新设计：eBPF 身份识别与两个新协程模式

## TL;DR

- psrl 仓库最近 3 个提交（`feat: strengthen environment admission, attribution and recovery` 等）给 `env_service` 加了一组新模块：用 **eBPF**（而不是读 `/proc`）在内核态确认"这个 GPU 进程到底属于哪个任务"，用一条**显存会计公式**把"已用显存"和"还没兑现的显存预约"合并成一个不会重复计费的准入判断，用**容器预算**把一个任务的 CPU/内存需求转成 docker-compose 的硬限制，用一个**健康状态机**在节点失联时判断能不能安全重建。
- 核心动机一句话：**当"运行 env_service 的节点本身也是一个容器"时，容器看到的 `/proc` 只是它自己的命名空间视角，不能保证里面的 PID 8 就是宿主上真正该被记账的那个进程；eBPF 方案绕开这个视角问题，直接在内核里追踪进程的真实身份和它所属的 cgroup。**
- 本文只讲这组改动里**新出现**的知识点（eBPF/BTF/raw_tracepoint/libbpf FFI/cgroup v1 devices 控制器桥接技巧、两个新的 asyncio 清理模式），协程/线程/进程的基础三层模型已经在 并发/并行全景（本仓库尚未收录） 讲过，这里只做必要的回顾和延伸，不重复推导。
- psrl 仓库自带的 [`process-identity-without-procfs.md`](process-identity-without-procfs.md)、[`gpu-memory-admission.md`](gpu-memory-admission.md) 已经把这组设计的背景和取舍写得很细，本文是以"零基础读者"为口径的补充讲解，侧重把那两篇文档里**假设读者已经懂**的术语（eBPF 本身是什么、BTF 是什么、为什么需要 FFI）展开讲清楚，并把两份文档串成一条可以顺着读的线。

## 专题导航

已拆分为 2 篇子文档。按完整章节迁移，未重新运行历史实验。

- [1. 先还原一个具体问题：同一个容器里，这笔显存该记给谁](gpu-admission-attribution-ebpf-identity/identity-and-accounting.md)
- [8.3 新模式 B：多持有者引用计数 + 取消安全清理（routing_pause.py）](gpu-admission-attribution-ebpf-identity/cancellation-and-experiment.md)
