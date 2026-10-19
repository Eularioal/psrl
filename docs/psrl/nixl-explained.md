# NIXL 是什么

## 一句话结论

NIXL（**NVIDIA Inference Xfer Library**）是 NVIDIA Dynamo 项目下的开源库（[github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)），提供一套统一的点对点内存搬运 API，底层基于 UCX 自动挑选最快的传输方式（同节点共享内存 / GPU IPC / RDMA 网络等），最初为"拆开的推理服务"（prefill/decode 分离、KV cache 跨实例迁移）设计，psrl 用它做训练侧和 rollout 侧之间的权重同步、以及 rollout 实例间的 KV cache 迁移。

## NIXL 本身解决什么问题

在分布式 GPU 系统里，经常需要把一大块内存（模型权重、KV cache block）从 A 搬到 B，但 A、B 的物理关系不确定：

- 同一张卡内 → 直接指针
- 同节点不同卡 → CUDA IPC / NVLink
- 同节点 CPU↔GPU → pinned memory DMA
- 跨节点 → RDMA（InfiniBand/RoCE）或退化成 TCP

如果每种情况都手写一套传输代码会很繁琐。NIXL 把这些抽象成统一的"注册内存 + 发起 read/write"API，具体走哪种传输由底层 UCX 在建连时根据双端硬件自动选择，调用者不需要关心。

## psrl 里 NIXL 的两处用法

结合仓库 `psrl/docs/design/parameter_server.md` 和 `psrl/docs/design/kv_cache.md`：

### 1. Parameter Server 权重同步（`psrl.ps_mode=nixl_cpu`）

- 权重常驻在 **PS Worker** 的 pinned CPU 内存里（不占 GPU 显存，给 vLLM 的 KV cache 让路）。
- **Push**：训练侧每个 rank 把自己那部分 sharded 权重通过 NIXL write 流式写进对应 PS Worker，写完后只用一次很轻的 Ray RPC 去 bump 版本号（控制面和数据面分离，字节流不经过 PS Manager）。
- **Pull**：rollout 实例的每个 GPU rank 从（优先本地的）PS Worker NIXL read 自己的 shard，同节点走 DMA/共享内存，跨节点走 RDMA。
- 效果：权重同步不占用 Ray 对象序列化/反序列化的 CPU 开销，也不用等一次全量 all-gather，训练和生成两个物理分离的集群可以异步往前跑，互不阻塞。详见 [psrl-run-guide.md](psrl-run-guide.md)。

### 2. KV Cache 跨实例迁移（LMCache 的 `p2p_transfer_channel=nixl`）

- 当某个请求要从一个 rollout 实例迁移到另一个（比如做负载均衡/长尾调度），源实例把 KV cache blocks 通过 NIXL push 到目的实例，走同一套 UCX 传输选择逻辑；没有 NIXL/UCX 环境时会退化成 TCP。

**为什么负载均衡会牵扯到 KV cache 迁移**：LLM 推理请求的代价（生成长度）在到达时未知，且不是纯计算时间，而是"占着一块随生成长度线性增长的 KV cache 显存"直到结束——少数长请求会一直挤占某个实例的显存/batch 空间，拖慢同一实例上其他本该很快的短请求，形成长尾热点。路由层发现某实例过载后，必须能把一个**已经跑了一部分**的请求挪到别的实例，而不能只在请求到达时一次性静态分配。

**为什么迁移必须搬 KV cache，而不是换个实例重新跑**：KV cache 是该请求到目前为止每层每个 head 上所有历史 token 的 K/V 张量，是自回归生成"记忆"的物理体现。不搬 KV cache 直接换实例，新实例就要对已生成的前缀重新做一次完整 prefill——不仅浪费已花掉的算力，长上下文场景下重算前缀的代价（$O(n^2)$ attention）可能比继续生成还大，用户会感知到明显的延迟突刺，等于把这条生成"重启"了一次。直接搬运 KV cache 张量则是把已经算好的数值原样拷贝过去（无精度漂移），目标实例可以从断点无感续写。

迁移的目的可以归纳为：①消除长尾热点、把显存/算力挪给排队的其他请求；②避免重算前缀的高昂代价；③配合权重同步/扩缩容时的拓扑变化，让在飞请求能挪到新实例而不是被打断重启（对应 `kv_cache.md` 里 "sync-triggered re-routing"）。`psrl/docs/design/kv_cache.md` 的 P2P transfer 一节给了具体机制：Router/Coordinator 只决定"要不要搬、搬到哪"，源实例 LMCache worker 通过 NIXL 直接把 KV blocks 推给目标实例的 LMCache worker，Controller 仅在传输前查一次对端地址，不在数据路径上；`transfer_mode` 的 `async`/`sync`/`pin_sync` 三档，本质是在"传输延迟"和"重算前缀延迟"之间按前缀长短选更便宜的一边。

## 为什么要单独装（`scripts/install_nixl.sh`）

NIXL 依赖特定版本的 UCX（脚本要求 ≥1.20.0，会检测已有安装或从源码编译，Mellanox 卡会额外加 `--with-rdmacm --with-mlx5` 之类选项），然后从 `github.com/ai-dynamo/nixl`（脚本锁定 `v1.2.0`）编译安装 C++ 库和 Python 绑定。这是可选依赖——`psrl.ps_mode` 还有不依赖 NIXL 的 `cpu`/`cpu_ref` 兜底模式，生产环境才推荐 `nixl_cpu`。

## 一句话记忆

NIXL 不是训练算法的一部分，是一层"跨进程/跨节点内存搬运"的基础设施；psrl 的异步架构（训练和生成物理分离）之所以能不互相阻塞，靠的就是 NIXL 把权重/KV cache 的搬运做得足够快且不占 GPU 资源。

## 延伸阅读

- `psrl/docs/design/parameter_server.md`（权重同步的完整设计）
- `psrl/docs/design/kv_cache.md`（KV cache 迁移）
- [psrl-run-guide.md](psrl-run-guide.md)
