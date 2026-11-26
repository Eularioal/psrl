# 高速传输：NIXL内存搬运机制

NIXL（NVIDIA Inference Xfer Library，面向推理系统的内存传输库）负责把模型权重、KV 缓存等数据从一个进程搬到另一个进程。PSRL 用它连接分开的训练服务和生成服务：训练更新权重，生成服务取得新权重后继续回答问题。

## 目录

- [为什么需要专门的数据通道](#为什么需要专门的数据通道)
- [一次权重更新怎样传过去](#一次权重更新怎样传过去)
- [KV 缓存为什么也需要搬运](#kv-缓存为什么也需要搬运)
- [部署时需要什么](#部署时需要什么)
- [进一步阅读](#进一步阅读)

## 为什么需要专门的数据通道

假设训练进程得到 4 GB 新权重，要交给两个生成进程。如果把这 4 GB 当作普通远程函数参数，系统还要处理对象序列化、传递和接收。NIXL 让应用先登记可访问的内存，再直接读写这些内存区域。

<!-- figure:registered-memory-transfer -->
这张图沿同一个场景解释：发送函数返回后，为什么接收端还不能立刻使用新数据？

[![大块内存传输：先登记缓冲区，再确认搬运完成（1/2）](../_static/img/document-figures/registered-memory-transfer-screen.png)](../_static/img/document-figures/registered-memory-transfer.png)

[![大块内存传输：先登记缓冲区，再确认搬运完成（2/2）](../_static/img/document-figures/registered-memory-transfer-screen-02.png)](../_static/img/document-figures/registered-memory-transfer.png)

一个训练缓冲区向生成侧缓冲区传输参数。传输可以异步发起。 控制通知与数据完成是两个时刻。异步返回节省等待，但需要显式维护缓冲区生命周期。 生命周期教学。NIXL 采用的 RDMA、IPC 或其他后端由配置决定，不视为固定路径。 [查看矢量图](../_static/img/document-figures/registered-memory-transfer.svg)。
<!-- /figure:registered-memory-transfer -->

这里有两类消息：**控制消息**说明“版本 12 已准备好”；**数据传输**搬运真正的权重字节。小消息可以交给 Ray（分布式任务框架），大数据通过专门通道传递，方便分别优化。

底层使用哪条通道取决于硬件、后端和配置：同节点可能利用共享内存或 GPU 间互访，跨节点可能利用 RDMA。RDMA 是让网卡直接访问已登记内存的传输方式，可以减少主机 CPU 参与的数据复制。实际吞吐还受网络、内存带宽和两端负载影响。

## 一次权重更新怎样传过去

在 `psrl.ps_mode=nixl_cpu` 模式下，Parameter Server（参数服务器，简称 PS）在固定页的 CPU 内存中保存权重；“固定页”使这块内存适合设备直接传输。

```text
训练 GPU → PS Worker 的 CPU 权重缓冲区 → 生成 GPU
                  ↑
             版本与就绪通知
```

以前面的教学数字为例：4 GB 权重分为两个 2 GB 分片。训练侧将对应分片写入 PS Worker，完成后更新版本；生成侧各个 rank（参与并行计算的进程）读取自己所需的分片，再交给模型使用。协调器负责版本与位置，权重字节通过数据通道传递。

这种分工减少了大对象经过协调进程的开销。权重复制仍消耗带宽、缓冲区和执行时间，异步训练也仍需控制版本差距；这些成本应结合具体部署测量。

机制与源码入口见[参数服务器](../design/parameter_server.md)和 [NIXL 客户端的内存登记](../../psrl/utils/nixl/client.py#L301)。

## KV 缓存为什么也需要搬运

KV 缓存保存模型已经处理过的历史 token 的注意力中间结果。token 是模型处理文字的基本片段；保留缓存后，下一次生成可以接着计算。

例如请求已经处理了 20,000 个 token，需要从繁忙实例 A 转到空闲实例 B。B 可以重新计算这 20,000 个 token，也可以接收 A 已算好的缓存。迁移缓存能省下重复计算，但也要付传输和协调成本；前缀很短时，重算可能更合适。

PSRL 的 KV 迁移设计通过 LMCache 管理缓存、通过传输通道搬运数据。是否迁移、目的实例和等待方式由上层策略决定，详见 [KV 缓存设计](../design/kv_cache.md)。LMCache 的 NIXL 通道与 TCP 通道是配置选择，部署时应核实所选通道及其依赖。

## 部署时需要什么

仓库的[安装脚本](../../scripts/install_nixl.sh)指定 NIXL `v1.2.0`，并检查 UCX 至少为 `1.20.0`。UCX 是底层通信库，用来对接共享内存、网络等传输能力。这是脚本的版本约定；安装前还要核对当前驱动、CUDA 和网络环境。

PSRL 也提供 `cpu`、`cpu_ref` 等参数服务器模式。选择哪种模式，应先看部署依赖、权重规模和同步开销，再看是否需要 NIXL。

核对范围：2026-11-16 阅读本地安装脚本、NIXL 客户端及参数服务器设计；本次未执行安装或传输实验。

## 进一步阅读

- 想知道“训练更新后，生成进程什么时候切换版本”：读[参数服务器](../design/parameter_server.md)。
- 想知道“换生成实例时怎样接着回答”：读 [KV 缓存设计](../design/kv_cache.md)。
- 想从整条训练链路定位代码：读[训练脚本与源码地图](07-源码阅读_从Science脚本到训练流程.md)。
- 库本身的接口和后端支持以 [NIXL 官方仓库](https://github.com/ai-dynamo/nixl)为准。
