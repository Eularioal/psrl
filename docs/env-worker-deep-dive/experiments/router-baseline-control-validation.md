# EnvRouter 修复后的低负载固定基线对照

> **核心结论**：应引用同步准备与连接复用修复之后的 OFF/ON/ON/OFF 同期配对。32 项任务和 64 条 CUDA 命令全部成功。工作阶段合计 ON 少 6.31%，包含初始化和收尾的总时间多 1.49%，两对方向不同，尚未证明稳定提速。修复前 40.37/55.83 秒已移入历史记录，不再表示当前开销。

## 目录

- [条件与版本](#条件与版本)
- [Results](#results)
- [已修复问题与证据](#已修复问题与证据)
- [复核与边界](#复核与边界)

## 条件与版本

这里的OFF运行固定旧版Docker路径，ON运行Env资源调度路径。每项任务都按“CPU准备 → GPU计算 → CPU等待 → GPU计算”执行，CUDA指NVIDIA的GPU计算平台。低负载时两组容量都足够，问题主要是调度与环境准备多花多少时间。

实验日期 2026-10-20，2026-11-11 只读复核。OFF 为 `0eeead861ffafed6bf31c962d259fd847775dcf9` 加已披露运行修复，没有整卡锁。ON 为 `37a06ea` 加同步准备优化。每组 8 项任务、4 并发，每任务两段真实 CUDA 工作，包含中间 CPU 等待。两对顺序为 OFF、ON、ON、OFF。没有模型生成或 PSRL 策略更新，不把人工保持显存视为持续满负荷计算。

## Results

| 计时口径 | OFF | ON |
|---|---:|---:|
| 第一对工作阶段 | 89.46 秒 | 77.83 秒 |
| 第二对工作阶段 | 82.23 秒 | 83.02 秒 |
| 两对工作阶段合计 | 171.68 秒 | 160.84 秒 |
| 两对驱动总时间，含初始化及收尾 | 213.96 秒 | 217.15 秒 |
| 单任务平均延迟 | 41.741 秒 | 39.045 秒 |
| 单 GPU 命令平均调用延迟 | 3.827 秒 | 3.718 秒 |

原始 [paired-analysis.json](../env-service-verification/results/router-paired-review-20261020/paired-analysis.json) 保留每组和每条样本。只有两对，组内任务共享运行环境，不等于 16 次独立重复。工作阶段含容器准备、执行、评分和任务清理，驱动总时间还含初始化及最终收尾，不含 Python 导入前的进程启动。

## 已修复问题与证据

早期低负载开销先后涉及同步 Docker 查询、连接重复创建和同步准备阻塞事件循环。当前 runner 已使用异步准备入口，具体机制及 CPU 回归见[同步准备修复档案](history/router-baseline-control-validation-chronology.md#13-results同步准备路径移出事件循环)。后续配对比早期单次测量更适合作为低负载参考，不能把旧高开销继续当作未修问题。

早期高压力合成任务 22.02% 的工作阶段收益仍是其版本条件下的记录，但总体性能判断优先引用[三步真实训练链压力对照](science-single-gpu-training-control.md)与[常驻模型对照](gpu-benchmarks/FINDINGS.md#episode-记账修复后的整卡压力对照)。新结果不是把所有旧工作负载重测了一遍，不能改写旧样本。

## 复核与边界

[配对分析脚本](../env-service-verification/scripts/analyze_paired_router.py)读取已完成结果。[详细配对条件](router-paired-latency-and-design-review.md#2-results低负载同期配对)保留配置和延迟口径。完整排查过程在[历史档案](history/router-baseline-control-validation-chronology.md)，已从本文移除重复的失败、尝试与跨轮参考表。

本次核对 JSON 与代码，未重跑 GPU。低负载没有触发超额排队，不能用它证明高压力保护。当前完整版本在所有自然任务上的延迟仍需单列验收，也不能把约 1% 的差异当作固定协议成本。
