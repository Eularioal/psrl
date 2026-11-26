# Science训练：三步对照与显存压力

> **核心结论**：固定旧版与显存准入修复后的 ON 都完成 3 步、12 条接受轨迹。人工高压力下，OOM 从 18 次降为 0，三步计时减少 54.24%，含初始化清理总时间减少 27.59%。它取代早期整任务占卡的负面结果作为这一修复方向的有效证据，但不是所有自然任务的通用提速率。
>
> 2026-11-11 只读复核保存的 JSON 和现有准入公式，没有新跑训练。本文移除早期失败和修复前实验的重复叙述，原始条件完整保存在[历史档案](history/06-历史过程_Science显存压力与原生任务.md)。

## 目录

- [Results：三步训练对照](#results三步训练对照)
- [当前引用边界](#当前引用边界)

## Results：三步训练对照


本组已于 2026-10-26 晚完成，2026-10-27 核验结果与现场释放状态。OFF 为用户指定的 `0eeead861ffafed6bf31c962d259fd847775dcf9` 加已披露的必要运行补丁，仍无整卡锁。ON 为 `b52a82d8e6ccd24fc147d7cc355c9db04587d317` 加[显存准入公式修复](../knowledge/02-资源与调度/01-显存准入_实测占用与预约预算.md)，六个生产文件按已验证指纹冻结到实验专用工作树。主目录的其他并行修改未纳入。补丁、指纹和结果已留档，实验工作树暂保留用于查阅复现；两组没有运行进程或 GPU 占用。

<!-- figure:measured-science-three-steps -->
这张图沿同一个场景解释：相同工作量与计时边界是什么？

[![Science三步对照：训练时间与完整墙钟分别看（1/3）](../../_static/img/document-figures/measured-science-three-steps-screen.png)](../../_static/img/document-figures/measured-science-three-steps.png)

[![Science三步对照：训练时间与完整墙钟分别看（2/3）](../../_static/img/document-figures/measured-science-three-steps-screen-02.png)](../../_static/img/document-figures/measured-science-three-steps.png)

[![Science三步对照：训练时间与完整墙钟分别看（3/3）](../../_static/img/document-figures/measured-science-three-steps-screen-03.png)](../../_static/img/document-figures/measured-science-three-steps.png)

重画原报告：三步训练合计1100.80与503.69秒，含初始化和清理的总墙钟1954.11与1415.01秒。两组均接受12条轨迹，OOM尝试分别18与0；数字属于该报告的压力条件和版本，不是本次新运行。 [查看矢量图](../../_static/img/document-figures/measured-science-three-steps.svg)。
<!-- /figure:measured-science-three-steps -->

两组顺序执行 OFF → ON，每组两个 LAPS 修复任务、每题两条轨迹、三次训练更新，模型为 Qwen3.5-4B。训练与推理使用三张卡，环境另用一张卡，持续注入 70 GiB 外部显存占用；每条轨迹的 CUDA 阶段分配 14 GiB、保持 15 秒，ON 按 15 GiB 需求和 90 GiB 单元预算准入。沿用相同重试、两轮代理交互及 4096 回复长度限制。压力是人为注入，不能等同于原生 science 程序的 GPU 需求。

每组初始化上限 900 秒、开始采样后的训练阶段上限 1440 秒。整对监督器上限 5100 秒，外层最终看门狗上限 6000 秒，均留出清理时间，不自动增加训练步数或重跑已完成组。监督器在独立会话运行、标准输入关闭、输出持久化，SSH 或聊天断开不依赖本轮工具继续轮询。正常结束和预算退出均按 PID 出生时间及实验目录归属清理进程、容器与网络，再核验物理显存释放。

结果目录为 [science-resident-three-step-20261026](../env-service-verification/results/science-resident-three-step-20261026)。可先读取 [status.json](../env-service-verification/results/science-resident-three-step-20261026/status.json) 查看阶段；[后台日志](../env-service-verification/results/science-resident-three-step-20261026/background.log) 与各组 `off/launcher.log`、`command/launcher.log` 保存运行输出。监督脚本是结果目录内的 `background_job.py` 和 `run.sh`，包含本次实际路径与预算，不属于生产代码改动。

两组结束后已自动调用现有 [分析脚本](../env-service-verification/scripts/analyze_single_gpu_training.py)，生成 `comparison.json`、`comparison.png`、`per-step.png` 与 `matched-prefix.png`。双方均完成三步、十二条接受轨迹，退出码零，未触发超时，物理资源已释放。以下为本次完整三步结果，一次启动样本不能证明 ON 固有的初始化开销。

| 指标 | 固定 OFF | 修复后 ON | 本次观察 |
|---|---:|---:|---|
| 初始化至首个采样槽开始 | 844.85 秒 | 897.22 秒 | ON 多 52.37 秒 |
| 第一步 | 390.08 秒 | 183.51 秒 | ON 少 206.57 秒 |
| 第二步 | 359.91 秒 | 170.37 秒 | ON 少 189.54 秒 |
| 第三步 | 350.81 秒 | 149.80 秒 | ON 少 201.00 秒 |
| 三步训练计时之和 | 1100.80 秒 | 503.69 秒 | 减少 54.24% |
| 三步等待采样数据之和 | 1044.43 秒 | 443.00 秒 | 减少 57.58% |
| 三步模型更新之和 | 41.67 秒 | 49.57 秒 | ON 多 7.90 秒 |
| 总墙钟，含初始化和清理 | 1954.11 秒 | 1415.01 秒 | 少 539.10 秒，减少 27.59% |
| 完成步数 / 接受轨迹 | 3 / 12 | 3 / 12 | 完成目标相同 |
| 总尝试 / OOM 尝试 | 30 / 18 | 12 / 0 | OFF 每步六次 OOM，ON 无重试 |
| 有效训练回复 token | 7166 | 8852 | ON 多 23.53%，生成长度不完全相同 |

总墙钟是各组驱动开始至清理结束的实测时差，OFF 为 19:40:06–20:12:40，ON 为 20:12:40–20:36:15，时区 UTC+8。两组开始前共用的数据准备和镜像预热不在各组计时里。初始化定义为驱动开始至第一个采样槽开始，包含常驻压力、Ray 与模型准备；它不是 EnvRouter 构造函数独占耗时。训练器 step 计时与采样槽采用不同起止边界，因此初始化与 step 之和不能精确当作总墙钟分解。

用户关心的初始化摊销在本次已经出现：ON 初始化多 52.37 秒，三步训练累计节省 597.11 秒，最终完整运行实测节省 539.10 秒，约 8 分 59 秒。第一步训练计时单独省 206.57 秒，已经大于这次的初始化差额。不能将此外推为所有负载都在一步回本，也不能将单次 52.37 秒视为固定 Router 启动成本。

为什么排队反而更快，可以直接看本次每步的工作量。每步要四条成功轨迹，OFF 每次共尝试十条，其中六条因 OOM（显存不足）失败，重建环境和重新采样后才凑齐四条。ON 每步只尝试四条，全部成功；观测到了四个会话同时存活、一个 GPU 命令运行、三个命令等待。两组成功 CUDA 阶段的并发峰值都是一，ON 的收益是减少无效争抢，而非提高成功 GPU 命令的同时执行数量。

将本次配置忽略少量运行时开销作可手算示例：外部已经占 70 GiB，第一条申请 15 GiB 预约，可以进入；在它尚未分配实际显存时，第二条也需 15 GiB，新的公式按 `70 + 15 + 15 = 100 GiB > 90 GiB` 拒绝第二条并让它等待。第一条结束、预约归还后，再放行下一条。真实观测中的显存含驱动和 CUDA 上下文开销，不能把 70/15/90 的示例当作逐次遥测值。

这一解释对应现有源码：[准入与等待判断](../../../psrl/workers/env_service/command_gpu.py#L87)、[显存加未兑现预约公式](../../../psrl/workers/env_service/gpu_memory.py#L9)、[两组共同采用的 OOM 重试等待](../env-service-verification/scripts/science_fixed_work_agent.py#L115)。完整调用图和阅读顺序见[历史执行链与复现](history/06-历史过程_Science显存压力与原生任务.md#1-真实执行链路)及[准入公式文档](../knowledge/02-资源与调度/01-显存准入_实测占用与预约预算.md)；这些代码如有不清楚的地方，可以沿对应位置继续追问。CPU 可运行例子是 [explain_resident_admission_window.py](../env-service-verification/scripts/explain_resident_admission_window.py)，它是教学模拟，真实训练证据仍以本节结果为准。

三个边界需要同时保留。第一，ON 有效训练回复 token 更多，模型更新耗时也更长，不能把总训练提速解释为反向传播算得更快。第二，双方 `reward_repair` 和 `grad_norm` 均为零，虽然执行了三次训练更新且有有效 token，但本次没有证明策略学习或科学任务修复效果改善。第三，本轮仅重新测了高压力；上一轮低压力未见加速，而且它使用修复前 ON，不能拿它代替修复后低压力性能验证。

本次测试支持“70 GiB 常驻压力下，修复后的准入避免了这一轮观察到的过量准入 OOM”。节点仍缺少可信宿主进程归属信息，运行的是保守记账路径；不据此声称所有未知外部占用、超申报任务、多卡组合都不会 OOM。OFF → ON 顺序运行且没有重复统计，未观察到其他 GPU 超过 64 MiB 的活动，但这不能排除 CPU、共享存储、缓存和生成随机性的影响。OFF 两次、ON 一次 GPU 遥测错误保留为缺失，没有补零。

收尾核验：后台训练与分析退出码均为零，两个常驻压力进程正常退出，按归属回收容器和网络，结束日志确认显存释放；次日现场全部 GPU 为 0 MiB，Docker 无容器，后台看门狗与监督器 PID 已消失。没有追加训练。数据准备前的 position_ids 三项测试通过，两组实际 worker 源码路径及 ON 六文件指纹核验见 [completion-audit.json](../env-service-verification/results/science-resident-three-step-20261026/completion-audit.json)。

证据入口：[完整比较 JSON](../env-service-verification/results/science-resident-three-step-20261026/comparison.json)、[逐步曲线](../env-service-verification/results/science-resident-three-step-20261026/per-step.png)、[累计对比图](../env-service-verification/results/science-resident-three-step-20261026/matched-prefix.png)。只复算本次日志、不启动训练的命令：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_single_gpu_training.py \
  docs/env-worker-deep-dive/env-service-verification/results/science-resident-three-step-20261026
```


## 当前引用边界

该组 ON 为 `b52a82d` 加显存准入修复，不能标成今天 HEAD 重测。旧版存在静态设备配置和进程内并发限制，必要运行补丁单独记录。原始任务是 LAPS CPU 科学程序修复，GPU 压力双方共同注入。两组奖励和梯度为零，生成 token 长度不同，结果支持该压力条件下的执行效率，不支持学习质量或纯模型计算加速。

低压力优先查看[同期配对报告](02-基线对照_低负载EnvRouter修复验证.md)，真实常驻 GPU 模型优先查看[修复后的整卡对照](gpu-benchmarks/01-任务验收_结果与证据导航.md#episode-记账修复后的整卡压力对照)。它们的工作量与计时口径分别成立，不把不同实验百分比叠加。
