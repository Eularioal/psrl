# 评分复用：PostTrain准备成本与整批耗时

> **核心结论**：保留已加载的基础模型，能把后续请求从约 **9.5 秒降到 4.4 秒**。但本次三个请求的完整批次为 **113.84 秒**，仍慢于“保留进程、每次重载模型”的 **91.96 秒**。目前最明确的结论是存在可复用的模型准备成本，完整批次收益仍需超过普通进程复用这个基线。

实验日期为 **2026-11-13**，基于 `6aa328d` 加当时未提交研究脚本及冻结的 PostTrain 本地评分程序。共完成九次真实 GPU 评分、跨组结果核对和物理清理。2026-11-16 整理本文，只读原证据；该研究原型由独立脚本调用。

## 目录

- [真实任务为什么存在重复准备](#真实任务为什么存在重复准备)
- [三种执行方式分别省什么](#三种执行方式分别省什么)
- [Results：真实候选的有界对照](#results真实候选的有界对照)
- [怎样确保切换后没有用错参数](#怎样确保切换后没有用错参数)
- [检查的价值与代价](#检查的价值与代价)
- [结果适用于什么范围](#结果适用于什么范围)
- [跟着图读源码与运行例子](#跟着图读源码与运行例子)
- [验证范围与停止条件](#验证范围与停止条件)

## 真实任务为什么存在重复准备

PostTrain 让 agent 更新已有语言模型，再交出参数供独立评分。这里使用固定 Qwen3-1.7B-Base 的本地小预算协议：固定模型主体，只训练最后归一化层的 `model.norm.weight`，即调整模型内部数值尺度的 **2048 个数**。

本实验读取此前已训练好的两份候选：A 来自 GSM8K 数学任务，B 来自 AIME 数学任务。每份都经过两次真实参数更新，各为 **4184 字节**。基础模型则有十几亿参数。因此每次只是换一个很小的提交物，重建整个模型可能重复准备大部分共同内容。

独立评分在各任务的四条隐藏记录上，计算模型给标准答案分配的概率。指标叫**负对数似然**，可以理解为“给正确答案的概率代价”，越低越好。A、B 面向不同数据，实验比较的是**同一候选在不同执行方式下能否得到相同结果**。

## 三种执行方式分别省什么

三种方式都按 **A → B → A** 执行相同评分程序，检查开销也采用相同规则。

| 方式 | 每个请求重新做什么 | 可以保留什么 |
|---|---|---|
| 每次新进程 | 启动容器和 Python、导入库、建立模型、载入候选、评分 | 磁盘上的固定资产 |
| 同进程重载 | 重新建立基础模型、载入候选、评分 | Python 程序和已导入的库 |
| 同进程保留模型 | 完整替换本次候选、评分；基础身份变化时才重建 | 程序、库、分词器和已加载的基础模型 |

分词器把文本转成模型读取的数字编号；基础模型是共同的大部分参数；候选是本次训练产出的小参数文件。重载模型读取已有权重，训练则根据数据修改参数。

## Results：真实候选的有界对照

在一张启动前空闲 H20 上，每组固定三个请求。**完整时长**包括容器内初始化、请求和退出；**后续请求时长**从请求进入到结束处理，包含内容核对与评分。两种指标回答不同问题。

<!-- figure:measured-posttrain-reuse -->
这张图沿同一个场景解释：相同工作量与计时边界是什么？

[![PostTrain复用：单个请求变快，整组仍需计入准备（1/3）](../../../_static/img/document-figures/measured-posttrain-reuse-screen.png)](../../../_static/img/document-figures/measured-posttrain-reuse.png)

[![PostTrain复用：单个请求变快，整组仍需计入准备（2/3）](../../../_static/img/document-figures/measured-posttrain-reuse-screen-02.png)](../../../_static/img/document-figures/measured-posttrain-reuse.png)

[![PostTrain复用：单个请求变快，整组仍需计入准备（3/3）](../../../_static/img/document-figures/measured-posttrain-reuse-screen-03.png)](../../../_static/img/document-figures/measured-posttrain-reuse.png)

重画原报告A/B/A完整容器时长274.27、91.96、113.84秒；同进程重载与保留模型的B请求分别9.58与4.36秒。整组与单请求放在不同面板，原测量边界和条件见正文。 [查看矢量图](../../../_static/img/document-figures/measured-posttrain-reuse.svg)。
<!-- /figure:measured-posttrain-reuse -->

| 执行方式 | A/B/A 完整容器时长 | B 请求 | 再回到 A |
|---|---:|---:|---:|
| 每次新进程 | 274.27 秒 | 新进程分别计入完整时长 | 新进程分别计入完整时长 |
| 同进程重载 | **91.96 秒** | 9.58 秒 | 9.42 秒 |
| 同进程保留模型 | 113.84 秒 | **4.36 秒** | **4.53 秒** |

新进程三次分别耗时 144.98、63.07、66.22 秒，首次明显较慢。保留模型后，后续两请求分别减少约 **55%、52%**，可见重复准备确实存在。

**完整批次为何仍慢？** 保留模型组的进程准备为 54.22 秒，重载组为 39.80 秒；第一请求分别为 39.74、29.77 秒，外层退出等时间也有差异。两次后续请求的节省被这些成本盖过。本次每种方式一组、固定执行顺序，文件缓存与节点波动都可能参与，现有计时只能定位差异，完整因果贡献尚未分离。

九次评分的参数与结果核对均通过：A 六次负对数似然都是 `0.9280166029930115`，B 三次都是 `8.390504956245422`，与各自历史结果一致。两个候选的参数也确实不同。

全部流程含组间核查与清理为 **486.18 秒**，在 600 秒预算内结束。固定结果见[跨组汇总](../../env-service-verification/results/posttrain-verifier-reuse-20261113/summary.json)、[状态与清理](../../env-service-verification/results/posttrain-verifier-reuse-20261113/status.json)及[冻结来源](../../env-service-verification/results/posttrain-verifier-reuse-20261113/provenance.json)。

## 怎样确保切换后没有用错参数

```text
请求指定：基础模型版本 + 评分源码 + 候选 + 隐藏输入
                         ↓
核对共同内容身份 → 命中已有模型，或重建
                         ↓
完整载入本次候选的 2048 个参数 → 独立评分
                         ↓
回读实际参数与结果 → 处理下一请求，最后清理模型和容器
```

A/B/A 是检查串候选的具体例子：首次 A 完整载入 A；B 到达时完整覆盖为 B；再回 A 时又完整覆盖为 A。第三次参数与第一次一致，分数也应一致。基础模型虽然保留，每次评分仍实际读取本次提交并运行。

实验使用 SHA256 内容指纹，即按文件实际字节计算的标识。它检查基础模型、配置、源码、候选与输入，基础身份变化时重建对象。每次还比较提交文件参数、加载函数返回参数和实际模型参数，检查数值、形状和存储类型。评分入口从源码字节编译，避免旧字节码误读。

资产与候选副本在本实验中固定。只读挂载可阻止容器修改宿主输入，但宿主其他生产者仍可能覆盖文件；跨多个文件的一致版本还需受控发布约定。

## 检查的价值与代价

**约四秒的全量内容扫描，是本次实验的审计方法。** 每请求重读约 3.45 GB 基础资产，用于取得直接版本证据。原评分程序没有这项大扫描，生产执行可以在受控发布时检查一次，再引用固定版本。

| 工作 | 实测或实现情况 | 后续应怎样处理 |
|---|---|---|
| 大基座内容扫描 | 后两次约 4.04、4.21 秒，是保留模型请求的主要部分 | 固定内容版本后，移到发布或首次加载 |
| 模型重载 | 后两次传 GPU 约 4.11、4.04 秒，模型加载约 0.64、0.62 秒 | 基础版本相同时可复用 |
| 载入小候选 | 每次读 4184 字节，检查参数并完整替换 | 属于评分语义，持续保留 |
| 参数回读、重复 A、跨组对比 | 取得候选切换正确性的直接证据 | 用于测试与诊断，按目的启用 |

保留模型后，原样评分 `execute()` 的后两次耗时约 0.162、0.160 秒，完整请求仍需四秒多，正是扫描成本的影响。小候选加载及包装审计三次合计约 7–8 毫秒，模型参数回读未单独计时。

两个同进程组都做三次大扫描，累计内容身份时间仅差 **0.314 秒**，所以它解释了请求成本的大部分，却只解释完整组差距 **21.88 秒**中的很小部分。将总时长简单减掉扫描，也只是估算；真正低扫描方案需实现后再测。

可以采用以下发布约定，这是后续设计建议：

1. 写完 `base-v1` 后校验并固定内容。
2. 评分进程首次加载该版本，A、B 请求明确引用它。
3. 每请求完整载入小候选，复用共同基座。
4. 基座更新时另发 `base-v2`，按新版本建立模型对象。

这依赖旧版本内容保持固定，以及可信清单或发布约束。约束成立后才有减少扫描的依据。原审计说明与源码复核见[口径核验](../../env-service-verification/results/posttrain-verifier-reuse-20261113/audit-clarification-validation.json)。

<a id="收益意味着什么以及不意味着什么"></a>

## 结果适用于什么范围

该结果说明本地“小候选、大共同模型”的评分结构存在重复加载。当前最强的简单对照是普通进程复用，保留模型的完整短批次仍需胜过它。自然 agent 的请求间隔、候选变化频率、缓存容量竞争与完整微调的收益，属于后续问题。

保留模型也会持续占显存。教学假设：重载需 3 秒，下一请求 30 秒后到达。保留能省这 3 秒，但其它任务可能因这段显存占用而等待。需要结合真实请求间隔与等待任务，才知道保留是否划算。

本次比较同一研究执行器的三种方式；固定旧 PSRL 提交 `0eeead8` 的调度对照另见[实验入口](../01-实验总览_调度收益与证据范围.md)。生产 EnvRouter 的跨请求模型缓存、版本发布和驻留记账仍需按专门设计推进。

## 跟着图读源码与运行例子

| 图中步骤 | 实现与输入输出 |
|---|---|
| 冻结请求与分组 | [run_study.main](../../env-service-verification/scripts/posttrain_reuse_study/run_study.py#L54)冻结源码与候选副本，各组按相同请求执行；组内请求串行 |
| 核对身份 | [Study.request](../../env-service-verification/scripts/posttrain_reuse_study/bounded_worker.py#L121)读取真实内容，确定本次版本 |
| 命中或重建 | [Slot.get](../../env-service-verification/scripts/posttrain_reuse_study/bounded_worker.py#L76)返回可复用对象，身份改变时重新建立 |
| 载入与评分 | [原样 execute](../../../../examples/posttrain_bench/local_model.py#L61)完整替换候选，并检查四条隐藏输入 |
| 核对与汇总 | [参数核对](../../env-service-verification/scripts/posttrain_reuse_study/bounded_worker.py#L220)、[summarize](../../env-service-verification/scripts/posttrain_reuse_study/summarize.py#L9)比较实际参数和跨组分数 |

先用标准库教学脚本理解 A/B/A，仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/posttrain_reuse_study/explain_reuse.py
```

预期只建一次基座，参数依次为 `[2,3]`、`[5,7]`、`[2,3]`。然后保持基础文件大小、修改时间相同，只改内容，第四次请求应发现变化并第二次建立基座。该教学示例与[五项 CPU 回归](../../env-service-verification/scripts/posttrain_reuse_study/test_bounded_worker.py)已有历史运行记录，GPU 数字仍来自前述实机证据。

只分析已有结果时运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/posttrain_reuse_study/summarize.py \
  --run "$RUN_DIR" --output "$SUMMARY_JSON"
```

`RUN_DIR` 必须是 driver 生成的完整结果目录，`SUMMARY_JSON` 使用新路径。GPU driver 另需固定镜像、Python、基础模型、候选、隐藏输入和现场确认的设备，参数结构见 [input.json](../../env-service-verification/results/posttrain-verifier-reuse-20261113/input.json)：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/posttrain_reuse_study/run_study.py \
  --config "$RUN_CONFIG" --output "$RUN_DIR"
```

## 验证范围与停止条件

固定实验完成九次评分、结果核对与资源释放后结束，未追加训练或重跑追求更短时长。初次 Docker `--gpus` 探针因 NVIDIA runtime 配置失败，改用项目设备/驱动挂载并先通过小 CUDA 内核后才评分；主环境激活挂起的处理与探针失败均保留，宿主激活时间单列。

五项 CPU 回归通过，教学脚本已运行，完整验证状态见[validation](../../env-service-verification/results/posttrain-verifier-reuse-20261113/validation.json)。实际候选来源见[准备审计](../../env-service-verification/results/benchmark-state-audit-20261113/posttrain-preparation-audit.json)。范围外全仓格式问题与本实验结果分别记录。本次文档整理没有运行新实验。
