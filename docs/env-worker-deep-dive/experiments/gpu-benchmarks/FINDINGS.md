# GPU 任务验收：结果与证据导航

> **核心结论**：GPU 任务的“能够运行”“策略能够使用”“调度带来收益”分别有不同证据。本页帮助定位对应记录；当前题库数量与逐类缺口看[任务路线图](../../milestones/science-gpu-bench-status-and-roadmap.md)，首次了解项目可先读[简版介绍](../../milestones/science-gpu-bench-brief.md)。

2026-11-16 按主题整理已有结果，未重跑实验。下文保留原有章节入口，完整旧叙述、版本、失败和补丁条件归入[接入历史档案](../history/gpu-benchmark-onboarding-chronology.md)。原始 JSON 仍在本目录。

## 目录

- [异质任务与独立运行验证 Results](#异质任务与独立运行验证-results)
- [2026-11-11 AIRS 非 Kaggle GPU 任务](#2026-11-11-airs-非-kaggle-gpu-任务)
- [2026-11-11 完成六类本地题库接入](#2026-11-11-完成六类本地题库接入)
- [2026-11-10 本地完整接入验收](#2026-11-10-本地完整接入验收)
- [接入范围](#接入范围)
- [KernelBench Results](#kernelbench-results)
- [官方资产准备](#官方资产准备)
- [autoresearch 真实任务](#autoresearch-真实任务)
- [原生 Env 的 episode 长任务验证](#原生-env-的-episode-长任务验证)
- [episode 记账修复后的整卡压力对照](#episode-记账修复后的整卡压力对照)
- [评分教学脚本验证](#评分教学脚本验证)
- [修复与验证边界](#修复与验证边界)
- [复现入口](#复现入口)
- [KernelBench command 接入验证](#kernelbench-command-接入验证)
- [2026-11-09 接入复核](#2026-11-09-接入复核)

## 异质任务与独立运行验证 Results

**要回答的问题：新任务能否被策略实际调用，并把评分交回训练链路？** 策略是负责选择工具和编写方案的语言模型；任务内训练的分类器或小模型是另一层。

2026-11-12 的独立运行覆盖 MLGym 强化学习、视觉与 SWE 共8题：32508个有效生成token，一次策略更新，版本1取得全部8题后续评分。初始策略有工具误用，随后补充了命令指引。[结构化证据](heterogeneous-validation-20261112.json)保留初始失败、提示变化及受控收尾。

另两次独立策略运行共同提供 PostTrain 9组合、MLGym NLP 2题及 FastKernels Linear 的逐题证据。11题主运行达到3000秒预算时，版本1评分完成10题；后续独立两题运行补齐 AIME 与 Linear。各次运行的初始权重、通过状态及梯度分开保存，见[状态审计](onboarding-status-audit-20261112.json)和[验收详情](../../milestones/science-gpu-full-onboarding.md#results-与验证边界)。

## 2026-11-11 AIRS 非 Kaggle GPU 任务

19项全部通过真实GPU执行、独立评分与奖励转换，总用时365.68秒。SICK代表题另完成真实策略采样、一次优化器步骤及版本1评分；672个有效token，组内奖励相同，策略梯度为0。

这些是有界本地数据协议，适合验证接线。每项使用的数据量与评分含义见[AIRS recipe](../../knowledge/airs-gpu-recipe.md)，原始证据见[airs-gpu-validation](airs-gpu-validation-20261111.json)。GPU运行当时的123项CPU回归，与11月13日补充的164项CPU回归按日期分别保存，见[AIRS修复报告](../../airs-bench-audit/fixes-and-results.md)。

## 2026-11-11 完成六类本地题库接入

此阶段发布476个任务：Kernel 270、FlashInfer 190、RE 7、Inference 4、MLE 4、autoresearch 1。新增任务通过逐题独立验证，既有代表题策略证据按原条件复用。逐类协议、范围与后续扩展见[路线图](../../milestones/science-gpu-bench-status-and-roadmap.md)，固定运行结果见[complete-validation](complete-validation-20261111.json)。

## 2026-11-10 本地完整接入验收

六类代表任务完成混合策略链路：语言模型生成工具调用 → 独立验收器返回奖励 → PSRL更新策略 → 发布版本1 → 继续采样评分。一次更新使用11525个有效token，loss约0.003347，梯度范数约0.798689。20条结束轨迹包含17条正分和3条失败，另4条在达到目标后取消。

完整的 digits 分类例子、任务内模型与策略模型的区别、环境和清理见[接入与验收](../../milestones/science-gpu-full-onboarding.md)，原始记录见[full-onboarding](full-onboarding-20261110.json)。此组版本为 `bc5665f` 加当时未提交实现。

## 接入范围

题库接入分三层：**任务定义**说明要做什么，**独立验收**检查指定方案能否执行并评分，**策略闭环**检查语言模型能否使用任务并完成更新后采样。例如，458个FastKernels目标中29个已有CUDA快检，Linear另有策略闭环。这三种数量需分别阅读。

现有范围只维护在[任务路线图](../../milestones/science-gpu-bench-status-and-roadmap.md#目前有哪些任务及可用边界)。官方完整协议与本地协议的差别，在该文逐类说明；早期官方recipe的资产条件保存在[历史接入范围](../history/gpu-benchmark-onboarding-chronology.md#接入范围)。

## KernelBench Results

2026-10-29，官方4096×4096矩阵乘法题真实编译并通过3/3正确性检查。候选约7.80毫秒，参考约4.55毫秒，加速比0.583，说明候选更慢。本地奖励1取自正确性。

见[kernel-result](kernel-result.json)、[summary](summary.json)和[历史完整条件](../history/gpu-benchmark-onboarding-chronology.md#kernelbench-results)。原始结果字段 `runtime_us` 的值实际为毫秒；历史记录保留原值，当前适配器使用 `runtime_ms`。

## 官方资产准备

官方资产和本地快速协议分别记录。autoresearch取得官方单训练分片、固定验证分片与tokenizer；InferenceBench完成Mistral模型和样本完整性核验。MLE正式竞赛数据在Kaggle认证阶段受阻。

见[autoresearch-assets](autoresearch-assets.json)、[inference-assets](inference-assets.json)和[资产历史](../history/gpu-benchmark-onboarding-chronology.md#官方资产准备)。当前可直接上手的本地任务及凭据要求见[接入指南](../../knowledge/gpu-benchmark-recipes.md)。

## autoresearch 真实任务

2026-10-29，独立验收器重放原始 `train.py`，在300秒训练预算内完成306个任务内优化器步骤，官方指标 `val_bpb≈1.157411`，PSRL任务奖励约0.463519。BPB衡量验证文本每字节的预测代价，越低越好。

这里更新的是任务内约50.3M参数模型。完整配置、393.6秒脚本用时、约479.8秒任务生命周期和清理见[历史报告](../history/gpu-benchmark-onboarding-chronology.md#autoresearch-真实任务)，结果见[autoresearch-result](autoresearch-result.json)。

## 原生 Env 的 episode 长任务验证

2026-10-30，把原始autoresearch训练与独立评分接到原生Env后端，检查长任务期间持续持有GPU预算、结束后回收。episode指完整任务生命周期。

首次镜像缺虚拟环境符号链接，下一次长命令又在约300秒时提前中断；故障修复后的结果与原失败都保存在[长任务验证历史](../history/gpu-benchmark-onboarding-chronology.md#原生-env-的-episode-长任务验证)。它为接入可靠性提供证据，性能对照另见下节。

## episode 记账修复后的整卡压力对照

**同一张H20、同样7项常驻模型任务和14次请求：固定旧版发生一次显存不足并重试，Env提前按预算排队，首轮全部通过。** 驱动总时间158.48→141.73秒，减少10.6%。

独立报告[常驻模型显存压力](resident-model-pressure.md)解释服务为何持续占显存、四项20GiB预约为何成为并发上限，以及如何抵扣已加载模型的重复记账。原始数据为[comparison](episode-pressure-comparison.json)、[samples](episode-pressure-samples.json)、[provenance](episode-pressure-provenance.json)。

## 评分教学脚本验证

[评分说明](../../knowledge/inferencebench-scoring.md)使用CPU教学数字解释质量门槛和速度奖励。例如参考400/500题正确、候选390/500题正确，质量比例0.975达到门槛后，首token延迟0.2秒给出速度分5；质量降到360/500则奖励0。示例调用真实本地奖励函数，观测和运行条件见[教学验证历史](../history/gpu-benchmark-onboarding-chronology.md#评分教学脚本验证)。

## 修复与验证边界

原接入过程中曾遇到空闲卡显存误读、预算与设备容量相等导致等待、验收脚本挂载和执行位缺失。失败日志和回归口径保存在[历史修复记录](../history/gpu-benchmark-onboarding-chronology.md#修复与验证边界)。11月10日以后与提交物、评分及隔离相关的修复，按主题整合在[验收相关修复与回归](../../milestones/science-gpu-full-onboarding.md#验收相关修复与回归)。

## 复现入口

开始前先确定目标：查看已有结果只需读取本页JSON；准备任务按[GPU recipe](../../knowledge/gpu-benchmark-recipes.md)；复现完整策略接入按[验收命令](../../milestones/science-gpu-full-onboarding.md#运行入口)。GPU重跑使用新输出目录，现场检查资源、依赖、资产和回收条件。

## KernelBench command 接入验证

2026-11-04，独立验收器经过command准入，官方矩阵乘法题3/3正确。command指显式申请GPU的一段执行区间；结束后任务会话可继续，但这段GPU预算已归还。候选7.82毫秒、参考4.55毫秒。只读测试上传和chmod两次前置失败、实际提交 `cd09114` 加补丁及物理清理见[历史记录](../history/gpu-benchmark-onboarding-chronology.md#kernelbench-command-接入验证)，机制见[command与性能测量](../../knowledge/gpu-benchmark-recipes.md#kernelbench-command-与性能测量)。

## 2026-11-09 接入复核

固定版本 `bc5665f`，顺序复核KernelBench、autoresearch、固定常驻服务与RE CUDA smoke。前两项分别取得正确性与真实训练评分；常驻服务确认两请求复用；RE smoke确认CUDA路径。各自物理清理通过。

[science-audit](science-audit-20261109.json)保存源码、上游与结果指纹，[复核历史](../history/gpu-benchmark-onboarding-chronology.md#2026-11-09-接入复核)保存Docker预检失败、104通过/27跳过的CPU回归、单位修正和凭据边界。后来的正式本地RE题与策略验收见前文对应新证据。
