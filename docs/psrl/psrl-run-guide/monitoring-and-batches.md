# PSRL 监控与批量：先看进展，再看质量和耗时

监控需要回答三个问题：任务推进到哪里，模型输出是否改善，时间花在哪个阶段。进程、GPU 与训练指标分别提供部分信息，放在一起看才能判断下一步。

2026-11-16 核对本地批量解析和示例脚本，整理监控方法；未启动训练或读取新的运行指标。下面样本行与数字为教学数据。

## 目录

- [如何阅读训练日志、抓取每步指标](#如何阅读训练日志抓取每步指标)
  - [先按角色与阶段读日志](#先按角色与阶段读日志)
  - [把指标提取成表](#把指标提取成表)
- [`train_prompt_bsz` 对应哪个配置、是什么意思](#train_prompt_bsz-对应哪个配置是什么意思)
  - [从题目数算到一次更新](#从题目数算到一次更新)
- [查一个训练 pid 的运行情况：去哪儿看、看什么](#查一个训练-pid-的运行情况去哪儿看看什么)
  - [怎样区分推进缓慢与缺少进展](#怎样区分推进缓慢与缺少进展)

## 如何阅读训练日志、抓取每步指标

### 先按角色与阶段读日志

合并日志常交织多个 Ray actor 的输出。先看行首角色：TaskRunner/Trainer 多用于阶段和汇总；训练 worker 查看模型加载与更新；vLLM 查看生成执行；PS/协调器查看版本和样本等待；环境日志查看工具与评分。

| 问题 | 重点字段 | 怎样解释 |
|---|---|---|
| 到第几步 | `training/global_step`、退出码 | 对照目标步数与完成记录 |
| 更新是否数值稳定 | loss、`actor/grad_norm` | 查 NaN/Inf 与突变，结合概率比和输入定位 |
| 输出是否有变化 | `actor/entropy`、真实回答 | 熵是分布多样程度，升降需结合正确率 |
| 反馈是否改善 | `critic/score/mean`、rewards、验证指标 | 先确认评分键与奖励整形，再比较趋势 |
| 组内是否有区分 | `group/zero_variance_fraction` | 同分组的相对奖励优势可能为零 |
| 长度是否触顶 | `response_length`、截断比例 | 抽取回答与结束原因，区分长推理和重复 |
| 时间花在哪里 | `timing_s/step`、生成等待、update_actor | 按相同计时范围比较采样与训练 |

例如训练分数上涨、验证准确率下降，同时回答越来越长，应该检查评分与输出内容，而不能用单一上升曲线判断效果。真实历史联合退化见[DAPO 诊断](../dapo-training-collapse-diagnosis.md)。

### 把指标提取成表

下面是自包含 Python 示例。默认解析两条教学日志并打印表格；将 `lines` 换成实际日志迭代器即可分析保存的数据。先去掉 ANSI 颜色控制码，再提取 `key:value`。

```python
import re

lines = [
    "step:1 - training/global_step:1 - actor/entropy:0.30 - critic/score/mean:0.25",
    "step:2 - training/global_step:2 - actor/entropy:0.28 - critic/score/mean:0.50",
]
keys = ["training/global_step", "actor/entropy", "critic/score/mean"]
ansi = re.compile(r"\x1b\[[0-9;]*m")
print("\t".join(keys))
for raw in lines:
    line = ansi.sub("", raw)
    if not re.search(r"\bstep:\d+", line):
        continue
    fields = {}
    for item in line.split(" - "):
        if ":" in item:
            key, value = item.split(":", 1)
            fields[key.strip()] = value.strip()
    print("\t".join(fields.get(key, "-") for key in keys))
```

预期输出两步，熵为 0.30/0.28，分数为 0.25/0.50；`-` 表示该行缺失的字段。实际日志格式与版本有关，先抽几行确认，再批量提取。已有 W&B 或 TensorBoard 时，也可直接查看曲线并保留配置与原始日志。

## `train_prompt_bsz` 对应哪个配置、是什么意思

示例脚本将 `train_prompt_bsz` 传给 `data.train_batch_size`，表示一批多少道题。每题采样数为 `rollout.n`，因此目标题目组数和轨迹条数需要分别算。

### 从题目数算到一次更新

教学例子：64 道题，每题 8 条回答，目标共 512 条轨迹。若 mini-batch 配为 16 道题，对应每个 mini-batch 128 条轨迹，整批可划为 4 个 mini-batch。

| 层次 | 作用 | 本例 |
|---|---|---|
| 全局题目批量 | 一批数据含多少题 | 64 题 × 8 = 512 条 |
| Mini-batch，优化小批量 | 划分优化器更新的数据范围 | 16 题 × 8 = 128 条，4 组 |
| Micro-batch，显存小批量 | 将一次更新拆为可放入显存的小块 | 多次前后向累积梯度，随后更新 |

这是常规完整批处理路径的教学说明；策略、PPO epoch、过滤与分块方式会改变实际更新安排。动态 token 打包按 token 预算形成计算块，和“需要多少题形成算法分组”是不同层次。

本地[批量解析](../../../psrl/utils/config.py#L21)计算 `full_batch_samples = train_batch_size * rollout_n`、`mini_samples = ppo_mini_batch_size * rollout_n`，并按所选策略检查分组和并行对齐。配置修改后应核对最终解析值；实际采样量还可能受冗余和过滤影响。

更详细的组内统计与计算拆分，见[四条回答的例子](../ppo-grpo-dapo-sample-requirements.md)。

## 查一个训练 pid 的运行情况：去哪儿看、看什么

先确认 PID 属于哪台节点及哪个任务。仅在终端所在机器执行 `ps`，可能看不到远端 worker。

| 层面 | 取什么证据 | 回答什么 |
|---|---|---|
| 进程 | PID、创建时间、命令和退出码 | 本次进程是否仍在 |
| Ray | 节点列表、资源池和 actor 状态 | 角色是否已创建，实际在哪台机器 |
| 阶段日志 | 初始化、采样、评分、更新与清理标记 | 推进到哪一步 |
| 设备 | UUID、归属进程、显存和计算利用率 | 本次角色是否在实际用卡 |
| 质量 | 有效 token、奖励、验证与样本内容 | 已完成什么学习或验证 |

GPU 显存占用高可来自模型权重与缓存预留，利用率低可来自等待、初始化或工具阶段，均应结合时间段解释。Ray dashboard 主要查看资源与任务，模型质量查看训练记录系统及真实样本。

### 怎样区分推进缓慢与缺少进展

按预先设置的阶段预算，看新 actor、模型加载、完成轨迹或训练 step 是否出现。进度条短时不动时先检查当前阶段；超过约定预算仍缺少推进证据时，保存对应 actor 日志并按任务停止条件处理。

运行中单次遥测超时记为缺失；训练失败依据独立的错误、退出码和预算判断。清理按本次进程和容器归属执行，最后核实物理资源。完整规则见[集群检查](cluster-preflight.md)。
