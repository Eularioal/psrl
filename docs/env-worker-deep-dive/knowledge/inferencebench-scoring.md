# InferenceBench 如何评分，以及 Env ON/OFF 实验验证了什么

> **核心结论**：InferenceBench 要求 agent 优化推理服务，在回答质量达标的前提下提高速度。官方 Harbor 奖励是按场景计算的延迟倒数或吞吐量，不是答题准确率，也不是统一的 0 到 1 分。本地 PSRL 还对裁判结果加门槛。常驻服务 ON/OFF 实验的 `reward=1` 是另一套功能检查，只说明请求成功和服务复用。
>
> 官方评分依据本地固定上游版本 `108ec747fab0bd92fd9de501f62157d074c50ed5`，不是随主分支变化的推测。本文补充 [benchmark 导览](../experiments/gpu-benchmarks/README.md) 和 [接入 recipe](gpu-benchmark-recipes.md)。

## 目录

- [先看完整评分流程](#先看完整评分流程)
- [四种场景的分数](#四种场景的分数)
- [质量门槛与手算例子](#质量门槛与手算例子)
- [PSRL 奖励和功能实验分数](#psrl-奖励和功能实验分数)
- [ON 与 OFF 的证据边界](#on-与-off-的证据边界)
- [跟着图读源码](#跟着图读源码)
- [不占 GPU 的评分演示](#不占-gpu-的评分演示)

## 先看完整评分流程

任务给 agent 一个基础语言模型和请求场景。agent 修改服务代码或配置，交付可以启动的服务。独立 verifier，即验收程序，重新运行提交，避免只相信 agent 自报的速度。这里优化的是推理服务，不要求重新训练基础模型。

```text
[1] agent 交付服务方案 → verifier 接收提交
          ├→ [2] judge 审核轨迹与方案，生成有效性、污染和禁用行为标记
          └→ [3] verifier 启动最终服务
                    ├→ [4] 发性能请求，统计延迟、吞吐与成功情况
                    └→ [5] 发质量题目，与参考准确率比较
                  停服务
                    ↓
[6] 性能数据有效且质量达标 → 按 A/B/C/D 公式写 Harbor reward
          judge 标记随 reward 一起保存，官方这里不据此直接清零
                    ↓
[8] 所属容器完成清理并归还租约
                    ↓
[7] PSRL 外层读取 reward 和标记 → 本地完整性检查 → 训练可用 score
```

图表示数据依赖，不表示所有工作并发。当前官方 `test.sh` 先执行 judge，再恢复提交、启动服务、测性能与质量、停服务及写奖励。PSRL runner 在返回结果前的 `finally` 中清理环境，所以实际清理先于外层消费最终奖励，不是在训练评分后才释放 GPU。

图例：图里每个部分对应的源码位置。上游链接指向本工作区已实际核对的 checkout，这些文件不随 PSRL 仓库分发。没有此 checkout 时使用 [官方固定版本目录](https://github.com/aisa-group/InferenceBench/tree/108ec747fab0bd92fd9de501f62157d074c50ed5/src)。

| 编号 | 职责、函数与源码 |
|---|---|
| 1 | 本地 [prepare](../../../examples/gpu_benchmarks/prepare.py#L76) 导入官方任务及 verifier，不自行替换评分公式 |
| 2、3、6 | 上游 [test.sh](../../../../benchmark-sources/InferenceBench/src/harbor_adapter/template/tests/test.sh#L230) 编排 judge、提交恢复、服务启动、最终评测、停服及奖励输出 |
| 4 | 上游 [run_evaluation](../../../../benchmark-sources/InferenceBench/src/eval/inference/runner.py#L1021) 调用性能评测，[统计函数](../../../../benchmark-sources/InferenceBench/src/eval/inference/runner.py#L737) 计算请求指标 |
| 5 | 上游 [compute_quality_gate](../../../../benchmark-sources/InferenceBench/src/eval/inference/quality_gate.py#L130) 比较准确率，默认阈值见 [参数定义](../../../../benchmark-sources/InferenceBench/src/eval/inference/runner.py#L1092) |
| 6 | 上游 [primary_metric](../../../../benchmark-sources/InferenceBench/src/eval/inference/hpo_search_baselines.py#L298) 选择场景公式，[write_reward](../../../../benchmark-sources/InferenceBench/src/harbor_adapter/template/tests/test.sh#L198) 写 `reward.json` |
| 7 | 本地 [runner 读取奖励](../../../examples/sciaccel_rl/runner.py#L569)、[agent loop 传递指标](../../../examples/sciaccel_rl/agent_loop.py#L254)、[compute_score](../../../examples/gpu_benchmarks/reward.py#L6) |
| 8 | 本地 [runner finally](../../../examples/sciaccel_rl/runner.py#L605) 清理所属容器、归还租约，发生在外层取得结果前 |

这些源码如果有理解上的问题，可以随时按编号追问。这里描述的是正式 recipe 的执行和评分接口，不宣称本次完成了官方整套 GPU 评测。

## 四种场景的分数

先把三个指标拆开。token 是模型生成文本的基本单位，可能是一个词或词的一部分。

- **TTFT，首 token 延迟**：从发送请求到收到第一段生成内容的时间。它回答“用户等多久才看到开始输出”。这里以秒计，越低越好。
- **TPOT，后续每 token 平均耗时**：单条请求中，首 token 之后的生成时间除以剩余 token 数。它回答“开始输出后，每个 token 平均多快”。例如首 token 后又生成 100 个 token，共用 2 秒，则 TPOT 为 0.02 秒。
- **p50，中位数**：将多次请求的指标排序后取中间位置。它反映典型请求，不等于最慢请求。TPOT p50 是先算每条请求 TPOT，再取这些数的中位数。
- **请求吞吐量**：成功请求数除以该批请求的实际总时间，单位为请求/秒。它不是输出 token/秒。

| 场景 | 鼓励的改进 | 官方 Harbor 原始奖励，越大越好 |
|---|---|---|
| A | 更快开始回答 | `1 / TTFT_p50` |
| B | 更快持续生成 | `1 / TPOT_p50` |
| C | 在不同请求到达方式下处理更多请求 | `(Q_burst × Q_poisson × Q_constant)^(1/3)` |
| D | 兼顾开始输出、持续生成和吞吐 | `((1 / TTFT_p50) × (1 / TPOT_p50) × Q_burst)^(1/3)` |

A、B、D 取 burst 性能记录。C 的 burst 是集中到达，poisson 是随机到达，constant 是固定间隔到达。表中 Q 是相应批次的请求吞吐量。几何平均就是把正数相乘后开相应次数的根，例如吞吐为 8、27、64 请求/秒，C 得分为 `(8×27×64)^(1/3)=24`，不会把某一种到达方式单独当作全部成绩。

分数不是统一百分制。A 得 5 和 B 得 50 没有直接大小比较的意义，原始值的量纲和优化目标不同。单项得分也不是“相对官方机器快了几倍”。例如同一 A 场景从 2.5 分变为 5 分，可以另算延迟指标改善 2 倍，但 5 分本身不表示 5 倍加速。

`speed_pass` 容易被误解。当前 [profile_complete](../../../../benchmark-sources/InferenceBench/src/eval/inference/quality_gate.py#L112) 检查请求全部成功、没有失败或空输出。它**不是必须快过参考服务**。`primary_metric` 还要求性能数据满足上游完整性和物理合理性检查。缺项、无效指标或不合法的场景不能直接获得上述正分数。

## 质量门槛与手算例子

速度优化不能靠回答变差、直接输出空内容来换分。默认质量评估使用 500 道 MMLU-Pro 题，MMLU-Pro 是多学科多选题数据集。评测解析模型答案、与正确选项比较，得到准确率。实际题量、种子和门槛由生成任务的配置确定，默认值不是所有自定义任务的强制常量。

当前默认规则是：`候选准确率 / 参考准确率 ≥ 0.95`，并且质量请求全部成功且非空。这里的 0.95 是保留参考质量的比例，**不是要求绝对答对 95% 的题**。质量参考需与数据集、样本数和种子匹配，不能随便填一个数。

**教学假设，不是实测成绩**：参考模型 500 题答对 400 题，准确率 80%。候选方案同样 500 题，A 场景 TTFT p50 为 0.2 秒。假设性能完整性和裁判均通过。

| 候选答对数 | 准确率 | 相对参考质量 | 门槛结果 | 官方 A 奖励 |
|---:|---:|---:|---|---:|
| 390 | 78% | `78% / 80% = 97.5%` | 通过 | `1 / 0.2 = 5` |
| 380 | 76% | `76% / 80% = 95%` | 通过 | `5` |
| 360 | 72% | `72% / 80% = 90%` | 不通过 | `0` |

同一个模型优化前 TTFT p50 为 0.4 秒且质量通过，对应 2.5 分。优化后质量通过且延迟减半，对应 5 分。第三行虽然同样很快，但质量掉得太多，因此为 0。

官方适配器写奖励时检查 eval、speed 和 quality。交付缺失或模型下载失败等基础设施错误可能直接没有奖励文件，不能把“没有完成评测”解释成一个正常的零分提交。本文也不把跳过质量检查的调试选项当作正式成绩。

**与我们的 Env 对照的关系**：官方的质量 reference 用来防止服务变快却答错，不是要求拿 H20 速度与别人 H100 的成绩比较。框架实验可以在同一 H20、同一模型与同一数据上固定工作量比较 ON/OFF。如果自行重建参考或缩小评测，必须记录协议变更，不能称为原版官方成绩。

## PSRL 奖励和功能实验分数

必须区分三个层次：

| 分数 | 来源与含义 | 不能推出什么 |
|---|---|---|
| 官方 Harbor `reward` | 性能和质量通过后按场景公式计算，judge 标记另存 | 不能仅凭正分推断 judge 没有检测到违规 |
| 本地 PSRL `score` | 读取官方 reward，再要求 `judge_valid`、`quality_pass`、`eval_valid` 为 1，污染及禁用行为标记明确为 0 | 不能因 H20 得分较低就判断 Env 接入失败 |
| 常驻 workload `reward=1` | [固定实验 verifier](../../../examples/inference_bench/resident_ab.py#L64) 只检查两次非空响应及相同服务 PID，即同一进程 | 不代表答案全对，不代表官方速度分，也不代表完成 agent 自主优化 |

本地 `compute_score` 不单独再次检查 `speed_pass`，因为上游写 reward 已处理它。对裁判字段则采用明确通过才接受的条件，缺失也会归零。因此同一份上游 `reward=5`，若污染标记为 1，本地 score 为 0。这个额外门槛是 PSRL 当前实现，不能反过来说上游 Harbor 本身也这样算。

## ON 与 OFF 的证据边界

已完成的 [整卡压力对照](../experiments/gpu-benchmarks/FINDINGS.md#episode-记账修复后的整卡压力对照) 使用真实模型与真实题目，但固定服务和动作，测的是资源管理。双方都跨请求保持模型驻留，没有在请求后的 command 边界 offload/reload，即卸载 GPU 状态再恢复。

ON 控制同时获准运行的 episode 数，释放后再接纳等待项。OFF 在压力下有一次显存不足，即 OOM，随后按既定规则重试。相同最终完成量下，避免这次失败和重复加载是 ON 收益的明确机制。记录中的初始化、后端路径和共享计算干扰并未逐项隔离，因此不能声称全部耗时差都由 OOM 单一因素精确解释。

这组结果支持“保持驻留的工作负载需要可信准入和归属记账”，**不支持“episode 比 command 加卸载恢复更快”**。要回答后者，需额外对照状态可恢复的 command 方案，测量保存、退出 GPU 进程、恢复及排队的全部成本，并检查真实释放。

**教学假设**：同一模型下一次使用前有 30 秒空档，保存和恢复共花 8 秒，理论上只剩约 22 秒可借给别的任务，还要扣掉回收及调度成本。若空档只有 3 秒，该方案可能增加延迟。若常驻能与其他任务安全共存，保留模型又可能更合适。这些数字不来自本次实验，也不是已实现的自动卸载策略。

## 跟着图读源码

| 执行顺序 → 图中编号 | 阅读入口 | 输入、输出与状态 |
|---|---|---|
| 任务导入 → 1 | `prepare`，见图例 | 官方 Harbor 任务转换为本地任务和数据入口 |
| 正式验收 → 2、3 | `test.sh`，见图例 | 接收提交，先审核，再恢复服务并启动 |
| 性能与质量 → 4、5 | `run_evaluation` → `compute_quality_gate` | 请求结果变成延迟、吞吐、准确率与通过标记 |
| 停服务并出分 → 6 | `test.sh` → `primary_metric` → `write_reward` | 停最终服务，将数值和裁判标记写入奖励文件 |
| 返回前清理 → 8 | runner `finally` | 物理清理容器后归还租约，异常也进入收尾 |
| 外层消费奖励 → 7 | agent loop → `compute_score` | 保留原始指标，本地检查后得到 PSRL score |

这是调用与数据传递说明，不代表本次运行了完整 PSRL 优化器更新。常驻实验绕开官方优化评分链，必须回到上一节的功能验收定义理解其分数。

## 不占 GPU 的评分演示

在 PSRL 仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_inferencebench_score.py
```

[脚本](../env-service-verification/scripts/explain_inferencebench_score.py) 只用标准库。图中 4、5、6 使用上述教学数字模拟输入与公式，图中 7 调用真实本地 `compute_score`。它不启动服务、不下载数据、不访问裁判，也不替代官方性能数据的完整性检查。

预期输出：参考 A 为 2.5 分，质量通过的候选 A 为 5 分，质量下降的候选为 0 分。同一个原始 5 分带污染标记时，真实 PSRL 函数返回 0。C 的教学吞吐例子得 24 分。具体运行结果记在 [验证记录](../experiments/gpu-benchmarks/FINDINGS.md#评分教学脚本验证)。
