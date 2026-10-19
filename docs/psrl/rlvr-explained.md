# RLVR 是什么

## 一句话结论

RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励的强化学习）指用 RL 训练模型时，reward 来自一段**确定性、可程序化验证**的规则函数（答案是否匹配、代码是否通过单测），而不是像 RLHF 那样训练一个拟合人类偏好的 Reward Model 来打分。

## 和 RLHF 的核心区别

| | RLHF | RLVR |
|---|---|---|
| reward 来源 | 训练一个 Reward Model，拟合人类偏好标注 | 规则/程序化验证器（exact match、单测通过与否、符号等价判断） |
| 是否需要额外训练一个网络 | 需要（RM 单独训练、单独占显存） | 不需要，reward 是一段确定性代码 |
| 适用场景 | 主观质量判断（有没有帮助、语气、创造性写作） | 有客观对错标准的任务（数学、代码、逻辑题、部分 agentic 任务） |
| reward hacking 风险 | 较高——RM 是近似模型，优化久了模型会学着"讨好 RM"而非真的答对 | 较低——规则验证器不会被"讨好"，但可能被验证器自身漏洞钻空子（比如格式取巧） |

RLVR 描述的是"reward 从哪来"，和用什么 RL 算法（PPO、GRPO、DAPO...）是两个正交的维度——GRPO/DAPO 是优化算法，RLVR 是奖励信号的性质。

## 具体例子：GSM8K 的 reward 函数

PSRL 仓库里 `examples/dapo_trainer/qwen2.5_3b_fsdp.sh` 跑的 DAPO + GSM8K，就是一次 RLVR 训练。reward 函数在 `verl/utils/reward_score/gsm8k.py`：

```python
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    answer = extract_solution(solution_str, method)   # 正则提取 "#### 答案" 格式的最终数字
    if answer is None:
        return 0                      # 没按格式给出答案：0 分
    return score if answer == ground_truth else format_score   # 数字对得上真值：1 分，否则 0 分
```

全程是字符串/数值比较，没有神经网络参与打分——这正是"verifiable"的字面含义：reward 能被一段确定性代码验证，而不是靠另一个模型的主观判断。

## 为什么最近很火（DeepSeek-R1、o1 之后）

验证器是确定性的、算力便宜（不用再养一个和 policy 同量级的 RM），也几乎不可能被模型"钻营"出高分却答案错误——这让大规模、长时间的纯 RL 训练变得可行且稳定，是 DeepSeek-R1 等工作能靠 RL（而非海量人工标注）激发长链推理能力的关键前提。代价是 RLVR 只能用在"能程序化判断对错"的任务上，创造性写作、开放式对话这类主观任务仍需 RLHF/RM。

## 记忆点

RLVR 不是一种 RL 算法，是一种 reward 来源；判断一个训练是不是 RLVR，看它的 reward 函数是不是一段确定性验证代码，而不是一个学出来的打分网络。

## 相关链接

- [psrl-run-guide.md](psrl-run-guide.md) — DAPO/GSM8K 示例脚本的完整拆解
- [dapo-training-collapse-diagnosis.md](dapo-training-collapse-diagnosis.md) — 一次 RLVR 训练在优化压力下崩溃的实例分析
- [ppo-grpo-dapo-sample-requirements.md](ppo-grpo-dapo-sample-requirements.md) — PPO/GRPO/DAPO 算法本身的区别
