"""Illustrate agent self-checks and final scoring without a model or GPU."""


def main():
    events = []

    def emit(step, message):
        events.append((step, message))
        print(f"[{step}] {message}")

    # V1 establishes the environment before the agent starts.
    emit("V1", "创建一条 rollout 的环境，最终奖励尚不存在")
    candidate = {"small": False, "large": False}
    for turn in (1, 2):
        # V2 and V3 repeat within the same rollout.
        emit("V2", f"turn {turn}：模型生成修改方案，CPU 命令写文件")
        candidate["small"] = True
        candidate["large"] = turn == 2
        emit("V3", f"GPU 自测：{candidate}，结果作为下一轮模型的输入")
        emit("V3", "自测命令结束并确认归还 GPU，尚未产生最终奖励")

    emit("V4", "Agent 提交当前文件，停止继续生成命令")
    official_results = []
    for name in ("small", "large"):
        # V5 has two required units inside one final verifier.
        official_results.append(candidate[name])
        emit("V5", f"最终 verifier 的 {name} 单元完成")
        if len(official_results) == 1:
            emit("提案边界", "这里可以更新剩余成本估计，但最终评分还没结束")
    reward = int(all(official_results))
    emit("V6", f"全部必做测试完成，最终 reward={reward}，清理后交付训练数据")
    assert len(official_results) == 2 and reward == 1, "All required checks must pass."
    assert next(i for i, event in enumerate(events) if event[0] == "V5") > next(
        i for i, event in enumerate(events) if event[0] == "V4"
    ), "Final verification must follow submission in this example."


if __name__ == "__main__":
    main()
