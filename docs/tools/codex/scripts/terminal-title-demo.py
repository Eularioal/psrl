"""Simulate terminal title loss during reconnection without changing a terminal."""


def main():
    # 图 1：聊天名称保存在会话存储中。
    thread_name = "排查 Codex 的 Bus error"
    process_name = "codex"
    transcript = ["用户提问", "助手回答"]

    # 图 2 和 3：应用发来标题后，界面可以显示聊天名称。
    title_event = thread_name
    view = {"sequence": title_event, "text": list(transcript)}
    print("收到标题后:", view["sequence"] or process_name)

    # 图 4：教学假设，只恢复正文，没有恢复应用标题。
    restored = {"sequence": "", "text": list(view["text"])}
    print("只恢复正文后:", restored["sequence"] or process_name)
    print("保存的会话名称:", thread_name)
    print("恢复的正文条数:", len(restored["text"]))

    # 图 2 再到图 3：重新发送标题后，界面又能显示聊天名称。
    restored["sequence"] = title_event
    print("重新收到标题后:", restored["sequence"] or process_name)


if __name__ == "__main__":
    main()
