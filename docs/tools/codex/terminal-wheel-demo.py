"""教学模拟：不操作真实终端，只比较滚轮事件的两类结果。"""

from dataclasses import dataclass


@dataclass
class View:
    first_line: int = 81
    input_text: str = ""


def wheel_up(view: View, mode: str) -> None:
    # 图 2：假设滚一格代表向上查看三行。
    if mode == "scrollback":
        view.first_line = max(1, view.first_line - 3)
    elif mode == "arrow-key":
        # 图 2 → 4：假设终端发出向上键，应用选中上一条问题。
        view.input_text = "上一条问题"
    elif mode == "tmux-copy":
        # 图 3：仅模拟复制模式移动历史视口，不模拟真实鼠标协议。
        view.first_line = max(1, view.first_line - 3)
    else:
        raise ValueError(mode)


def main() -> None:
    for mode in ("scrollback", "arrow-key", "tmux-copy"):
        view = View()
        # 图 1：各自从相同初始状态接收一次向上滚轮。
        wheel_up(view, mode)
        print(f"{mode}: 显示第 {view.first_line}–{view.first_line + 19} 行，输入框={view.input_text!r}")


if __name__ == "__main__":
    main()
