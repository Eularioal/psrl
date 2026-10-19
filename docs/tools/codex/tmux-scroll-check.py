"""真实 tmux 隔离实验：需要本机 tmux 和 Python 标准库，不运行 Codex。

图中 1–5 见 codex-session-management.md 第五节。
使用独立 socket；只清理本实验服务器，不修改已有会话。
"""

import fcntl
import os
import pty
import select
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path


def main():
    name = f"codex-scroll-check-{os.getpid()}"
    config = Path(__file__).with_name("tmux-scroll.conf")

    def tm(*args):
        return subprocess.check_output(
            ["tmux", "-L", name, *args],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        ).strip()

    with tempfile.TemporaryDirectory(prefix="tmux-scroll-check-") as directory:
        directory = Path(directory)
        received = directory / "input"
        received.write_bytes(b"")
        app = directory / "app.py"
        # 图 4：产生真实屏幕历史；这个程序仅记录输入，不实现聊天功能。
        app.write_text(
            "import os, tty\ntty.setraw(0)\n"
            "for i in range(150): os.write(1, f'line {i:03d}\\r\\n'.encode())\n"
            "while True:\n data = os.read(0, 1024)\n"
            f" with open({str(received)!r}, 'ab') as f: f.write(data)\n"
        )
        master = slave = client = None

        def drain(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], max(0, deadline - time.monotonic()))
                if ready:
                    try:
                        os.read(master, 65536)
                    except OSError:
                        break

        try:
            tm("-f", str(config), "new-session", "-d", "-s", "check", "-x", "80", "-y", "24", "python3 " + str(app))
            # 图 1：伪终端提供真实的终端接口，以字节模拟用户鼠标事件。
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
            env = os.environ.copy()
            env.pop("TMUX", None)
            env["TERM"] = "xterm-256color"
            client = subprocess.Popen(
                ["tmux", "-L", name, "attach-session", "-t", "check"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=env,
                start_new_session=True,
            )
            drain(0.7)

            def state():
                return tm("display-message", "-p", "-t", "check", "#{pane_in_mode}:#{scroll_position}")

            before = state()
            # 图 2→3→4：SGR 鼠标编码，64 表示向上滚，坐标为 (10, 10)。
            os.write(master, b"\x1b[<64;10;10M")
            drain(0.5)
            up = state()
            # 图 4→5：向下滚 5 行到底，copy-mode -e 自动退出。
            os.write(master, b"\x1b[<65;10;10M")
            drain(0.3)
            down = state()
            assert before.startswith("0:"), before
            assert up == "1:5", up
            assert down.startswith("0:"), down
            assert received.read_bytes() == b"", received.read_bytes()
            # 图 5：再次向上滚后，用 Esc 主动退出历史阅读。
            os.write(master, b"\x1b[<64;10;10M")
            drain(0.3)
            os.write(master, b"\x1b")
            # tmux 2.7 默认等待 500 ms 区分单独 Esc 和转义键序列。
            drain(0.8)
            assert state().startswith("0:")
            print(f"before={before} wheel_up={up} wheel_down={down}")
            print(f"application_input={received.read_bytes()!r}")
            print("PASS: 上滚 5 行、下滚到底退出、Esc 退出；滚轮未传给应用。")
        finally:
            # 只销毁本脚本唯一命名的服务器；临时目录自动删除。
            socket_file = None
            try:
                socket_file = Path(tm("display-message", "-p", "#{socket_path}"))
                tm("kill-server")
            except subprocess.SubprocessError:
                pass
            if client is not None:
                try:
                    client.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    client.terminate()
                    client.wait(timeout=3)
            for fd in (master, slave):
                if fd is not None:
                    os.close(fd)
            # tmux 2.7 可能留下已无监听者的 socket 文件；确认失效后删除。
            if socket_file is not None and socket_file.exists():
                import socket

                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(1)
                    try:
                        probe.connect(str(socket_file))
                    except ConnectionRefusedError:
                        socket_file.unlink()


if __name__ == "__main__":
    main()
