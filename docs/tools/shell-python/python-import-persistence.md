# Python `import` 会不会在多次脚本运行之间"留下来"

一句话结论：**不会**。每次 `python xxx.py` 都是全新的独立进程，`import` 只是把包加载进这一次进程自己的内存，进程退出后这份状态就被完全丢弃；下一个脚本能不能 `import` 成功，只取决于当前激活的 Python 环境（conda env / venv）的 site-packages 目录里有没有装这个包，跟"之前跑没跑过某个脚本"没有关系。

## 场景

`vllm_experiments/` 目录下先跑 `00_check_env.py`（检测 `torch`/`vllm`/`transformers`/`huggingface_hub` 能否正常 import，见 `00_check_env.py:20-32`），再跑 `01_bandwidth_and_decode_scaling.py`。疑问：`00_check_env.py` 里 import 过的包，`01_xx.py` 还能用到吗？

## 结论展开

1. **两次运行是两个完全独立的进程**。`00_check_env.py` 的 `import torch` 只把 torch 加载进这一次进程的 `sys.modules`，脚本结束、进程退出，这份内存状态整个消失，不会传递给任何后续进程。这跟 `source` 命令的行为正好相反：`source script.sh` 是在**同一个 shell 进程**里执行，所以对环境变量的修改能留下来；而 `python a.py` 和 `python b.py` 是**两个子进程**，天然互不残留（详见 [[bash-source-command]]）。

2. **真正决定 import 成不成功的是磁盘上的 site-packages，不是"跑没跑过某个脚本"**。只要 `01_xx.py` 运行时激活的是和 `00_check_env.py` **同一个** conda env / venv，这些包本来就一直可用——因为它们是通过 `pip install` 写入到该环境的 site-packages 目录里的，这份安装状态是持久化在磁盘上的，与进程运行历史无关。

3. **`00_check_env.py` 本身不做任何安装**，读一下源码就能确认：它只是循环调用 `importlib.import_module(mod)` 做检测，try/except 抓 `ImportError`，没有任何 `pip install` 或写入环境的操作。所以它对环境是**只读**的，作用仅仅是"提前帮你确认环境是否就绪"，不具备"安装"或"为后续脚本准备好包"的效果。

4. **什么情况下会失败**：如果两次运行之间切换了 Python 环境（新开一个 shell 忘记 `conda activate`、换了不同的 conda env、进了不同的容器等），即使 `00_check_env.py` 之前检测通过，`01_xx.py` 也可能 import 失败——因为两次用的根本不是同一份 site-packages。判断是否同一环境，看 `which python` / `conda env list` 里当前激活的环境名是否一致即可。

## 相关链接

- [[bash-source-command]] —— `source` 与直接执行子进程的区别，是理解"进程间状态不共享"这个概念的另一个例子
- [[activate-script-notes]] —— conda + 代理激活脚本，涉及环境切换的具体操作
