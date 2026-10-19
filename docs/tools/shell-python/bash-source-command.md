# `source` 命令是做什么的

一句话结论：`source`（等价于 `.`）是 bash 内置命令，在**当前 shell 进程**里执行指定脚本，脚本对环境变量、`PATH`、cwd、函数、别名的修改会保留在当前 shell 中；这一点和"直接运行脚本会另开子进程"完全不同。

## 和直接执行脚本的区别

```bash
./script.sh      # 新开一个子进程执行，子进程结束后它对环境的修改全部消失
bash script.sh   # 同上，显式指定用 bash 开子进程执行
source script.sh # 在当前 shell 里逐行执行，脚本对环境变量/函数/别名的修改都保留下来
. script.sh      # 和 source 完全等价，是 POSIX 标准写法
```

区分记忆：**`source` = 把脚本内容"读进来"在自己身上跑；直接执行 = "叫别人"去跑，别人跑完了效果不会带回来。**

## 典型使用场景

- **激活环境**：`source venv/bin/activate`、`conda activate` 内部机制、`source /path/to/lhy/activate`（见 [[activate-script-notes]]）—— 这类脚本需要修改当前 shell 的 `PATH`、代理变量等，必须用 `source` 执行，直接运行没有任何效果，因为改动都留在了已经退出的子进程里。
- **加载/重载配置**：`source ~/.bashrc`、`source ~/.bash_profile` —— 改完配置文件后立即在当前终端生效，不用重新开一个终端。
- **引入函数库**：把公共函数定义写在一个文件里，其他脚本用 `source` 引入并复用（类似其他语言的 `import`/`include`）。

## 常见坑

- 脚本里如果有 `exit`，被 `source` 执行时会直接关闭当前 shell/终端（因为执行的是当前进程，不是子进程），直接运行脚本则只会退出那个子进程。
- 被 `source` 的脚本如果写坏了（比如清空/污染了 `PATH`），影响会留在当前 shell 里，不像子进程出错那样互不影响，需要重开终端或手动修复。
- `.` 和 `source` 行为完全一致，`.` 是更早的 POSIX 标准写法，`source` 是 bash（以及 zsh 等）提供的别名。

## 相关链接

- [[activate-script-notes]] —— 一个必须用 `source` 执行的实际脚本案例（conda + 代理激活）
