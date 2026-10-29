# 开发工具知识索引

一句话结论：这里收录开发工具、Shell、Python 与 Linux 文件系统基础，配套可操作的说明和实验。

目录：

- [Claude Code CLI](#claude-code-cli)
- [Codex CLI](#codex-cli)
- [Shell / Python 基础](#shell--python-基础)
- [Linux 文件系统](#linux-文件系统)
- [延伸](#延伸)

## Claude Code CLI

| 文档 | 内容 |
|---|---|
| [claude-code/claude-code-permission-modes.md](claude-code/claude-code-permission-modes.md) | 权限模式一览（default/acceptEdits/plan/bypassPermissions）、Shift+Tab 切换、`--dangerously-skip-permissions` 风险、用 `/permissions` 或 settings.json 白名单常用命令的推荐做法 |
| [claude-code/claude-code-run-terminal-commands.md](claude-code/claude-code-run-terminal-commands.md) | 手动运行终端命令看报错的三种方式对比：`!` 前缀（输出自动进入上下文）、手动粘贴报错、让 Claude 直接用 Bash 工具跑 |

## Codex CLI

| 文档 | 内容 |
|---|---|
| [codex/codex-agents-md-scope.md](codex/codex-agents-md-scope.md) | `AGENTS.md` 是什么、作用范围如何按目录层级决定、全局规则 vs 项目规则、`AGENTS.override.md` |
| [codex/codex-session-management.md](codex/codex-session-management.md) | 会话恢复、问答浏览与搜索、发送图片、TCodex 安装与共享会话、输入历史的区别、历史版本命名限制、`/goal`，以及终端回滚与 tmux 排查 |
| [codex/codex-config-backup-restore.md](codex/codex-config-backup-restore.md) | `config.toml` 实际读 `CODEX_HOME` 而非 `~/.codex`、没有 git 时靠备份副本回退（`cp -p` 先存当前再覆盖）、`diff` 选版本、整文件覆盖的代价与更稳的做法 |

## Shell / Python 基础

| 文档 | 内容 |
|---|---|
| [shell-python/activate-script-notes.md](shell-python/activate-script-notes.md) | `lhy/activate` 环境初始化脚本逐行说明：Conda 清理与接管、`PATH` 重建、缓存隔离、代理与镜像变量、安全注意事项 |
| [shell-python/bash-source-command.md](shell-python/bash-source-command.md) | `source`（`.`）命令在当前 shell 进程里执行脚本，环境变量/`PATH`/cwd/函数/别名的修改会保留，和直接执行脚本另开子进程完全不同 |
| [shell-python/python-import-persistence.md](shell-python/python-import-persistence.md) | Python `import` 不会在多次脚本运行之间"留下来"：每次 `python xxx.py` 都是全新进程，下一个脚本能不能 import 成功只取决于当前激活环境的 site-packages |

## Linux 文件系统

| 文档 | 内容 |
|---|---|
| [inode 与 Docker 写回阻塞](linux/inode-basics.md) | 用真实文件实验解释 inode、硬链接、重命名、删除后的打开引用，区分容量耗尽与写回阻塞 |

## 延伸

- [知识文档总索引](../README.md)
