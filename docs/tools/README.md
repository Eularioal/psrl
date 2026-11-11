# 开发工具知识索引

一句话结论：这里收录开发工具、Shell、Python 与 Linux 文件系统基础，配套可操作的说明和实验。

目录：

- [Claude Code CLI](#claude-code-cli)
- [Codex CLI](#codex-cli)
- [Shell / Python 基础](#shell--python-基础)
- [Linux 文件系统](#linux-文件系统)
- [文档维护](#文档维护)
- [延伸](#延伸)

## Claude Code CLI

| 文档 | 内容 |
|---|---|
| [claude-code/claude-code-permission-modes.md](claude-code/claude-code-permission-modes.md) | 权限模式一览（default/acceptEdits/plan/bypassPermissions）、Shift+Tab 切换、`--dangerously-skip-permissions` 风险、用 `/permissions` 或 settings.json 白名单常用命令的推荐做法 |
| [claude-code/claude-code-run-terminal-commands.md](claude-code/claude-code-run-terminal-commands.md) | 手动运行终端命令看报错的三种方式对比：`!` 前缀（输出自动进入上下文）、手动粘贴报错、让 Claude 直接用 Bash 工具跑 |

## Codex CLI

| 文档 | 内容 |
|---|---|
| [Rule、Memory、Skill 简介](codex/rule-memory-skill.md) | 用工作规范、交接笔记和操作手册解释三者用途，附同一任务中的具体例子与容易混淆之处 |
| [codex/codex-agents-md-scope.md](codex/codex-agents-md-scope.md) | `AGENTS.md` 是什么、作用范围如何按目录层级决定、全局规则 vs 项目规则、`AGENTS.override.md` |
| [Codex 会话与故障排查导航](codex/codex-session-management.md) | 已按功能拆分为 8 篇子文档，见下方入口 |
| [Codex 会话使用：恢复、命名、回看与图片](codex/session-management/session-usage.md) | 恢复、会话命名与终端标签的区别、回看问答与图片输入 |
| [Codex 持续目标：使用 /goal 推进任务](codex/session-management/goals.md) | 持续目标、停止条件与具体示例 |
| [Codex 终端滚动与 tmux 排查](codex/session-management/terminal-scrolling.md) | 终端回滚、tmux 鼠标路由与共享账号配置范围 |
| [TCodex 安装与已有会话共享](codex/session-management/tcodex-setup.md) | 启动器安装、认证边界与历史会话共享 |
| [Codex 会话数据库恢复与沙箱排查](codex/session-management/session-database-recovery.md) | 历史分页错误、数据库恢复、独立存档与沙箱检查 |
| [Codex Bus error：崩溃证据与文件映射机制](codex/session-management/bus-error.md) | SIGBUS 转储、文件映射机制与验证边界 |
| [Codex 日志数据库与剩余问题修复记录](codex/session-management/log-repair.md) | 日志本地化、旧路径维护、Shell 快照与修复验收 |
| [Codex 修复思路 Map](codex/session-management/repair-map.md) | 一页阅读修复背景、定位方法与结论边界 |
| [codex/codex-config-backup-restore.md](codex/codex-config-backup-restore.md) | `config.toml` 实际读 `CODEX_HOME` 而非 `~/.codex`、没有 git 时靠备份副本回退（`cp -p` 先存当前再覆盖）、`diff` 选版本、整文件覆盖的代价与更稳的做法 |

## Shell / Python 基础

| 文档 | 内容 |
|---|---|
| [shell-python/activate-script-notes.md](shell-python/activate-script-notes.md) | `lhy/activate` 环境初始化脚本逐行说明：Conda 清理与接管、`PATH` 重建、缓存隔离、代理与镜像变量，以及新 tunnel 的 source / CODEX_HOME 自启复现 |
| [shell-python/bash-source-command.md](shell-python/bash-source-command.md) | `source`（`.`）命令在当前 shell 进程里执行脚本，环境变量/`PATH`/cwd/函数/别名的修改会保留，和直接执行脚本另开子进程完全不同 |
| [shell-python/python-import-persistence.md](shell-python/python-import-persistence.md) | Python `import` 不会在多次脚本运行之间"留下来"：每次 `python xxx.py` 都是全新进程，下一个脚本能不能 import 成功只取决于当前激活环境的 site-packages |

## Linux 文件系统

| 文档 | 内容 |
|---|---|
| [inode 与 Docker 写回阻塞](linux/inode-basics.md) | 用真实文件实验解释 inode、硬链接、重命名、删除后的打开引用，区分容量耗尽与写回阻塞 |
| [文件映射、SQLite 与 SIGBUS](linux/mmap-sqlite-sigbus.md) | 从 Codex 崩溃理解进程、线程、虚拟内存和 WAL，配删除与截断的 CPU 教学实验 |

## 文档维护

- [docs 文档时效巡检](docs-freshness-audit.md)：全目录静态扫描、架构现状矛盾修订、源码核验依据与未覆盖范围。

## 延伸

- [知识文档总索引](../README.md)
