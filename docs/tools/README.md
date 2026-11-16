# 开发工具知识索引

一句话结论：这里收录开发工具、Shell、Python 与 Linux 文件系统基础，配套可操作的说明和实验。

## 目录

- [Claude Code CLI](#claude-code-cli)
- [Codex CLI](#codex-cli)
- [Shell / Python 基础](#shell--python-基础)
- [Linux 文件系统](#linux-文件系统)
- [文档维护](#文档维护)
- [延伸](#延伸)

## Claude Code CLI

| 文档 | 内容 |
|---|---|
| [命令执行与权限设置](claude-code/claude-code-usage.md) | `!` 手动命令、交给 Claude 执行和权限模式怎样配合，附常用规则示例 |

## Codex CLI

| 文档 | 内容 |
|---|---|
| [Rule、Memory、Skill 简介](codex/rule-memory-skill.md) | 用工作规范、交接笔记和操作手册解释三者用途，附同一任务中的具体例子与容易混淆之处 |
| [codex/codex-agents-md-scope.md](codex/codex-agents-md-scope.md) | `AGENTS.md` 是什么、作用范围如何按目录层级决定、全局规则 vs 项目规则、`AGENTS.override.md` |
| [Codex 会话管理](codex/codex-session-management.md) | 恢复、会话命名与终端标签的区别、回看问答与图片输入 |
| [Codex 持续目标：使用 /goal 推进任务](codex/session-management/goals.md) | 持续目标、停止条件与具体示例 |
| [Codex 终端滚动与 tmux 排查](codex/session-management/terminal-scrolling.md) | 终端回滚、tmux 鼠标路由与共享账号配置范围 |
| [TCodex 安装与已有会话共享](codex/session-management/tcodex-setup.md) | 启动器安装、认证边界与历史会话共享 |
| [Codex 会话故障恢复](codex/session-management/session-recovery.md) | 按故障定位，再查会话数据库、SIGBUS、日志与修复验收证据 |
| [codex/codex-config-backup-restore.md](codex/codex-config-backup-restore.md) | 定位 `CODEX_HOME`（未设置时默认 `~/.codex`），用备份与差异选择恢复范围 |

## Shell / Python 基础

| 文档 | 内容 |
|---|---|
| [Shell 与 Python 环境](shell-python/python-environments.md) | 从 `source`、独立 Python 进程和 `import` 串起环境作用范围，再解释项目激活脚本、PATH、Conda 与启动配置 |

## Linux 文件系统

| 文档 | 内容 |
|---|---|
| [inode 与 Docker 写回阻塞](linux/inode-basics.md) | 用真实文件实验解释 inode、硬链接、重命名、删除后的打开引用，区分容量耗尽与写回阻塞 |
| [文件映射、SQLite 与 SIGBUS](linux/mmap-sqlite-sigbus.md) | 从 Codex 崩溃理解进程、线程、虚拟内存和 WAL，配删除与截断的 CPU 教学实验 |

## 文档维护

- [历史文档时效巡检](docs-freshness-audit.md)：2026 年 11 月三轮检查的方法、发现与归档依据。
- [本轮文档组织审查](docs-organization-audit.md)：本次可读性整理、分类与逐篇审查入口。

## 延伸

- [知识文档总索引](../README.md)
