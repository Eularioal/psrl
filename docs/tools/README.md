# 开发工具知识索引

这里按用途组织智能体工具与[计算机系统基础](系统基础/README.md)。系统基础涵盖运行环境、进程、文件、网络通信、GPU 和 SSH 远程登录。工具文档按配置、会话使用和故障排查选择。

<!-- figure:map-2f31c326af -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![开发工具知识索引：阅读地图（1/2）](../_static/img/document-figures/map-2f31c326af-screen.png)](../_static/img/document-figures/map-2f31c326af.png)

[![开发工具知识索引：阅读地图（2/2）](../_static/img/document-figures/map-2f31c326af-screen-02.png)](../_static/img/document-figures/map-2f31c326af.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../_static/img/document-figures/map-2f31c326af.svg)。
<!-- /figure:map-2f31c326af -->

## 目录

- [Claude Code CLI](#claude-code-cli)
- [Codex CLI](#codex-cli)
- [Shell / Python 基础](#shell--python-基础)
- [Linux 文件系统](#linux-文件系统)
- [网络与远程登录](#网络与远程登录)
- [文档维护](#文档维护)
- [延伸](#延伸)

## Claude Code CLI

| 文档 | 内容 |
|---|---|
| [命令执行与权限设置](claude-code/01-ClaudeCode_命令执行与权限设置.md) | `!` 手动命令、交给 Claude 执行和权限模式怎样配合，附常用规则示例 |

## Codex CLI

| 文档 | 内容 |
|---|---|
| [Rule、Memory、Skill 与 AGENTS.md](codex/01-智能体配置_Rule与Memory及Skill.md) | 同篇解释三者用途、规则放置、目录作用范围及加载验证，附完整任务例子 |
| [Codex 会话使用](codex/session-management/01-Codex会话_恢复命名与历史回看.md) | 恢复、命名、终端标签、回看问答、图片、持续目标与 TCodex 共享历史 |
| [Codex终端：滚动与Tmux排查](codex/session-management/02-Codex终端_滚动与Tmux排查.md) | 终端回滚、tmux 鼠标路由与共享账号配置范围 |
| [Codex 会话故障恢复](codex/session-management/03-Codex故障_会话数据库与SIGBUS排查.md) | 按故障定位，再查会话数据库、SIGBUS、日志与修复验收证据 |
| [Codex配置：备份与按需恢复](codex/02-Codex配置_备份与按需恢复.md) | 定位 `CODEX_HOME`（未设置时默认 `~/.codex`），用备份与差异选择恢复范围 |

## Shell / Python 基础

| 文档 | 内容 |
|---|---|
| [Shell 与 Python 环境](系统基础/01-运行环境_Shell与Python及Conda.md) | 从 `source`、独立 Python 进程和 `import` 串起环境作用范围，再解释项目激活脚本、PATH、Conda 与启动配置 |

## Linux 文件系统

| 文档 | 内容 |
|---|---|
| [inode 与 Docker 写回阻塞](系统基础/03-文件系统_inode与链接及Docker写回.md) | 用真实文件实验解释 inode、硬链接、重命名、删除后的打开引用，区分容量耗尽与写回阻塞 |
| [文件映射、SQLite 与 SIGBUS](系统基础/02-操作系统_进程与文件映射及SQLite.md) | 从 Codex 崩溃理解进程、线程、虚拟内存和 WAL，配删除与截断的 CPU 教学实验 |

## 网络与远程登录

- [SSH 连接与 IP、网卡、端口](系统基础/06-远程登录_SSH连接与IP网卡端口.md)从 `SSH_CONNECTION` 四列出发，解释连接端点、接口地址、监听端口和 tunnel 的影响。

## 文档维护

- [历史文档时效巡检](文档维护/02-文档时效_2026年11月历史巡检.md)：2026 年 11 月三轮检查的方法、发现与归档依据。
- [文档组织与可读性审查](文档维护/01-文档组织_阅读结构与维护规范.md)：目录整理、[图文解释复核](文档维护/01-文档组织_阅读结构与维护规范.md#图文可读性复核)、覆盖范围及验证记录。

## 延伸

- [知识文档总索引](../README.md)

## 目录补充导航

- [Claude Code 使用](claude-code/README.md)
- [Codex 使用、规则与故障](codex/README.md)
- [Linux 系统基础与故障](linux/README.md)
- [计算机系统基础：环境、进程、文件、网络与GPU](系统基础/README.md)
