# Codex 会话与故障排查导航

> **核心结论**：本文已按功能拆分为 8 篇子文档。只想了解本次修复方法，直接读[修复思路 Map](session-management/repair-map.md)。详细使用方法与故障证据分别查阅下表。

## 目录

- [按任务选择文档](#按任务选择文档)
- [阅读顺序与验证边界](#阅读顺序与验证边界)
- [相关入口](#相关入口)

## 按任务选择文档

| 文档 | 解决的问题 |
|---|---|
| [Codex 会话使用：恢复、命名、回看与图片](session-management/session-usage.md) | 恢复、会话命名与终端标签的区别、回看问答与图片输入 |
| [Codex 持续目标：使用 /goal 推进任务](session-management/goals.md) | 持续目标、停止条件与具体示例 |
| [Codex 终端滚动与 tmux 排查](session-management/terminal-scrolling.md) | 终端回滚、tmux 鼠标路由与共享账号配置范围 |
| [TCodex 安装与已有会话共享](session-management/tcodex-setup.md) | 启动器安装、认证边界与历史会话共享 |
| [Codex 会话数据库恢复与沙箱排查](session-management/session-database-recovery.md) | 历史分页错误、数据库恢复、独立存档与沙箱检查 |
| [Codex Bus error：崩溃证据与文件映射机制](session-management/bus-error.md) | SIGBUS 转储、文件映射机制与验证边界 |
| [Codex 日志数据库与剩余问题修复记录](session-management/log-repair.md) | 日志本地化、旧路径维护、Shell 快照与修复验收 |
| [Codex 修复思路 Map](session-management/repair-map.md) | 一页阅读修复背景、定位方法与结论边界 |

## 阅读顺序与验证边界

- 了解这次怎么修复：先读 [Map](session-management/repair-map.md)，需要证据时再看[日志修复记录](session-management/log-repair.md)。
- 理解为什么会 Bus error：读[崩溃排查](session-management/bus-error.md)，基础概念见[文件映射与 SQLite 教程](../linux/mmap-sqlite-sigbus.md)。
- 日常操作：从[会话使用](session-management/session-usage.md)进入，按需查终端滚动、持续目标或 TCodex。

2026-11-10 本次只整理文档结构。各页保留原始核对日期、版本和实测边界，不把历史验证刷新成当前保证。正文已迁入子文档，原页仅保留导航。

## 相关入口

- [开发工具索引](../README.md)
- [知识文档总览](../../README.md)
- [Codex 配置备份与回退](codex-config-backup-restore.md)
