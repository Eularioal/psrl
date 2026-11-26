# Codex 会话归档

> 核心结论：随仓库保存会话文件、名称索引和数据库备份。包含聊天正文，不含独立认证文件或配置文件。

- [2026-11-09 归档](codex-20261109-180614.tar.gz)：54 份会话文件、名称和元数据、状态库与历史库在线备份。
- [SHA-256 校验文件](codex-20261109-180614.tar.gz.sha256)。
- [会话管理与恢复说明](../session-management/01-Codex会话_恢复命名与历史回看.md)。

在此目录执行 `sha256sum -c codex-20261109-180614.tar.gz.sha256` 校验。恢复时先解压到独立目录，不覆盖正在使用的存储。会话文件与数据库不是跨文件原子快照，后续会话不会自动进入此包。

<!-- figure:session-history-branches -->
这张图沿同一个场景解释：从同一步分叉后，哪些历史相同，后续修改又属于哪条讨论？

[![会话历史：共同前缀与分支后的新消息分开保存（1/2）](../../../_static/img/document-figures/session-history-branches-screen.png)](../../../_static/img/document-figures/session-history-branches.png)

[![会话历史：共同前缀与分支后的新消息分开保存（2/2）](../../../_static/img/document-figures/session-history-branches-screen-02.png)](../../../_static/img/document-figures/session-history-branches.png)

原讨论已有消息 1 至 4，随后分出 A、B 两条讨论，各自继续追问不同问题。 对话分叉描述历史关系，不能把它等同于复制文件系统、Python 内存或正在运行的实验。 概念图。实际命令、继承范围与恢复方式以正文核对过的客户端版本为准。 [查看矢量图](../../../_static/img/document-figures/session-history-branches.svg)。
<!-- /figure:session-history-branches -->
