# Codex `AGENTS.md` 的用途与作用范围

整理日期：2026-09-10。

## 它是什么

`AGENTS.md` 是 Codex 的持久化指令文件，用来告诉 Codex 在某个项目中应遵守的工作约定和上下文。例如：构建与测试命令、代码规范、目录职责、提交或审查要求，以及与用户沟通的固定偏好。

它不是对话记录，不保存聊天内容，不负责给 session 命名、恢复、归档或管理 session 列表。会话的恢复由 `codex resume` 和 Codex 的会话存储处理。

## `lw/AGENTS.md` 是否管理所有对话

不管理所有对话。它的作用范围取决于启动 Codex 时所在的项目目录。

当前目录结构可概括为：

```text
lw/                         项目根目录
├── AGENTS.md               对 lw 及其子目录中的任务有效
├── psrl/
│   └── AGENTS.md           对 psrl 中的任务追加更具体的规则
└── .codex/                 当前环境的 Codex 主目录
```

如果从 `lw/`、`lw/psrl/` 或其他 `lw/` 子目录启动新的 Codex session，`lw/AGENTS.md` 会被读取。因此它适合放置希望在这个工作区内始终遵守的规则。

从 `lw/psrl/` 启动时，Codex 会先加载 `lw/AGENTS.md`，再加载更靠近当前目录的 `psrl/AGENTS.md`；后者优先级更高。每个目录若同时有 `AGENTS.override.md`，则该目录的普通 `AGENTS.md` 会被替代。

它不会自动影响：

- 在 `lw/` 之外启动的 Codex session。
- 已有会话的聊天历史和会话名称。
- 其他机器、其他用户，或使用不同 Codex 主目录的环境。
- 已经在运行的 session。Codex 通常在新 session 启动时构建指令链；修改文件后应新建或重新进入会话，使其重新加载。

## 全局规则和项目规则

| 放置位置 | 典型用途 | 作用范围 |
|---|---|---|
| `$CODEX_HOME/AGENTS.md` | 个人长期偏好，如回答语言、沟通方式、默认检查习惯 | 使用这个 `CODEX_HOME` 启动的所有项目 |
| `lw/AGENTS.md` | 当前工作区共有规则 | `lw/` 及其子目录 |
| `lw/psrl/AGENTS.md` | PSRL 专属架构、代码规范和工作流程 | `psrl/` 及其子目录 |
| 子目录的 `AGENTS.override.md` | 需要替代父目录规则的局部例外 | 该子目录及其下级目录 |

普通终端环境中，`CODEX_HOME` 默认是 `~/.codex`，所以全局文件通常是 `~/.codex/AGENTS.md`。本次环境将 `CODEX_HOME` 设为 `lw/.codex`，因此这里的全局文件对应 `lw/.codex/AGENTS.md`。

## 对当前规则的含义

当前 `lw/AGENTS.md` 中“知识性或总结性回答后提供 Markdown”的规则，会影响在 `lw` 工作区内启动的新会话。`psrl/` 下的会话也会继承它，除非更近目录的规则明确覆盖或冲突。

若该偏好只适用于这个工作区，保留在 `lw/AGENTS.md` 是合适的。若希望它适用于你在任意代码仓库中开启的 Codex 对话，应将同类个人偏好放到全局 `$CODEX_HOME/AGENTS.md`。

## 依据

- [OpenAI 官方文档：使用 AGENTS.md 设置自定义指令](https://learn.chatgpt.com/es-419/docs/agent-configuration/agents-md)
- [OpenAI 官方文档：Codex 自定义概览](https://learn.chatgpt.com/zh-Hans/docs/customization/overview)
- 当前工作区中的 AGENTS.md（本地资料，未随仓库发布） 与 [PSRL AGENTS.md](../../../AGENTS.md)。

## 相关文档

- [Codex 会话管理：恢复与命名](codex-session-management.md)
- Claude Code 对应的规则文件是 CLAUDE.md（本地资料，未随仓库发布），机制类似但文件名不同。
