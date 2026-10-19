# Claude Code 的权限模式（"auto mode"怎么开）

整理日期：2026-09-13。

> **一句话结论**：Claude Code 没有单一的"auto mode"开关，而是 `default → acceptEdits → plan → bypassPermissions` 几档权限模式，风险逐级递增；更推荐的做法不是整体跳过确认，而是用 `/permissions` 或 settings.json 白名单放行"确实安全的高频命令"。

## 权限模式一览

| 模式 | 效果 | 开启方式 |
|---|---|---|
| `default` | 每个工具调用都询问确认 | 默认状态 |
| `acceptEdits` | 自动接受文件编辑（Edit/Write），但 Bash 等仍会询问 | 会话内按 **Shift+Tab** 循环切换；或启动时加 `--permission-mode acceptEdits` |
| `plan` | 只读探索，不执行任何修改 | 同上循环切换，或启动时加 `--permission-mode plan` |
| `bypassPermissions` | **跳过所有权限确认**，包括任意 Bash 命令 | 启动时加 `--dangerously-skip-permissions`（⚠️ 高风险，建议仅在隔离/沙箱环境使用） |

## 更安全的替代方案：白名单常用命令

不整体跳过确认，而是把"确实安全、高频"的命令加入白名单：

- 会话内用 `/permissions` 命令添加规则
- 或直接编辑项目 `.claude/settings.json` / `.claude/settings.local.json`，例如：
  ```json
  {
    "permissions": {
      "allow": [
        "Bash(conda env *)",
        "Bash(curl *)"
      ]
    }
  }
  ```
- 也可以用内置的 **`fewer-permission-prompts`** skill：自动扫描最近会话中常见的只读命令，写入项目 settings.json 白名单，说"帮我精简权限确认"即可触发

## 风险提示

`--dangerously-skip-permissions` 会让 Claude 不经确认执行任意命令（包括删除、覆盖等破坏性操作），不建议在多人共享或生产相关环境中使用。

## 相关链接

- [知识文档索引](../../README.md)
