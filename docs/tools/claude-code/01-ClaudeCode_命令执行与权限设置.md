# ClaudeCode：命令执行与权限设置

> **核心结论**：自己运行命令并让 Claude 读取输出，可以使用 `!`；让 Claude 选择并执行操作时，由权限规则决定可自动执行的范围。下面把两种工作方式和设置放在一起说明，产品核对日期沿用各节记录。

## 目录

- [Claude Code 中怎样运行终端命令](#claude-code-中怎样运行终端命令)
  - [三种工作方式](#三种工作方式)
  - [连续输入命令](#连续输入命令)
- [Claude Code 权限模式：怎样减少操作确认](#claude-code-权限模式怎样减少操作确认)
  - [常用模式怎样选择](#常用模式怎样选择)
  - [只放行一条常用命令的例子](#只放行一条常用命令的例子)

## Claude Code 中怎样运行终端命令

在 Claude Code 输入框中输入 `!命令`，命令输出会进入对话，Claude 可以接着分析。需要它自行选择命令、修改并再次验证时，直接用自然语言交代任务即可。

<!-- figure:terminal-scroll-layers -->
这张图沿同一个场景解释：终端外层能滚动，为什么内部列表仍停在原地？

[![终端滚动：外层回滚、tmux历史和应用内部列表是三层（1/2）](../../_static/img/document-figures/terminal-scroll-layers-screen.png)](../../_static/img/document-figures/terminal-scroll-layers.png)

[![终端滚动：外层回滚、tmux历史和应用内部列表是三层（2/2）](../../_static/img/document-figures/terminal-scroll-layers-screen-02.png)](../../_static/img/document-figures/terminal-scroll-layers.png)

鼠标在一个终端中操作，里面先运行 tmux，再运行具有自身列表的交互应用。 同一个滚轮事件可能在不同层被消费。外层响应不能证明内层收到事件。 绘制的教学界面。快捷键、鼠标选项与版本差异按正文中的具体环境确认。 [查看矢量图](../../_static/img/document-figures/terminal-scroll-layers.svg)。
<!-- /figure:terminal-scroll-layers -->

### 三种工作方式

| 方式 | 例子 | 输出怎样交给 Claude |
|---|---|---|
| 在输入框直接运行 | `!python --version` | 自动加入当前对话 |
| 在外部终端运行 | 自己运行后粘贴完整报错 | 手动粘贴 |
| 交给 Claude 执行 | “运行这个测试，定位并修复失败原因” | Claude 读取工具输出并继续处理 |

例如先输入 `!python --version`，终端输出 `Python 3.12.x` 后，Claude 就能基于这个版本分析依赖问题。示例版本只说明输出形式。

`!` 是用户直接运行命令的入口，适用该入口自身的执行机制；交给 Claude 的工具调用则按[权限模式](#claude-code-权限模式怎样减少操作确认)处理。

### 连续输入命令

在空输入框按 `!` 可以进入 shell 输入模式。需要退出时，先清空当前输入，再使用退格、Esc 或 Ctrl+U 等界面支持的退出操作。命令在当前会话工作目录执行，运行前注意目录是否符合预期。

核对范围：2026-11-16 阅读 [Claude Code 官方交互说明](https://code.claude.com/docs/en/interactive-mode#bash-mode-with--prefix)，本次未启动命令会话。若要进一步理解“为什么 source 后的环境只影响这个终端”，读 [source 的进程示例](../系统基础/01-运行环境_Shell与Python及Conda.md#source让环境设置留在当前终端)。


## Claude Code 权限模式：怎样减少操作确认

Claude Code 的权限模式决定哪些操作自动执行、哪些需要确认。如果只是想减少重复确认，可以先允许固定的常用命令；如果希望自动修改项目文件，可以选择 `acceptEdits`。`auto` 模式也已存在，是否可用取决于安装版本与账户支持。

### 常用模式怎样选择

以下按 2026-11-16 的[官方权限模式文档](https://code.claude.com/docs/en/permission-modes)整理；本次核对文档，未修改本机设置。

| 模式 | 主要行为 | 适合的工作方式 |
|---|---|---|
| `default` | 按权限规则执行，读取等已允许操作可直接进行，其他操作按需确认 | 一边工作一边确认需要的能力 |
| `acceptEdits` | 自动接受文件编辑及文档所列的常见文件系统操作，其他命令仍按规则处理 | 已决定让 Claude 修改项目 |
| `plan` | 以分析和制定方案为主，限制修改 | 希望先看方案 |
| `auto` | 由后台分类器评估操作，减少人工确认 | 支持该模式的版本与账户 |
| `dontAsk` | 使用已允许的操作，其余操作直接拒绝 | 运行范围固定的自动化任务 |
| `bypassPermissions` | 跳过大部分权限提示，仍受部分既有约束影响 | 由外部隔离环境约束执行范围 |

它们对应不同工作方式。启动时可指定模式，例如：

```bash
claude --permission-mode acceptEdits
```

会话中也可以通过权限界面或模式切换操作调整。实际可选项以当前 CLI 帮助及界面为准。

### 只放行一条常用命令的例子

例如你经常让 Claude 查看修改，希望 `git status` 自动执行，修改文件和其他命令继续走原有规则。可以在 `/permissions` 中增加这条允许规则，也可以在项目设置文件中配置：

```json
{
  "permissions": {
    "allow": ["Bash(git status)"]
  }
}
```

这样做把授权范围直接对应到常用操作。规则语法、匹配方式与设置文件作用范围见[官方权限配置](https://code.claude.com/docs/en/permissions)。

如果你只是自己运行一条命令，并希望 Claude 看到输出，读[在会话中运行终端命令](#claude-code-中怎样运行终端命令)。
