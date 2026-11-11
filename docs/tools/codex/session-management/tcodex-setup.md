# TCodex 安装与已有会话共享

> **核心结论**：启动器可共享已有 Codex 会话存储，但登录身份和模型网关独立。读取旧会话成功不等于腾讯认证及模型调用已验证。

## 目录

- [TCodex 安装与共享已有会话](#tcodex-安装与共享已有会话)

[返回总览](../codex-session-management.md)。本页沿用各节原有版本、日期和验证边界，拆分不表示重新验证运行状态。

## TCodex 安装与共享已有会话

TCodex 是使用腾讯登录与网关的 Codex 启动器。此处记录本工作区已配置好的使用入口，安装参考用户提供的《TCodex与TClaude使用文档笔记.md》。安装包与启动脚本保存在工作区，不作为 PSRL 项目依赖发布。

**直接使用：**

```bash
tcodex login
tcodex resume --all
```

第一条命令在终端打印腾讯账号登录链接，需要使用者在浏览器完成认证。第二条打开历史会话选择器，`--all` 取消工作目录过滤。需要精确指定会话时，使用 `tcodex resume` 后接该会话的真实 ID，避免 `--last` 在多会话环境中选错。已在另一端运行的同一会话，应先结束那一端的任务再接续。

如果共享系统账号，优先在自己的终端通过 `TCODEX_AUTH_TOKEN` 提供个人凭据。可用 Bash 隐藏输入读取，无需把 Token 写进 shell 历史：

```bash
read -r -s -p 'TCodex API Token: ' TCODEX_AUTH_TOKEN
printf '\n'
export TCODEX_AUTH_TOKEN
tcodex resume --all
```

`TCODEX_AUTH_TOKEN` 只对这个 shell 和后续子进程生效。会话共享不等于登录身份共享，原版 Codex 的账号凭据不能替代腾讯认证。`tcodex login` 的腾讯状态使用启动器自己的用户级目录，不能仅凭 `CODEX_HOME` 相同推断腾讯登录态也存在那里。

**本地安装配置：**

| 项目 | 配置 |
|---|---|
| TCodex 包 | `@tencent/tcodex`，版本 `0.1.4` |
| 安装位置 | 工作区 `.local/share/tcodex/` |
| 启动脚本 | 工作区 `.local/bin/tcodex`，通过用户命令目录中的同名链接调用 |
| 会话目录 | 启动脚本固定 `CODEX_HOME` 为本工作区原有 `.codex` |
| 实际 Codex 程序 | `TCODEX_CODEX_PATH` 指向现有的 `0.163.0` 可执行文件 |
| 后台服务 | 转发给 Codex 时加入 `--no-daemon`，为 TCodex 使用独立运行进程 |
| 网关配置 | TCodex 在本次启动参数中注入腾讯提供方与本地网关地址 |

这样共享的是同一份配置及会话存储，不是复制一份历史，也不是只链接旧版 `sessions/` 目录。现有记录还使用 `state_5.sqlite` 和 `thread_history_1.sqlite`，只链接旧目录不足以覆盖这些会话。启动器注入的腾讯网关参数只影响本次启动，原有配置文件不需要改成腾讯地址。

例如，原来在 `codex` 中讨论过 EnvRouter 的会话，其 ID 在共享目录中保持不变。腾讯登录完成后，把同一个 ID 交给 `tcodex resume` 就能请求恢复这条会话，后续实际模型访问使用 TCodex 的网关。能读取旧记录与腾讯模型调用成功是两项不同的验证。

**版本检查：**

```bash
tcodex --version
tcodex -- -V
```

第一条显示 TCodex `0.1.4` 及包内标注的上游 `0.154.0`。第二条透传给实际执行的 Codex，输出 `codex-cli 0.163.0`。本地脚本使用 `TCODEX_CODEX_PATH` 覆盖包内默认运行时，所以两项版本不同是预期现象。升级 TCodex 的配套运行时不会自动改变脚本固定的路径，后续升级时需重新核对版本与会话兼容性。

**验证结果与边界。** 安装后实际启动 TCodex 的本地 app-server 接口，初始化返回实际版本 `0.163.0` 和预期 `CODEX_HOME`。只读列举命中了已有的全部 35 条未归档会话，并读取其中一条的 5 轮、91 个条目。检查前后原有 `config.toml` 与 `auth.json` 内容一致。验证进程已退出。本次没有完成腾讯账号认证，没有发送模型请求，因此不将本地历史读取成功描述为模型对话已验证。
