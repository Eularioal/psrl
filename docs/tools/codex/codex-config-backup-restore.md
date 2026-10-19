# Codex 配置文件的备份与回退：一次 `config.toml` 回退的完整过程

一句话结论：没有 git、也没有"撤销"按钮时，回退配置文件的本质就是**拿一份以前存好的完整副本，整个覆盖掉当前文件**；做之前先把当前版本另存一份，这样这次回退本身也能反悔。

目录：

- [1. 背景：为什么需要回退](#1-背景为什么需要回退)
- [2. 先搞清楚：codex 到底读哪个配置](#2-先搞清楚codex-到底读哪个配置)
- [3. 回退之前：找到可用的"旧版本"](#3-回退之前找到可用的旧版本)
- [4. 实际执行的命令，逐行解释](#4-实际执行的命令逐行解释)
- [5. 回退前后对照（具体例子）](#5-回退前后对照具体例子)
- [6. 这种回退方式的局限](#6-这种回退方式的局限)
- [7. 更稳的做法：只回退想回退的部分](#7-更稳的做法只回退想回退的部分)
- [8. 如何恢复"回退前"的版本](#8-如何恢复回退前的版本)

## 1. 背景：为什么需要回退

场景（真实发生的）：用 `codex` 发请求时报"缺少环境变量"，怀疑 `config.toml` 被人改过，希望恢复到改动之前的样子。

要回退的文件是一个普通文本文件。普通文本文件**自己不带历史记录**：你保存之后，旧内容就被覆盖了，没有任何地方会替你记住"上一次长什么样"。所以回退能不能做，完全取决于**有没有别处存着旧内容**。常见的来源有四种：

| 旧内容的来源 | 怎么取回 | 本次是否可用 |
|---|---|---|
| git 仓库 | `git checkout` / `git restore` / `git show 旧提交:文件` | 否，工作目录 `lw/` 不是 git 仓库 |
| 编辑器本地历史（VS Code Timeline 等） | 在编辑器里点"还原" | 本次没有用，这个只对在 IDE 里改的文件有效 |
| 文件系统快照 | 联系存储管理员 | 本次没有 |
| **手工或工具留下的备份副本** | `cp` 覆盖回来 | **可用，本次就是这个** |

本次目录里恰好有两份备份副本，所以可以回退。

## 2. 先搞清楚：codex 到底读哪个配置

很容易犯的错误：以为配置在 `~/.codex/config.toml`（`~` 就是 `/root`）。

Codex 的规则是：优先看环境变量 `CODEX_HOME`，没设置才默认用 `~/.codex`。本机 `~/.bashrc` 里有一行：

```bash
export CODEX_HOME=${WORKSPACE_ROOT}/.codex
```

所以真正生效的配置是 `lw/.codex/config.toml`，而 `~/.codex/` 下根本没有 `config.toml`。回退之前如果改错了目录，就会出现"改了半天没效果"的情况。验证方法：

```bash
echo $CODEX_HOME        # 看环境变量
ls -la $CODEX_HOME      # 看里面有没有 config.toml 和备份
```

## 3. 回退之前：找到可用的"旧版本"

`lw/.codex/` 里有两份以 `config.toml.backup.<时间戳>` 命名的文件：

```
config.toml.backup.20261015-123622-563378   # 675 字节，model_provider = "OpenAI"
config.toml.backup.20261015-142939-378403   # 1007 字节，model_provider = "zhicloud"
config.toml                                  # 1006 字节，当前版本
```

名字里的 `20261015-123622` 是 2026-10-15 12:36:22，后面一串数字是区分用的序号。

> 说明（推断，没有逐一核实）：这种命名看起来是某个工具在**切换 provider 之前自动留的备份**。证据是：12:36 的备份里 provider 还是 OpenAI，而 `zhicloud_api_key` 文件正好也是 10-15 12:36 创建的，说明有人（或脚本）在那个时间点把 provider 切到了 zhicloud，切之前把旧文件存成了备份。具体是哪个程序做的，没有确认。

用 `diff` 比较各版本，能看出每次改了什么，也就能决定回退到哪一份：

```bash
diff 备份A 备份B       # 左边有、右边没有的行用 < 标，右边新增的行用 > 标
```

本次比较的结论：

| 对比 | 差异 |
|---|---|
| 12:36 备份 → 14:29 备份 | provider 从 OpenAI 换成 zhicloud；model 换成 gpt-6-luna；effort 从 high 改 medium；新增 `[model_providers.zhicloud]` 段；新增两个 psrl-wip 项目信任项 |
| 14:29 备份 → 当前 | 只有 model（luna → astra）和 effort（medium → high）两行 |

差别大到影响行为（换了 provider），所以**不能替用户猜**，最后由用户选定回退到 12:36 版本。

## 4. 实际执行的命令，逐行解释

```bash
cd $CODEX_HOME && \
cp -p config.toml config.toml.backup.20261016-before-revert && \
cp -p config.toml.backup.20261015-123622-563378 config.toml && \
chmod 600 config.toml
```

用 `&&` 连起来的意思是：前一条成功才执行后一条。任何一步失败，后面的就不会运行，避免"备份没存成功就先把原文件覆盖了"。

| 步骤 | 作用 |
|---|---|
| `cd $CODEX_HOME` | 进到真正生效的配置目录，后面全部用相对文件名，不会写错地方 |
| `cp -p config.toml config.toml.backup.20261016-before-revert` | **第一步先保险**：把当前这份（被改过的）另存成新备份。这样这次回退万一选错了，还能回来 |
| `cp -p config.toml.backup.20261015-123622-563378 config.toml` | **真正的回退**：用旧备份的内容整个覆盖 `config.toml` |
| `chmod 600 config.toml` | 把权限设成只有所有者能读写。配置文件里可能含密钥路径或 token，不应对其他用户可读 |

`cp -p` 里 `-p` 的含义是**保留原文件的权限、所有者和修改时间**。这有一个容易让人困惑的副作用：回退之后 `ls -la` 看到的 `config.toml` 修改时间是 `Oct 13 11:43`（那是备份内容当年的时间），**不是**回退发生的时间。如果你之后想用"修改时间"判断文件是什么时候被动过，要记得这次是被 `-p` 保留成旧时间的。回退发生的真实时间，要看那份 `before-revert` 备份的创建，或者 shell 历史。

回退后用一条命令确认关键行：

```bash
grep -n -E '^model_provider|^model =|^model_reasoning_effort' config.toml
```

预期（本次实际输出）：

```
1:model_provider = "OpenAI"
2:model = "gpt-6-astra"
8:model_reasoning_effort = "high"
```

## 5. 回退前后对照（具体例子）

| 项 | 回退前（当前版本） | 回退后（12:36 版本） |
|---|---|---|
| `model_provider` | `zhicloud` | `OpenAI` |
| 接口地址 | `https://api.zcloudapi.com/v1` | `https://api.abaoyun.org` |
| 密钥来源 | 环境变量 `ZHICLOUD_API_KEY`（`env_key`） | `requires_openai_auth = true`，走 `auth.json` 登录信息 |
| `[model_providers.zhicloud]` 段 | 有 | 没有 |
| `psrl-wip` 两个项目信任项 | 有 | **没有（被一并去掉）** |

这里最后一行就是"整文件覆盖"的代价：旧备份里没有那两条信任项，覆盖后它们也消失了，即使它们可能是后来合理添加的。

## 6. 这种回退方式的局限

1. **它是回到某个快照，不是撤销某一次修改。** 想撤销"今天 00:43 那次改动"，得有一份恰好在那次改动前存的备份；如果没有，只能退到更早的快照，同时丢掉中间所有正当改动。
2. **备份本身可能不可信。** 备份就是普通文件，谁都能改、能删。如果怀疑"被人改了"，备份也可能被动过。回退前用 `diff` 看过内容，比盲目覆盖安全。
3. **看不到是谁改的。** 回退只恢复内容，不会告诉你来源。文件所有者都是 `root`，系统里没有额外审计日志的话，无法从文件本身查出是谁。
4. **codex 通常在启动时读配置**（推断，未专门验证），所以已经打开的 codex 会话可能不会立刻采用新配置，需要新开会话。

## 7. 更稳的做法：只回退想回退的部分

如果只是想撤销个别几行，不要整文件覆盖，而是先看差异再手工合并：

```bash
diff -u config.toml config.toml.backup.20261015-142939-378403
```

输出里以 `-` 开头的是当前文件有而备份没有的行，以 `+` 开头的是备份有而当前没有的行。看完后在编辑器里只改需要的几行即可。

更好的长期方案是**把配置放进 git 管理**：

```bash
cd $CODEX_HOME
git init && git add config.toml && git commit -m "baseline"
# 之后任何改动都能 git diff 看到、git checkout -- config.toml 一键撤销
```

注意：`.codex` 里除了 `config.toml` 还有 `auth.json`、`zhicloud_api_key` 等密钥文件，以及体积很大的 sqlite 和 sessions 目录。如果这么做，只 `add` `config.toml`，并用 `.gitignore` 排除其余内容，避免把密钥提交进历史。

## 8. 如何恢复"回退前"的版本

如果发现回退错了：

```bash
cd $CODEX_HOME
cp -p config.toml.backup.20261016-before-revert config.toml
```

## 相关文档

- [codex-agents-md-scope.md](codex-agents-md-scope.md)：`AGENTS.md` 的作用范围（同样位于 `CODEX_HOME` 下的另一类配置）
- [codex-session-management.md](codex-session-management.md)：会话恢复与历史
- [../shell-python/bash-source-command.md](../shell-python/bash-source-command.md)：`.bashrc` 里 `export` 为什么能让所有新 shell 都读到 `CODEX_HOME`
