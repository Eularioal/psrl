# Codex 配置备份与回退：先确认文件，再恢复需要的设置

回退配置的关键是找到一份已知可用的旧内容，并确认这次要恢复哪些设置。完整覆盖会同时恢复模型、服务提供方和项目设置；只想改一项时，比较差异后局部修改更合适。

## 目录

- [先找到实际使用的配置](#先找到实际使用的配置)
- [备份和回退怎样做](#备份和回退怎样做)
- [为什么有时只改几行](#为什么有时只改几行)
- [2026-10-16 的历史案例](#2026-10-16-的历史案例)
- [进一步阅读](#进一步阅读)

## 先找到实际使用的配置

Codex 个人配置位于 `$CODEX_HOME/config.toml`。没有设置 `CODEX_HOME` 时，默认目录是 `~/.codex`；项目配置等其他层级还可能覆盖部分值，详见[官方配置说明](https://learn.chatgpt.com/docs/config-file/config-advanced)。

```bash
codex_config_dir="${CODEX_HOME:-$HOME/.codex}"
printf '%s\n' "$codex_config_dir/config.toml"
```

第一行把实际目录存入任务变量，第二行只打印路径。先确认路径，再检查文件与备份，能避免修改另一份同名配置。

## 备份和回退怎样做

下面是操作示例，需要把 `config.toml.backup.已确认的版本` 换成实际选定的备份文件名。本次文档整理未执行回退。

```bash
codex_config_dir="${CODEX_HOME:-$HOME/.codex}"
codex_backup_stamp=$(date +%Y%m%d-%H%M%S)
cp -p "$codex_config_dir/config.toml" \
  "$codex_config_dir/config.toml.before-restore.$codex_backup_stamp" &&
cp -p "$codex_config_dir/config.toml.backup.已确认的版本" \
  "$codex_config_dir/config.toml"
```

第一次复制保存当前状态，第二次复制恢复选定版本。`&&` 保证第一步成功后才执行覆盖；`cp -p` 尽可能保留原文件的权限和时间等属性。因此，恢复后的修改时间可能仍是备份原来的时间，操作发生时间应另行记录。

恢复前应检查差异。配置可能包含认证相关内容，查看和分享时只摘取与问题相关的设置。恢复后在新启动的会话中核对模型、服务连接与预期操作。

## 为什么有时只改几行

假设旧文件设置了模型 A，之后既改成模型 B，又增加了项目路径。现在只想恢复模型 A：

| 方法 | 结果 |
|---|---|
| 整文件覆盖 | 模型恢复为 A，新增项目设置也回到旧状态 |
| 比较后局部修改 | 模型恢复为 A，保留新增项目设置 |

可以先比较：

```bash
diff -u "$codex_config_dir/config.toml" \
  "$codex_config_dir/config.toml.backup.已确认的版本"
```

输出中 `-` 表示当前文件中的内容，`+` 表示备份中的内容。选定真正需要恢复的字段后，用编辑器修改。需要撤回这次恢复时，使用刚保存的 `before-restore` 副本。

## 2026-10-16 的历史案例

当时 Codex 报缺少环境变量，现场发现两份 2026-10-15 的配置备份。较早版本使用一种服务配置，较晚版本切换了服务提供方，并增加了两项项目信任设置。用户选择较早版本，操作前另存当前文件，再执行整文件覆盖。

记录中的验证结果是：选定的服务提供方、模型和推理设置已恢复；较晚添加的两项项目设置也随之回退。该案例说明了完整快照恢复的实际影响。原记录没有确认是谁或哪个程序改动了配置，因此移除了根据时间戳推断操作者的叙述。

本节保留排障经过及方法，不作为当前机器配置清单。2026-11-16 仅核对官方配置路径规则并整理文字，未读取或修改个人配置。

## 进一步阅读

- 配置中的规则文件如何生效：[AGENTS.md 的作用范围](codex-agents-md-scope.md)。
- 聊天记录和配置分别存什么：[会话修复地图](session-management/session-recovery.md#codex-修复思路-map)。
- 环境变量怎样留在当前终端：[source 的进程示例](../shell-python/python-environments.md#source让环境设置留在当前终端)。
