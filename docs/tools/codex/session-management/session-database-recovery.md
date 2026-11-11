# Codex 会话数据库恢复与沙箱排查

> **核心结论**：先区分状态库、历史库和原始会话文件。保全原文件、在副本验证恢复，并协调所有使用者停止写入后切换。数据库故障与沙箱启动故障分别检查。

## 目录

- [Results：恢复时报 bounded thread item page](#results恢复时报-bounded-thread-item-page)
- [resume 数据库错误与终端启动失败](#resume-数据库错误与终端启动失败)
  - [Results：2026-11-09 只读排查](#results2026-11-09-只读排查)
  - [Results：从本地会话文件验证恢复](#results从本地会话文件验证恢复)
  - [Results：恢复后的独立会话存档](#results恢复后的独立会话存档)
  - [Results：沙箱启动提示的再次核查](#results沙箱启动提示的再次核查)

[返回总览](../codex-session-management.md)。本页沿用各节原有版本、日期和验证边界，拆分不表示重新验证运行状态。

## Results：恢复时报 bounded thread item page

2026-11-09 后续恢复两个长会话时出现 `failed to load a bounded thread item page`。服务进一步返回 SQLite 错误 `database disk image is malformed`。只读检查发现 `thread_history_1.sqlite` 多处页面异常，`state_5.sqlite` 也随后报损坏。两个原始 JSONL 会话文件逐行解析通过。名称字段可读不等于历史分页可读，[会话名称检查](session-usage.md#results名称混乱的只读核查)没有覆盖这个执行路径。

可以把 JSONL 理解为原始记录，历史库是界面分页所用的数据。只移开历史库后直接列消息会得到空页，必须恢复会话，触发程序重新处理原始记录。本轮隔离验证包含真实 `thread/resume` 及 `thread/items/list`：两个会话分别返回 10 条非空历史消息，重建历史库通过完整性检查。未运行新模型轮次，也未截短原始会话。

状态恢复候选来自当天较早的自动数据库备份，修复索引后完整性检查通过，保留 50 条会话元数据记录，并重新写入已确认的 25 个原名称。元数据时间、计数等可能回到备份时点，不能称为无差别恢复到最新状态。原始对话文件保持不动，损坏数据库另存备份。数据库反复损坏的底层原因仍未确定。

以下为当时待切换阶段的记录，后续结果见[恢复后的独立会话存档](#results恢复后的独立会话存档)。切换前必须先退出使用同一个 `CODEX_HOME` 的 Codex，再停止其 daemon，避免打开中的数据库与替换文件产生冲突。[修复脚本](../scripts/repair-session-history.py)会拒绝仍有本机进程打开目标数据库的情况。它仅在主状态库检查失败时安装预先验证的候选库，并移开历史库供下一次 resume 重建。无法检查其他节点的打开状态，多节点使用者必须先全部退出。

在工作区根目录的普通终端中执行，使用本次已准备好的候选库：

```bash
codex app-server daemon stop
python psrl/docs/tools/codex/scripts/repair-session-history.py \
  --codex-home "$CODEX_HOME" \
  --replacement-state "$CODEX_HOME/backups/resume-repair-20261109/repaired-state.sqlite" \
  --apply
codex resume 01a18571-7d79-7890-bbb7-d7c35c736a01
```

脚本在临时目录验证了活动文件占用时拒绝执行、损坏状态库替换、历史库隔离和原文件保留。隔离副本恢复通过不等于用户前台已经恢复成功，正式切换及前台恢复仍待执行。此流程修复可恢复性，不承诺解决底层重复损坏原因。

---

## resume 数据库错误与终端启动失败

`Unable to check resumed folder` 后面如果跟着 `thread/read failed`，先检查会话信息读取。它本身不能证明项目目录不存在。SQLite 是保存在文件中的数据库，文件存在也不代表内部结构可读。

### Results：2026-11-09 只读排查

本节限于 Codex CLI 0.167.0 的本次检查，不重新验证本文其他章节。本轮未替换活动数据库、未重启后台服务。

| 检查对象 | 实际观察 | 结论边界 |
|---|---|---|
| 终端执行 | 沙箱内 Bash、sh 和不同工作目录均报进程创建 ENOENT，用户关闭沙箱后 `pwd` 成功 | 当前终端可用，故障与沙箱执行路径有关，不能说 shell 文件缺失 |
| 状态库 `state_5.sqlite` | 只读打开后查询报 `database disk image is malformed` | 状态库存在结构损坏，足以解释会话列表读取失败 |
| 历史库 `thread_history_1.sqlite` | 能读 schema，但 `PRAGMA quick_check(3)` 返回 `Failed to read ptrmap` | 部分内容可读不等于完整性正常，历史可恢复范围待核实 |
| 日志库 `logs_2.sqlite` | 元数据读取及历史写入投影失败，底层为 SQLite code 11 | 保存也受到影响，界面仍能回答不代表已可靠持久化 |
| 文件系统 | `findmnt` 显示 `fuse.dop-fuse` | 不是已确认的 NFS，尚未证明共享存储或并发是损坏诱因 |
| 后台服务环境 | 继承 PATH 首项指向已不存在的 `tmp/arg0-linux/codex-arg0…` 目录 | 沙箱辅助入口过期是具体嫌疑，未捕获失败调用的实际可执行路径，仍不能定为最终根因 |

例如本次 `resume` 需要读取会话所属目录，而这个字段属于会话元数据。读取元数据时数据库报错，界面就显示“无法检查恢复目录”。项目目录本身仍存在，修改工作目录或反复加 `--all` 不能修复数据库内部结构。这个解释依据报错层次及只读查询，没有查到对应版本的本地 Rust 源码，不是逐函数源码追踪。

以下最小诊断只用 Python 标准库，每份数据库最多检查五秒，不创建缺失数据库、不执行修复。`CODEX_HOME` 使用当前终端设置，未设置时使用用户主目录下的 `.codex`。本次已执行等价的只读检查，得到上表结果。

```bash
python - <<'PY'
import os
import sqlite3
import time
from pathlib import Path

root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
for name in ("state_5.sqlite", "thread_history_1.sqlite"):
    path = root / name
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro",
                             uri=True, timeout=2) as db:
            deadline = time.monotonic() + 5
            db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            print(name, db.execute("PRAGMA quick_check(3)").fetchall())
    except sqlite3.Error as error:
        print(name, type(error).__name__, str(error))
PY
```

正常结构返回 `ok`。`malformed` 或页面检查错误表示结构异常，`locked` 表示锁冲突，`interrupted` 表示检查预算耗尽，不能混为一类。只读检查不保证活动进程随后不会改变数据库。

恢复前应协调使用该存储的所有客户端停止写入，保全数据库、WAL 和 SHM 伴随文件，再在副本上检查备份与可恢复记录。WAL 是预写日志，可能包含尚未合入主文件的事务，不能先删除它。恢复后仍需验证会话列表、指定会话读取和继续保存。关闭沙箱只恢复了本次命令执行，没有修复数据库。本轮另观察到其他进程正在准备恢复脚本，因此未并行执行数据库替换。

[官方共享存储说明](https://learn.chatgpt.com/docs/environments/network-file-systems)解释 NFS 与 WAL 的限制，以及迁移前停止全部客户端的要求。它只能作为机制参考，不能证明本次 FUSE 挂载是损坏原因。数据库最初如何损坏、沙箱辅助目录为何消失、历史可恢复范围仍待核实。

### Results：从本地会话文件验证恢复

同日后续检查找到 53 份 `sessions/` 下的 JSONL 会话文件，合计约 210 MB。这里“本地”指服务器 `CODEX_HOME` 下的文件，该目录位于共享挂载，并不是用户笔记本上的副本。JSONL 是逐行 JSON 记录，可以保存会话元信息、用户输入、助手输出及工具事件。状态数据库保存用于列表与恢复的元数据，历史数据库保存可分页读取的历史。文件保留与数据库可用是不同条件。

例如某条旧对话的 JSONL 文件还在，但状态库内部索引损坏，列表查询仍会失败。恢复可以先从健康副本获得会话 ID 和文件路径，再由 Codex 读取会话文件生成新的历史数据库，不需要重新询问模型生成旧答案。文件中从未成功保存的内容则无法凭此补回。

这次在独立本机临时目录执行，所有恢复读写均针对副本，未启动模型生成：

1. 复查已有修复候选，状态库完整性通过，原含 50 条会话索引。
2. 复制全部 53 份会话文件。对三个原候选未收录的新会话，通过原文件路径调用当前 Codex 服务的 `thread/resume`。
3. 为历史使用过的不同 provider 分组运行隔离服务。最初配置缺失和 provider 不匹配导致拒绝恢复，补齐隔离配置后再验证。测试 provider 的网络地址指向本机未提供服务的端口，不调用模型服务。
4. 53 条会话均完成 `thread/resume`，其中 46 条通过 `thread/items/list` 读出历史页。另 7 条返回空页，核查源文件只有会话元信息及事件，没有聊天消息，因此不把它们算成成功恢复非空历史。
5. 新状态库有 53 条索引，新状态库和历史库完整性检查均返回 `ok`。该证据证明可打开会话及读取历史页，没有逐条对比全部消息，也不保证故障期间最后几条消息已落盘。
6. 在另一份隔离副本上执行安装脚本，确认旧库与伴随文件保留到备份目录、候选状态库安装成功、完整性通过。活动目录的试运行则因后台服务仍持有数据库而主动拒绝，没有替换活动数据。

恢复包保存在用户自己的 `CODEX_HOME/backups/resume-recovery-20261109-verified/`，没有提交到项目仓库。包括 53 条索引的候选库、会话文件快照、损坏文件的证据副本、验证结果和安装脚本。活动数据库文件的直接拷贝只作证据，不冒充事务一致的数据库备份。

最终安装不直接搬回隔离验证生成的历史库。验证时副本会话可能继续产生元信息，而原会话也还在增长，两边的文件偏移不应混用。安装脚本保留旧历史库，之后由 Codex 从原始会话文件重建，尽量纳入停止前已经写入的内容。

**执行下面命令前，先结束所有使用同一目录的 Codex 客户端，包括其他机器和当前对话。** 当前对话也依赖待停止的后台服务，不能在它继续运行时安全换库。下面的确认参数表示执行者已核实所有机器停止写入，脚本另外检查本机打开文件的进程。

```bash
codex app-server daemon stop
python "$CODEX_HOME/backups/resume-recovery-20261109-verified/apply.py" \
  --codex-home "$CODEX_HOME" \
  --replacement-state "$CODEX_HOME/backups/resume-recovery-20261109-verified/replacement-state.sqlite" \
  --all-hosts-stopped --apply
```

只有安装成功后再启动 Codex，恢复指定会话并核对最后一条已保存消息。安装脚本运行期间不要重开客户端。如果报告仍有持有数据库的 PID，应先核实并关闭对应客户端，不绕过检查。`--all-hosts-stopped` 不能代替跨机器核查。安装包只修复这次已验证的状态库与历史库故障，不声称解决共享存储的根因，也不修复目标、记忆等其他数据库。

本次交付状态：副本恢复和隔离安装验证完成，活动库仍未替换。待全部使用者停止写入后才能完成切换与前台恢复验证。

### Results：恢复后的独立会话存档

用户确认 resume 和名称恢复后，再次只读检查状态库及历史库，两者均通过完整性检查。已在工作区 `session-archives/` 建立时间戳存档，包含 54 份会话 JSONL、名称索引、输入历史、完整会话元数据导出，以及通过 SQLite 在线备份接口生成的状态库和历史库。新增会话使文件数从前次检查的 53 增至 54。

压缩包约 81 MB，保存逐文件 SHA-256 摘要和压缩包摘要，并已从压缩包重新读取会话文件核对摘要。未包含认证文件或配置。会话本身可能含敏感内容，应按私有资料保存，不提交到项目仓库。

此存档在 Codex 运行时生成，各数据库分别使用一致的在线备份，会话文件只保存读取时已完成的 JSON 行。数据库与会话文件之间不是同一瞬间的原子快照，不能声称覆盖备份后继续产生的消息。

工作区 `session-archives/archive-sessions.py` 可重新生成一份时间戳存档，不覆盖旧包。该脚本绑定本工作区存储位置，不是仓库发布工具。本次没有配置定时执行。存档与原目录仍位于同一共享存储，可防范再次修改或数据库损坏，不能防范整块存储丢失。需要后者时应将压缩包与摘要另外下载或复制到独立存储。

### Results：沙箱启动提示的再次核查

后续检查发现后台服务已更换进程，继承的沙箱辅助目录及 `codex-linux-sandbox` 链接均存在，不能继续把此前的入口目录缺失当作当前事实。此前关闭沙箱时普通命令可执行。用户随后切回受限沙箱，已实际验证 Bash 和 Python 能启动，工作区及 `/tmp` 内创建、读取、删除临时文件成功，对 `/root` 的临时文件创建被 `Read-only file system` 拒绝。上述调用未请求沙箱外执行，证明本次受限沙箱的基本执行与文件写入边界正常。成功创建的测试文件已清理，网络限制、其他会话及长期稳定性未在本次验证范围内。

当前启动日志明确提示系统 PATH 找不到 bubblewrap，同时说明暂时使用内置 bubblewrap。bubblewrap 是 Linux 上用于限制进程文件系统及其他资源视图的辅助工具，这条提示本身不等于沙箱执行失败。例如只看到启动警告、后续命令正常，与出现 `Failed to create unified exec process` 导致命令没有启动，是不同状态。还需要用户提供反复出现的原始提示来确定属于哪种情况。

启动日志也报告三个不识别的设置被忽略：`disable_response_storage`、顶层 `network_access` 和 `windows_wsl_setup_acknowledged`。这是配置兼容性提示，不能直接归因于某条会话损坏。本轮未安装系统包或修改配置。另在只读访问日志数据库时出现 `disk I/O error`，尚未证明与沙箱同源，不据此宣称数据库再次损坏。
