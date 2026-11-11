# `lhy/activate` 脚本说明与新 tunnel 启动配置

> **核心结论**：环境脚本要用 `source` 在当前 Bash 中加载。新 tunnel 的终端需在自己的 `~/.bashrc` 中配置环境脚本和 `CODEX_HOME`，共享工作区存在并不代表新入口的启动文件已配置。

目录：

- [执行流程](#执行流程)
- [安全与副作用](#安全与副作用)
- [按代码解释](#按代码解释)
- [如何让每个新 shell 自动执行该脚本](#如何让每个新-shell-自动执行该脚本)
- [新 tunnel 两项配置的复现](#新-tunnel-两项配置的复现)

目标脚本：`${PSRL_ENV_SCRIPT}`

该文件是供 Bash **以 `source` 方式执行**的环境初始化脚本。它不只是把 `conda` 加入 `PATH`，还会清理当前 Conda 状态，并配置代理、模型下载镜像和项目变量。

推荐调用方式：

```bash
source ${PSRL_ENV_SCRIPT}
```

或在脚本所在目录执行：

```bash
source ./activate
```

不能直接以 `./activate` 运行：那会在子 shell 中设置变量，结束后不会留在当前终端。脚本特别提示，在该目录使用 `source activate` 可能会优先命中 `PATH` 中其他 Conda 的同名脚本，因而应写成 `source ./activate`。

## 执行流程

1. **指定 Conda 根目录**

   设置 `CONDA_HOME` 为该用户目录下的 `anaconda3`。后续 Conda hook、包缓存和 `PATH` 都基于这个目录。

2. **退出并清除已有 Conda 环境**

   脚本依据 `CONDA_SHLVL` 循环执行 `conda deactivate`，尽可能退出所有已叠加的 Conda 环境；随后删除 `CONDA_EXE`、`CONDA_PREFIX`、`CONDA_DEFAULT_ENV` 等变量及当前 shell 中的 `conda` 函数。目的在于避免机器上的多个 Anaconda/Miniconda 安装混用。

3. **从 `PATH` 剔除其他 Conda 路径**

   它逐项检查冒号分隔的 `PATH`，删除包含 `anaconda3` 或 `miniconda` 的目录，再将目标安装的 `anaconda3/bin` 加回开头。这使得随后执行的 `python`、`pip`、`conda` 会优先来自指定安装。

   注意：该匹配规则比较宽泛，只要目录名包含这些字符串就会被移除；如果某个非 Conda 工具恰好位于这种目录名下，也会暂时从 `PATH` 消失。

4. **隔离 Python 和 Conda 缓存**

   - `CONDA_PKGS_DIRS` 指向当前安装内的 `pkgs`，避免与别的 Conda 安装共用下载包缓存。
   - `PYTHONNOUSERSITE=1` 禁用用户级 site-packages，避免 `~/.local` 中的 Python 包覆盖环境内的包。

5. **加载 Conda 的 Bash hook**

   首选执行 `conda shell.bash hook` 并 `eval` 其输出，使 `conda activate` 在当前 shell 可用。若失败，则回退到加载 `etc/profile.d/conda.sh`；再失败时只把 Conda 的 `bin` 加入 `PATH`。最后还会再次确保目标 `bin` 位于 `PATH`。

   这里仅初始化 Conda 命令，**没有自动执行 `conda activate <环境名>`**。因此执行后通常处于 base 或未激活状态，取决于 Conda hook 的行为和原 shell 状态；如项目需要特定环境，仍应另行运行 `conda activate <env>`。

6. **配置网络、下载和项目变量**

   脚本设置 HTTP/HTTPS 代理、UV 下载超时、Hugging Face 镜像地址，以及项目工作目录 `PSRL_WORKSPACE`。这会覆盖当前 shell 原有的 `http_proxy`、`https_proxy`、`UV_HTTP_TIMEOUT`、`HF_ENDPOINT` 和 `PSRL_WORKSPACE`。

   它还会从可选的 `NODE_IP_LIST` 中移除 `:8`，追加到已有 `no_proxy`，并强制保证 `localhost` 和 `127.0.0.1` 不走代理，最后打印最终的 `no_proxy` 和 Conda 路径。

## 安全与副作用

- 脚本中直接导出了一个 W&B API 凭据。不要将该文件复制到公开仓库、日志或聊天记录；更理想的做法是从受权限控制的环境变量或密钥管理服务读取它。
- 由于它会清空 Conda 相关变量、重写 `PATH`、覆盖代理相关变量，适合在新开终端或明确要切换到该项目环境时使用；不适合希望保留现有 Python/Conda 环境的终端。
- 文件中的说明和实际可执行路径使用 `${WORKSPACE_ROOT}`，而当前访问路径是 `${WORKSPACE_ROOT}`。当前两者下的目标 `conda` 都可执行，因而脚本此刻能工作；维护时仍建议确认这是否是预期的挂载别名，避免未来环境迁移后路径失效。

## 按代码解释

### 第 1--5 行：调用说明和固定配置

```bash
CONDA_HOME="${WORKSPACE_ROOT}"
```

这是 shell 变量赋值。等号两侧不能有空格；双引号使路径中即使有空格也能安全表示。这里没有 `export`，所以 `CONDA_HOME` 只供本脚本和随后由它 `source` 的代码使用，不会自动传给之后启动的子进程。第 1--3 行是注释，强调必须由当前 shell `source`；脚本本身修改的 `PATH`、函数和环境变量才会留在当前终端。

### 第 8--15 行：卸载原有 Conda

```bash
case "${CONDA_SHLVL:-0}" in ''|*[!0-9]*) CONDA_SHLVL=0 ;; esac
```

`${CONDA_SHLVL:-0}` 是带默认值的参数展开：变量未设置或为空时取 `0`。`case ... in` 根据模式匹配：

- `''` 匹配空字符串；
- `*[!0-9]*` 匹配含有任意非数字字符的值；
- 两个模式由 `|` 表示“任一匹配”。

匹配到异常值时，把层数归零，避免下一行的整数比较报错。`;;` 是该 `case` 分支的结束符。

```bash
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate >/dev/null 2>&1 || break
    ...
done
```

`[` 是 `test` 命令，`-gt` 表示“整数大于”。只要 Conda 嵌套层数大于零，就调用一次 `conda deactivate`；通常每次会使 `CONDA_SHLVL` 减一。`>/dev/null 2>&1` 同时丢弃标准输出和标准错误，`|| break` 表示命令失败时退出循环，防止死循环。循环内再次校验变量是数字，防御某个异常 hook 改坏该变量的情况。

```bash
unset CONDA_EXE ... _CE_CONDA
unset -f conda __conda_activate __conda_reactivate __conda_hashr >/dev/null 2>&1
```

第一条删除 Conda 留在环境中的变量。第二条的 `-f` 表示删除 **shell 函数**，不是删除磁盘文件；错误输出被隐藏，因为这些函数不一定存在。这样做后，后续 `conda` 只会由目标 Anaconda 的初始化代码重新定义。

### 第 18--29 行：重建 `PATH`

```bash
_ifs_save=$IFS
IFS=:
for _entry in $PATH; do
```

`IFS` 是 shell 对未加引号文本做分词时的分隔符。先保存其原值，再设为冒号，令 `for` 将 `PATH` 按目录逐项拆分。`_entry` 是每一个路径条目。

```bash
case "$_entry" in
    *anaconda3*|*miniconda*) continue ;;
esac
```

若条目包含 `anaconda3` 或 `miniconda`，`continue` 会跳过本次循环，因此该路径不会进入新 `PATH`。其余条目通过下面的条件表达式被用冒号重新串接到 `_clean_path`；第一个条目不加前导冒号。

```bash
IFS=$_ifs_save
PATH="$_clean_path"
unset _clean_path _ifs_save _entry
```

循环后恢复 `IFS`，以免影响用户后续输入；再把清理结果赋给当前 shell 的 `PATH`，并删除临时变量。这里的 `PATH=` 尚未 `export`，但在当前 shell 中它原本通常已具有 export 属性；随后第 42 或 48 行会明确以 `export PATH=...` 把目标 Conda bin 加入。

### 第 32--33 行：隔离缓存和用户包

```bash
export CONDA_PKGS_DIRS="$CONDA_HOME/pkgs"
export PYTHONNOUSERSITE=1
```

`export` 令变量对子进程也可见。Conda 会把已下载的包缓存到 `CONDA_PKGS_DIRS`，Python 看到 `PYTHONNOUSERSITE=1` 时不加载用户目录中的第三方包，避免环境内外依赖混用。

### 第 35--49 行：让目标 Conda 接管当前 shell

```bash
__conda_setup="$("$CONDA_HOME/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
```

`$(...)` 是命令替换：运行目标 Conda 的 `shell.bash hook`，并将它打印的 Bash 代码保存到 `__conda_setup`。`2> /dev/null` 只隐藏标准错误。紧随其后的 `$?` 是上一条命令的退出码，零表示成功。成功时 `eval` 在**当前 shell**执行这段 Conda 生成的代码，通常会定义 `conda` 函数及激活/退出辅助函数。这是能够使用 `conda activate` 的关键。

```bash
else
    if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
        . "$CONDA_HOME/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_HOME/bin:$PATH"
    fi
fi
```

若 hook 运行失败，先检查兼容初始化文件 `conda.sh` 是否存在。`.` 是 `source` 的简写，同样在当前 shell 加载它；若连该文件都没有，只能将 `bin` 加入 `PATH`，此时可执行 `conda` 二进制，但 `conda activate` 不一定可用。

第 46--49 行把 `PATH` 包在冒号中做 `case` 匹配，从而准确判断 `CONDA_HOME/bin` 是否已经是一个完整条目；缺失时才前置添加，避免重复。随后 `unset __conda_setup` 删除存放生成脚本的临时变量。

### 第 51--53 行：被关闭的编译器配置

三行均以 `#` 注释，当前完全不会执行。去掉注释后，会将 C/C++ 编译器指向 GCC Toolset 13，并加载该工具链环境；这通常是为编译含原生扩展的 Python 包准备的。

### 第 55--60 行：网络和项目设置

```bash
export http_proxy=...
export https_proxy=...
export UV_HTTP_TIMEOUT=3000
export HF_ENDPOINT=...
export WANDB_API_KEY=...
export PSRL_WORKSPACE=...
```

这些赋值没有条件判断，会覆盖已有同名变量。前两个会让支持小写代理变量的工具经指定代理访问网络；`UV_HTTP_TIMEOUT` 调整 uv 的 HTTP 超时；`HF_ENDPOINT` 让 Hugging Face 客户端改用镜像；`PSRL_WORKSPACE` 提供项目根路径。API 密钥已存在于原脚本中，但不应在终端输出、提交或转发。

### 第 62--71 行：构建不走代理名单

```bash
NODE_IPS=$(echo "${NODE_IP_LIST:-}" | sed 's/:8//g')
```

先读取可选变量 `NODE_IP_LIST`（不存在或为空则是空串）。管道 `|` 把 `echo` 输出交给 `sed`，`s/:8//g` 会删除每一次出现的文本 `:8`。例如 `10.0.0.1:8,10.0.0.2:8` 会变为 `10.0.0.1,10.0.0.2`。这不是通用的“删除端口”逻辑，只会删除端口值恰为 `8` 的字面量。

```bash
EXISTING_NO_PROXY="${no_proxy:-}"
if [ -n "$EXISTING_NO_PROXY" ]; then
    NEW_NO_PROXY="$EXISTING_NO_PROXY,$NODE_IPS"
else
    NEW_NO_PROXY="$NODE_IPS"
fi
export no_proxy="localhost,127.0.0.1,$NEW_NO_PROXY"
```

`-n` 判断字符串是否非空。若用户原先已有 `no_proxy`，便在其末尾追加逗号和节点 IP；否则只使用节点 IP。最后无论如何都以前缀形式加入本机地址。若 `NODE_IP_LIST` 为空，结果末尾会留下一个逗号，通常无害。最后两条 `echo` 用于让用户确认最终的 `no_proxy` 及实际采用的 Conda 路径。

## 如何让每个新 shell 自动执行该脚本

将 `source "$PSRL_ENV_SCRIPT"` 放在 `~/.bashrc` 末尾的交互式判断内。这里的变量必须先定义，或将它替换为已确认的脚本路径。不要把未定义的变量直接写入启动文件。非交互登录 Bash 也可能读取这个文件，应避免它重复执行较重的 Conda 初始化。

- 普通交互式 Bash 默认读取 `~/.bashrc`。`bash --norc` 等显式禁用启动文件的方式除外。
- 登录 Bash 读取登录配置。在本次检查的入口中，`~/.bash_profile` 已加载 `~/.bashrc`，因此无需再追加一次 source。其他入口需要先核对，不能假定相同。
- 普通 `bash script.sh` 不保证读取 `.bashrc`。自动任务应显式加载环境，不能仅凭工具名称或 SSH 连接方式推断它是否读取了启动文件。
- 放在末尾是为了让目标环境在已有 Conda 初始化和 PATH 设置之后生效。新入口未必有旧入口相同的 `conda init` 段落。
- 反复 source 会重新初始化 Conda，并可能重复追加 `no_proxy` 或库路径。配置文件不重复追加，不等于环境脚本本身完全幂等。
- 自动工具命令默认继承启动它的进程环境。需要特定训练环境的脚本仍应显式 source 指定脚本，不能把跳过重复初始化理解为自动具备完整训练环境。

修改工作区外的启动文件时，若工具要求授权，应通过其正式授权机制完成。

## 新 tunnel 两项配置的复现

本节是终端配置操作说明。脚本解析部分解释基础 `activate`，本节集中维护新入口的复现方法。

### 确定路径和脚本

| 配置 | 如何选择 | 用途 |
|---|---|---|
| `PSRL_ENV_SCRIPT` | 基础环境用共享目录的 `lhy/activate`，完整训练环境用同目录下的 `env/psrl.sh` | 前者初始化 Conda、代理和镜像，后者还会激活 `psrl_smg_new` 并设置训练变量 |
| `WORKSPACE_ROOT` | 包含 `.codex` 和 `psrl` 的个人工作区目录 | 用于定位共享配置和自己的仓库 |
| `CODEX_HOME` | `WORKSPACE_ROOT` 下的 `.codex` | 本次沿用的 Codex 环境变量 |

`env/psrl.sh` 内部已经 source 基础 `activate`，二者选择一个，不需要都写一遍。脚本可能覆盖 `PSRL_WORKSPACE`，因此在加载后恢复为自己的 `psrl` 仓库路径。

例如，新入口中工作区文件仍可访问，但终端没有加载环境脚本。补齐配置后，新交互式 Bash 先加载脚本，再设置个人仓库路径和 `CODEX_HOME`，随后启动的命令继承这些变量。非交互 Bash 跳过较重的初始化，仍设置个人路径。已经运行的进程不会被追溯修改。

### 写入启动文件

以下命令在 Bash 终端执行。交互输入实际路径，避免把未定义占位符写入 `.bashrc`。脚本会备份 `.bashrc`，并管理一个可重复更新的配置块。执行前查看已有配置，删除或合并旧的同用途 source/export 行，避免后面的旧配置覆盖新值。

```bash
read -r -e -p '环境脚本的完整路径: ' PSRL_ENV_SCRIPT
read -r -e -p '包含 .codex 和 psrl 的工作区路径: ' WORKSPACE_ROOT
export PSRL_ENV_SCRIPT WORKSPACE_ROOT
python3 - <<'PYCODE'
import os
import re
import shutil
import time
from pathlib import Path

script = Path(os.environ['PSRL_ENV_SCRIPT']).expanduser().resolve()
workspace = Path(os.environ['WORKSPACE_ROOT']).expanduser().resolve()
assert script.is_file(), f'环境脚本不存在: {script}'
assert (workspace / '.codex').is_dir(), '未找到 .codex 目录'
assert (workspace / 'psrl').is_dir(), '未找到 psrl 仓库'
rc = Path.home() / '.bashrc'
old = rc.read_text() if rc.exists() else ''
if rc.exists():
    backup = rc.with_name(f'.bashrc.backup-{time.time_ns()}')
    shutil.copy2(rc, backup)
    print(f'备份: {backup}')

# shell 引号确保路径中的空格和特殊字符被当作普通字符。
import shlex
begin = '# BEGIN personal shell environment'
end = '# END personal shell environment'
block = '\n'.join([
    begin,
    'case $- in',
    f'    *i*) source {shlex.quote(str(script))} ;;',
    'esac',
    f'export PSRL_WORKSPACE={shlex.quote(str(workspace / "psrl"))}',
    f'export CODEX_HOME={shlex.quote(str(workspace / ".codex"))}',
    end,
])
pattern = re.escape(begin) + r'.*?' + re.escape(end)
clean = re.sub(pattern, '', old, flags=re.S).rstrip()
rc.write_text(clean + '\n\n' + block + '\n')
print(f'已更新: {rc}')
PYCODE
bash -n ~/.bashrc
```

生成的配置先判断当前 shell 是否交互式，仅在交互式 shell 中 source 所选环境，再恢复个人仓库路径、设置 `CODEX_HOME`。`$-` 含 `i` 表示交互式。后两行属于个人路径配置，非交互 shell 也会执行。`bash -n` 只检查语法，不执行环境初始化。

2026-11-10 针对 Codex shell 快照超时的修复只修改了个人配置块，没有修改共享环境脚本。非交互登录 Bash 的现场启动耗时从约 6.5 秒降到约 0.7 秒，交互式启动仍能找到 Conda 和 Codex。修复后观察到真实会话生成的 shell 快照文件。详情见 [Codex 剩余问题修复记录](../codex/session-management/log-repair.md#剩余问题定位与修复)。这些是本次环境下的观测，不是所有入口的性能保证。

### 生效和验证

在当前终端执行一次：

```bash
source ~/.bashrc
```

再启动一个新交互式 Bash 验证，清除继承的关键变量以免把父进程配置误判成自启成功：

```bash
env -u CODEX_HOME -u PSRL_WORKSPACE bash -ic '
    printf "CODEX_HOME=%s\n" "$CODEX_HOME"
    printf "PSRL_WORKSPACE=%s\n" "$PSRL_WORKSPACE"
    printf "CONDA_DEFAULT_ENV=%s\n" "${CONDA_DEFAULT_ENV:-}"
    command -v python
    test -d "$CODEX_HOME" && test -d "$PSRL_WORKSPACE"
'
```

预期前两项分别指向个人工作区的 `.codex` 和 `psrl`。基础 `activate` 没有显式选择 `psrl_smg_new`，不能以未进入该环境判定失败。若选择 `env/psrl.sh`，则应看到 `CONDA_DEFAULT_ENV=psrl_smg_new`，且 Python 来自该环境。

检查登录 Bash 时，将 `bash -ic` 改为 `bash -lic`。不要用完整 `env` 输出检查配置，避免打印脚本加载的凭据。以上验证只确认 shell 初始化，不证明网络连通、GPU 可用或训练依赖全部正常。

### 本次验证范围

2026-11-09 在新入口实际采用基础 `lhy/activate`。`bash -ic` 与 `bash -lic` 均通过检查，个人工作区和 `CODEX_HOME` 正确，Python 来自该脚本指定的基础 Conda 安装。本节对应仓库 `bc5665f` 上的未提交文档修改，实际启动文件位于仓库之外。复现代码另在临时目录检查了语法和重复更新，不会重复生成配置块。未验证训练或网络请求。

### 取消自动加载

从 `.bashrc` 删除两个标记之间的整个块，保留其他设置。新开终端检查结果。删除文件中的配置不会撤销当前进程已加载的变量，必要时关闭旧终端。

相关说明：[source 的作用](bash-source-command.md)、[Codex 配置备份](../codex/codex-config-backup-restore.md)。
