# Gate B 新手版讲解：SessionRouter 出口的 GPU 空闲节流到底在干什么

## TL;DR

- 本文把 [`sciaccel-gpu-gate.md` §16.5](sciaccel-gpu-gate.md#165-修复已部署gate-a设备互斥-gate-b回复节流代码位置与验证脚本) 里 Gate B 那张 5 行的表，拆成"一个故事 + 一张图 + 逐行讲解"。表里每一行都是一个零件，五个零件合起来只做一件事：**当 AI 要在一张 GPU 上执行命令时，如果这张卡上上一件事还没干完，就把模型给 AI 的下一条回复先扣住，等卡空下来（最多约 600 秒）再放行。**
- 五个零件：①给 session（一次对话）的档案卡加一个 `gpu_device` 字段；②一个 HTTP 端点 `POST /control/gpu_bind` 让别的进程填这个字段；③用 `nvidia-smi` 读 GPU 的忙闲；④在回复出口处检查字段，非空就等；⑤`runner.py` 在拿到卡/还卡时各调一次端点。
- 三个"不会把系统卡死"的保险：**没绑定的 session 完全不受影响**；**`nvidia-smi` 读不到时按"空闲"处理（失败开放）**；**最长等 600 秒，到点无论如何放行**。
- 诚实声明：Gate B 的真实代码本机**没有运行**（缺 `aiohttp`、没有 GPU）；本文附的脚本是**复刻判定逻辑**的演示，用假遥测跑通，不是真实 GPU 上的实测。"为什么要有这道门"一节是设计意图，来自代码注释与设计提案，不是我验证出来的因果。

## 目录

- [0. 先把几个名词讲明白](#0-先把几个名词讲明白)
- [1. 一个故事：为什么要有 Gate B](#1-一个故事为什么要有-gate-b)
- [2. 全局图：五个零件在图上的位置](#2-全局图五个零件在图上的位置)
- [3. 图例：每个部分对应的源码位置](#3-图例每个部分对应的源码位置)
- [4. 第一行：给 session 的档案卡加一个 GPU 字段](#4-第一行给-session-的档案卡加一个-gpu-字段)
- [5. 第二行：绑定端点](#5-第二行绑定端点)
- [6. 第三行：怎么判断 GPU 空闲](#6-第三行怎么判断-gpu-空闲)
- [7. 第四行：出口钩子](#7-第四行出口钩子)
- [8. 第五行：runner 侧的绑定与解绑](#8-第五行runner-侧的绑定与解绑)
- [9. 用一个例子把五行串起来](#9-用一个例子把五行串起来)
- [10. 跟着图读源码的执行路径](#10-跟着图读源码的执行路径)
- [11. 可运行实验](#11-可运行实验)
- [12. 容易误解的地方与没有验证的部分](#12-容易误解的地方与没有验证的部分)
- [相关文档](#相关文档)

---

## 0. 先把几个名词讲明白

只解释理解后文**必须**用到的，路过的不展开。

| 名词 | 一句话解释 | 在本文里的角色 |
|---|---|---|
| **模型 / vLLM** | 一个负责"想下一步该干什么"的大语言模型服务，部署在专门的推理机器上 | AI 每走一步都要去问它 |
| **agent（AI 解题员）** | 跑在 Docker 容器里的程序：问模型 → 拿到一条命令 → 在容器里执行 → 把结果再问模型，循环往复 | Gate B 要"拦"的对象 |
| **episode** | agent 做完**一道题**的整个过程，可能持续几分钟到几小时 | 一个 episode = 一个容器 = 一个 session |
| **session（会话）** | 一个 episode 和模型之间的**一整段对话**，有唯一的 id（比如 `s-abc`） | Gate B 是"按 session"生效的 |
| **SessionRouter（会话路由器）** | 一个小型 Web 服务，夹在 agent 和模型之间，**所有问答都从它这里过**；它给每个 session 记一张"档案卡" | Gate B 的代码就住在它里面 |
| **HTTP 端点（endpoint）** | Web 服务对外开放的一个"窗口"，用 `方法 + 网址` 标识，比如 `POST /control/gpu_bind`；别的进程发一个请求过来，就等于敲了这个窗口 | 让 runner 能"远程填档案卡" |
| **GPU 利用率 / 显存** | 利用率：GPU 在忙的时间占比；显存：GPU 自带的内存，被程序占着就不能给别人 | 判断"这张卡空不空" |
| **`nvidia-smi`** | NVIDIA 驱动自带的命令行工具，一敲就打印每张 GPU 的状态，相当于"GPU 版任务管理器" | Gate B 的"眼睛" |
| **失败开放（fail open）** | 守卫自己坏了的时候，**默认放行**；反义词"失败关闭"是守卫坏了就**一律拦住** | 没有 GPU 的开发机不会被卡死 |
| **Gate A** | 先于 Gate B 的另一道门：episode 开工前，给它分一张**整卡**，保证两个 episode 不会拿到同一张卡 | 本文只用到结论，细节见 [`sciaccel-gpu-gate.md` §16.5](sciaccel-gpu-gate.md#165-修复已部署gate-a设备互斥-gate-b回复节流代码位置与验证脚本) 和 [`commit-72ca50c-gate-ab-walkthrough.md` §4](commit-72ca50c-gate-ab-walkthrough.md#4-gate-a-代码逐段讲解) |

## 1. 一个故事：为什么要有 Gate B

先用餐厅后厨做类比（类比只帮你建立直觉，边界在下面标出）：

```
  厨师（agent）      传菜窗口（SessionRouter）      大师傅（模型）
     │  "下一步？"  ──────►  转交  ──────────────►   想一想
     │                                                 │
     │  ◄────── 先在窗口扣住 ◄──────────────────  "开大火炒！"
     │            （灶台上还有火在烧？
     │              等火熄了再递给厨师）
     ▼
  厨师拿到指令，去自己的灶台（GPU）执行
```

对应关系：

| 后厨 | 真实系统 |
|---|---|
| 灶台 | 一张物理 GPU，比如 `/dev/nvidia1` |
| 排号领灶台（领到后整个灶台只归你） | Gate A |
| 厨师，带着自己的小料理间 | agent + 它的 Docker 容器 |
| 大师傅 | 模型（vLLM） |
| 传菜窗口 | SessionRouter |
| 窗口服务员"灶台还有火就先别递指令" | **Gate B** |
| 灶台上的火力表 | `nvidia-smi` 读出的利用率和显存 |

**这个类比的边界**：①真实代码里被扣住的是"模型的回复"，不是厨师本人，agent 不知道自己被节流，只觉得"模型今天回得慢"；②"火熄了"在代码里是"利用率 ≤ 5% **且** 显存 ≤ 256 MB"这个具体判据；③扣的时间有上限，**到点必放**，不会真的把厨师饿死。

**为什么要扣？** 这是**设计意图**，来源是代码注释（[`session_router.py:120-138`](../../psrl/workers/gen/session_router.py#L120)）和设计提案，我没有验证过这个因果：模型刚刚下达的一条工具命令（比如又跑一个程序），如果直接放行，可能会和"马上要复用同一张卡的评分程序（verifier）"撞车；而且设计提案担心 GPU 遥测读的是**整张卡**的计数器，邻居的负载会"污染"判断（见 设计提案译文 03 §5.4（本地资料，未随仓库发布））。

**为什么扣"回复"而不是扣"下一次请求"？** 因为 SessionRouter 能控制的就是"什么时候把模型的回复交回去"。扣住回复，agent 那边看起来只是"这次 HTTP 请求耗时更长"。这也带来一个硬约束：agent 的 LLM 请求超时是 900 秒且不重试（[`runner.py:316-317`](../../examples/sciaccel_rl/runner.py#L316)），扣的时间必须小于它，所以默认上限是 600 秒，**超时放行而不是报错**——超出软目标只损失一点吞吐，让 episode 失败会毁掉最长四小时的工作（设计提案 §6.3 的逻辑，见 译文 04（本地资料，未随仓库发布））。

## 2. 全局图：五个零件在图上的位置

我们跟着一个 episode 走一遍。它的 session id 是 `s-abc`，Gate A 分给它的卡是 `/dev/nvidia1`（这些是教学例子里的名字）：

```
 AgentLoopWorker 进程（跑 runner.py 的地方）            SessionRouter 进程（档案卡在这里）
 ─────────────────────────────────────────           ───────────────────────────────────────
                                                      states = {
 先前已经 create_session → 路由器里有了 s-abc 的档案卡    "s-abc": SessionState(gpu_device=None) ①
                                                      }
 Gate A 拿到卡 /dev/nvidia1
   │
   │ ⑤ _bind_gate_b(url, "s-abc", "/dev/nvidia1")
   └──── POST /control/gpu_bind ──────────────────►  ② control_gpu_bind
         [{"session_id":"s-abc",                          找到 s-abc 的档案卡，写入
           "gpu_device":"/dev/nvidia1"}]                  gpu_device = "/dev/nvidia1"

 容器里的 agent 反复：问模型 ──────────────────────►  session_chat_completions
                                                          │ 转发给模型，拿到回复
                                                          ▼
                                                      ④ 出口钩子：state.gpu_device 非空？
                                                          │是
                                                          ▼
                                                      ③ _wait_for_gpu_idle("/dev/nvidia1")
                                                          ├─ _read_gpu_utilization
                                                          │    (1 秒缓存 → nvidia-smi → {1:(util, mem)})
                                                          ├─ util≤5% 且 mem≤256MB → 放行
                                                          ├─ 读不到 → 放行（失败开放）
                                                          ├─ 等够 600 秒 → 放行
                                                          └─ 否则睡 2 秒再读
                                                          ▼
 agent 拿到回复，执行命令 ◄────────────────────────  返回回复

 episode 结束，容器已删
   │ ⑤ _bind_gate_b(url, "s-abc", None)
   └──── POST /control/gpu_bind  gpu_device=null ──►  ② 档案卡 gpu_device 清回 None
```

> 图里的圈号对应你那张表的五行，但**顺序是按时间线排的**：①字段是"地基"，②端点和⑤runner 调用配对（⑤是调用方，②是被调用方），③读忙闲是④钩子里调用的工具。

## 3. 图例：每个部分对应的源码位置

由于代码块里不能放可点击链接，这里按图的顺序列出（行号核对于当前工作区）：

| 图上 | 内容 | 源码 |
|---|---|---|
| ① | `SessionState.gpu_device` 字段（默认 `None`） | [`session_router.py:203-210`](../../psrl/workers/gen/session_router.py#L203) |
| ① 之前 | 本 episode 的 session 在 episode 开始时就创建好了，所以后面绑定时一定找得到 | [`agent_loop.py:140`](../../examples/sciaccel_rl/agent_loop.py#L140) |
| ② 路由注册 | 把网址 `/control/gpu_bind` 接到处理函数 | [`session_router.py:270-272`](../../psrl/workers/gen/session_router.py#L270) |
| ② 处理函数 | `control_gpu_bind` | [`session_router.py:535-573`](../../psrl/workers/gen/session_router.py#L535) |
| ③ 调参数 | 四个环境变量阈值与缓存/超时常量 | [`session_router.py:43-52`](../../psrl/workers/gen/session_router.py#L43) |
| ③ 设备名转编号 | `/dev/nvidia1` → `1` | [`session_router.py:55-61`](../../psrl/workers/gen/session_router.py#L55) |
| ③ 读遥测 | `_read_gpu_utilization` | [`session_router.py:64-107`](../../psrl/workers/gen/session_router.py#L64) |
| ③ 等空闲 | `_wait_for_gpu_idle` | [`session_router.py:120-172`](../../psrl/workers/gen/session_router.py#L120) |
| ④ 出口钩子 | `session_chat_completions` 末尾 | [`session_router.py:425-435`](../../psrl/workers/gen/session_router.py#L425) |
| ⑤ 调用函数 | `_bind_gate_b` | [`runner.py:41-68`](../../examples/sciaccel_rl/runner.py#L41) |
| ⑤ 绑定时机 | Gate A 拿到卡之后立刻绑 | [`runner.py:245`](../../examples/sciaccel_rl/runner.py#L245) |
| ⑤ 解绑时机 | 容器确认删除、释放设备之后 | [`runner.py:402-404`](../../examples/sciaccel_rl/runner.py#L402) |

这些代码如果有理解上的问题，可以随时追问。

## 4. 第一行：给 session 的档案卡加一个 GPU 字段

> 原文："`SessionState` 新增 `gpu_device` 字段，默认 `None`（未绑定的 session 完全不受影响）"

**SessionState 是什么**：SessionRouter 给每个 session 记的一张"档案卡"，里面已经有好多信息——这个 session 有几个请求正在飞（`inflight`）、有没有被协调器"挂起"（`hang_state`）、粘在哪个推理实例上（`base_worker_id`）等。**新增 `gpu_device`，就是在这张卡上多加一栏**："这个 session 的容器正在用哪张 GPU"。

```python
@dataclass(slots=True)
class SessionState:
    ...
    gpu_device: str | None = None
```
—— 节选自 [`session_router.py:175-210`](../../psrl/workers/gen/session_router.py#L175)（省略了其他字段，所以不用行号块）

逐个拆开：

- **`str | None`**：值要么是形如 `"/dev/nvidia1"` 的字符串，要么是 `None`（Python 里的"空"）。
- **默认 `None`**：新建的档案卡这一栏是空的，表示"没有 GPU 绑定"。
- **"完全不受影响"是怎么保证的**：出口钩子里是 `if gpu_device:`（见第 7 节），`None` 是"假值"，条件不成立，整段 Gate B 代码被跳过。所以所有**非 GPU 任务**，以及根本不会去调 `gpu_bind` 的其他例子，对它们来说 Gate B 就像不存在。唯一的"开销"是钩子里多读一次这个字段时拿一下 `state.lock`（微秒级，见 [`session_router.py:429-430`](../../psrl/workers/gen/session_router.py#L429)）。
- **为什么要显式加在类里**：`@dataclass(slots=True)` 的意思是这个类只允许有声明过的字段，运行时不能随手 `state.xxx = ...` 新增属性，所以必须先在类里声明一栏。

## 5. 第二行：绑定端点

> 原文："新控制端点 `POST /control/gpu_bind`：Gate A 拿到/释放设备时各调用一次，绑定或清空某个 session 的 `gpu_device`"

**先问一个问题：为什么需要一个"端点"，不能路由器自己知道哪张卡吗？** 不能，原因有两层：

1. **路由器看不到卡号**：agent 问模型的 URL 形如 `{路由器}/sessions/s-abc/v1/chat/completions`（见 [`session_agent_loop.py:95`](../../psrl/workers/agent_loop/loops/session_agent_loop.py#L95)），请求里只有 session id，没有卡号。
2. **分卡的人和路由器不在同一个进程**：分卡（Gate A）发生在 `AgentLoopWorker` 的进程里，档案卡在 SessionRouter 的进程里。两个进程不共享内存（甚至可能在两台机器上），唯一的沟通办法是**发网络请求**。所以路由器开一个窗口，谁分了卡，谁就过来说一声："s-abc 现在用 `/dev/nvidia1`"。

**请求长什么样**（示意，格式来自 [`control_gpu_bind` 的文档字符串](../../psrl/workers/gen/session_router.py#L535) 和 [`runner.py:57-61`](../../examples/sciaccel_rl/runner.py#L57)；下面是教学例子，不是从线上抓到的）：

```
POST {session_router_url}/control/gpu_bind
[{"session_id": "s-abc", "gpu_device": "/dev/nvidia1"}]     ← 绑定
[{"session_id": "s-abc", "gpu_device": null}]               ← 解绑
```

**处理函数做了什么**（[`session_router.py:535-573`](../../psrl/workers/gen/session_router.py#L535)），按顺序：

| 步骤 | 代码做什么 | 为什么 |
|---|---|---|
| 1 | 读请求体 JSON；读不出来返回 HTTP 400 | 拒绝格式错误的请求 |
| 2 | 兼容两种外形：直接是列表，或 `{"sessions": [...]}` | 和同文件里 `hang`/`continue` 两个控制端点的请求格式保持一致 |
| 3 | 逐条取出 `session_id`，用 `self.states.get(sid)` 查档案卡 | 注意是 `get`，**查不到不会新建**档案卡 |
| 4 | 查不到的记进 `missing` 列表 | 比如 session 已经被删了 |
| 5 | 在 `state.lock` 保护下写 `state.gpu_device = str(device) if device else None` | 加锁防止和同一 session 的其他处理同时改字段；`null` 或空字符串都会变成 `None`，这就是"解绑" |
| 6 | 返回 `{"bound": [...], "missing": [...]}` | 调用方可以知道哪些成功 |

**"绑定"和"清空"为什么共用一个端点**：只是字段值不同（字符串 vs `null`），逻辑是同一行赋值，所以"各调用一次"就是：拿到卡时绑，还卡时清。

**一个可能的疑问：绑定时 session 一定存在吗？** 从代码看是的：`agent_loop.py:140` 先 `create_session`，之后才把 `session_id` 传给 `run_harbor_episode`（[`agent_loop.py:151-156`](../../examples/sciaccel_rl/agent_loop.py#L151)），所以 `runner.py:245` 绑定时路由器里已经有这张档案卡。

## 6. 第三行：怎么判断 GPU 空闲

> 原文："真实 GPU 空闲判断：`_read_gpu_utilization` 用 `nvidia-smi --query-gpu=index,utilization.gpu,memory.used` 读整卡计数器，带 1 秒缓存避免多个 session 同一瞬间各发一次子进程；`nvidia-smi` 不存在或超时时'失败开放'（当成空闲），不会把没有 GPU 的开发机卡死"

这一行信息最密，拆成四小段。

### 6.1 读什么：`nvidia-smi` 的这条命令

```
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
```
—— [`session_router.py:82-88`](../../psrl/workers/gen/session_router.py#L82)

| 片段 | 含义 |
|---|---|
| `--query-gpu=index,utilization.gpu,memory.used` | 只要三列：**卡编号**、**GPU 利用率（%）**、**已用显存** |
| `--format=csv` | 输出成逗号分隔，方便程序解析 |
| `noheader` | 不要表头那一行 |
| `nounits` | 数字后面不要带 `%`、`MiB` 这样的单位 |

在一台 2 卡机器上，输出会是这样（**教学假设**：我这台 Mac 没有 `nvidia-smi`，没有实际运行；格式是按这条命令的参数推出来的）：

```
0, 0, 0
1, 97, 18000
```

读法：0 号卡利用率 0%、显存 0；1 号卡利用率 97%、显存 18000。

**"利用率"到底是什么**：按我对 NVIDIA 文档的理解，`utilization.gpu` 是"过去一小段采样时间里，**至少有一个 GPU 程序在运行的时间占比**"，不是"算力用了百分之几"。所以 97% 说明这张卡几乎一直有程序在跑，并不表示还剩 3% 的算力可用。这是我引述的外部定义，本地没有重新核对文档。另外 nvidia-smi 的显存单位其实是 MiB，代码里叫 MB，约差 4.8%，对 256 这个阈值的判断没有实质影响。

**"整卡计数器"的含义**：这些数字是**整张卡**的总和，分不出是哪个进程贡献的。这也是它偏"保守"的原因：只要卡上有任何东西在跑，就判忙。

### 6.2 怎么解析

```python
data: dict[int, tuple[float, float]] = {}
for line in stdout.decode(errors="replace").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        continue
    try:
        data[int(parts[0])] = (float(parts[1]), float(parts[2]))
    except ValueError:
        continue
```
—— [`session_router.py:96-104`](../../psrl/workers/gen/session_router.py#L96)

手算一遍 `"1, 97, 18000"`：`split(",")` 得到 `["1", " 97", " 18000"]` → `strip()` 去掉空格 → `data[1] = (97.0, 18000.0)`。最终 `data = {0: (0.0, 0.0), 1: (97.0, 18000.0)}`。**读不懂的行被直接跳过**（列数不对，或者不是数字），这样那张卡就不在字典里，后面会走"读不到 → 放行"的分支。

### 6.3 为什么要缓存 1 秒、还要加一把锁

场景：一台节点上同时有 20 个 GPU session，它们的回复恰好在同一瞬间到达出口，**每个都要看一眼 GPU 忙不忙**。如果每个都去启动一次 `nvidia-smi` 子进程，一瞬间就是 20 个子进程，既浪费又可能拖慢 `nvidia-smi` 本身（这是个"惊群"问题：大家同时醒来干同一件事）。

代码的做法（[`session_router.py:74-80`](../../psrl/workers/gen/session_router.py#L74)）：

1. 一个模块级的 `asyncio.Lock`：同一时刻只允许一个协程进入"读遥测"这段；
2. 拿到锁之后先看缓存：距离上次读取不到 1 秒（`_GATE_B_SMI_CACHE_TTL_SEC = 1.0`），直接返回上次的结果；
3. 否则才真的去启动 `nvidia-smi`，读完把结果和时间存进缓存。

于是 20 个协程排队过锁：第 1 个真的读，其余 19 个进去时都发现缓存是新的，直接拿现成结果。本文的演示脚本会数出"子进程只启动了 1 次"（见第 11 节）。**代价**：最多会拿到 1 秒前的数据，对"判断一张卡是不是空"这个粒度够用。

### 6.4 读不到怎么办：失败开放

```python
except (OSError, asyncio.TimeoutError) as exc:
    ...
    _gate_b_smi_cache_data = {}
    return _gate_b_smi_cache_data
```
—— [`session_router.py:90-94`](../../psrl/workers/gen/session_router.py#L90)

- `OSError`：命令不存在（比如没装驱动的开发机、纯 CPU 机器，Python 会抛 `FileNotFoundError`，它属于 `OSError`）；
- `TimeoutError`：5 秒内 `nvidia-smi` 没返回（`_GATE_B_SMI_TIMEOUT_SEC = 5.0`）。

两种情况都返回**空字典 `{}`**。在 `_wait_for_gpu_idle` 里，空字典意味着"这张卡的编号不在遥测里"，直接放行（[`session_router.py:146-147`](../../psrl/workers/gen/session_router.py#L146)）。这就是**失败开放**。

打个比方：门禁刷卡机坏了。"失败关闭"是门死锁，谁也进不去；"失败开放"是门默认敞开。Gate B 选了后者，因为它本来就只是个软节流，不能因为"监控设备坏了"就让所有 GPU 任务停摆。

**代价**要看清楚（以下是我读代码的推断，没有在真实 GPU 机器上验证）：在真正有 GPU 的机器上，如果 `nvidia-smi` 偶尔卡超过 5 秒，Gate B 会在这一段时间里**悄悄失效**。另外代码没有检查 `nvidia-smi` 的退出码，如果它运行了但输出为空，结果同样是 `{}`，同样放行。

## 7. 第四行：出口钩子

> 原文："出口 hook：`session_chat_completions` 返回响应前，若该 session 绑定了 `gpu_device`，先 `await _wait_for_gpu_idle(...)`，最长等待 `PSRL_GATE_B_MAX_WAIT_SEC`（默认 600 秒…）"

**"出口 hook"是什么**：钩子（hook）就是在流程某个固定位置挂上去的一小段额外代码。`session_chat_completions` 是处理"agent 问模型"这个请求的函数，流程大致是：

```
收到 agent 的请求 → 检查 session 状态 → 转发给模型 → 拿到回复 → 记录/更新状态 → 【出口钩子】 → 把回复返回给 agent
```

钩子就挂在"把回复交回 agent"的**前一刻**：

```python
async with state.lock:
    gpu_device = state.gpu_device
    gpu_shared = state.gpu_shared
if gpu_device:
    await _wait_for_gpu_idle(gpu_device, sid, shared=gpu_shared)

return self.build_response(result)
```
—— [`session_router.py:429-435`](../../psrl/workers/gen/session_router.py#L429)

两个细节：

1. **先在锁里把字段读出来，再到锁外面等**。等待可能长达几分钟，如果在锁里等，这个 session 的其他操作（比如协调器要查询/挂起它）都会被堵住。
2. **`await`**：这是协程的"礼貌等待"。`_wait_for_gpu_idle` 在 `asyncio.sleep` 的时候，会把 CPU 让给路由器里其他请求，所以**这一个 session 被扣住，不影响别的 session 被正常处理**。

### `_wait_for_gpu_idle` 的四个出口

```python
index = _parse_gpu_index(device)
if index is None:
    return
deadline = time.monotonic() + _GATE_B_MAX_WAIT_SEC
while True:
    telemetry = await _read_gpu_utilization()
    if index not in telemetry:
        return
    util, mem = telemetry[index]
    if util <= _GATE_B_IDLE_UTIL_PCT and mem <= _GATE_B_IDLE_MEM_MB:
        return
    if time.monotonic() >= deadline:
        return
    await asyncio.sleep(_GATE_B_POLL_INTERVAL_SEC)
```
—— 节选自 [`session_router.py:120-172`](../../psrl/workers/gen/session_router.py#L120)（省略日志语句，所以不用行号块）

> **关于这段节选（2026-10-12 校正）：** 上面是提交 `72ca50c` 里的**单设备版本**，便于入门；工作区当前的真实源码在此基础上多了两点：①`device` 可以是逗号连接的多个设备，先用 `_parse_gpu_indices`（[`:110`](../../psrl/workers/gen/session_router.py#L110)）拆成编号列表，**全部**设备都空闲才放行；②多了 `shared` 参数，共享卡（同一个 unit 的多个容器共用一张卡）只看利用率、不看显存，因为常驻的模型权重会让显存一直不为零。下文的讲解和表格仍按单设备、非共享的情况成立；这两点是工作区新增的行为，详见 [Env Service 实现讲解](env-service-implementation.md)。文中所有行号都已校正为工作区当前行号。

这个循环可以看成"轮询"（每隔一会儿看一眼）。每看一次，四选一：

| 看到什么 | 动作 | 为什么 |
|---|---|---|
| 设备名不是 `/dev/nvidiaN` 的形式 | 立刻放行 | 解析不了就不管 |
| 遥测里没有这张卡 | **立刻放行**（失败开放） | 见 6.4 |
| `util ≤ 5` **且** `mem ≤ 256` | 放行 | 卡"看起来空了" |
| 已经等过 `deadline`（默认 600 秒） | 打 WARNING 日志，**照样放行** | 保险：一个卡死的 episode 不能永远堵住 |
| 以上都不是 | 睡 2 秒，回到循环开头再读 | 继续等 |

四个阈值/参数都是环境变量，默认值如下（[`session_router.py:43-48`](../../psrl/workers/gen/session_router.py#L43)）：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `PSRL_GATE_B_MAX_WAIT_SEC` | 600 | 最长等多久 |
| `PSRL_GATE_B_POLL_INTERVAL_SEC` | 2 | 每隔多久再看一次 |
| `PSRL_GATE_B_IDLE_UTIL_PCT` | 5 | 利用率不超过多少算空 |
| `PSRL_GATE_B_IDLE_MEM_MB` | 256 | 显存占用不超过多少算空 |

**为什么两个条件是"且"**：利用率只反映"现在有没有程序在算"，而一个刚算完的程序可能还没把显存还回去（残留）。只看利用率，会在显存还没释放干净时就放行。所以必须**两项同时满足**。具体"残留多少"取决于驱动和程序，没有实测。

**实际会等多久**：判断顺序是"先读遥测，再判断是否过期，再睡"，所以在 600 秒的期限之后，会在**下一次读到遥测的时候**才放行，大约是 600 秒再加上最多一个轮询周期（2 秒），遇到 `nvidia-smi` 慢的情况还要再加上读取的耗时。

## 8. 第五行：runner 侧的绑定与解绑

> 原文："`runner.py` 侧的绑定/解绑调用（best-effort，失败只记日志，不影响 episode 本身）"

**runner.py 是什么角色**：它是 SciAccel 里"真正把一个 episode 跑起来"的代码，Gate A 也在这里。**Gate B 的调用方是 Gate A 的同一段代码**：拿到卡就绑，还卡就解。

```python
async def _bind_gate_b(session_router_url: str, session_id: str, device: str | None) -> None:
    if not session_router_url or not session_id:
        return
    try:
        await post(
            f"{session_router_url.rstrip('/')}/control/gpu_bind",
            payload=[{"session_id": session_id, "gpu_device": device}],
            max_retries=1,
        )
    except Exception as exc:  # noqa: BLE001
        psrl_logger.debug(...)
```
—— 节选自 [`runner.py:41-68`](../../examples/sciaccel_rl/runner.py#L41)（省略了文档字符串和日志参数，所以不用行号块）

**"best-effort（尽力而为）"是什么意思**：做得到最好，做不到也不算出错。这里体现在 `try/except Exception`：不管是路由器连不上、session 已被删，还是别的错，**统统吞掉，只打一条 debug 日志**，episode 照常往下走。

**为什么敢这么吞**：Gate B 本来就是软节流，**漏绑一次的后果只是这个 episode 没被节流**；而如果让绑定失败去中断 episode，会白白毁掉一道题的运行。损失和风险不对称，所以选"宁可漏拦，不可误伤"。

这个函数被调用两次：

| 时机 | 代码 | 传的值 |
|---|---|---|
| Gate A 拿到设备后 | [`runner.py:245`](../../examples/sciaccel_rl/runner.py#L245) `_bind_gate_b(session_router_url, session_id, allocated_gpu.device)` | `"/dev/nvidia1"`（绑定） |
| 容器确认删除、设备释放之后 | [`runner.py:402-404`](../../examples/sciaccel_rl/runner.py#L402) `_bind_gate_b(..., None)` | `None`（解绑） |

解绑放在释放设备**之后**，是因为在这之前容器还可能在使用这张卡，Gate B 还应该继续生效。

## 9. 用一个例子把五行串起来

教学设定（数字是我编的，用来说明机制，不是实测）：机器 2 张卡，`s-abc` 这个 episode 拿到 `/dev/nvidia1`，Gate B 参数用默认值。时间线：

| 时刻 | 发生了什么 | Gate B 的反应 |
|---|---|---|
| t=0 | Gate A 分到 `/dev/nvidia1`，`runner` 调 `gpu_bind` | 档案卡 `gpu_device` 从 `None` 变成 `"/dev/nvidia1"` |
| t=10 | agent 第 6 轮：问模型，模型回复"先编译" | 出口读 `nvidia-smi`：`1, 0, 0` → 空闲 → 立即放行 |
| t=40 | agent 第 7 轮：模型回复 `nohup ./solver > log &`（后台启动一个跑 GPU 的程序） | 此刻 GPU 还空 → 放行；agent 把 solver 丢到后台继续跑 |
| t=45 | agent 第 8 轮：问模型"看看日志？"，模型回复"`tail log`" | 读到 `1, 97, 18000` → 忙 → **扣住这条回复**，每 2 秒重读一次 |
| t=300 | solver 跑完，显存残留 300 | `1, 0, 300`：利用率 OK 但 300 > 256 → **仍然扣着** |
| t=302 | 显存释放干净 | `1, 0, 0` → 放行，agent 终于收到 `tail log` |
| t=900 | episode 结束，容器删除，设备释放 | `gpu_bind(null)`，档案卡清回 `None` |

这条时间线里值得注意的一点（**这是从代码推断的，没有在真实训练里验证**）：Gate B 读的是整卡计数器，而 Gate A 已经保证这张卡只有本 episode 在用，所以它多半是在**等 agent 自己放到后台的 GPU 程序**。这是否符合设计意图我没有确认（代码注释强调的是防止和 verifier 撞车），但它和"agent 在后台跑任务然后看日志"这种正常用法有张力。600 秒加上生成时间仍小于 900 秒的请求超时，所以不会因此直接超时。

## 10. 跟着图读源码的执行路径

| 步骤 | 对应图上 | 调用位置 |
|---|---|---|
| 1 | 前置 | [`agent_loop.py:140`](../../examples/sciaccel_rl/agent_loop.py#L140) `create_session` → [`:151-156`](../../examples/sciaccel_rl/agent_loop.py#L151) 把 `session_id` 交给 runner |
| 2 | Gate A | [`runner.py:240-241`](../../examples/sciaccel_rl/runner.py#L240) 拿到 `allocated_gpu` |
| 3 | ⑤ 绑定 | [`runner.py:245`](../../examples/sciaccel_rl/runner.py#L245) → [`:41-61`](../../examples/sciaccel_rl/runner.py#L41) `post(.../control/gpu_bind)` |
| 4 | ② 端点 | [`session_router.py:272`](../../psrl/workers/gen/session_router.py#L272) 路由 → [`:535`](../../psrl/workers/gen/session_router.py#L535) `control_gpu_bind` → [`:567`](../../psrl/workers/gen/session_router.py#L567) 写 `state.gpu_device` |
| 5 | ① 字段 | [`session_router.py:210`](../../psrl/workers/gen/session_router.py#L210) |
| 6 | ④ 钩子 | 每轮回复：[`session_router.py:429-433`](../../psrl/workers/gen/session_router.py#L429) |
| 7 | ③ 等空闲 | [`:120`](../../psrl/workers/gen/session_router.py#L120) `_wait_for_gpu_idle` → [`:145`](../../psrl/workers/gen/session_router.py#L145) `_read_gpu_utilization` → [`:82`](../../psrl/workers/gen/session_router.py#L82) `nvidia-smi` |
| 8 | ⑤ 解绑 | [`runner.py:402-404`](../../examples/sciaccel_rl/runner.py#L402) |

## 11. 可运行实验

脚本：`docs/psrl/scripts/gate_b_throttle_demo.py`（本地资料，未随仓库发布）。它把 `_parse_gpu_index`、`nvidia-smi` 输出解析、带锁和 TTL 的读取、`_wait_for_gpu_idle` 的四个出口**复刻**出来（不是 `import` 真实模块，因为后者依赖 `fastapi`/`aiohttp`，本机没装），只把"真的启动 `nvidia-smi`"换成"读一个随时间变化的假遥测"。时间整体缩小 10 倍。脚本注释里标了每段对应图上哪一步。

```bash
cd /Users/liuwei/CodeBuddy/20260811161322
python3 docs/psrl/scripts/gate_b_throttle_demo.py
```

本机实际运行输出：

```
假遥测：index=1  t<0.5 空闲 | 0.5~1.5 solver 在跑(97%,18000MB) | 1.5~2.0 残留(0%,300MB) | >=2.0 空闲

场景 1：回复在 GPU 空闲时到达 / 忙时到达 / 残留显存时到达
  A 回复到达时 GPU 空闲                     到达 t=0.00s  放行 t=0.00s  等了 0.00s  | 空闲 (util=0%, mem=0MB) -> 放行
  B 回复到达时 solver 正在跑                 到达 t=1.00s  放行 t=2.01s  等了 1.01s  | 空闲 (util=0%, mem=0MB) -> 放行
  C 回复到达时只剩残留显存                      到达 t=1.70s  放行 t=2.10s  等了 0.40s  | 空闲 (util=0%, mem=0MB) -> 放行

场景 2：20 个 session 在同一瞬间都要读遥测 -> 到底开了几次 nvidia-smi？
  => 20 个 session 同时读取，nvidia-smi 子进程只启动了 1 次（缓存 + 锁的效果）

场景 3：nvidia-smi 不存在（开发机没有 GPU）-> 失败开放
  D 无 nvidia-smi                     到达 t=2.61s  放行 t=2.61s  等了 0.00s  | 读不到这张卡 -> 失败开放，直接放行

场景 4：卡一直忙，最长等待 1.0s（真实默认 600s）-> 超时放行
  E 一直忙 + 超时                         到达 t=2.76s  放行 t=3.77s  等了 1.01s  | 超时 1.0s (util=97%, mem=18000MB) -> 照样放行
```

读法：

- **B** 在 t=1.0 到达，GPU 正忙，直到 t=2.0 才放行：忙时被扣住。
- **C** 在 t=1.7 到达，此时利用率已是 0 但显存还剩 300（> 256），仍被扣到 t=2.0 之后：对应"且"的判据。
- 场景 2 对应缓存 + 锁；场景 3 对应失败开放；场景 4 对应超时放行。

**诚实说明**：这些全是假遥测跑出来的判定逻辑，没有真的 GPU，也没有真的 `nvidia-smi`。真实代码本身本机没运行过（`import psrl.workers.gen.session_router` 需要 `aiohttp` 等依赖，当前环境没有）。

## 12. 容易误解的地方与没有验证的部分

| 常见误解 | 实际 |
|---|---|
| "Gate B 会拦住 agent 执行命令" | 它扣的是**模型的回复**；agent 已经在跑的命令不会被打断，后台程序也照样在跑（代码注释自己也说这是软措施） |
| "等不到就报错" | **超时照样放行**，只打一条 WARNING |
| "所有 session 都会被检查" | 只有被 `gpu_bind` 绑过的 session；其他全部跳过 |
| "绑定失败 episode 就挂了" | 绑定调用吞掉所有异常，最多是这个 episode 没被节流 |
| "`nvidia-smi` 读到的是这个容器的用量" | 是**整卡**的数字，分不出进程；它偏保守，宁可多等也不会把"忙"误判成"空" |

没有验证的部分：

- Gate B 在真实 GPU 集群上的行为没有验证；`util ≤ 5%` 与 `mem ≤ 256 MB` 的阈值在真实卡上是否合适没有验证。
- `nvidia-smi` 的真实输出格式、`utilization.gpu` 的精确定义，是按命令参数与我对 NVIDIA 文档的理解写的，本地没有核对文档。
- Gate B 的 `nvidia-smi` 运行在 **SessionRouter 所在的节点**，不一定是容器所在的节点；如果两者不在同一台机器，读到的可能是别的机器上同编号的卡。这是 [`commit-72ca50c-gate-ab-walkthrough.md` §7 第 3 条](commit-72ca50c-gate-ab-walkthrough.md#7-从代码读出的-6-处注意点推断) 已经记录的推断，本文没有额外验证。

## 相关文档

- [从 Gate B 学 HTTP](http-basics-for-gate-b.md)：本文第 5 节、第 8 节涉及的 HTTP 请求、端口、URL 设计、"POST 之后为什么有回应"的网络基础补课。
- [提交 72ca50c 代码讲解](commit-72ca50c-gate-ab-walkthrough.md)：本文的"专业版"，含 Gate A 的逐段讲解、与设计提案的对照和 6 处注意点。
- [SciAccel GPU Gate 实测与部署](sciaccel-gpu-gate.md)：§16.5 是 Gate B 那张 5 行表格的出处。
- Env Service 设计文档译文（本地资料，未随仓库发布）：Gate A / Gate B 的设计动机（§6.3 两道门）。
- [PSRL 笔记总索引](../README.md)
