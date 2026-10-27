# 从 Gate B 学 HTTP：端口、URL、POST 与"为什么发了就有回应"

## TL;DR

- 本文是 [`gate-b-beginner-guide.md`](gate-b-beginner-guide.md) 的网络基础补课篇，只讲 `runner.py` 向 SessionRouter 发 `POST /control/gpu_bind` 这一件事背后的计算机网络知识。
- **先纠正一个前提：`/control/gpu_bind` 没有单独开端口。** 单独占一个端口的是 **SessionRouter 整个服务**（和 SMG 路由器各占一个端口）；`gpu_bind` 只是这个服务里的一个"路径"，和 `/sessions/...` 共用同一个端口。**端口区分的是"哪个程序"，路径区分的是"这个程序里的哪项功能"。**
- 一次请求的全貌：`URL 拆成 协议/IP/端口/路径` → 用 IP 找到机器、用端口找到进程 → 建立 TCP 连接 → 在连接里发一段**按约定格式写成的文本**（请求行 + 头 + 空行 + 体）→ 服务端按 `方法 + 路径` 找到处理函数 → 把返回值也写成约定格式的文本，**从同一条连接写回去**。
- "POST 之后为什么有回应"：不是魔法，是三件事叠加——TCP 连接天然双向；HTTP 协议规定"一问一答"；服务端框架（uvicorn + FastAPI）负责把函数返回值包装并写回连接，客户端 `await` 就是在连接上等那段字节。
- **想快速查"往 `POST /sessions/{sid}/v1/chat/completions` 发一次请求会发生什么、为什么要区分 sid"，直接看 [§5.5 速查小结](#55-速查小结往数据面端点-post-一下会发生什么为什么要区分-sid)。**
- 实验脚本 `http_gpu_bind_demo.py` 用裸 socket 手写 HTTP 报文，全部输出已在本机跑过；**真实服务端（FastAPI/uvicorn）和真实客户端（aiohttp）本机没运行**，脚本是教学复刻。

## 目录

- [0. 先纠正一个前提：gpu_bind 没有单独的端口](#0-先纠正一个前提gpu_bind-没有单独的端口)
- [1. 全景图：一次 gpu_bind 请求怎么走](#1-全景图一次-gpu_bind-请求怎么走)
- [2. 图例：图里每个部分对应的源码位置](#2-图例图里每个部分对应的源码位置)
- [3. 网络基础：IP、端口、TCP、HTTP 各管什么](#3-网络基础ip端口tcphttp-各管什么)
- [4. 为什么 SessionRouter 要有自己的端口](#4-为什么-sessionrouter-要有自己的端口)
- [5. URL 格式：怎么拆，为什么这么设计](#5-url-格式怎么拆为什么这么设计)
- [6. 一条 HTTP 请求的原文长什么样](#6-一条-http-请求的原文长什么样)
- [7. POST 之后为什么就会有回应](#7-post-之后为什么就会有回应)
- [8. 回到 Gate B：扣住回复在网络层面是什么样](#8-回到-gate-b扣住回复在网络层面是什么样)
- [9. 跟着图读源码的执行路径](#9-跟着图读源码的执行路径)
- [10. 可运行实验](#10-可运行实验)
- [11. 没有验证的部分与注意点](#11-没有验证的部分与注意点)
- [相关文档](#相关文档)

---

## 0. 先纠正一个前提：gpu_bind 没有单独的端口

你的问题是"为什么要单独开一个端口"。从代码看，实际情况是：

```python
self.app.post("/sessions")(self.create_session)
...
self.app.post("/sessions/{sid}/v1/chat/completions")(self.session_chat_completions)
...
self.app.post("/control/gpu_bind")(self.control_gpu_bind)
```
—— 节选自 [`session_router.py:261-272`](../../../psrl/workers/gen/session_router.py#L261)（省略了中间的路由，所以不用行号块）

`self.app` 只有一个，它被 `uvicorn.run(router.app, host=host, port=port)` 挂在**一个**端口上（[`rollout_gateway.py:184`](../../../psrl/workers/gen/rollout_gateway.py#L184)）。创建 session、转发聊天、`gpu_bind` 全都从这同一个端口进来，靠**路径**区分。

所以真正的问题有两个，分开回答：

| 问题 | 答案在哪 |
|---|---|
| SessionRouter 为什么要有自己的端口（以及为什么是独立进程） | [第 4 节](#4-为什么-sessionrouter-要有自己的端口) |
| `gpu_bind` 为什么不需要自己的端口，只用一个路径 | [第 4.4 节](#44-为什么-gpu_bind-不需要自己的端口) 和 [第 5 节](#5-url-格式怎么拆为什么这么设计) |

## 1. 全景图：一次 gpu_bind 请求怎么走

场景延续前一篇：episode 的 session id 是 `s-abc`，Gate A 分到了 `/dev/nvidia1`。`runner.py` 要告诉 SessionRouter 这件事。教学假设：SessionRouter 的地址是 `http://10.0.0.7:8200`（IP 和端口是我编的）。

```
 AgentLoopWorker 进程（runner.py 在这里）                    SessionRouter 进程（档案卡在这里）
 ───────────────────────────────────────                  ──────────────────────────────────────
                                                           uvicorn.run(app, host="10.0.0.7", port=8200)
 ① 拼 URL                                                    └─ 启动时：bind + listen（"我在 8200 等客人"）
    "http://10.0.0.7:8200" + "/control/gpu_bind"
                │
 ② 拆 URL：协议=http  地址=10.0.0.7  端口=8200  路径=/control/gpu_bind
                │
 ③ TCP 连接：connect(10.0.0.7, 8200) ─────── 三次握手 ───────►  accept → 得到一条专属连接
                │                                              │
 ④ 在连接里写出请求（纯文本）：                                 │
    POST /control/gpu_bind HTTP/1.1                            │
    Host: 10.0.0.7:8200                                        │
    Content-Type: application/json                             │
    Content-Length: 55                                         │
    (空行)                                                     │
    [{"session_id":"s-abc","gpu_device":"/dev/nvidia1"}] ─────►⑤ uvicorn 读字节、按格式解析出
                │                                                 方法=POST 路径=/control/gpu_bind 体=...
                │                                              ⑥ FastAPI 按 方法+路径 查路由表
                │                                                 → control_gpu_bind(request)
                │                                              ⑦ 函数改 state.gpu_device，
                │                                                 return JSONResponse({"bound":[...]})
 ⑩ await 返回，拿到 JSON，函数结束  ◄── ⑨ 从同一条连接读回 ────  ⑧ 把返回值写成响应文本，写回连接：
                                                                  HTTP/1.1 200 OK
                                                                  Content-Type: application/json
                                                                  (空行)
                                                                  {"bound":["s-abc"],"missing":[]}
```

## 2. 图例：图里每个部分对应的源码位置

| 图上 | 内容 | 源码 |
|---|---|---|
| 启动 | 找空闲端口（从 8200 往上试） | [`rollout_gateway.py:147`](../../../psrl/workers/gen/rollout_gateway.py#L147)；逻辑在 [`http_utils.py:18-35`](../../../psrl/utils/common/http_utils.py#L18) |
| 启动 | 监听 IP = SMG 所在节点 IP | [`rollout_gateway.py:112`](../../../psrl/workers/gen/rollout_gateway.py#L112)（取 Ray 节点 IP）、[`:148`](../../../psrl/workers/gen/rollout_gateway.py#L148)（赋给 `session_ip`） |
| 启动 | 子进程里跑 `uvicorn.run`（bind + listen 发生在这里面） | [`rollout_gateway.py:184`](../../../psrl/workers/gen/rollout_gateway.py#L184)，子进程创建 [`:187-204`](../../../psrl/workers/gen/rollout_gateway.py#L187) |
| 启动 | 对外公布的 URL 字符串 | [`rollout_gateway.py:207`](../../../psrl/workers/gen/rollout_gateway.py#L207) |
| ① | runner 拼 URL | [`runner.py:57-59`](../../../examples/sciaccel_rl/runner.py#L57) |
| ①前 | runner 手里的 `session_router_url` 是从 agent_loop 传来的 | [`agent_loop.py:358`](../../../examples/sciaccel_rl/agent_loop.py#L358) |
| ③④⑨ | 通用 `post` → `request_json` → aiohttp 的 `session.request` | [`http_utils.py:701-719`](../../../psrl/utils/common/http_utils.py#L701)、[`:260-290`](../../../psrl/utils/common/http_utils.py#L260) |
| ⑥ | 路由表：`方法 + 路径 → 函数` | [`session_router.py:261-276`](../../../psrl/workers/gen/session_router.py#L261) |
| ⑦ | `control_gpu_bind` 处理并返回 | [`session_router.py:535-573`](../../../psrl/workers/gen/session_router.py#L535) |

这些代码如果有理解上的问题，可以随时追问。

## 3. 网络基础：IP、端口、TCP、HTTP 各管什么

这一节是通用的计算机网络知识（不是本项目代码特有的），我没有逐条去查标准文档，但都是很基础、稳定的概念。

### 3.1 用"公寓楼"建立直觉

```
  公寓楼地址        = IP 地址     （找到哪一台机器）
  房间号            = 端口        （找到这台机器上的哪个程序）
  住户              = 进程        （一个正在运行的程序）
  敲门 → 开门       = 建立 TCP 连接
  隔着门说话         = 在连接上收发字节
  说话的语言规范      = HTTP 协议   （双方事先约定的"话术格式"）
```

类比的边界：真实世界里一个房间可以住很多人，但网络里**同一个 (IP, 端口) 同一时刻只能有一个程序在"监听"**（第 4.2 节会实验）；另外"敲门"之后得到的是一条**专属于这一次通话的连接**，不是进房间。

### 3.2 四层各干什么

```
  应用层  HTTP        "说什么"：POST /control/gpu_bind + JSON      ← 本文重点
  传输层  TCP         "怎么可靠地送到哪个程序"：端口 + 重传 + 顺序
  网络层  IP          "送到哪台机器"：IP 地址 + 路由
  链路层  以太网/Wi-Fi  "这一跳怎么传"：本文不涉及
```

每一层只关心自己的事，上层把数据交给下层"打包"。程序员写 HTTP 请求时，TCP 和 IP 由操作系统自动处理。

### 3.3 TCP 的三个必须知道的特点

| 特点 | 含义 | 为什么和本文有关 |
|---|---|---|
| **面向连接** | 通信前先"握手"（三次：客户端问好、服务端应答、客户端确认），建立一条专属连接 | 图上 ③ |
| **可靠、有序** | 丢包会重传，到达顺序和发送顺序一致 | 所以 HTTP 不用操心丢包 |
| **字节流，没有消息边界** | 对 TCP 来说，只是一串字节在流动；你发两次，对方可能一次收到，也可能分三次收到 | **这就是为什么 HTTP 必须自己规定"一条请求到哪儿结束"**，见第 6 节 `Content-Length` |

### 3.4 HTTP 是什么

HTTP 是**应用层协议**：规定了"请求和响应的文本长什么样"。核心规则只有一条：**客户端发一个请求，服务端回一个响应**。它是"请求—响应"模式，服务端不会主动发起（这个点在第 7 节回答"为什么发了就有回应"时是关键）。

## 4. 为什么 SessionRouter 要有自己的端口

### 4.1 先看 runner 和路由器是不是"一伙"的

- `runner.py` 跑在 `AgentLoopWorker` 里。它是 [`@ray.remote`](../../../psrl/workers/agent_loop/worker.py#L45) 声明的 Ray Actor，即**独立的操作系统进程**（见 [`sciaccel-gpu-gate.md` §16.1 补充](sciaccel-gpu-gate.md#161-补充2026-10-07这里说的多进程具体指什么从哪行代码能看出来)）。
- SessionRouter 由 `RolloutGateway` Actor 用 `multiprocessing.Process` 拉起一个**另外的子进程**（[`rollout_gateway.py:187-204`](../../../psrl/workers/gen/rollout_gateway.py#L187)），并监听该 Actor 所在**节点的 IP**（[`:112`](../../../psrl/workers/gen/rollout_gateway.py#L112)、[`:148`](../../../psrl/workers/gen/rollout_gateway.py#L148)）。

两个进程**不共享内存**，不能互相调用对方的函数、读对方的变量，而且可能在**两台不同的机器**上（具体是否同机取决于部署，见 [`commit-72ca50c-gate-ab-walkthrough.md` §7 第 3 条](commit-72ca50c-gate-ab-walkthrough.md#7-从代码读出的-6-处注意点推断)）。

跨进程、跨机器要通信，**网络请求是最通用的办法**：只需要知道对方的 `IP:端口`。这就是为什么 `session_router_url` 要一层层传到 runner 手里。

### 4.2 端口是什么，为什么每个服务要有自己的

操作系统收到一个网络数据包时，只知道"发给这台机器的 8200 号"。**端口号是操作系统用来决定"这个包交给哪个进程"的编号。** 一个进程想接收请求，必须向操作系统登记"我来负责 8200"，这个动作分两步：

| 步骤 | 系统调用 | 含义 |
|---|---|---|
| `bind` | 绑定 | 登记"我要用 (IP, 8200) 这个地址" |
| `listen` | 监听 | 声明"我开始接客了"，之后 `accept` 一次返回一个新连接 |

**同一个 (IP, 端口) 同一时刻只能被一个进程监听**，否则系统就不知道包该交给谁。实验里亲手验证了（见 [第 10 节](#10-可运行实验)）：

```
  再开一个监听同一端口 -> OSError: [Errno 48] Address already in use
```

这就是 SMG 路由器和 SessionRouter **必须各用各的端口**：代码里 SMG 从 8100 往上找（[`rollout_gateway.py:115`](../../../psrl/workers/gen/rollout_gateway.py#L115)），SessionRouter 从 8200 往上找（[`:147`](../../../psrl/workers/gen/rollout_gateway.py#L147)）。

**怎么挑端口**：[`find_available_port`](../../../psrl/utils/common/http_utils.py#L18) 从基准端口开始，用"试着 bind + listen 一下，成功就说明空闲"的办法逐个往上试。有两个小细节（我读代码的观察）：
1. 检查和真正启动之间有一小段时间窗口，期间端口可能被别的进程抢走（和 [`sciaccel-gpu-gate.md` §19.2](sciaccel-gpu-gate.md#192-背景知识什么是文件锁advisory-锁和-mandatory-锁有什么区别) 讲的 TOCTOU 是同一类问题），代码在启动后 `sleep(1)` 再断言进程还活着（[`:205-206`](../../../psrl/workers/gen/rollout_gateway.py#L205)）来兜底。
2. 它是用通配地址 `""` 去试绑的，而真正的服务绑在具体 IP 上。**不同操作系统对"通配地址 vs 具体地址同端口"是否冲突的判定不同**：我在 macOS 上实测，服务绑在 `127.0.0.1:8200` 时，`find_available_port(8200)` 仍然返回 8200（看不出冲突）；我没有在 Linux 上验证。生产环境一般是 Linux，所以这更像是开发机上要留意的细节。

### 4.3 为什么是"独立进程 + 单独端口"，而不是塞进别的进程

直接的原因是第 4.1 节：runner 和路由器本来就不在同一个进程里，必须靠网络。至于**为什么当初把 SessionRouter 设计成独立的子进程**（而不是和 RolloutGateway 放在同一个进程的线程里），代码里没有写明理由。以下是我的推断，不是作者原话：`uvicorn.run` 是一个会一直阻塞的调用，放到子进程里可以让它拥有独立的事件循环，不和 `RolloutGateway` Actor 抢同一个 Python 解释器。

### 4.4 为什么 gpu_bind 不需要自己的端口

端口是稀缺且需要"登记"的系统资源，而路径只是**请求里的一段字符串**，同一个程序可以随便分配多少。把功能拆成路径有两个好处：

1. **零额外成本**：新加一个功能只是多注册一行路由（[`session_router.py:272`](../../../psrl/workers/gen/session_router.py#L272)），不用再抢端口、再公布一个地址、再让 runner 多持有一个 URL。
2. **共用同一份状态**：`gpu_bind` 要修改的是 `self.states` 里的档案卡，聊天转发读的也是这份。同一个进程里，这是一个普通的字典访问；如果拆成另一个进程，就还得再想办法同步。

## 5. URL 格式：怎么拆，为什么这么设计

### 5.1 拆开看

```
   http://10.0.0.7:8200/control/gpu_bind
   └─┬─┘  └───┬───┘└─┬─┘└──────┬───────┘
   协议     主机地址   端口      路径
```

| 部分 | 作用 | 这里的值 |
|---|---|---|
| 协议（scheme） | 用哪种应用层协议。`http` 默认端口 80（这里显式写了 8200 所以用不上默认） | `http` |
| 主机 | 找到哪台机器。可以是 IP，也可以是域名（域名要先通过 DNS 翻译成 IP）。这里直接是 IP，不需要 DNS | `10.0.0.7`（教学假设） |
| 端口 | 找到机器上的哪个进程 | `8200` |
| 路径 | 告诉服务端这次要哪项功能/资源 | `/control/gpu_bind` |

URL 字符串是 `runner` 自己拼的：

```python
f"{session_router_url.rstrip('/')}/control/gpu_bind"
```
—— 节选自 [`runner.py:57-59`](../../../examples/sciaccel_rl/runner.py#L57)

`rstrip('/')` 去掉基础地址末尾可能多出的斜杠，避免拼出 `...8200//control/gpu_bind`。基础地址本身在 [`rollout_gateway.py:207`](../../../psrl/workers/gen/rollout_gateway.py#L207) 用 `f"http://{session_ip}:{session_port}"` 生成。

### 5.2 这套路径是怎么分类的

把 SessionRouter 的路由整理成一张表（注册处 [`session_router.py:261-276`](../../../psrl/workers/gen/session_router.py#L261)）：

| 方法 + 路径 | 属于 | 谁调用 |
|---|---|---|
| `POST /sessions` | 会话管理：新建一个 session | agent_loop（[`session_agent_loop.py:64-73`](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L64)） |
| `GET /sessions/{sid}` | 会话管理：查 | 同上 |
| `DELETE /sessions/{sid}` | 会话管理：删 | 同上（[`:75-78`](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L75)） |
| `POST /sessions/{sid}/v1/chat/completions` | **数据面**：真正的"问模型" | agent（容器里） |
| `GET /control/sessions`、`POST /control/hang`、`POST /control/continue`、`POST /control/gpu_bind` | **控制面**：给协调器/调度方用的"遥控器" | 协调器、runner |

"控制面 vs 数据面"是网络和系统设计里的常见分法：**数据面**搬运真正的业务数据（对话内容），**控制面**只调整数据面的行为（挂起、继续、绑定 GPU）。代码注释也把 `hang/continue` 叫作 "Coordinator-facing hang/continue control plane"（[`session_router.py:266`](../../../psrl/workers/gen/session_router.py#L266)）。`gpu_bind` 放进 `/control/` 下，是和它们同类。

### 5.3 几个"为什么这样设计"（部分是推断）

| 设计点 | 代码里能确认的 | 我的解读（推断，非作者原话） |
|---|---|---|
| `session_id` 放在路径里：`/sessions/{sid}/v1/...` | agent 拿到的"模型地址"就是 `.../sessions/{sid}/v1`（[`session_agent_loop.py:93-95`](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L93)） | 外部 agent 框架通常只让你配置一个"模型 base URL"，把 session id 嵌进地址，就不用改 agent 框架去加自定义头；`/v1` 结尾是兼容 OpenAI 风格的客户端，它们会自己往后补 `/chat/completions` |
| `gpu_bind` 的 `session_id` 放在**请求体**里，而不是 `/sessions/{sid}/gpu` | 请求体是 `[{"session_id":..., "gpu_device":...}]`，可以放多条（[`:535-573`](../../../psrl/workers/gen/session_router.py#L535)） | 和同在 `/control/` 下的 `hang`、`continue` 保持**批量**格式一致（它们的解析函数见 [`:575-627`](../../../psrl/workers/gen/session_router.py#L575)）。runner 实际只传一条 |
| 用 `POST` | 绑定与解绑都走 POST | HTTP 方法的语义惯例：`GET` 读、`DELETE` 删、`POST` 提交一个动作。绑定/解绑是"让服务端执行一个动作"，所以用 POST。这是惯例，不是技术强制 |
| 路径用小写加下划线 `gpu_bind` | — | 与同类端点的风格一致 |

### 5.4 方法 + 路径 共同决定由谁处理

注册路由时写的是 `self.app.post("/control/gpu_bind")(self.control_gpu_bind)`，意思是"**方法是 POST 且路径是 `/control/gpu_bind` 的请求，交给 `control_gpu_bind`**"。这里 `app.post(path)` 返回一个装饰器，紧跟着的 `(self.control_gpu_bind)` 是把函数交给它登记。实验（第 10 节）里手写了三种不匹配：路径写错得到 `404 Not Found`；路径对但方法不对在**我的教学服务端**里我让它返回 `405 Method Not Allowed`——我认为 FastAPI 真实行为也是 405，但没有运行真实服务验证。

### 5.5 速查小结：往数据面端点 POST 一下会发生什么，为什么要区分 sid

针对 `POST /sessions/{sid}/v1/chat/completions` 这个数据面端点的两个常见疑问（2026-10-12 补，行号已按当前工作区核对）。

**问题 1：往这里 POST 一下，模型就会回复吗？**

会，但这是一次**同步的"你问我答"**：agent 发请求后一直等，模型的回复就是这次 HTTP 响应的响应体，没有"模型自己找上门说话"这回事。中间经过的环节：

| 步骤 | 谁 | 做了什么 | 位置 |
|---|---|---|---|
| ① | agent（容器里） | `POST .../sessions/s-xxx/v1/chat/completions`，体里是完整对话历史 | — |
| ② | SessionRouter | 按 `sid` 取出这场会话的 `SessionState`（档案卡）；会话正在关闭则直接回 `409` | [`:322-345`](../../../psrl/workers/gen/session_router.py#L322) |
| ③ | SessionRouter | 往请求头里补：会话 id、上次用的推理实例、版本号 | [`:356-362`](../../../psrl/workers/gen/session_router.py#L356) |
| ④ | SessionRouter → SMG | 转发到 SMG 的 `v1/chat/completions`，**转发路径里已经没有 `sid`，身份搬进了请求头** | [`:374-379`](../../../psrl/workers/gen/session_router.py#L374) |
| ⑤ | SMG → vLLM | 选实例、生成、记录本轮 token（TITO）。这一段的源码在外部 Rust 项目里，本工作区没有，按官方设计文档转述 | — |
| ⑥ | SessionRouter | 记下实际服务的实例和版本，供下一轮用 | [`:398-406`](../../../psrl/workers/gen/session_router.py#L398) |
| ⑦ | SessionRouter | 若该会话绑了 GPU，先等卡空闲（Gate B），再返回响应 | [`:425-435`](../../../psrl/workers/gen/session_router.py#L425) |

**问题 2：为什么要区分不同的 sid？**

同时有很多个 episode 在跑，每个是一场独立对话。`sid` 像银行的取号单：没有它，柜台分不清谁办到哪一步。代码里 `sid` 至少承担这些作用：

| 作用 | 没有 sid 会怎样 | 代码依据 |
|---|---|---|
| 录像归档（TITO） | 几百场对话的 token 混在一起，取不回属于某个 episode 的训练样本 | 请求头 `x-smg-tito-session-id`（[`:33`](../../../psrl/workers/gen/session_router.py#L33)）；episode 结束后 `GET /sessions/{sid}` 一次取回 |
| 版本钉住 | 后续轮次可能落到比第 1 轮更旧的权重版本上 | `state.version_tag`（[`:404-406`](../../../psrl/workers/gen/session_router.py#L404)） |
| 记住上次的实例 | 没处存"这场会话上次用了哪个实例" | `state.base_worker_id` / `target_dp_rank`（[`:399-403`](../../../psrl/workers/gen/session_router.py#L399)） |
| 会话级控制 | 协调器没法只暂停某一场；删除会话时无法等它的在途请求排空；Gate B 不知道这场对话占着哪张卡 | `hang_state`、`inflight`、`gpu_device` 都存在 `SessionState` 里 |
| 防串号 | agent 可能自称别的会话 | 客户端自带的会话 id 请求头会被丢掉，只认路由器自己写入的（[`:739`](../../../psrl/workers/gen/session_router.py#L739)） |

**两处容易说过头的地方（已核对）：**

1. **"粘性路由"默认是关的。** `enable_trajectory_sticky` 的仓库默认值是 `False`（见 官方设计译注 04 §6（本地资料，未随仓库发布））。路由器代码**无条件**记录并回传实例信息（`:399-403`），但粘性关闭时 SMG 是否真的忽略这份提示，是官方设计文档的说法，SMG 源码不在本工作区，我没有验证。版本钉住不受这个开关影响。
2. **用一个没创建过的 `sid` 去 POST，路由器自己不会拒绝。** `_ensure_state`（[`:642-651`](../../../psrl/workers/gen/session_router.py#L642)）找不到就**当场新建**一张只含会话 id 头的档案卡，不带 `create_session` 时从请求头抄下来的 `x-prompt-id`、`x-version-tag` 等默认值。之后 SMG 会怎么处理这个它没登记过的会话，我没有验证。

## 6. 一条 HTTP 请求的原文长什么样

下面是脚本在本机实际发出和收到的内容（`\r\n` 换行已显示成换行）。**这是教学服务端的响应，不是真实的 uvicorn 响应**，但请求格式是 HTTP/1.1 的标准格式，真实服务端接收的请求与此同形。

```
POST /control/gpu_bind HTTP/1.1                ← 请求行：方法  路径  协议版本
Host: 127.0.0.1                                ← 头部（headers）：一行一个 "名字: 值"
Content-Type: application/json                 ← 告诉对方：体是 JSON
Content-Length: 55                             ← 告诉对方：体有 55 个字节
Connection: close                              ← 这次说完就关连接
                                               ← 空行：头部结束的标志
[{"session_id": "s-abc", "gpu_device": "/dev/nvidia1"}]    ← 请求体（body）：正好 55 字节
```

响应：

```
HTTP/1.0 200 OK                                ← 状态行：协议版本  状态码  文字说明
Server: BaseHTTP/0.6 Python/3.14.7
Date: Sun, 11 Oct 2026 16:51:02 GMT
Content-Type: application/json
Content-Length: 35

{"bound": ["s-abc"], "missing": []}            ← 响应体
```

要点：

1. **整个 HTTP 就是这样的文本**。没有什么神秘的二进制结构。浏览器、aiohttp、curl 做的都是帮你拼这段文本并解析返回的文本。
2. **`Content-Length: 55` 为什么必须有**：TCP 是没有消息边界的字节流（第 3.3 节）。服务端读完头部（遇到空行）后，必须知道体有多长才能停下。实验第 6 步故意把体只发 5 个字节就暂停 1 秒，服务端**一直等**，到齐后才回复，总耗时 1.00 秒。
3. **状态码**是服务端告诉客户端"结果怎么样"的数字：`2xx` 成功，`4xx` 是客户端的请求有问题，`5xx` 是服务端自己出错。`control_gpu_bind` 里：JSON 读不出来返回 `400`（[`session_router.py:547-550`](../../../psrl/workers/gen/session_router.py#L547)），其余情况返回 `200`——**包括 session 不存在，这时只是把它放进响应体的 `missing` 列表**，状态码仍是 200（实验第 5 步演示了这个行为，复刻自 [`:561-573`](../../../psrl/workers/gen/session_router.py#L561)）。

## 7. POST 之后为什么就会有回应

这个问题很多初学者会觉得像魔法：我只管发，怎么就有东西回来？拆成四个层面，一层一层看。

### 7.1 第一层：TCP 连接本来就是双向的

三次握手后得到的连接，**两端都可以读也可以写**。客户端写进去的字节，服务端 `read` 得到；服务端 `write` 的字节，客户端 `read` 得到。就像电话：一方说话，另一方也能说话，是同一条线路。所以"回应"不需要另外建一条路，直接**沿着同一条连接写回来**。图上 ⑧⑨ 就是这回事。

### 7.2 第二层：HTTP 规定了"一问一答"

HTTP 是**协议**，协议就是双方事先说好的规矩。规矩之一："客户端发完一个完整请求之后，服务端必须回一个完整响应，客户端才算这次通信结束。"所以"POST 之后会有回应"，是因为服务端**按规矩**要回；客户端也**按规矩**在等。

### 7.3 第三层：服务端框架替你做了"收—找—回"

服务端不用自己处理 socket。流程（图上 ⑤–⑧）：

| 步骤 | 谁做 | 做了什么 |
|---|---|---|
| ⑤ | uvicorn | 从连接上读字节，按第 6 节的格式解析出方法、路径、头、体 |
| ⑥ | FastAPI | 用方法 + 路径查路由表，找到要调用的函数 |
| ⑦ | 你写的函数 | `control_gpu_bind(request)`：改档案卡，`return JSONResponse(...)` |
| ⑧ | FastAPI + uvicorn | 把返回值包装成响应文本（状态行、头、体），写回同一条连接 |

关键：**你写的函数只管 `return` 一个值。`return` 之后，框架负责把它变成字节写回去。** 这就是"我函数里没写发送，为什么对方收到了回应"的答案。如果函数里没有 `return`，FastAPI 会把返回的 `None` 序列化成 JSON `null` 并以 200 回应（这是我对 FastAPI 的了解，没有在本机运行验证）。

### 7.4 第四层：客户端的 `await` 就是在连接上等字节

runner 里是 `await post(...)`。这个 `await` 的意思是：**发出请求后，把这个协程挂起，让事件循环去做别的事；等响应的字节到了再回来继续**（asyncio 的基础，见 `psrl-concurrency-landscape.md`（本地资料，未随仓库发布））。调用链：

```
_bind_gate_b → post(url, payload, max_retries=1)
   → request_json_maybe_distributed → request_json → request_raw/_request_once
      → aiohttp: session.request("POST", url, json=payload)  → 读响应 → 状态码/体/头
```
—— 对应 [`runner.py:57-61`](../../../examples/sciaccel_rl/runner.py#L57)、[`http_utils.py:701-719`](../../../psrl/utils/common/http_utils.py#L701)、[`:657-697`](../../../psrl/utils/common/http_utils.py#L657)、[`:260-290`](../../../psrl/utils/common/http_utils.py#L260)

两个我从代码读出来的细节：

- **`max_retries=1` 实际意味着"不重试"**：函数内部是 `attempts = max(1, max_retries)`，出错后 `retry_count += 1`，`retry_count >= attempts` 就抛出（[`http_utils.py:462`](../../../psrl/utils/common/http_utils.py#L462)、[`:435-437`](../../../psrl/utils/common/http_utils.py#L435) 对应 `request_raw` 的同款逻辑）。`post` 走的是 `request_json`，我读了 `request_raw` 的完整代码，`request_json` 的 `attempts` 行是同样的写法，但后面的重试分支没有逐行读完，这里按结构类比。
- **`post()` 默认会因为 4xx/5xx 抛异常**（`raise_for_status` 默认是 `True`，见 [`http_utils.py:666`](../../../psrl/utils/common/http_utils.py#L666)）。所以如果 SessionRouter 回了 `400`，`await post(...)` 会抛出，被 `_bind_gate_b` 的 `except Exception` 吞掉，只记一条 debug 日志。这正是上一篇说的 best-effort。

### 7.5 如果没有回应会怎样

| 情况 | 现象 | 结果 |
|---|---|---|
| 服务端进程不在、端口没人监听 | 连接被拒绝（TCP 握手失败） | aiohttp 抛连接错误 → 被 `_bind_gate_b` 吞掉，episode 照跑，只是没绑定 Gate B |
| 服务端收到了但一直不回 | 客户端一直等 | 取决于超时配置。`create_aiohttp_client` 里的总超时来自 `_runtime_config.total_timeout_secs`，其**默认值是 `None`**（[`http_utils.py:90`](../../../psrl/utils/common/http_utils.py#L90)、[`:201-202`](../../../psrl/utils/common/http_utils.py#L201)），我没有去查别处是否会覆盖它，所以"默认可能无限等"只是代码默认值的读法 |
| 中间网络断了 | 连接被重置/超时 | 同第一行 |

## 8. 回到 Gate B：扣住回复在网络层面是什么样

现在可以用网络语言复述上一篇的 Gate B 了。agent 向 SessionRouter 发的 `POST /sessions/s-abc/v1/chat/completions`，也是一条请求，也要等响应。**Gate B 所谓"扣住回复"，在网络层面就是：服务端已经拿到了模型的结果，但故意先不把响应文本写回连接**（[`session_router.py:425-435`](../../../psrl/workers/gen/session_router.py#L425) 的 `await _wait_for_gpu_idle` 发生在 `return self.build_response(result)` **之前**）。

从 agent 视角：连接开着，请求发出去了，**迟迟没有收到响应的字节**，和"模型生成得慢"没有区别。这也解释了为什么上一篇强调等待上限必须小于 agent 的 LLM 请求超时 900 秒：那个超时就是客户端"等响应字节"的最长时间，超过了就会判定请求失败。

两条请求的对比：

| | `POST /control/gpu_bind` | `POST /sessions/{sid}/v1/chat/completions` |
|---|---|---|
| 谁发 | runner | 容器里的 agent |
| 服务端处理时长 | 毫秒级（改个字段就返回） | 秒到分钟（等模型生成，再叠加 Gate B 的等待） |
| 两者共用同一个端口 | 是 | 是 |
| 会互相阻塞吗 | 不会：每个请求在服务端是一个独立的协程，`await` 时让出控制权 | 同左 |

## 9. 跟着图读源码的执行路径

| 步骤 | 对应图上 | 调用位置 |
|---|---|---|
| 1 | 启动 | [`rollout_gateway.py:147`](../../../psrl/workers/gen/rollout_gateway.py#L147) `find_available_port(8200)` |
| 2 | 启动 | [`:187-204`](../../../psrl/workers/gen/rollout_gateway.py#L187) 拉子进程 → [`:178-184`](../../../psrl/workers/gen/rollout_gateway.py#L178) 构造 `SessionRouter` 并 `uvicorn.run`（bind + listen） |
| 3 | 启动 | [`:207-209`](../../../psrl/workers/gen/rollout_gateway.py#L207) 返回 `session_router_url` |
| 4 | ① | URL 传到 runner：[`agent_loop.py:358`](../../../examples/sciaccel_rl/agent_loop.py#L358) → [`runner.py:245`](../../../examples/sciaccel_rl/runner.py#L245) → [`:41-61`](../../../examples/sciaccel_rl/runner.py#L41) |
| 5 | ③④⑨⑩ | [`http_utils.py:701`](../../../psrl/utils/common/http_utils.py#L701) `post` → [`:657`](../../../psrl/utils/common/http_utils.py#L657) → [`:260-290`](../../../psrl/utils/common/http_utils.py#L260) `_request_once` |
| 6 | ⑥ | 路由表 [`session_router.py:272`](../../../psrl/workers/gen/session_router.py#L272) |
| 7 | ⑦ | [`session_router.py:535`](../../../psrl/workers/gen/session_router.py#L535) `control_gpu_bind` → [`:567`](../../../psrl/workers/gen/session_router.py#L567) 写字段 → [`:573`](../../../psrl/workers/gen/session_router.py#L573) `return JSONResponse` |

## 10. 可运行实验

脚本：`docs/psrl/scripts/http_gpu_bind_demo.py`（本地资料，未随仓库发布）。只用标准库：服务端是 `http.server`，客户端是裸 socket、手写 HTTP 文本。它复刻了 `find_available_port`、`control_gpu_bind` 的绑定/解绑/`missing`/400 逻辑，注释里标了每一步对应图上的哪一段。

```bash
cd /Users/liuwei/CodeBuddy/20260811161322
python3 docs/psrl/scripts/http_gpu_bind_demo.py
```

本机（macOS）实际输出的关键部分：

```
[步骤 2] 同一个 (地址, 端口) 不能被两个服务同时监听
  再开一个监听同一端口 -> OSError: [Errno 48] Address already in use
  另起一个监听 0.0.0.0:8300 的占位者后，find_available_port(8300) -> 8301（顺延）

[步骤 3] ... POST /control/gpu_bind（绑定）→ 响应 {"bound": ["s-abc"], "missing": []}
  服务端档案卡现在是: {'gpu_device': '/dev/nvidia1'}
[步骤 3b] 解绑：gpu_device = null
  服务端档案卡现在是: {'gpu_device': None}

[步骤 4] 方法 + 路径 共同决定由谁处理
  路径写错 POST /control/gpu_bnd         -> HTTP/1.0 404 Not Found
  方法不对 GET /control/gpu_bind         -> HTTP/1.0 405 Method Not Allowed
  GET /sessions/s-abc                -> HTTP/1.0 200 OK

[步骤 5] 状态码：同一个端点，不同结果
  session 不存在  -> HTTP/1.0 200 OK | {"bound": [], "missing": ["nope"]}
  请求体不是合法 JSON -> HTTP/1.0 400 Bad Request | {"error": "invalid JSON body"}

[步骤 6] ... 请求体只发 5 字节后停 1 秒，再发剩余部分；总耗时 1.00s
  服务端在请求体到齐之前一直在等，到齐后才回复 -> HTTP/1.0 200 OK
```

你也可以在另一个终端里自己开真实环境：用 `curl -v -X POST http://<router>/control/gpu_bind -H 'Content-Type: application/json' -d '[...]'` 观察真实 SessionRouter 的请求和响应头（`-v` 会把请求/响应原文打印出来）。这条命令我没有运行，因为本机没有运行中的 SessionRouter。

## 11. 没有验证的部分与注意点

- **真实服务端（FastAPI + uvicorn）和真实客户端（aiohttp）本机没有运行**（缺 `aiohttp` 等依赖）。真实响应是 HTTP/1.1 还是 1.0、响应头有哪些，以真实运行为准；教学服务端是 HTTP/1.0 风格、写完就关连接。
- 第 5.4 节 "FastAPI 对路径对而方法不对返回 405"、第 7.3 节 "函数返回 `None` 时回 `null`"，都是我对 FastAPI 的了解，**没有在本机验证**。
- **`/control/gpu_bind` 在这份代码里没有任何鉴权**：我读 [`control_gpu_bind`](../../../psrl/workers/gen/session_router.py#L535) 没有看到身份校验，任何能连到这个 `IP:端口` 的进程都能调用。是否由内网/防火墙隔离不在代码里，需要看部署。
- `post` 在配置了"分布式 POST"时，会改由一组 Ray Actor 代发（[`http_utils.py:669-681`](../../../psrl/utils/common/http_utils.py#L669)），失败再回落到本地请求。本文按本地请求路径讲解；开启后实际发请求的进程会变，但 URL、报文格式、服务端行为不变。
- 第 3 节的网络基础（三次握手、四层模型、端口语义）是通用知识，没有逐条对照 RFC 或操作系统文档。

## 相关文档

- [Gate B 新手版讲解](gate-b-beginner-guide.md)：本文的上篇，讲 Gate B 五个零件各做什么。
- [提交 72ca50c 代码讲解](commit-72ca50c-gate-ab-walkthrough.md)：Gate A/B 的专业版代码讲解。
- 并发/并行全景（本地资料，未随仓库发布）：协程、线程、进程三层，解释 `await` 与事件循环。
- [PSRL 笔记总索引](../../README.md)
