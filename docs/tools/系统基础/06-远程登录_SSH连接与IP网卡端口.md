# 远程登录：SSH 连接与 IP、网卡、端口

> **核心结论**：IP 地址用于把数据送到连接的一端，网卡是操作系统用来收发数据的网络接口，端口让操作系统把数据交给对应的服务或连接。`SSH_CONNECTION` 的四列记录了一段 SSH 连接两端各自的 IP 和端口。

## 目录

- [1. 一次登录的四个数字](#1-一次登录的四个数字)
- [2. IP 怎样和网卡关联](#2-ip-怎样和网卡关联)
- [3. 一个服务端口怎样接待多条连接](#3-一个服务端口怎样接待多条连接)
- [4. 有 tunnel 时客户端是谁](#4-有-tunnel-时客户端是谁)
- [5. 在 Shell 里逐项核对](#5-在-shell-里逐项核对)

## 1. 一次登录的四个数字

假设 SSH 客户端 A 主动登录 Linux 机器 B。B 上的 SSH 服务程序 `sshd` 已经在 TCP 端口 `36000` 等待连接。TCP 是操作系统用来在两端之间传递有序字节流的协议。登录后，在 B 的远程 Shell 里查看 `SSH_CONNECTION`，假设看到下面四列。全文 IP 使用文档专用示例地址。 [OpenSSH 手册](https://man.openbsd.org/ssh#SSH_CONNECTION) · [RFC 5737](https://www.rfc-editor.org/rfc/rfc5737.html#section-3)

```text
198.51.100.20 42640 192.0.2.40 36000
```

[![SSH 连接的两个端点、各自的网络接口，以及一个监听口怎样接收两条连接](../../_static/img/document-figures/ssh-connection-endpoints-screen.png)](../../_static/img/document-figures/ssh-connection-endpoints.svg)

从图的上半部分看第一条连接。A 的操作系统通常会为这次连接分配一个临时源端口，这里是 `42640`。它发送给 B 的数据以 `192.0.2.40` 为目标 IP，以 `36000` 为目标 TCP 端口。数据到达后，B 的操作系统把连接交给正在监听的 `sshd`。B 返回的终端输出沿同一条连接送回 A，其目标端口是 `42640`。

| 列 | 示例值 | 在这次登录里的含义 |
|---:|---|---|
| 1 | `198.51.100.20` | B 看到的 SSH 客户端 IP |
| 2 | `42640` | B 看到的客户端源端口，回复数据发往这个端口 |
| 3 | `192.0.2.40` | 这条连接到达 B 时使用的本地 IP |
| 4 | `36000` | 这条连接到达 B 时使用的 SSH 服务端口 |

因此，在这次远程登录里问“我登录到了哪个 IP”，看第三列。要写连接目标则用第三、四列，例如 `ssh -p 36000 user@192.0.2.40`。这里的 A、B 是同一条连接的两个角色，B 回送数据时角色仍保持不变。

## 2. IP 怎样和网卡关联

Linux 把收发网络数据的接口称为 **网络接口**，命令里通常用 `dev` 表示。物理以太网卡可以提供接口，软件也可以创建接口。例如 `bond1` 可以是将多个物理链路组织起来的逻辑接口，`docker0` 通常是 Docker 用来连接容器的网桥，`lo` 是把数据留在本机的回环接口。用 `ip -d link show` 可以查看接口类型细节。

**IP 是配置在接口上的地址。** B 可以在 `bond1` 上配置 `192.0.2.40/26`，在另一个接口上配置其他地址。一块接口也可以有多个地址。于是机器的地址清单和某条连接实际使用的地址是两项不同的信息。前者看 `ip address`，后者看连接两端的记录。 [ip-address(8)](https://man7.org/linux/man-pages/man8/ip-address.8.html)

`/26` 表示 IPv4 地址的前 26 位是网络前缀。在这个例子里，前 24 位确定 `192.0.2`，再用最后一段的前 2 位划分地址块。剩 6 位可变化，所以一个块有 `2^6 = 64` 个地址值，`.40` 落在 `.0` 到 `.63` 这一块。B 发数据时，内核查路由规则来选择接口和下一步交给谁。若查询结果含有 `via`，它给出的就是下一跳地址，例如把发往 A 的数据先交给网关 `192.0.2.1`。 [ip-route(8)](https://man7.org/linux/man-pages/man8/ip-route.8.html)

## 3. 一个服务端口怎样接待多条连接

假设同一个 A 又开一个终端，发起第二条独立 TCP 连接。B 仍然监听 `192.0.2.40:36000`，A 这次使用 `42641`。冒号左边是 IP，右边是端口，合在一起表示一个端点。

| TCP 连接 | 客户端端点 | 服务端端点 |
|---|---|---|
| 第一条 | `198.51.100.20:42640` | `192.0.2.40:36000` |
| 第二条 | `198.51.100.20:42641` | `192.0.2.40:36000` |

**监听**是服务先在本地地址和端口上登记，等待新连接。客户端发起连接时，内核根据目标地址和端口找到匹配的监听入口。连接建立后，内核用两端的 IP 和端口四项组合区分这两条连接。B 因此可以在同一个 `36000` 上接待多个客户端，也知道每段终端输出该送回哪条连接。程序通过 **socket** 读写这些连接，socket 是操作系统提供给程序的通信接口。 [TCP 规范 §3.4.1](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.4.1)

监听地址也会影响连接匹配。`192.0.2.40:36000` 表示绑定这个具体地址，`127.0.0.1:36000` 只接受本机回环地址上的连接，`0.0.0.0:36000` 表示在当前网络环境的所有本地 IPv4 地址上监听。远端客户端要能连上，还需要有通往对应地址的路由，并满足网络访问规则。 [Linux ip(7)](https://man7.org/linux/man-pages/man7/ip.7.html)

## 4. 有 tunnel 时客户端是谁

`SSH_CONNECTION` 从 **接受这条 SSH 连接的服务端**的视角记录端点。A 直接连到 B 时，第一列对应 B 看到的 A 的源地址。如果笔记本先连跳板机，再由跳板机向 B 发起 TCP 连接，B 看到的客户端端点就是跳板机出站连接的地址和端口。如果中间设备做了网络地址转换，B 看到的是转换后的地址。要追溯最初发起操作的机器，需要结合 tunnel 或跳板机的连接方式判断。 [OpenSSH 跳板连接选项 `-J`](https://man.openbsd.org/ssh#J)

`SSH_CONNECTION` 是由 SSH 登录设置、后续进程可以继承的环境变量。这意味着 tunnel 提供的 Shell 可能没有该变量，也可能继承外层 SSH 登录的值。例如先 SSH 登录 B，再从 B 启动 tunnel 服务，tunnel 内的 Shell 若继承到了该值，它描述的是那次 SSH 登录。要确认 Shell 当前在哪台机器上运行，可同时用 `hostname` 查看主机名，用 `ip address` 核对当前环境管理的接口地址。 [OpenSSH 环境变量](https://man.openbsd.org/ssh#ENVIRONMENT)

## 5. 在 Shell 里逐项核对

下面用 Linux 上的 Bash 和 `iproute2` 工具。先确认这一段 SSH 连接的端点，再看哪些地址配置在本机接口上。

```bash
printf '%s\n' "${SSH_CONNECTION:-}"
printf '%s\n' "${SSH_CONNECTION:-}" | awk '{print $3}'
hostname
ip -4 -brief address
```

第一行显示四列，第二行只取服务端 IP。最后一行列出 IPv4 接口及其地址，示例中会有 `bond1 UP 192.0.2.40/26`。`bond1` 是接口名，`UP` 是该行展示的接口状态，`192.0.2.40/26` 是接口上配置的地址和前缀。可用 `ip -d link show dev bond1` 进一步查看该接口。

如果第一行有四列且是 IPv4，可以继续在同一个 Shell 查当前路由和端口。

```bash
read -r ssh_peer_ip ssh_peer_port ssh_local_ip ssh_local_port <<< "$SSH_CONNECTION"
ip -4 route get "$ssh_peer_ip"
ss -ltn "sport = :$ssh_local_port"
ss -tn state established "( sport = :$ssh_local_port and dport = :$ssh_peer_port )"
```

| 命令 | 怎样读结果 |
|---|---|
| `ip -4 route get "$ssh_peer_ip"` | 查询当前系统到该对端的出站路由，看 `dev` 选了哪个接口、`src` 选了哪个源地址。它执行一次路由查询，历史入站路径需要其他证据。 |
| `ss -ltn ...` | 查本服务端口上的 TCP 监听记录。`-l` 只看监听，`-t` 选 TCP，`-n` 保留数字地址和端口。 |
| `ss -tn state established ...` | 查当前已建立的 TCP 连接。在 B 上，`Local Address:Port` 对应 B，`Peer Address:Port` 对应 A。结合两端 IP 核对是否为第一条连接。 |

如果最后一条命令只显示表头，当前网络环境没有找到符合过滤条件的已建立连接。环境变量可能仍保留此前登录的四列值。`ss -ltn` 查到监听记录，只说明该端口此刻在等待连接。

在 B 上查询它自己的地址时，`ip -4 route get "$ssh_local_ip"` 可能显示 `local ... dev lo`，因为此时目的地就在本机。要确认这个 IP 配在哪个接口，回到 `ip -4 -brief address`。 [ip-route(8)](https://man7.org/linux/man-pages/man8/ip-route.8.html) · [ss(8)](https://man7.org/linux/man-pages/man8/ss.8.html)

同样的 IP 和端口关系也适用于 HTTP 服务。HTTP 在连接建立后还会按请求方法和路径选择具体处理函数。 [网络通信：HTTP 请求与 Gate B 调用链](04-网络通信_HTTP请求与GateB调用链.md)
