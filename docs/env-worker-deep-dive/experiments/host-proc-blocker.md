# GPU 进程归属的 procfs 挂载障碍与实验处理

> **核心结论：** 当前节点能运行 Docker CUDA，但 procfs 后端缺少与 NVML 宿主进程号对应的宿主视图，不等于驱动损坏。新增可选 eBPF 后端已接入节点任务遥测和显存核销，并通过双容器 CUDA 归属及回收验证，可以绕开宿主挂载。完整 science 严格评分尚未验收，已有 `observe` 结果仍须标记 GPU 归因未验证。

## 目录

- [给平台同事的说明](#给平台同事的说明)
- [实验如何继续](#实验如何继续)
- [潜在风险与替代方式](#潜在风险与替代方式)
- [实现与验证](#实现与验证)

## 给平台同事的说明

PSRL 运行于平台提供的容器环境，内部再用 Docker 创建 science 任务容器。NVIDIA 管理接口 NVML 返回宿主机进程号，而节点当前 `/proc` 使用容器的进程编号范围，两者无法可靠关联。

```text
NVIDIA 驱动 → GPU 进程号 ──────────────┐
                                   ↓
节点服务 → 宿主 procfs → 进程所属容器 → 任务 GPU 活动
                ↑
          当前缺少这个视图
```

图例：进程采样和归属关联由 [TaskGpuTelemetry](../../../psrl/workers/env_service/task_telemetry.py#L70)实现，能力检查在 [check_gpu_attribution_support](../../../psrl/workers/env_service/command_gpu_node.py#L164)。图中展示 procfs 后端的挂载依赖，可选 eBPF 后端通过内核事件维护身份。平台挂载位于外层节点部署，不由内部任务容器负责。

**希望平台提供：** 将物理宿主机 `/proc` 只读挂载给可信节点运行容器，例如挂到 `/host/proc`，并为节点服务设置 `PSRL_GPU_HOST_PROC=/host/proc`。不需要向 science 任务容器开放此挂载。不必为此升级 GPU 驱动。若不允许挂载，可以考虑由宿主机采集服务提供可信的进程归属信息。

复现错误为 `GPU attribution needs host PID procfs`。例如，监控返回宿主 PID 2000，但节点只看得到同一进程的本地 PID 300，就不能直接查询 `/proc/2000` 判断它属于哪个任务。这两个数字是教学示例。已有只读诊断与部署片段见[详细记录](../knowledge/gpu-jobs-and-task-telemetry.md#89-设计要求当前检查与修复边界)。

## 实验如何继续

在本次实验实际使用的 science agent 配置中设置：

```yaml
harbor:
  gpu_attribution_mode: observe
```

这是 `SciAccelAgentLoop` 的 `harbor` 配置，不是任务的 `task.toml`，也不是 `psrl.env_service` 下的字段。仓库默认配置保持 `required`，本节点后续实验按用户要求显式使用 `observe`。

`observe` 跳过原生 CUDA command 路径的可信归因启动门槛，保留任务 verifier 返回的评分。verifier 是检查任务结果并计算分数的程序。结果新增 `gpu_attribution_enforced=0`，`gpu-job-reports.json` 记录 `task_verifier_unverified`、跳过原因及原始遥测错误，即使没有 GPU 作业报告也保存标记。

容量准入、设备授权、进程取消、GPU 归还等待和容器清理继续执行。缺少可信归属时，当前显存准入实现保留未能核销的预算，不把未知占用当成零。`observe` 不是忽略所有 GPU 或 Docker 错误的开关。

## 潜在风险与替代方式

- **评分风险：** 若任务原始 `gpu_active` 使用整卡利用率，邻居的 GPU 计算可能让本任务的 CPU 计算也获得 GPU 活动分。保留的 `reward_gpu` 必须标为未经可信归因验证。优先用数值正确性、完成量、OOM 和耗时评价本轮调度实验。
- **性能解释风险：** 无法准确拆分每个任务的显存和利用率，保守预算可能减少并发。整卡曲线不能写成某条任务的曲线，观察到的耗时变化也不能全部归因于调度器。
- **适用边界：** 如果验收目标就是准确 GPU 评分或进程记账，需要恢复 `required` 并补齐能力。当前没有宣称这个问题已经解决。

| 替代方式 | 能解决什么 | 当前限制 |
|---|---|---|
| `observe` 模式 | 让实验不再被挂载条件阻断 | 不解决可信进程归属 |
| 宿主采集服务 | 在正确的进程视图内建立 PID 与容器关系 | 需要平台部署，尚未实现 |
| eBPF 内核追踪 | 不挂宿主 procfs，按内核身份、资源组及命令注册号与 NVML 关联 | 已接入 command 节点接口并通过双容器 CUDA 验证，完整 science 评分、其他内核及长期运行仍需验收 |
| 任务程序内采样 | 可记录自身计算阶段和部分显存数据 | 要适配程序，可能遗漏子进程，任务自行上报不能直接作为可信评分 |
| 独占 GPU 并做前后对照 | 减少邻居干扰，可做受控功能验证 | 必须保证实际独占，牺牲共享并发，也不等价于逐进程归属 |

单纯改用 Docker PID、容器名或 cgroup 不能自动得到 NVML 宿主 PID 的映射。保留容器归属标签仍然有用，但它只解决映射链的一侧。

两种方式的通俗比较、配置与实现见[进程身份专题](../knowledge/process-identity-without-procfs.md)。`psrl.env_service.gpu_identity_backend: ebpf` 选择可选内核后端，默认仍为 procfs。新后端已经通过双容器子进程的独立记账、取消后失效和归还验证，不再依赖随机名字。启用之前仍需实际节点严格预检，既有 observe 报告不能改标为可信归因。

## 实现与验证

跟着流程读源码：配置由 [HarborConfig](../../../examples/sciaccel_rl/config.py#L17)校验，进入 [runner](../../../examples/sciaccel_rl/runner.py#L263)的启动分支，再由 [gpu_grading](../../../examples/sciaccel_rl/gpu_grading.py#L46)选择评分策略并保存证据，最后仍经过 runner 的归还等待与清理。这些代码如有理解问题，可以继续追问。

CPU 回归覆盖严格模式拒绝、观察模式继续、未验证标记落盘以及回收错误仍然生效。运行命令如下，容器与节点接口使用测试替身，不是实机训练：

```bash
pytest -q tests/env_service/test_gpu_attribution_policy.py
```

后台验证的日志、状态和交接说明保存在 [procfs-observe 后台任务目录](../env-service-verification/results/procfs-observe-20261026)。测试结果以该目录的 `status.json` 为准。本次没有启动新 GPU 训练，没有修改平台部署。
