# 历史存档：原生后端修复前生命周期复现

> **核心结论**：旧版本 `8d4de8f` 在文件尚未上传完成时就允许命令进入；初始化与删除先后失败后，再次启动还会错误返回成功。本文保留这个故障机制和CPU注入输出，修复与真实容器验收见[原生后端报告](../native-backend-fixes-and-science.md)。


## 目录

- [7. 后续修复优先级与生命周期故障复现](#7-后续修复优先级与生命周期故障复现)
  - [7.1 第一优先：修复生命周期的就绪状态](#71-第一优先修复生命周期的就绪状态)
  - [7.2 第二优先：接入现有资源准入和回收，再扩展 science GPU](#72-第二优先接入现有资源准入和回收再扩展-science-gpu)
  - [7.3 第三优先：把同卡共享的性能边界测出来](#73-第三优先把同卡共享的性能边界测出来)
  - [7.4 暂缓：队列回填与主动 offload](#74-暂缓队列回填与主动-offload)
  - [7.5 Results 与可运行复现](#75-results-与可运行复现)

## 7. 后续修复优先级与生命周期故障复现

本节保留修复前版本 `8d4de8f` 的故障复现。ready意为环境已准备好，可以对外执行命令；旧图和下方输出解释错误如何发生。历史代码可从[保存的补丁](../../env-service-verification/results/native-fixes-20261029/final-source.patch#L710)对照修改前后的行，现行机制看[当前设计](../../knowledge/current-design.md)。

先看确认的问题如何发生。A、B 是两种分别触发的路径，不要求同时发生：

```text
[1] start → create_sandbox 成功 → _started=True
                                 │
            A：环境上传仍在等待 ─┼→ [2] 并发 exec 通过检查，进入后端
                                 │
            B：环境上传失败 ────┴→ [3] destroy_sandbox 再次失败
                                          │
                         _started=False / _closed=True 没有执行
                                          ↓
                              [4] 再次 start 直接返回成功
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 代码与职责 |
|---|---|
| 1 | [旧版start与修复差异](../../env-service-verification/results/native-fixes-20261029/final-source.patch#L768)：旧版在创建后立即设置 `_started`，随后才上传；修复后移到上传完成 |
| 2 | [旧版执行检查与新增门禁](../../env-service-verification/results/native-fixes-20261029/final-source.patch#L838)：旧检查只看 `_started` 和 `_closed` |
| 3 | [异常处理与清理差异](../../env-service-verification/results/native-fixes-20261029/final-source.patch#L790)：旧版先等待清理，再更新状态；修复引入明确失效和待清理状态 |
| 4 | [旧版启动检查](../../env-service-verification/results/native-fixes-20261029/final-source.patch#L770)：已有 `_started=True` 时直接返回 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，从 1 的正常启动读到上传等待，分支 A 跳到并发调用 2。分支 B 则从上传异常进入 3，清理再次异常后返回调用者，调用者重试才进入 4。第二条调用链不是“容器成功初始化两次”。

### 7.1 第一优先：修复生命周期的就绪状态

假设容器已创建，但题目数据上传到一半失败，Docker 又暂时无法执行删除。当前适配器会留下 `_started=True`、`_closed=False`。调用者第二次要求启动时，没有重新准备环境，却收到正常返回。这会让上层误以为环境可用，掩盖第一次初始化失败。

另外，数据还没上传完时，有另一个协程调用 exec，会进入后端执行。正常 Harbor 顺序调用通常先等待 start，因此本次不能声称生产 science 已发生该并发故障。但后端接口确实没有防住它，扩展并发执行前应该修复。

**修复前的建议方案：** 显式区分 `starting`（准备中）、`ready`（可用）、`closing`（清理中）、`cleanup_failed`（清理未确认）与 `closed`（已关闭）。只有目录、权限、环境文件全部准备成功才发布 ready。初始化失败时先撤销可用状态，再尝试清理。清理失败保留所有权和重试入口，不能把环境伪装成 ready，也不能未经确认就记为 closed。exec、上传和下载统一检查就绪状态，内部启动步骤通过专用内部方法执行，避免因公共接口的新门禁使启动自己卡住。

这不是简单地把 `_started=True` 挪到函数最后：当前启动中的 `ensure_dirs` 和上传过程也会调用适配器方法，需要一并理清“内部准备”和“对外可用”。命令超时导致底层销毁后，也应同步适配器状态。收尾复查发现并行工作已在未提交改动中加入 `_invalidate` 和 `exec_shell`，本节不将这部分算成本轮修复；上述 start 提前发布状态及清理失败问题仍未被覆盖。

**最小验收：** CPU 注入初始化失败、清理失败、并发 exec，检查拒绝错误就绪且能够重试清理。修复后再用一个真实 CPU 容器验证启动中断及删除结果，不需要训练或 GPU。

### 7.2 第二优先：接入现有资源准入和回收，再扩展 science GPU

修复前每个 `EnvWorkerEnvironment` 自行实例化一个 EnvWorker。单容器 Docker CPU/内存上限已经生效，但当时这些实例没有共同参加 EnvRouter 的节点预算核算。当前 science 入口已补齐整次任务准入。这是明确的接入缺口，不是本轮复现的新 OOM。

教学例子：一台机器可用于环境的内存是 32 GiB，8 个任务分别配置 6 GiB。每个容器的 6 GiB 限制都正确，不妨碍总申请达到 `8 × 6 = 48 GiB`。单容器限额回答“一个容器最多用多少”，节点准入回答“现在还能不能再启动一个”。二者都需要。

建议复用既有资源租约，而不是再增加一份独立总账：先申请 CPU/内存/容器名额，创建时绑定所有权，销毁并确认后释放。再让 GPU 命令复用既有 command 的设备授权、预算和物理归还协议。现有 science 接线入口见 [runner.py:100](../../../../examples/sciaccel_rl/runner.py#L100)，不能直接解除适配器拒绝 GPU 参数的检查，就宣称接入完成。

最小验收先做受控容量下的两个 CPU 容器：第二个必须等第一个物理回收后才能取得名额。之后根据目标 science 任务实际需要补齐镜像或网络能力，再验证一条真实 GPU 命令与正常归还。跨节点 worker 池和多服务不必同时全部实现。可信进程归属沿用另一项工作的接口，不在这里重复建设。

### 7.3 第三优先：把同卡共享的性能边界测出来

这是测量与保护策略的缺口，不是已确认的错误调度。现有功能实验主要持有显存，不能量化环境持续计算使模型变慢多少。详见[计算竞争与控制等待](../../knowledge/vllm-env-opportunistic-sharing.md#11-计算竞争与控制等待已经验证到哪里)。

先把排队、GPU 命令、归还、模型控制门禁的时间边界分开记录。使用一个模型、小量固定输入输出、有限轮数，比较模型独占、环境仅持有显存、环境持续计算三组。分别报告生成耗时和控制等待，不用包含模型加载的总时间替代。训练吞吐不是这一轮必要验收目标。

有了数据，再决定哪些环境命令可以进入共享卡、命令期限要多短、是否需要限制共享并发。缺少测量时，不能通过减小显存安全余量来解决计算竞争，更不应把主动 offload 当作必然更快的方案。

### 7.4 暂缓：队列回填与主动 offload

[command_gpu.py:82](../../../../psrl/workers/env_service/command_gpu.py#L82) 当前按 FIFO，即先到先服务，选择队首等待者。教学例子：只剩 8 GiB，队首需要 12 GiB，后面需要 2 GiB，也可能跟着等待。允许小请求绕过队首称为回填，确实可能提高利用率，但必须限制绕过次数或保留预约，避免大任务一直没有机会。这是优化取舍，不等同于当前队列实现错误。

相比之下，就绪状态错误能直接让调用者误判环境成功，应先修。主动 offload 还需要空窗判断、恢复成本和提前返回处理，当前也没有充分性能证据支持优先扩展。

### 7.5 Results 与可运行复现

配套 [review_native_harbor_lifecycle.py](../../env-service-verification/scripts/review_native_harbor_lifecycle.py) 调用真实适配器，用测试替身代替 Docker 创建、删除和上传，内部 5 秒预算，不触碰共享服务：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/review_native_harbor_lifecycle.py
```

本次实际运行得到：

```json
{
  "exec_accepted_before_upload_finished": true,
  "first_start_error": "docker unavailable",
  "second_start_returned_success": true,
  "create_calls_after_two_starts": 1,
  "started_after_failed_cleanup": true,
  "closed_after_failed_cleanup": false
}
```

第一项对应图 A，后几项对应图 B，明确证明接口的状态问题。`docker unavailable` 是人为注入的异常，不表示实验节点的 Docker 真的故障。本次没有启动真实容器、模型或训练，也没有把上述接入建议写成已完成修复。
