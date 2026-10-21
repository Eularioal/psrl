# 知识文档索引

本目录收录对话中产生的知识性 / 总结性解释，便于日后查阅。规则见 [AGENTS.md](../AGENTS.md)。

> **核心结论**：按主题查阅下列索引。环境执行与回收专题另列入口，AIRS-Bench 接入审查与修复报告包含源码证据、CPU 对照实验和实机验收边界。

按主题组织文档；下列入口提供主题索引，已拆分的长文同时列出子文档。

目录：

- [PSRL 框架与相关知识](#psrl-框架与相关知识)
- [环境执行与回收专题](#环境执行与回收专题)
- [开发工具](#开发工具)
- [其他位置的文档](#其他位置的文档)

## PSRL 框架与相关知识

PSRL 训练框架本身（运行指南、源码地图、Agent Loop/Harbor/TITO 工具调用、Env Service/EnvWorker 设计提案）、以及理解它必须懂的 RL 算法知识（GAE、PPO/GRPO/DAPO、RLVR）和底层推理系统知识（vLLM、NIXL）、还有一次真实的训练崩溃复盘。包含以下两组已拆分专题；其他文档详见 [psrl/README.md](psrl/README.md)。

- [vllm-and-training-parallelism](psrl/vllm-and-training-parallelism.md)：已拆分为 5 篇子文档，原文件保留为索引。
- [vLLM 推理机制：缓存、调度与合批](psrl/vllm-and-training-parallelism/inference-mechanisms.md)：从 prefill/decode 的计算差异出发，理解 KV 缓存、PagedAttention、前缀复用、连续合批和选择性合批；原有算例与性能边界保留。
- [训练并行、通信量与带宽分析](psrl/vllm-and-training-parallelism/training-parallelism.md)：按切分对象理解训练并行，再用每卡通信量和互联带宽判断并行策略应如何排布。
- [PSRL 训推协作与 PPO、GRPO 基础](psrl/vllm-and-training-parallelism/rl-training-workflow.md)：串联轨迹生成、概率重算、优势估计、参数更新与权重同步，并保留 PPO/GRPO 的逐步推导和长尾分析。
- [PSRL 分组采样、DAPO 与训练损失](psrl/vllm-and-training-parallelism/rl-loss-and-sampling.md)：解释同一 prompt 的分组采样、DAPO 改动、熵与 KL 项，以及多轮 response_mask、奖励标签和 rollout correction。
- [推理与训练并行：公式速查及实验记录](psrl/vllm-and-training-parallelism/experiments-and-reference.md)：集中查阅符号、公式、机制对应的实验脚本与历史测量条件；历史结果不代表当前环境的重新验证。

- [psrl-run-guide](psrl/psrl-run-guide.md)：已拆分为 5 篇子文档，原文件保留为索引。
- [PSRL 基础运行与示例脚本](psrl/psrl-run-guide/getting-started.md)：以 DAPO 示例说明安装、模型与数据准备、启动参数和单机 Demo；执行前先核对集群检查专题中的环境约束。
- [PSRL 的 Ray 调度与生成长尾](psrl/psrl-run-guide/ray-and-rollout.md)：分别说明 Ray 的角色、资源调度流程和 rollout 长尾，区分集群调度职责与模型计算职责。
- [PSRL 日志监控、批量配置与进程排查](psrl/psrl-run-guide/monitoring-and-batches.md)：从训练日志提取指标，理解 train_prompt_bsz 的配置映射，并确认训练进程所在节点及运行状态。
- [PSRL 单机与多机启动故障复盘](psrl/psrl-run-guide/startup-incident-reviews.md)：保留 IP、数据格式、环境变量覆盖及上下文长度问题的历史故障链、修复过程与验证边界。
- [PSRL 共享环境与集群检查](psrl/psrl-run-guide/cluster-preflight.md)：源码依赖、节点与 Docker 预检、Docker 文件系统阻塞排查，以及 science 训练复盘、监控容错、显存交接、对照设计和停止条件；历史状态每次重新核实。

## 环境执行与回收专题

- [GPU 作业入口、任务遥测与执行后端统一](env-worker-deep-dive/gpu-jobs-and-task-telemetry.md)：显式异步 GPU 作业、按容器进程归因、宿主机 PID 视图要求、science 评分接线，以及 Harbor 与 EnvWorker 的 shell 状态对照和统一路线。

- [implementation-and-gaps.md](env-worker-deep-dive/implementation-and-gaps.md) —— EnvWorker 计数与回收缺口，含 AIRS-Bench 标签隔离、MLGym 适配、shell 超时、排队与清理问题审查。
- [AIRS-Bench 审查与复现](env-worker-deep-dive/implementation-and-gaps.md#17-airs-bench-接入批评核对与最小复现) —— 2026-10-13 修复前代码裁定、真实 Bash 复现、上游 MLGym 契约检查和修复顺序建议。

- [AIRS-Bench 修复与小实验](env-worker-deep-dive/airs-bench-audit/fixes-and-results.md) —— 普通 CSV 评分隔离、最终奖励取值、shell 有界 I/O、资源回收，以及真实 MLGym 回放、远端 Docker 小实验和持续复核结果。
- [APPS 判题隔离实现](env-worker-deep-dive/airs-bench-audit/apps-isolation.md) —— 可信 judge 与无答案容器分离、评分错误分类、正式镜像小样本对照与 MLGym 回放，以及全量吞吐的验收边界。
- [Docker 借用 rollout GPU](env-worker-deep-dive/rollout-gpu-lending.md) —— 区分 Ray 配额与物理显存，复核现有实验边界，实现 science 命令批次与模型调用交替借卡，附静态 unit 配额说明、CPU 回归、两轮真实 GPU 借还结果，以及普通任务阶段租约、抢占和预测回填的设计边界。
- [EnvRouter 命令阶段配额](env-worker-deep-dive/command-gpu-reservations.md)：保留容器状态、按命令归还 GPU 配额及后台归还与规划重叠的实现，包含慢查询定位与直接接口修复的 GPU 对照、失败验收、CPU 教学模拟及跨节点调度提案。
- [EnvRouter 当前流程与优化总览](env-worker-deep-dive/env-router-current-flow.md)：串联任务放置、会话账本、GPU 命令借还、节点预热与清理，区分独立环境卡和 vLLM 借卡分支。
- [修复后 Router 与固定 OFF 的单卡对照](env-worker-deep-dive/router-baseline-control-validation.md)：在相同物理 GPU 和固定成功数量下，比较低压力开销、高压力 OOM 补采与命令阶段安全回填，补充 NVML 连接复用后的低负载优化实测及私有节点预热、固定 OFF 的实机复验，明确合成任务和训练的区别。

- [Science 训练资源账本对照](env-worker-deep-dive/science-env-router-comparison.md) —— 解释 14 GiB 任务的 unit 余量，比较旧整卡锁与 EnvRouter 预算共享，附真实 GPU 并发与显存图、完整训练对照未完成的原因和修复验证边界。
- [真实 LAPS 训练的 Router 开关对照](env-worker-deep-dive/science-training-router-ab.md) —— 两组各 8 条轨迹、2 step 的 entropy/loss 与资源曲线；第 10 节汇总 EnvRouter 的意义、共享预算与回收收益，区分 GPU 容器加速和训练尚无性能优势的证据，附源码路径与 CPU 演示。

- [Science GPU 压力与 Router 训练对照](env-worker-deep-dive/science-gpu-pressure-ab.md) —— 最新按指定旧提交对照四题各四条、OOM 补采：两组完成，OOM 7/0，但 Router 采样慢 64.3%；保留历史结果、HTTP 修复、等完成量图表及整任务租约空闲占用证据。

## 开发工具

- [inode 与 Docker 写回阻塞](tools/linux/inode-basics.md)：用真实六字节文件实验解释文件名、inode、硬链接、删除后的打开引用，以及 inode 容量与写回等待的区别。

和具体研究内容无关的开发工具与系统基础：Claude Code 权限模式与执行命令方式、Codex 的 `AGENTS.md` 作用范围、会话管理与配置备份回退、Shell `source`、Python `import` 和 Linux inode。详见 [tools/README.md](tools/README.md)。

## 其他位置的文档

- [文档站点构建指南](#documentation-site)

## Documentation site

Source for the PSRL documentation site, built with [Sphinx](https://www.sphinx-doc.org/)
and published on [Read the Docs](https://psrl.readthedocs.io/en/latest/).

This README is a guide for people editing the docs. It is excluded from the built
site.

---

## Build locally

From the repository root:

```bash
# 1. Install the documentation toolchain
python -m pip install -r docs/requirements.txt

# 2. Build the HTML site
cd docs
make html

# 3. Preview it
python -m http.server -d _build/html 8000
```

Then open <http://localhost:8000>.

Building the docs does not import PSRL itself, so a full PSRL environment (vLLM, Ray,
Megatron) is not required. The packages in `requirements.txt` are enough and install
cleanly into a bare virtualenv.

For an auto-reloading dev server that rebuilds on save:

```bash
python -m pip install sphinx-autobuild
cd docs
make livehtml
```

`make clean` removes `_build/`, and `make help` lists every target Sphinx offers.

To catch broken references the way CI does, promote warnings to errors:

```bash
cd docs
make html SPHINXOPTS="-W --keep-going"
```

---

## Layout

```
docs/
├── conf.py            # Sphinx configuration
├── Makefile           # make html / livehtml / clean
├── requirements.txt   # Sphinx, theme, and extensions
├── README.md          # This guide (not part of the site)
├── index.md           # Landing page and top-level toctree
├── _static/           # CSS, JS, and figures (SVG, PNG, MP4)
├── _templates/        # Theme template overrides
├── overview/          # What PSRL is and how it fits together
├── tutorial/          # Installation, quickstart, configuration reference
├── design/            # Architecture and subsystem deep dives
└── examples/          # End-to-end recipes (RLVR, agentic RL, reward models)
```

`.readthedocs.yaml` lives at the repository root, where Read the Docs expects it.

---

## Editing

### Add a page

1. Create the `.md` file in the right section, for example `docs/design/new_feature.md`.
2. Register it in that section's `index.md` toctree, otherwise Sphinx warns that the
   page is orphaned:

   ````markdown
   ```{toctree}
   :maxdepth: 1

   architecture
   parameter_server
   new_feature
   ```
   ````

3. Rebuild and check for warnings.

### Markdown flavor

Pages are MyST Markdown. Enabled extensions include `colon_fence`, `deflist`,
`dollarmath`, `amsmath`, `attrs_inline`, `substitution`, and `tasklist`, so
definition lists, `$...$` math, and `:::`-fenced directives all work.

Cross-reference other pages with `` {doc}`../design/architecture` `` instead of a raw
relative link, so Sphinx validates the target at build time.

### Diagrams

Mermaid renders client-side:

````markdown
```{mermaid}
graph LR
    A[Train Worker] -->|push| B[Parameter Server]
    B -->|pull| C[Rollout Instance]
```
````

### Figures

Put figures in `docs/_static/img/` and reference them with a root-relative path:

````markdown
```{figure} /_static/img/overview.svg
```
````

Prefer SVG. To convert a PDF figure, use `pdf2svg in.pdf out.svg` or
`inkscape in.pdf --export-filename=out.svg`.

The repository `.gitignore` excludes `**/*.png` as a runtime artifact, with an
explicit exception for `docs/_static/img/*.png`. Keep new raster figures in that
directory so they are not silently dropped from commits.

### Sphinx Design components

```markdown
:::{grid-item-card} Title
:link: target/page
:link-type: doc
Description text.
:::

::::{tab-set}
:::{tab-item} FSDP
FSDP instructions.
:::
:::{tab-item} Megatron
Megatron instructions.
:::
::::

:::{admonition} Title
:class: tip
Content.
:::
```

---

## Publishing

Read the Docs builds from `.readthedocs.yaml` at the repository root, which points at
`docs/conf.py` and installs `docs/requirements.txt` on Python 3.11. Pushing to the
default branch triggers a rebuild.

To validate docs on pull requests:

```yaml
# .github/workflows/docs.yml
name: Docs
on: [pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r docs/requirements.txt
      - run: make -C docs html SPHINXOPTS="-W --keep-going"
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not import extension myst_parser` | `pip install -r docs/requirements.txt` |
| `Theme error: no theme named 'pydata_sphinx_theme'` | Same, the theme ships in `requirements.txt` |
| Mermaid block renders as plain text | Confirm `sphinxcontrib-mermaid` is installed and the fence is `{mermaid}` |
| `document isn't included in any toctree` | Add the page to its section `index.md`, or list it in `exclude_patterns` |
| Figure missing from output | Check the file is under `docs/_static/img/` and the path starts with `/_static/img/` |
| Stale output after edits | `make clean && make html` |
