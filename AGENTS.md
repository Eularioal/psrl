# Project Contract

## ALWAYS

- Apply first-principles thinking. Do not assume that I always have a clear understanding of what I want or how to achieve it. Stay cautious and start from the fundamental needs and problem. If the motivation or objective is unclear, pause and discuss it with me. 
- When running scripts or inspecting the environment, please activate the conda environment by executing `source /apdcephfs_zwfy10/share_303541817/lhy/env/psrl.sh`. All dependencies and packages are installed within this environment.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.

## Coding Guidelines

Three reference files live under `.claude/`:

- **`.claude/coding-style.md`** — formatting rules, naming conventions, docstrings, logging, and annotation markers. Ordered by risk (silent-bug rules first).
- **`.claude/codebase-map.md`** — system architecture, directory tree, configuration hierarchy, quick-lookup indices, and import dependency graphs.
- **`.claude/readme-style.md`** — what belongs in a README and what does not. Read it before writing or editing any user-facing `.md`.

Claude must read and apply these guides when writing or modifying code.

## Documentation

- Keep all workspace documentation in this repository's `docs/` directory. From the parent workspace, this is `psrl/docs/`.
- Merge documentation into the existing topic directories. Do not create a sibling `../docs/` directory or a `docs/notes/` migration layer.
- Update `docs/README.md` and relative links when adding or moving documentation.

- A README tells a reader what to do and what will bite them. It is not the lab notebook that proves how we learned it.
- Do not paste measured forensics into documentation. Keep a number only if the reader acts on it, and drop the run-specific evidence that merely justifies a past decision. Experiment results belong in a `Results` section or a `FINDINGS.md`, tied to the script that reproduces them.
- Never write history into documentation or comments. `git log` owns what the code used to be.
- Everything here is published. No absolute paths, real hostnames, cluster IPs, or internal mirrors, and every command must run exactly as written.

## 架构与设计文档时效维护

- 修改架构实现、组件职责、接口、配置默认值、生命周期或能力边界，以及新增或更新设计文档前，先读 `docs/README.md`、专题索引和当前设计入口，搜索受影响的本地文档及反向引用。纯代码任务同样需要判断文档是否失效。
- 对照当前源码确认调用链、状态、配置条件和限制，标注核对日期、基准提交、未提交改动及核对范围。区分本地实现、已合并版本和实际部署，不能以刷新日期代替核验。
- 优先修改已有文档的原结论，同步更新核心结论、正文、图表、示例和源码链接。不得仅追加新文或进展说明，却保留相互矛盾的现状描述。提案落地后更新状态，不同配置分支分别说明。
- 固定版本分析和历史实验保留原始条件，明确历史范围并链接现状入口，不将历史结果改写成当前验证。无法确认的结论标注待核实及具体缺口。
- Env 环境执行的现状入口是 `docs/env-worker-deep-dive/knowledge/current-design.md`，分类入口是同目录 `README.md`。相关变化须同步入口和专题，其他主题沿用自己的既有入口。
- 完成前搜索受影响范围内的旧接口、旧配置和旧限制，检查引用、章节锚点及源码行号的语义。汇报维护了哪些文档、哪些结论待核实。无需更新时说明依据。文档核验不自动授权额外训练或 GPU 实验。

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)


## 基础知识文档必须面向初学者

- 凡新增或补充基础知识文档，默认读者第一次接触主题，不假设已经了解分布式系统、操作系统、GPU 或框架术语。此要求同样适用于对话讲解，除非用户明确要求简讲。
- 先交代真实场景、要解决的问题和为什么需要，再解释机制。不得用术语清单替代讲解。
- 术语、缩写和英文名首次出现时，解释中文含义、用途，以及在当前流程中实际做什么。关键术语还要说明内部工作方式，不能用另一个未解释术语替代。
- 至少给出一个完整例子，包含初始状态、输入、中间步骤和结果。涉及容量、重试或并发时用可手算数字，教学假设和实测结果明确区分。
- 多组件协作先给全局图，再说明谁调用谁、数据和状态如何变化。结合已核实源码提供跳转位置，并配自包含 CPU 教学实验，说明与真实系统的差异。
- 保存的文档必须保留讲解过程、例子、术语解释与验证边界，不能把详细聊天缩成结论提纲。简短操作 README 可链接到详细知识文档，但知识文档本身不能因 README 风格规则省略基础解释。
