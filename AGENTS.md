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

- Chapter figures must answer the section’s main questions through a readable scenario, state changes and causal evidence. Agree on the questions before drawing, organize dense material into a clear reading order, and inspect at the actual document width. Image presence alone is not a quality check.

- Follow the parent workspace `AGENTS.md` for audience, explanation depth, visual explanations, rendered figure checks, experiment evidence, documentation updates and link checks. Keep these shared rules there rather than duplicating them here.
- Every maintained document under `docs/`, including READMEs and short references, must embed a relevant explanatory color figure. Follow the parent rules for meaningful diagrams, sources, rendered inspection and the figure coverage check.
- Maintain documentation in this repository's `docs/`, merge into existing topics, and update the relevant indexes and relative links.
- Assume familiarity with machine learning, LLMs and agents, but less experience with computer systems. Explain the system mechanisms needed to follow the project, keep the main explanation self-contained, and offer optional topic links or follow-up questions for deeper learning.
- Choose structure by purpose: operational READMEs focus on running the project, tutorials explain mechanisms with examples, and experiment reports lead with the question, results and their explanation. Preserve experimental conditions and evidence in the report or linked evidence files.
- Keep closely related explanations together. Use headings to navigate a long topic, merge fragments that readers need together, and split only for distinct reading tasks rather than a line-count limit.
- Use `.claude/readme-style.md` for operational READMEs. Its short-format rules do not replace the explanation required in tutorials or experiment reports.
- Published documentation uses repository-relative paths or explicit placeholders. Keep real hostnames, cluster IPs and internal mirrors out of published examples. Reproduction commands must state their prerequisites.

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)
