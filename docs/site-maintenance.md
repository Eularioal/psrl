---
orphan: true
---

# 文档站点构建与维护

> **核心结论**：`docs/README.md` 是仓库内所有主题的总索引；`docs/index.md` 是 Sphinx 公开站点的首页。新增或移动文档时，先维护所属主题导航，再检查本地链接。公开站点页面还需要登记到 Sphinx 导航并通过构建。

## 目录

- [两种阅读入口](#两种阅读入口)
- [在独立环境构建](#在独立环境构建)
- [怎样新增或移动页面](#怎样新增或移动页面)
- [链接、图和资源文件](#链接图和资源文件)
- [常见构建问题](#常见构建问题)

## 两种阅读入口

| 入口或目录 | 用途 |
|---|---|
| [README.md](README.md) | 本地知识总览，覆盖所有主题 |
| [index.md](index.md) | Sphinx 站点首页及顶层导航 |
| [overview/](overview/index.md) | 项目介绍、中文入门和术语查询 |
| [tutorial/](tutorial/index.md) | 安装、运行和配置 |
| [design/](design/index.md) | 系统设计与组件机制 |
| [examples/](examples/index.md) | 算法和任务配方 |
| [psrl/](psrl/README.md)、[环境专题](env-worker-deep-dive/README.md)、[tools/](tools/README.md) | 仓库知识文档，目前由 `conf.py` 排除出公开站点 |

[Sphinx](https://www.sphinx-doc.org/) 把 Markdown 源文件生成 HTML 站点。[conf.py](conf.py) 定义纳入范围、主题和扩展；仓库根目录的 [.readthedocs.yaml](../.readthedocs.yaml) 定义 Read the Docs 构建。是否触发远端发布还取决于远端项目配置。

## 在独立环境构建

从仓库根目录执行，避免把站点依赖安装到共享训练环境：

```bash
python -m venv /tmp/psrl-docs-venv
/tmp/psrl-docs-venv/bin/python -m pip install -r docs/requirements.txt
/tmp/psrl-docs-venv/bin/sphinx-build -b html -W --keep-going docs /tmp/psrl-docs-html
/tmp/psrl-docs-venv/bin/python -m http.server -d /tmp/psrl-docs-html 8000
```

前三行准备依赖并生成页面，第四行启动本地预览；浏览器打开 `http://localhost:8000`。`-W` 将警告计为失败，`--keep-going` 继续收集问题。构建文档本身不要求启动 Ray、Docker、模型或 GPU。

若已在自己的文档环境中安装依赖，也可以在 `docs/` 执行 `make html`。输出位于 `docs/_build/html/`。`make clean` 清理该构建目录。

## 怎样新增或移动页面

1. 按主题选择目录。先搜索已有文档，能整合就直接改原内容。
2. 文档按“标题、核心结论、目录、正文”组织。先解释读者的问题和完整例子，再按需深入。
3. 更新所属分类 README，让总索引能访问到该页。
4. 对公开站点页面，在该专题 `index.md` 的 `toctree` 中添加文件名。
5. 检查相对路径、章节锚点与源码定位，最后构建站点。

例如新增 `docs/design/new-feature.md`，就在 `docs/design/index.md` 的导航中加入 `new-feature`。只写文件而不登记，读者可能找不到它，Sphinx 也会报未纳入导航。

同一问题的解释尽量放在一起，较长时用目录和章节导航。只有阅读用途确实独立时才拆分；合并后移除多余转发页，修正所有引用。历史结果保留原日期、代码版本和配置。

## 链接、图和资源文件

普通 Markdown 使用相对链接：

```markdown
[系统架构](design/architecture.md)
[源码函数](../psrl/workers/ps/ps_manager.py#L1194)
```

路径以当前 Markdown 所在目录为基准。文档间跳转使用标题锚点，源码定位使用 `#L行号`。示例中的行号只有在确实对应所述函数时才成立。

公开站点也支持 MyST 语法，如 `` {doc}`design/architecture` ``。移动页面时同样检查这些引用，不能只修普通 Markdown 链接。

源码中的相对 Markdown 链接供本地阅读。构建站点时，`conf.py` 会把指向站外仓库文档的实际文件链接转成对应 GitHub 源码链接，保留章节锚点；站内页面仍跳转到生成的 HTML。这使同一份正文兼顾两种阅读方式，站外资料继续保留原有发布范围。

Mermaid 流程图可用普通 `mermaid` 代码块在支持的 Markdown 阅读器查看；Sphinx 页面使用 `{mermaid}` 指令。图先帮助串起流程，后面用源码表说明对应实现。

图片放在 `docs/_static/img/`。站点指令中的 `/_static/img/...` 从文档根解析，普通 Markdown 则按页面位置使用相对路径。新增位图要确认 Git 没有把它当作运行产物忽略。优先复用已有图，保留可重新生成的来源。

## 常见构建问题

| 现象 | 处理 |
|---|---|
| `Could not import extension` 或主题缺失 | 在独立文档环境安装 `requirements.txt`，确认构建使用同一 Python |
| 页面未包含在 `toctree` | 加到专题导航；有意不发布的资料放入既有排除范围 |
| 引用移动前的旧路径 | 同时检查 Markdown 链接、MyST 文档引用和卡片链接字段 |
| 页面能打开但章节跳转失败 | 核对目标标题锚点，包括重复标题和标点处理 |
| 源码链接行号超出文件或指向错误代码 | 重新读源码并修正定位，文件存在检查无法发现语义错误 |
| 本地结果仍显示旧页面 | 清理本次构建目录后重建，保留源码 |

全目录链接检查与站点构建回答不同问题：前者还覆盖不公开的知识文档，后者检查公开站点的 MyST 解析和页面导航。两者都通过后，仍需人工核对正文是否前后一致、结论是否有相应证据。
