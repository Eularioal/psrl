# Claude Code 使用

[上级导航](../README.md) · [总导航](../../README.md)

从下面的主题文档开始阅读，子目录入口列在文末。

<!-- figure:terminal-scroll-layers -->
这张图沿同一个场景解释：终端外层能滚动，为什么内部列表仍停在原地？

[![终端滚动：外层回滚、tmux历史和应用内部列表是三层（1/2）](../../_static/img/document-figures/terminal-scroll-layers-screen.png)](../../_static/img/document-figures/terminal-scroll-layers.png)

[![终端滚动：外层回滚、tmux历史和应用内部列表是三层（2/2）](../../_static/img/document-figures/terminal-scroll-layers-screen-02.png)](../../_static/img/document-figures/terminal-scroll-layers.png)

鼠标在一个终端中操作，里面先运行 tmux，再运行具有自身列表的交互应用。 同一个滚轮事件可能在不同层被消费。外层响应不能证明内层收到事件。 绘制的教学界面。快捷键、鼠标选项与版本差异按正文中的具体环境确认。 [查看矢量图](../../_static/img/document-figures/terminal-scroll-layers.svg)。
<!-- /figure:terminal-scroll-layers -->

## 文件与子目录

- [ClaudeCode：命令执行与权限设置](01-ClaudeCode_命令执行与权限设置.md)
