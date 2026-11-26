# 站点图片

[上级导航](../README.md) · [总导航](../../README.md)

本目录保存站点架构图与各专题的彩色解释图。PNG便于直接阅读，SVG便于放大，子目录README提供对应正文、图注来源和复现脚本。

<!-- figure:raster-vector-source -->
这张图沿同一个场景解释：PNG、SVG 与绘图脚本分别解决显示、编辑还是复现的问题？

[![配图资源：PNG负责直接看，SVG和脚本负责放大与维护（1/2）](document-figures/raster-vector-source-screen.png)](document-figures/raster-vector-source.png)

[![配图资源：PNG负责直接看，SVG和脚本负责放大与维护（2/2）](document-figures/raster-vector-source-screen-02.png)](document-figures/raster-vector-source.png)

同一幅解释图既要在 Markdown 里直接显示，也要能修改和追溯数字来源。 显示文件不是全部来源。保留源脚本与原始证据，才能避免下一轮只剩无法核对的截图。 SVG 内嵌原始栅格面板时，该面板不会自动变成矢量。本轮证据图明确保留这一来源。 [查看矢量图](document-figures/raster-vector-source.svg)。
<!-- /figure:raster-vector-source -->

## 文件与子目录

- [论文原图与调度教学图集](literature-reviews/README.md)：作者原图、中文读图正文、来源清单与教学示意
- [全文档彩图：布局、时间线、曲线与阅读地图](document-figures/README.md)
- [PSRL_arch.svg](document-figures/explained-PSRL_arch-4d2b637.svg)
- [PSRL_character.png](PSRL_character.png)
- [PSRL_logo.png](PSRL_logo.png)
- [PSRL_logo_merge.png](PSRL_logo_merge.png)
- Ray 与 Science 轨迹的分工图为 800×554，根据项目调用关系绘制。修改 SVG 后，在安装 librsvg 命令行工具和 Droid Sans Fallback 中文字体的环境中，从本目录运行 `rsvg-convert -o psrl-ray-science-roles.png psrl-ray-science-roles.svg` 更新显示图。[PNG](psrl-ray-science-roles.png) · [可编辑 SVG](psrl-ray-science-roles.svg) · [对应正文](../../psrl/psrl-run-guide/01-PSRL运行_准备启动与监控.md#ray-角色与-rollout-流程)。
- [install_ffmpeg.sh](install_ffmpeg.sh)
- [overview.svg](document-figures/explained-overview-c46ca98.svg)
- [ps.svg](document-figures/explained-ps-d3858c3.svg)
- [rollout_techniques.svg](document-figures/explained-rollout_techniques-cf838b0.svg)
- [staleness_compatibility.svg](document-figures/explained-staleness_compatibility-ab32e0e.svg)
- [staleness_control.mp4](staleness_control.mp4)
