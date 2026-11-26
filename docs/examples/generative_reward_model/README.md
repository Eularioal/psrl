# 生成式奖励模型规划

[上级导航](../README.md) · [总导航](../../README.md)

从下面的主题文档开始阅读，子目录入口列在文末。

<!-- figure:model-derived-feedback -->
这张图沿同一个场景解释：独立反馈模型怎样影响策略学习，又有哪些角色不能混在一起？

[![模型反馈：评分者或教师为策略提供学习信号（1/2）](../../_static/img/document-figures/model-derived-feedback-screen.png)](../../_static/img/document-figures/model-derived-feedback.png)

[![模型反馈：评分者或教师为策略提供学习信号（2/2）](../../_static/img/document-figures/model-derived-feedback-screen-02.png)](../../_static/img/document-figures/model-derived-feedback.png)

设计方案：策略模型先回答，反馈模型再提供分数或教师目标，PSRL 使用对应学习信号。 反馈模型、环境任务模型和 Agent 策略是不同角色。先定义反馈协议，再讨论接入与训练。 提案示意，不表示 Judge、蒸馏或相关训练配方已全部接入正式路径。 [查看矢量图](../../_static/img/document-figures/model-derived-feedback.svg)。
<!-- /figure:model-derived-feedback -->

## 文件与子目录

- [Generative Reward Model](index.md)
