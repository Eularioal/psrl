# ReTool 工具调用

[上级导航](../README.md) · [总导航](../../../README.md)

从下面的主题文档开始阅读，子目录入口列在文末。

<!-- figure:rollout-batch-training -->
这张图沿同一个场景解释：一条 episode 结束后，为什么模型不一定马上开始训练？

[![异步采样与训练：一条轨迹完成后，先进入批次准备（1/2）](../../../_static/img/document-figures/rollout-batch-training-screen.png)](../../../_static/img/document-figures/rollout-batch-training.png)

[![异步采样与训练：一条轨迹完成后，先进入批次准备（2/2）](../../../_static/img/document-figures/rollout-batch-training-screen-02.png)](../../../_static/img/document-figures/rollout-batch-training.png)

固定批次教学：四条轨迹分别在第 3、5、6、8 秒完成，更新需要这四条。 优化单条延迟需要连到批次关键路径。完成一条轨迹与执行一次训练更新不是一一对应。 数字为教学假设。分块、冗余与异步配置可能改变消费条件，按实际配置说明。 [查看矢量图](../../../_static/img/document-figures/rollout-batch-training.svg)。
<!-- /figure:rollout-batch-training -->

## 文件与子目录

- [ReTool: Code Interpreter RL](index.md)
