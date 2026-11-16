# 上下文长度：模型能看多长、一次回答能写多长

**核心结论**：多轮任务要同时满足模型上下文、单次生成和训练序列的预算。先统计“当前完整历史 + 本轮计划输出”，再核对实际配方如何传递长度参数。单纯把推理窗口调大，会把更长轨迹交给训练侧，显存也要随之检查。

## 目录

- [用一个多轮任务理解三个长度](#用一个多轮任务理解三个长度)
- [三次历史故障分别发生在哪里](#三次历史故障分别发生在哪里)
- [`config.json` 逐字段说明](#configjson-逐字段说明)
- [RoPE 怎样表示位置](#rope-怎样表示位置)
- [怎样选择长度预算](#怎样选择长度预算)
- [延伸](#延伸)

## 用一个多轮任务理解三个长度

**上下文**是模型这次请求能看到的全部输入，包括初始问题、已有回答、工具结果和消息格式。文本经分词器编码后成为 **token**，它是长度预算的计量单位。

下面用教学数字说明。初始问题占 2000 token，第一轮模型写了 1000 token，工具返回 3000 token。第二轮再请求模型时，输入已经约有 6000 token。若模型窗口为 8000，本轮最多还能留约 2000 token 给新输出，另外还要计算模板等开销。

| 参数 | 主要用途 | 读配置时要核实的接线 |
|---|---|---|
| `max_prompt_length` | 数据加载时的初始问题长度预算 | 是否已经计入消息模板和工具说明 |
| `max_response_length` | 响应或训练轨迹的长度预算 | 具体 Agent Loop 怎样把它传给生成、截断及训练样本重建 |
| `max_model_len` | vLLM 配置的单请求总上下文上限 | 当前完整输入与允许输出之和是否放得下 |

这三个值在不同 Agent Loop 中可能有不同连接方式。当前 [Science 训练脚本](../../examples/sciaccel_rl/fsdp_qwen35_4b.sh#L68)把 `max_model_len` 设为 prompt 和 response 预算之和，再检查训练打包预算能覆盖它。为推理引擎额外放大窗口时，应同时核对 Agent 是否会花掉这段空间，以及 TITO 收回的轨迹是否仍符合训练预算。

## 三次历史故障分别发生在哪里

以下保留旧训练记录的配置和观察，说明排查思路。本次整理未重新运行它们，历史 `*_laps.sh` 名称也已由当前脚本入口替代。

| 故障阶段 | 当时条件与结果 | 从中学到什么 |
|---|---|---|
| 启动校验 | Qwen2.5-3B 配置声明 32768，脚本却请求 `2048+65536=67584`，8 个 vLLM server 初始化失败 | 先按具体 checkpoint 和引擎版本核对支持窗口 |
| 多轮请求 | 将总长降到 32768 后能启动，但部分实际 prompt 长到 32800–33600+，请求被拒绝 | 初始问题长度无法代表后续完整历史 |
| actor 反向传播 | 改用 Qwen3.5-4B 和较大窗口后，仍有超长请求，同时第一次训练更新 OOM | 推理长度控制和训练显存预算需要一起检查 |

第一类错误的原始提示为：

```text
ValueError: max_model_len (67584) should be less than or equal to max_position_embeddings (32768)
```

第二类提示为：

```text
The prompt (length 32851) is longer than the maximum model length of 32768.
```

当时较短轨迹仍能完成，记录中保留了 `global_step_10/20/30`，约每 5 小时一个 checkpoint。因此更准确的现象是“超长任务反复失败、训练低效推进”，而非整次训练始终完全停住。

第三类故障详见 [SciAccel OOM 复盘与五步验收](../sciaccel-grpo-oom-analysis.md)。推理请求被拒绝的长度，与最终进入训练的有效序列长度属于不同记录，后者仍需训练日志确认。原文把两者直接当作同一条超长训练样本，现改为分别解释。

## `config.json` 逐字段说明

以这次实际读到的 Qwen2.5-3B-Instruct `config.json` 为例：

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 32768,
  "max_window_layers": 70,
  "model_type": "qwen2",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

| 字段 | 含义 |
|---|---|
| `architectures` | 记录预期模型类（这里是 `Qwen2ForCausalLM`），具体加载分发还结合 `model_type` 和加载器 |
| `model_type` | `AutoConfig`/`AutoModel` 用来分发到对应建模代码的键；Qwen2.5 沿用的还是 `"qwen2"`，2.5 是训练数据/流程的升级，架构族没变 |
| `hidden_size` | 残差流（residual stream）的维度，也是 attention 输出、embedding 的维度 |
| `intermediate_size` | MLP（FFN）中间层维度，这里 `11008 / 2048 ≈ 5.37`，是这个模型的中间层尺寸比例 |
| `num_hidden_layers` | Transformer block 层数（深度） |
| `num_attention_heads` / `num_key_value_heads` | Query 头数 vs KV 头数。`16` vs `2` 说明是 **GQA（Grouped-Query Attention）**：每 8 个 query 头共享 1 组 K/V，KV cache 显存降到 MHA（`num_kv_heads=num_attention_heads`）的 1/8 |
| `max_position_embeddings` | 模型配置声明的最大位置数，具体支持长度还要结合模型说明和引擎校验 |
| `rope_theta` | RoPE 的频率基数，增大它会减慢多数频率通道，见后文公式 |
| `sliding_window` / `use_sliding_window` | 若开启，每个 token 只关注最近 `sliding_window` 个 token（省显存，牺牲长距离依赖）；这里 `use_sliding_window: false`，所以 `sliding_window` 的数值不生效，走的是全量 attention |
| `max_window_layers` | 部分 Qwen2 变体会让前若干层用 sliding window、后面的层用全量 attention 做混合；在 `use_sliding_window=false` 时这个字段不起作用 |
| `rope_scaling` | （本例没有）运行时长上下文扩展配置，见上文 |
| `tie_word_embeddings` | 输入 embedding 矩阵和输出 LM head 矩阵是否共享权重（参数量换效果的常见小模型技巧），`true` 表示共享 |
| `torch_dtype` | 默认加载精度 |
| `vocab_size` | 分词器词表大小，决定 embedding/LM head 矩阵的另一维 |
| `attention_dropout` / `initializer_range` / `rms_norm_eps` | 标准超参：注意力 dropout 概率、权重初始化标准差、RMSNorm 的 epsilon |
| `hidden_act` | MLP 激活函数，`"silu"` 对应 SwiGLU 结构里的门控激活 |
| `use_cache` | 生成时是否默认启用 KV cache（自回归推理的标准做法） |
| `bos_token_id` / `eos_token_id` | 特殊 token id |

## RoPE 怎样表示位置

**RoPE（旋转位置编码）**给注意力计算中的 query、key 向量加入位置相关旋转。可以把 query 理解为当前 token 寻找信息的特征，key 是其他 token 提供的匹配特征。位置差会影响它们的匹配分数。

常见形式中，第 $i$ 对维度在位置 $m$ 的旋转角度为：

$$
\phi_i(m)=m\,\theta^{-2i/d},\qquad i=0,\ldots,d/2-1.
$$

$d$ 是旋转维度数，$\theta$ 对应 `rope_theta`。取教学数字 $d=4$：第 0 个通道的速度为 1，第 1 个通道的速度为 $1/\sqrt\theta$。将 $\theta$ 从 10000 改为 1000000，第 1 个通道每走一步的旋转由 0.01 变为 0.001 弧度，第 0 个通道保持不变。

这说明增大基数会让多数通道旋转更慢，却不能由公式独立推出模型可可靠使用的最大长度。不同频率共同编码位置，单个通道转过一圈也不足以判定所有位置信息丢失。实际能力还取决于训练数据、模型权重和使用的扩展方法。

`max_position_embeddings` 是配置声明，不能据此还原预训练数据中每条序列的真实长度。当前 [Qwen2.5-3B 官方配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)仍列出 32768 和 `rope_theta=1000000`，具体支持方式应按模型说明核对。

**`rope_scaling`（位置编码扩展配置）**指定怎样调整位置或频率。线性插值、NTK 类方法和 YaRN 有不同做法，字段格式也随模型和 Transformers 版本变化。应采用该 checkpoint 与引擎明确支持的方案，再评估长输入上的效果。修改配置后能启动，只完成了配置兼容性检查。

## 怎样选择长度预算

先估算任务会花多少上下文。把初始问题、每轮模型输出、工具观察和模板开销分别记录，再检查实际最大值与分布。对多轮任务，优先使用 Agent 实际支持的总预算、每轮剩余预算或截断策略控制增长。

随后核对训练路径。工具返回即便由 `response_mask=0` 排除出直接输出损失，仍可能参与上下文表示计算，占用激活显存。训练打包预算决定一次放多少序列，序列并行决定怎样跨卡处理，单条序列的长度仍需单独控制。

例如窗口 8000、实际历史 6000 时，生成预算应依据剩余空间分配。若工具一次返回 5000 token，下一轮应在提交请求前按策略结束、裁剪或摘要。采用哪种策略会影响模型获得的信息和任务完成率，需要在报告中交代。

## 延伸

- [工具结果怎样拼回上下文](psrl-agent-loop-harbor-tool-calling.md)：沿修代码的例子看多轮累计。
- [KV 缓存与推理显存](vllm-and-training-parallelism/inference-mechanisms.md)：推理阶段的内存账。
- [训练并行与激活](vllm-and-training-parallelism/training-parallelism.md)：训练阶段怎样分担序列计算。
- [vLLM 模型配置参考](https://docs.vllm.ai/en/stable/api/vllm/config/model/)：按实际安装版本核对窗口推导和校验分支。

可以继续问：“剩余上下文怎样转成每轮 max_tokens？”或“RoPE 的相对位置关系怎样从旋转公式推出来？”
