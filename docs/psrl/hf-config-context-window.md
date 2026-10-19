# HuggingFace `config.json` 里的上下文窗口字段，与一次 vLLM 启动崩溃

## 一句话结论

vLLM 启动前会校验 `max_model_len ≤ max_position_embeddings`；`max_position_embeddings` 是模型预训练时位置编码覆盖的最大 token 数，超过它且没有配置 `rope_scaling`（长上下文扩展）时，vLLM 会直接拒绝启动（而不是默默跑出质量下降的结果）——这是 Qwen2.5-3B 训练任务崩溃的直接原因：脚本要的 `max_model_len=67584` 超过了模型原生的 `32768`。

## 背景：一次真实的崩溃

在 `psrl` 里用 `fsdp_qwen2.5_3b_laps.sh` 跑 GRPO 训练时，8 个 vLLM rollout/validate server 全部在初始化阶段抛出：

```
ValueError: max_model_len (67584) should be less than or equal to max_position_embeddings (32768)
```

脚本里：

```bash
max_prompt_length=2048
max_response_length=${MAX_RESPONSE_LENGTH:-65536}
max_model_len=$(( max_prompt_length + max_response_length ))   # = 67584
```

这个 `65536` 的默认值是从另一个脚本（针对 Qwen3.5，原生上下文窗口大得多）直接复制过来的，没有按 Qwen2.5-3B 的实际上限校正。Qwen2.5-3B-Instruct 的 `config.json`：

```json
{
  "max_position_embeddings": 32768,
  "sliding_window": 32768,
  "use_sliding_window": false,
  "rope_theta": 1000000.0,
  ...
}
```

没有 `rope_scaling` 字段，意味着这个 checkpoint **没有配置任何超出 32768 的扩展方式**，vLLM 的前置校验（`_validate_configs`）就会直接拒绝启动，而不是让模型在没见过的位置区间上跑一个质量不可控的结果。8 个 server 同时失败 → 对应的 Ray actor 在 `__init__` 里死亡 → 级联抛出 `ActorDiedError` 打断整个训练 driver。

**修复思路**：让 `max_prompt_length + max_response_length ≤ max_position_embeddings`（这里应 ≤ 32768），或者显式给推理引擎配置 `rope_scaling` 做长上下文扩展（见下文），二者是完全不同的取舍。

## RoPE 与 `max_position_embeddings` 到底是什么关系

Qwen2 系列用 **RoPE（Rotary Position Embedding，旋转位置编码）** 给每个 token 注入位置信息：对 query/key 向量的每一对维度，按位置 $m$ 旋转一个角度 $m\theta_i$，其中

$$\theta_i = \text{rope\_theta}^{-2i/d}, \quad i = 0, 1, \dots, d/2-1$$

两个 token 的注意力分数只取决于它们的**相对位置**，这是 RoPE 相比可学习绝对位置编码的核心优势（可以外推）。但"可以外推"不等于"外推质量有保证"——`rope_theta` 决定了旋转的"节奏"：$\theta$ 越大，同样的位置差对应的旋转角度越小，相当于把可分辨的位置范围拉长。Qwen2.5 把 `rope_theta` 设到 `1,000,000`（远高于 Llama 类模型常见的 `10,000`），就是预训练阶段为了让模型原生支持到 32768 这个长度做的调整。

`max_position_embeddings` 记录的就是**预训练阶段实际训练过的最大位置数**——模型在这个范围内见过真实数据分布下的位置模式，超出这个范围，旋转角度虽然在数学上可以继续算，但模型从未在这些角度组合上训练过，attention 分布会失真（这就是所谓的"长度外推能力差"）。

## 追问："$\theta$ 越大，同样的位置差对应的旋转角度越小"到底是什么意思

这句话说的是 `rope_theta`（公式里的底数 $\theta$）和每个频率通道实际旋转速度 $\theta_i$ 之间的关系。回到公式：

$$\theta_i = \theta^{-2i/d}, \quad i = 0, 1, \dots, d/2-1$$

- 位置为 $m$ 的 token，在第 $i$ 个频率通道上被旋转的角度是 $m \cdot \theta_i$。
- 两个 token 的位置差 $\Delta m$，对应的**相对**旋转角度是 $\Delta m \cdot \theta_i$——这是 attention 分数里唯一保留下来、只依赖相对位置的量。

关键在于 $\theta_i$ 怎么随 $\theta$（`rope_theta`）变化：对 $i \ge 1$，指数 $-2i/d$ 是负数，而 $\theta > 1$，所以 $\theta_i = \theta^{-2i/d}$ 是"底数越大、值越小"的量。也就是 $\theta \uparrow \implies \theta_i \downarrow$。代回 $\Delta m \cdot \theta_i$：同样的 $\Delta m$，$\theta_i$ 变小了，算出来的旋转角度自然也变小。

直觉上可以把每个频率通道想成一根转速不同的"表盘指针"，$\theta_i$ 是它每走一步（位置 +1）转过的角度：

- $\theta_i$ 大 → 指针转得快 → 走不了几步就转满 $2\pi$，之后的角度和之前重复（"混叠"，模型分不清 $\Delta m$ 和 $\Delta m + 2\pi/\theta_i$）。
- 把 `rope_theta` 调大 → 所有通道（尤其低频、$i$ 大的通道）的 $\theta_i$ 整体变小 → 指针转得更慢 → 需要更大的位置差才会转满一圈。

这正是"相当于把可分辨的位置范围拉长"的含义：指针转得慢，意味着在预训练见过的位置范围内（比如 Qwen2.5 的 32768），各通道的相位组合还没开始重复，模型能靠这些独特的相位组合区分更远的位置差。这也是 Qwen2.5 把 `rope_theta` 设到 `1,000,000`（远高于 Llama 常见的 `10,000`）的原因——为了让原生上下文长度撑到 32768。

## 追问：这里有数学上的关联吗？为什么不能靠调大 `rope_theta` 无限延长上下文

**有数学关联**。每个通道 $i$ 转满一圈（$2\pi$）需要的位置步数（"周期"）是：

$$T_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \theta^{\,2i/d}$$

$\theta$（`rope_theta`）越大，$T_i$ 越大——从这个公式看，$\theta$ 可以取任意大，$T_i$ 就能任意长，**数学上没有硬上限**。真正限制"最长能撑多远"的是最低频通道（$i=d/2-1$，转得最慢），它最早开始重复/混叠。

但实际不能靠"调大 `rope_theta`"免费换更长上下文，原因不在公式本身，而在三个工程约束：

1. **改 $\theta$ 会改变所有通道、所有位置的旋转角度，不只是新增的那一段**。base 调大后，连负责短距离精细分辨的高频通道也被"拉慢"了——同一个 $\Delta m=1,2,3\dots$ 对应的旋转角度和模型预训练时学到的分布不一样，这是对模型已学权重的一次分布偏移，不重新训练/微调会连累短距离的分辨能力。这也是为什么 `rope_theta=1000000` 是 Qwen2.5 在**预训练阶段**和训练数据的长度分布、模型权重一起调好的结果，不是一个能事后随便改的独立开关。
2. **均匀拉长所有频率是有代价的**。YaRN / NTK-aware 等 `rope_scaling` 方法不会简单把 base 换大，而是**只**拉伸低频（长周期、负责远距离）维度，尽量不动高频维度——精细地"该拉的拉、不该拉的不拉"，比粗暴调大 `rope_theta` 效果更好，说明单纯调大 base 本身会牺牲局部分辨率。
3. **真正的硬限制常常来自训练数据和算力，不是 RoPE 公式**。base 再大、理论周期再长，模型要真正学会利用这么长的依赖关系，还是需要预训练语料里有对应长度的真实长文档，以及能撑住这个序列长度 attention 计算的算力/显存（参见 [vllm-and-training-parallelism.md](vllm-and-training-parallelism.md) 的显存账本）。`rope_theta` 只解决"位置编码数学上能不能表达这么长的相对位置"，不解决"模型有没有见过、算力够不够训"的问题。

一句话：$\theta$ 本身可以无限调大、数学上没有硬上限，但它是和权重、训练数据绑定好的预训练超参，事后单纯调大会让所有位置的旋转角度分布偏离模型学过的分布、损害短距离分辨率，也绕不开训练数据和算力的真实限制——这才是不能无限调长的原因，也是长上下文扩展要靠专门设计的 `rope_scaling` 而不是直接改 `rope_theta` 的原因。

## `rope_scaling`：没有重新预训练也能延长上下文

`rope_scaling` 是一个可选字段（这次的 Qwen2.5-3B-Instruct 没有配），格式类似：

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 32768
}
```

作用是让推理引擎在运行时**重新计算旋转频率**，把原本只在 `[0, 32768)` 训练过的位置模式"拉伸"或"插值"到更长的范围（比如 131072），常见方法：

| 方法 | 思路 |
|---|---|
| Linear scaling (PI) | 把位置索引整体除以一个因子，相当于把序列"压缩"进模型见过的范围 |
| NTK-aware / Dynamic NTK | 改变 `rope_theta` 本身而不是简单缩放位置，高频维度少改、低频维度多改，兼顾局部和全局 |
| YaRN | NTK 思路的改进版，额外做了温度校正，是目前 Qwen/DeepSeek 等长上下文模型的主流选择 |

这是**推理时的近似扩展**，不是重新训练，会有一定质量损失，扩展倍数越大损失通常越明显。Qwen 官方对部分 checkpoint 提供了经过验证的 `rope_scaling` 配置（用于 "Processing Long Texts" 场景），但不是每个 checkpoint 都默认开启——这次用的 Qwen2.5-3B-Instruct 就没有。如果要用这条路径解决上下文不够的问题，应该参考模型官方文档给的扩展倍数，而不是随便设一个值。

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
| `architectures` | HF `AutoModel` 用它决定实例化哪个 Python 类（这里是 `Qwen2ForCausalLM`），不影响权重本身 |
| `model_type` | `AutoConfig`/`AutoModel` 用来分发到对应建模代码的键；Qwen2.5 沿用的还是 `"qwen2"`，2.5 是训练数据/流程的升级，架构族没变 |
| `hidden_size` | 残差流（residual stream）的维度，也是 attention 输出、embedding 的维度 |
| `intermediate_size` | MLP（FFN）中间层维度，这里 `11008 / 2048 ≈ 5.37`，是标准的 SwiGLU 扩张比例 |
| `num_hidden_layers` | Transformer block 层数（深度） |
| `num_attention_heads` / `num_key_value_heads` | Query 头数 vs KV 头数。`16` vs `2` 说明是 **GQA（Grouped-Query Attention）**：每 8 个 query 头共享 1 组 K/V，KV cache 显存降到 MHA（`num_kv_heads=num_attention_heads`）的 1/8 |
| `max_position_embeddings` | 预训练时位置编码覆盖的最大 token 数，见上文 |
| `rope_theta` | RoPE 旋转频率的基数，越大旋转越"慢"，原生可分辨的位置范围越长，见上文公式 |
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

## 第二次运行：改了参数但没留余量，运行时持续报错、训练卡死

修好第一次崩溃后，把 `MAX_RESPONSE_LENGTH` 覆盖成了 `30720`，于是 `max_model_len = max_prompt_length(2048) + max_response_length(30720) = 32768`，正好卡在 Qwen2.5-3B 的 `max_position_embeddings` 上限，vLLM 启动前的校验通过了，看起来"修好了"。

但这是一个 multi-turn agent 任务（`gen_actor_rollout_ref.rollout.multi_turn.max_turns=50`），每一轮的历史对话和工具调用输出都会累积进下一轮请求的 prompt，实际 prompt 长度会随轮数增长，远超预留的 `max_prompt_length=2048`。于是运行时大量请求的 prompt 长度滚到 32800~33600+，持续触发：

```
litellm.BadRequestError: OpenAIException - The prompt (length 32851) is longer than the maximum model length of 32768.
```

现象：
- 不再是启动时崩溃，而是**运行时持续报错**，超长 episode 以 `reward=0.000` 失败收场
- 训练没有完全停滞：短一些的 episode（轮数少、还没滚到撞墙长度）仍能正常跑完，靠这部分样本训练依然缓慢推进，实测 ckpts 目录里留下了 `global_step_10/20/30`（大约每 5 小时一个 checkpoint）——但同一批次里越长的 episode 越容易在还没产出有效 reward 前就被 vLLM 拒绝，相当于一部分训练信号被系统性丢弃，样本效率明显低于预期
- 某些时间窗口 GPU 利用率会掉到 0%——被拒绝的长 episode 请求在 vLLM 做 prompt 长度校验时就直接 reject，没有进入生成阶段

**根因对比**：第一次是初始配置就超过模型上限（连起点都不对）；第二次是初始配置刚好卡在模型上限、**没有给 multi-turn 场景下会持续增长的 prompt 留任何余量**。两次本质是同一类问题的两种表现。正确修法应该是让 `max_prompt_length + max_response_length` 明显小于 `max_model_len`（留出多轮累积的安全余量），而不是让二者刚好相等。

## `max_prompt_length` / `max_response_length` / `max_model_len` 三者的关口原理

这三个参数管的是流程里三个不同的关口，不是同一种机制：

**`max_prompt_length`** 是**数据/预处理层**的关口，跟 vLLM 无关。框架读训练数据集时把每条样本的 prompt 用 tokenizer 编码，检查 token 数是否超过它；超过了怎么处理由 `data.truncation` 决定（`error` 就是直接抛异常，不做静默截断）。它检查的是**训练集里原始 prompt** 的长度，不管推理引擎实际收到的请求会不会比这个长。

**`max_response_length`** 是**推理引擎的硬停止条件**，会被转成 vLLM/HF 生成接口里的 `max_tokens` 采样参数。解码循环每生成一个 token 就计数，一旦达到这个数量引擎强制停止（`finish_reason="length"`），跟模型是否输出 EOS 无关——纯粹是计数器式硬切断。

**`max_model_len`** 是 vLLM 的**架构级天花板**，对应模型 KV cache 能开多长、RoPE 位置编码训练时覆盖到多远（即 `max_position_embeddings`）。vLLM 在每次请求进来时都会实时校验：

```
当前 prompt 的 token 数 + 这次请求要生成的 max_tokens ≤ max_model_len
```

不满足就直接拒绝，抛 `BadRequestError`。**这个校验用的是"这次请求实际的 prompt token 数"，不是配置文件里写的 `max_prompt_length` 这个理论值**——两者在 multi-turn 场景下会严重脱节，这正是下一节问题的根源。

### 为什么 `max_prompt_length + max_response_length` 要明显小于 `max_model_len` 才安全

原因是两层系统性膨胀，`max_prompt_length` 这个理论预算从设计上就没覆盖到：

1. **chat template / tokenizer 开销**：套上 system 消息、role 标签、工具 schema 声明之后再编码，token 数会比裸文本编码时多出一截，配置 `max_prompt_length` 时往往没精确算进去。
2. **multi-turn 累积增长（这次问题的真正主因）**：`max_turns=50` 的场景下，每一轮的模型输出和工具调用结果都会拼回对话历史，成为下一轮请求 prompt 的一部分。第 16 轮时 vLLM 收到的"prompt"早就不是数据集里那条 2048 token 的初始 prompt，而是"初始 prompt + 前 15 轮全部对话和工具输出"——日志里 `num_turns=16, response_tokens=30548` 就是这种累积效应的直接证据。

如果 `max_prompt_length + max_response_length` 刚好等于 `max_model_len`（第二次运行的 2048+30720=32768），留给这种累积增长的余量是零，只要对话进行到有实质性工具交互的几轮，实时 prompt 长度必然超过这个卡满的天花板。留出余量，本质是专门预留出来吸收"多轮累积"和"模板开销"这两种系统性膨胀的缓冲区。

**但留余量只是缓解，不是根治**——`max_turns=50` 下累积长度理论上无上限增长，余量再大也会被吃掉，只是往后拖延撞墙的时间。真正根治需要在流程层面加轮次级别的预算控制（比如给全局累积 token 数设硬预算，用满就提前结束 trajectory）或对早期轮次做截断/摘要，不能只靠留余量。

## 第三次运行：`max_model_len` 留了余量之后，又在训练侧（backward）撞了显存 OOM——`max_turns` 如何影响训练显存

把 `max_model_len` 从卡满的 `32768`（Qwen2.5-3B）换成有真实余量的 `131072`（Qwen3.5-4B，见上面 §背景）之后，context 溢出的问题确实消失了。但换了模型、重新跑 `fsdp_qwen35_4b_laps.sh`（`max_turns=50`）之后，训练在**第一个 step 还没跑完**时就在 `loss.backward()` 里报了 `torch.OutOfMemoryError`（GPU 0 的 95GiB 里训练进程已经占了 90GiB+，backward 再要 1.64GiB 就爆了），日志里同时能看到不少 `prompt length 131159/131706 > max_model_len 131072` 的拒绝——说明即便留了余量，`max_turns=50` 下仍然有 episode 滚到了接近 `max_model_len` 上限的长度。这两个现象根子上是同一件事：**`max_turns` 越大，训练要吃进去的单条序列越长，训练侧的显存开销跟着涨**。具体机制：

1. **`max_turns` 直接决定一条 trajectory 最终有多长**：每一轮 harbor 的 `Terminus2` 把模型输出（assistant 消息）和终端工具的真实输出（截断到 `max_observation_bytes=4096` 字节的 env 消息）都拼回对话历史，作为下一轮请求的 prompt（见 [psrl-agent-loop-harbor-tool-calling.md](psrl-agent-loop-harbor-tool-calling.md) §4）。轮数越多，这条对话累积得越长，直到碰到 `max_model_len`（硬顶）或模型自己判定 `task_complete`（软停）为止。`max_turns=50` 相当于把这个"能滚多长"的软上限设得很宽松。
2. **训练时是把整条序列一起过模型做 forward + backward，不是只算 assistant 部分**：`response_mask` 只决定哪些 token 的 loss 参与梯度（`response_mask=1` 是 assistant 输出，`=0` 是环境/工具输出），不代表被 mask 掉的 token 不占显存——因果语言模型的每一层都要对序列里**每一个位置**算 hidden state，所以一条 trajectory 的显存开销由它的**总 token 数**（prompt + 历史所有轮次 + 当前 assistant 输出）决定，跟"多少 token 真正参与梯度"无关。轮数越多 → 总 token 数越大 → 每一层的中间激活（activation）越大，这个开销是随序列长度近似线性叠加在每一层上的，层数、`hidden_size` 越大，同样的轮数增量对应的显存增量也越大。
3. **`use_dynamic_bsz` + `ppo_max_token_len_per_gpu`（这里是 `131072`）决定"一次塞多少 token 到一张卡"**：这是训练侧的打包（packing）策略，尽量让每张卡处理的 token 总数逼近但不超过这个预算。如果某条 trajectory 本身已经滚到十万+ token（比如日志里那两条被拒绝的 `131159`/`131706`，说明确实有 episode 滚到了接近上限），它几乎会独占这张卡在这个 micro-batch 里的全部 token 预算——`TRAIN_SP=4`（Ulysses 序列并行）会把这条超长序列在 4 张卡间切分，单卡分到 `L/4` 的量级，但 `L` 本身已经逼近 `131072`，切完仍然是一个不小的数字，叠加上其他常驻显存（模型参数分片、这次还叠了 `colocate_validate_and_train=True` 下同卡驻留的 vLLM 引擎），就更容易把这张卡的显存账本挤爆。
4. **结论**：这次 OOM 发生在训练还没跑完第一个 step 就爆了，时间点和"至少有一条 episode 滚到了接近 `max_model_len` 上限"的证据（那两条 `BadRequestError`）吻合，说明大概率是某条/某几条贴着上限的超长 trajectory 被打包进了第一个 mini-batch，训练侧的 backward 显存需求因此被这少数几条极端样本拉爆。`max_turns` 从 `50` 降下来，直接压低"能滚多长"的软上限，减少出现这类贴着 `max_model_len` 的超长 trajectory 的概率和上限，是从根子上压低训练侧峰值显存需求的直接手段——跟 `gpu_memory_utilization`（管的是 vLLM 推理侧的显存预留）、`ppo_max_token_len_per_gpu`（管的是打包预算，调小会让长序列被拆得更细，但拆不动单条序列本身的长度）是两个不同层面的杠杆，`max_turns` 直接砍的是问题的源头（序列能有多长），后两者是在下游给一个已经很长的序列"腾地方"。

代价：`max_turns` 调低意味着需要很多轮才能定位/修复的复杂任务会在还没解决前被强制截断（提前进入 `overlong_filtering` 逻辑），可能牺牲一部分本来能拿到正向 reward 的长 episode。这是显存安全和任务上限之间的直接取舍，需要跑起来看实际的 episode 轮数分布（大多数 episode 到底几轮就 `task_complete`）来判断降到多少合适，不是一个能靠公式算出唯一答案的参数。

## 延伸

- 关于 vLLM 的 PagedAttention / KV cache 显存账本，见 [vllm-and-training-parallelism.md](vllm-and-training-parallelism.md)
- 关于这次崩溃所在的完整训练流程，见 `psrl/examples/sciaccel_rl/fsdp_qwen2.5_3b_laps.sh`
- Harbor 怎么把每轮工具调用结果拼回 prompt、TITO 怎么记录 token，见 [psrl-agent-loop-harbor-tool-calling.md](psrl-agent-loop-harbor-tool-calling.md)
