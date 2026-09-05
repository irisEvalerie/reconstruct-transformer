# Reconstruct Transformer

从最基础的张量运算出发，复现 [Vaswani et al. (2017) *Attention Is All You
Need*](https://arxiv.org/abs/1706.03762) 中的 Encoder–Decoder Transformer。

全程**不使用** `torch.nn.Transformer`、`torch.nn.MultiheadAttention`、
`torch.nn.functional.scaled_dot_product_attention`、Hugging Face `transformers`
或 `fairseq`。每个组件都由手写实现，并配有独立的单元测试与集成测试；完整模型
在合成的 copy task 上完成了端到端训练。

## 实现内容

| 组件 | 源码 |
|---|---|
| Scaled Dot-Product Attention | `src/reconstruct_transformer/attention.py` |
| Multi-Head Attention | `src/reconstruct_transformer/attention.py` |
| Token Embedding 与正弦位置编码 | `src/reconstruct_transformer/embedding.py` |
| 手写 Layer Normalization | `src/reconstruct_transformer/normalization.py` |
| Position-wise Feed-Forward Network | `src/reconstruct_transformer/feed_forward.py` |
| Padding / Causal / Cross-attention Mask | `src/reconstruct_transformer/masks.py` |
| Encoder 与 Decoder（post-norm） | `encoder.py`、`decoder.py` |
| 完整 Encoder–Decoder Transformer | `src/reconstruct_transformer/transformer.py` |
| 合成 copy task 数据管线 | `src/reconstruct_transformer/data.py` |
| 训练、Noam 调度、解码、checkpoint | `src/reconstruct_transformer/training.py` |

## 核心公式

Scaled Dot-Product Attention：

$$ \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V $$

Multi-Head Attention：

$$ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O,
\quad \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) $$

正弦位置编码：

$$ PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

Position-wise Feed-Forward Network：

$$ \text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2 $$

Layer Normalization（沿最后一维，含可学习参数 $\gamma$、$\beta$）：

$$ \text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$

## 安装

要求 Python 3.12。在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

如需 GPU 训练，请先从
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装匹配驱动的
CUDA wheel，再执行 editable install。

## 测试

```bash
python -m pytest           # 全部单元 + 集成测试
python -m ruff check .     # 代码检查
```

CUDA 相关测试在没有 GPU 时会自动 skip，因此整套测试在纯 CPU 环境和 CI 中都能运行。

## Copy Task

模型被训练来「复制」一段随机 token 序列：Encoder 读取源序列，Decoder 以
`[BOS] + source` 作为 teacher-forcing 输入，预测 `source + [EOS]`。

```bash
python scripts/train_copy.py                 # 完整训练（读取 configs/copy_task.yaml）
python scripts/train_copy.py --quick         # 极小规模冒烟
python scripts/train_copy.py --eval-only --checkpoint outputs/copy_task/best.pt --demo
```

结果（NVIDIA GeForce RTX 5060 Laptop GPU，约 60 秒）：

| 指标 | 数值 |
|---|---|
| 最终训练 loss | 0.042 |
| 最终验证 loss | 0.074 |
| 最终验证准确率 | 97.6% |
| 最佳验证准确率 | 98.7%（第 54 个 epoch）|
| Greedy token 准确率 | 96.6% |
| Greedy 完全匹配准确率 | 92.2% |

配置：词表大小 11，序列长度 2–8，训练集 2048 / 验证集 256 条，`d_model=64`、
4 头、`ffn_hidden=128`、2 层、dropout 0.0，Adam + Noam warmup 调度（400 步
warmup，60 个 epoch）。

## 项目结构

```text
reconstructTransformer/
├─ configs/                       # 模型与训练的 YAML 配置
├─ data/{raw,processed}/           # 本地数据（大文件不提交）
├─ docs/lessons/                  # 推导与设计说明
├─ notebooks/                     # 待补充的可视化
├─ scripts/                       # 环境、训练、评估入口
├─ src/reconstruct_transformer/   # 模型与训练源码
├─ tests/{unit,integration}/      # 组件与端到端测试
├─ checkpoints/                   # 本地权重（不提交）
└─ outputs/                       # 本地实验产物（不提交）
```

## 已知限制

- 位置编码的 `max_len` 固定，超长输入会报错。
- 解码仅实现了 greedy（无 beam search）。
- 目前唯一的训练实验是合成 copy task。

## Roadmap

- [ ] 在小型公开数据集上做机器翻译实验
- [ ] Beam search 与长度惩罚解码
- [ ] Attention 热力图与位置编码可视化
- [ ] BLEU 评估

## 致谢与说明

初始代码骨架是跟随 B 站「手搓 Transformer」教程编写的；之后每个组件都经过
独立推导、重写，并用数值、梯度与形状测试逐一验证，训练与解码流程为本项目
自行实现。本仓库不会把教程内容包装成原创。

## License

[MIT](LICENSE)
