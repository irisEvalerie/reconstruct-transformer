# 设计说明

实现细节以代码内 docstring 为准：每个模块的 docstring 都描述了输入/输出形状、
核心公式与实现决策。主 `README.md` 的「核心公式」一节汇总了关键数学。

| 主题 | 源码 |
|---|---|
| Scaled 与 Multi-Head Attention | `src/reconstruct_transformer/attention.py` |
| Embedding 与位置编码 | `src/reconstruct_transformer/embedding.py` |
| Layer Normalization | `src/reconstruct_transformer/normalization.py` |
| Feed-Forward Network | `src/reconstruct_transformer/feed_forward.py` |
| Mask | `src/reconstruct_transformer/masks.py` |
| Encoder / Decoder / 完整模型 | `encoder.py`、`decoder.py`、`transformer.py` |
| 训练、解码、评估 | `src/reconstruct_transformer/training.py` |

对应的单元与集成测试见 `tests/`。
