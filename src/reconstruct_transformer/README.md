# 模块边界

- `embedding.py`：词嵌入和正弦位置编码
- `attention.py`：缩放点积注意力和多头注意力
- `normalization.py`：手写 LayerNorm
- `feed_forward.py`：逐位置前馈网络
- `masks.py`：padding mask 和 causal mask
- `encoder.py`：编码器层及其堆叠
- `decoder.py`：解码器层及其堆叠
- `transformer.py`：顶层 Transformer，组装 mask 并调用编解码器

模块实现时统一约定张量布局为 `(batch, sequence, d_model)`。
