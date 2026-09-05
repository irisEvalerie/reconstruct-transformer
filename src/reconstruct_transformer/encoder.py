'''Transformer encoder stack with post-norm residual sublayers.'''

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding
from .feed_forward import PositionwiseFeedForward
from .normalization import LayerNorm


class EncoderLayer(nn.Module):
    '''One post-norm encoder sublayer pair.

    Applies ``x = LayerNorm(x + Dropout(SelfAttention(x)))`` followed by
    ``x = LayerNorm(x + Dropout(FFN(x)))``, exactly as described in the paper.
    The layer implements no masking of its own; a source padding mask is
    forwarded to self-attention unchanged. Inputs and outputs share shape
    ``(batch, seq_len, d_model)``.
    '''

    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        return x


class Encoder(nn.Module):
    '''Token/positional embeddings followed by ``num_layers`` encoder layers.

    ``forward`` embeds token ids and applies every layer in order, passing the
    same source padding mask to each one. Input tokens have shape
    ``(batch, seq_len)`` and the output has shape ``(batch, seq_len, d_model)``.
    Move the module with ``module.to(device)``; no device argument is accepted
    at construction time.
    '''

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError('num_layers must be greater than zero')

        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len, dropout, padding_idx=padding_idx
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, tokens: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
