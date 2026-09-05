'''Transformer decoder stack with masked self-attention and cross-attention.'''

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding
from .feed_forward import PositionwiseFeedForward
from .normalization import LayerNorm


class DecoderLayer(nn.Module):
    '''One post-norm decoder sublayer triple.

    Applies, in order, masked self-attention, encoder-decoder cross-attention,
    and the position-wise feed-forward network, each wrapped as
    ``x = LayerNorm(x + Dropout(Sublayer(x)))``. The target mask is forwarded
    only to self-attention and the source mask only to cross-attention. Inputs
    have shape ``(batch, target_len, d_model)`` and ``memory`` has shape
    ``(batch, memory_len, d_model)``.
    '''

    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attention(x, x, x, target_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.cross_attention(x, memory, memory, source_mask)
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + residual)

        return x


class Decoder(nn.Module):
    '''Decoder stack: embeddings, ``num_layers`` layers, and output projection.

    ``forward`` embeds the target tokens, applies every layer with the target
    mask for self-attention and the source mask for cross-attention, then maps
    the final ``d_model`` representations to ``vocab_size`` logits through the
    output projection. Input tokens have shape ``(batch, target_len)``, memory
    has shape ``(batch, memory_len, d_model)``, and logits have shape
    ``(batch, target_len, vocab_size)``.

    The output projection lives here (not in the top-level model) so ``Decoder``
    alone yields logits and can be tested in isolation. Move the module with
    ``module.to(device)``; no device argument is accepted at construction time.
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
                DecoderLayer(d_model, ffn_hidden, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, memory, target_mask, source_mask)
        return self.output_projection(x)
