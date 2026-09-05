'''Token embeddings and sinusoidal positional encodings.'''

from __future__ import annotations

import math

import torch
from torch import nn


class TokenEmbedding(nn.Embedding):
    '''Map token IDs to vectors and scale them by ``sqrt(d_model)``.

    Input shape is ``(batch, seq_len)`` and output shape is
    ``(batch, seq_len, d_model)``.
    '''

    def __init__(
        self, vocab_size: int, d_model: int, padding_idx: int | None = None
    ) -> None:
        if vocab_size <= 0:
            raise ValueError('vocab_size must be greater than zero')
        if d_model <= 0:
            raise ValueError('d_model must be greater than zero')
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''Embed a rank-2 token tensor and apply the Transformer scale.'''
        if tokens.ndim != 2:
            raise ValueError('tokens must have shape (batch, seq_len)')
        return super().forward(tokens) * self.scale


class PositionalEmbedding(nn.Module):
    '''Create the fixed sinusoidal positional encoding from the paper.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`` and
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))``. The encoding is a
    registered buffer and therefore follows ``module.to(device)``. Given a
    token-like input of shape ``(batch, seq_len)``, the returned shape is
    ``(1, seq_len, d_model)`` for broadcasting across the batch.
    '''

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError('d_model must be greater than zero')
        if max_len <= 0:
            raise ValueError('max_len must be greater than zero')

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
        frequency = torch.exp(even_dimensions * (-math.log(10000.0) / d_model))

        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        odd_count = encoding[:, 1::2].size(1)
        encoding[:, 1::2] = torch.cos(position * frequency[:odd_count])
        self.register_buffer('encoding', encoding)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''Return positions for a rank-2 token tensor.'''
        if tokens.ndim != 2:
            raise ValueError('tokens must have shape (batch, seq_len)')
        seq_len = tokens.size(1)
        if seq_len > self.encoding.size(0):
            raise ValueError(
                f'sequence length {seq_len} exceeds configured '
                f'max_len {self.encoding.size(0)}'
            )
        return self.encoding[:seq_len].unsqueeze(0)


class TransformerEmbedding(nn.Module):
    '''Combine scaled token and positional embeddings, then apply dropout.

    Input shape is ``(batch, seq_len)`` and output shape is
    ``(batch, seq_len, d_model)``. Dropout is active only in training mode.
    The compatibility-only ``device`` argument is ignored; call
    ``module.to(device)`` to move parameters and the positional buffer
    together.
    '''

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        drop_prob: float = 0.1,
        device: torch.device | str | None = None,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        del device
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError('drop_prob must be between 0 and 1')
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = PositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''Embed tokens, add positions, and apply dropout.'''
        token_embedding = self.tok_emb(tokens)
        positional_embedding = self.pos_emb(tokens)
        return self.dropout(token_embedding + positional_embedding)
