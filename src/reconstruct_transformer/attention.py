import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Compute ``softmax(Q @ K^T / sqrt(d_k)) @ V`` directly.

    Input shapes are ``(..., query_len, d_k)``, ``(..., key_len, d_k)``, and
    ``(..., key_len, d_v)``. A boolean mask must broadcast to
    ``(..., query_len, key_len)``; ``True`` permits attention. The returned
    output and weights have shapes ``(..., query_len, d_v)`` and
    ``(..., query_len, key_len)``. Fully masked rows produce zero weights and
    zero output. Returned weights include the optional dropout operation.
    '''
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError('query, key, and value must each have at least 2 dimensions')
    if query.size(-1) != key.size(-1):
        raise ValueError('query and key must have the same final dimension d_k')
    if key.size(-2) != value.size(-2):
        raise ValueError('key and value must have the same sequence length')
    d_k = query.size(-1)
    if d_k <= 0:
        raise ValueError('query and key dimension d_k must be greater than zero')
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError('mask must have dtype torch.bool')
    if dropout is not None and not isinstance(dropout, torch.nn.Module):
        raise TypeError('dropout must be a torch.nn.Module or None')

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    allowed: torch.Tensor | None = None
    if mask is not None:
        try:
            allowed = torch.broadcast_to(mask.to(scores.device), scores.shape)
        except RuntimeError as error:
            raise ValueError('mask cannot broadcast to attention scores') from error
        fully_masked = ~allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, float('-inf'))
        scores = scores.masked_fill(fully_masked, 0.0)

    attention_weights = torch.softmax(scores, dim=-1)
    if allowed is not None:
        attention_weights = attention_weights.masked_fill(~allowed, 0.0)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = attention_weights @ value
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, query_len, _ = q.shape
        key_len = k.size(1)
        value_len = v.size(1)
        if key_len != value_len:
            raise ValueError("key and value must have the same sequence length")

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch_size, query_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, value_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, 0.0)
        attention_weights = self.attention_dropout(attention_weights)

        context = attention_weights @ v
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        return self.w_combine(context)


# Backward-compatible alias for the original misspelling.
MutiHeadAttention = MultiHeadAttention
