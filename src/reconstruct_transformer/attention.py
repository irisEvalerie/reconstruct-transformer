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
    '''Multi-head scaled dot-product attention built on the verified core.

    Projects query, key, and value into ``num_heads`` independent subspaces,
    runs :func:`scaled_dot_product_attention` once across every head, then
    concatenates the heads and applies the output projection. No softmax or
    masking lives here — that's handled by ``scaled_dot_product_attention``.

    Inputs and outputs use ``(batch, seq_len, d_model)``. ``d_model`` must be
    divisible by ``num_heads``, giving each head dimension
    ``head_dim = d_model // num_heads``. Setting ``need_weights=True`` returns
    ``(output, attention_weights)`` where weights have shape
    ``(batch, num_heads, query_len, key_len)``.
    '''

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError('d_model must be greater than zero')
        if num_heads <= 0:
            raise ValueError('num_heads must be greater than zero')
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')
        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be between 0 and 1')

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Attend with ``num_heads`` heads over the projected inputs.'''
        self._validate_inputs(query, key, value)
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)
        value_len = value.size(1)

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, value_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attention_dropout
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        output = self.w_o(context)

        if need_weights:
            return output, weights
        return output

    def _validate_inputs(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                'query, key, and value must have shape (batch, seq_len, d_model)'
            )
        if query.size(0) != key.size(0) or query.size(0) != value.size(0):
            raise ValueError('query, key, and value must have the same batch size')
        if (
            query.size(-1) != self.d_model
            or key.size(-1) != self.d_model
            or value.size(-1) != self.d_model
        ):
            raise ValueError(
                'query, key, and value must have final dimension d_model'
            )
        if key.size(1) != value.size(1):
            raise ValueError('key and value must have the same sequence length')
