import math

import torch
import torch.nn as nn


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
