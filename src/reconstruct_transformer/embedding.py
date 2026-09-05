import math

import torch
from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens):
        return super().forward(tokens) * self.scale


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device=None):
        super().__init__()
        encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(max_len, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.encoding.size(0):
            raise ValueError("sequence length exceeds configured max_len")
        return self.encoding[:seq_len].unsqueeze(0)


class TransformerEmbedding(nn.Module):
    def __init__(
        self, vocab_size, d_model, max_len, drop_prob, device=None, padding_idx=None
    ):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)
