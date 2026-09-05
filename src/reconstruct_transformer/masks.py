"""Padding and causal attention masks."""

import torch


def make_pad_mask(query, key, key_pad_idx):
    """Return a key-padding mask shaped for multi-head attention.

    Query padding is deliberately not masked here. Padding outputs should be
    ignored by the loss; masking every key in a query row makes softmax
    ill-defined.
    """

    query_len = query.size(1)
    return key.ne(key_pad_idx).unsqueeze(1).unsqueeze(2).expand(
        -1, 1, query_len, -1
    )


def make_causal_mask(query_len, key_len=None, device=None):
    """Return a lower-triangular boolean attention mask."""

    key_len = query_len if key_len is None else key_len
    return torch.ones(query_len, key_len, dtype=torch.bool, device=device).tril()
