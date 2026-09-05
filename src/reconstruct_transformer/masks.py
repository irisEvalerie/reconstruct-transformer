'''Boolean padding and causal attention masks.'''

from __future__ import annotations

import torch


def _validate_token_matrices(query: torch.Tensor, key: torch.Tensor) -> None:
    if query.ndim != 2 or key.ndim != 2:
        raise ValueError('query and key tokens must have shape (batch, seq_len)')
    if query.size(0) != key.size(0):
        raise ValueError('query and key tokens must have the same batch size')
    if query.size(1) <= 0 or key.size(1) <= 0:
        raise ValueError('query and key sequence lengths must be greater than zero')


def make_pad_mask(
    query: torch.Tensor, key: torch.Tensor, key_pad_idx: int
) -> torch.Tensor:
    '''Return a key-padding mask shaped ``(batch, 1, query_len, key_len)``.

    ``True`` permits attention and ``False`` blocks a padded key. Query padding
    rows are not masked: masking an entire query row would make the softmax
    ill-defined. Outputs for padded query tokens must instead
    be excluded from the training objective with the loss ``ignore_index``.

    If every key in an example is padding, the returned rows are all ``False``;
    the attention implementation defines fully masked rows to produce zeros.
    '''
    _validate_token_matrices(query, key)
    query_len = query.size(1)
    key_is_valid = key.ne(key_pad_idx)
    return key_is_valid[:, None, None, :].expand(-1, 1, query_len, -1)


def make_causal_mask(
    query_len: int,
    key_len: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    '''Return a lower-triangular mask shaped ``(1, 1, query_len, key_len)``.

    A query at position ``i`` may attend only to keys at positions ``j <= i``.
    For decoder self-attention, ``query_len`` and ``key_len`` are equal.
    '''
    if query_len <= 0:
        raise ValueError('query_len must be greater than zero')
    key_len = query_len if key_len is None else key_len
    if key_len <= 0:
        raise ValueError('key_len must be greater than zero')

    mask = torch.ones(query_len, key_len, dtype=torch.bool, device=device).tril()
    return mask[None, None, :, :]


def make_decoder_self_attention_mask(
    target: torch.Tensor, target_pad_idx: int
) -> torch.Tensor:
    '''Combine target padding and causality into a decoder self-attention mask.'''
    padding_mask = make_pad_mask(target, target, target_pad_idx)
    causal_mask = make_causal_mask(target.size(1), device=target.device)
    return padding_mask & causal_mask


def make_cross_attention_mask(
    target: torch.Tensor, source: torch.Tensor, source_pad_idx: int
) -> torch.Tensor:
    '''Return a mask that blocks padded source keys during cross-attention.'''
    return make_pad_mask(target, source, source_pad_idx)
