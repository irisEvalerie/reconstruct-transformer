"""Unit tests for boolean Transformer attention masks."""

from __future__ import annotations

import pytest
import torch

from reconstruct_transformer.masks import (
    make_causal_mask,
    make_cross_attention_mask,
    make_decoder_self_attention_mask,
    make_pad_mask,
)


def test_padding_mask_values_shape_and_dtype() -> None:
    query = torch.tensor([[4, 0, 5], [6, 7, 0]])
    key = torch.tensor([[1, 2, 0, 0], [3, 0, 4, 5]])

    mask = make_pad_mask(query, key, key_pad_idx=0)

    expected_keys = torch.tensor(
        [[True, True, False, False], [True, False, True, True]]
    )
    expected = expected_keys[:, None, None, :].expand(-1, 1, 3, -1)
    assert mask.shape == (2, 1, 3, 4)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_query_padding_rows_are_not_fully_hidden() -> None:
    query = torch.tensor([[5, 0, 6]])
    key = torch.tensor([[1, 2, 0]])

    mask = make_pad_mask(query, key, key_pad_idx=0)

    expected_key_pattern = torch.tensor([True, True, False])
    torch.testing.assert_close(mask[0, 0, 1], expected_key_pattern)
    assert mask[0, 0, 1].any()


def test_causal_mask_exact_values_and_shape() -> None:
    mask = make_causal_mask(query_len=4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )[None, None, :, :]

    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_rectangular_causal_mask() -> None:
    mask = make_causal_mask(query_len=3, key_len=5)
    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ]
    )[None, None, :, :]

    assert torch.equal(mask, expected)


def test_decoder_mask_combines_padding_and_causality() -> None:
    target = torch.tensor([[1, 2, 0, 3]])

    mask = make_decoder_self_attention_mask(target, target_pad_idx=0)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, True],
        ]
    )[None, None, :, :]

    assert mask.shape == (1, 1, 4, 4)
    assert torch.equal(mask, expected)


def test_cross_attention_masks_source_keys_with_different_lengths() -> None:
    target = torch.tensor([[1, 2, 3], [4, 5, 0]])
    source = torch.tensor([[7, 8, 0, 0, 0], [9, 10, 11, 0, 0]])

    mask = make_cross_attention_mask(target, source, source_pad_idx=0)

    assert mask.shape == (2, 1, 3, 5)
    assert torch.equal(mask[0, 0, 0], torch.tensor([1, 1, 0, 0, 0]).bool())
    assert torch.equal(mask[1, 0, 2], torch.tensor([1, 1, 1, 0, 0]).bool())


def test_no_padding_allows_every_key() -> None:
    query = torch.tensor([[1, 2], [3, 4]])
    key = torch.tensor([[5, 6, 7], [8, 9, 10]])

    mask = make_pad_mask(query, key, key_pad_idx=0)

    assert mask.all()


def test_all_padding_keys_produce_fully_false_rows() -> None:
    query = torch.tensor([[1, 2]])
    key = torch.tensor([[0, 0, 0]])

    mask = make_pad_mask(query, key, key_pad_idx=0)

    assert mask.shape == (1, 1, 2, 3)
    assert not mask.any()


def test_masks_broadcast_over_multiple_attention_heads() -> None:
    query = torch.ones(2, 3, dtype=torch.long)
    key = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    mask = make_pad_mask(query, key, key_pad_idx=0)

    broadcast = torch.broadcast_to(mask, (2, 6, 3, 4))

    assert broadcast.shape == (2, 6, 3, 4)


def test_all_mask_constructors_return_boolean_tensors() -> None:
    source = torch.tensor([[1, 2, 0]])
    target = torch.tensor([[1, 0]])
    masks = [
        make_pad_mask(target, source, 0),
        make_causal_mask(2),
        make_decoder_self_attention_mask(target, 0),
        make_cross_attention_mask(target, source, 0),
    ]

    assert all(mask.dtype == torch.bool for mask in masks)


def test_masks_are_created_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    source = torch.tensor([[1, 2, 0]], device="cuda")
    target = torch.tensor([[1, 0]], device="cuda")

    masks = [
        make_pad_mask(target, source, 0),
        make_causal_mask(2, device="cuda"),
        make_decoder_self_attention_mask(target, 0),
        make_cross_attention_mask(target, source, 0),
    ]

    assert all(mask.is_cuda for mask in masks)
    assert all(mask.dtype == torch.bool for mask in masks)


@pytest.mark.parametrize(
    ("query", "key"),
    [
        (torch.ones(3), torch.ones(1, 3)),
        (torch.ones(1, 3), torch.ones(1, 3, 1)),
    ],
)
def test_rejects_non_matrix_token_inputs(
    query: torch.Tensor, key: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="batch, seq_len"):
        make_pad_mask(query, key, 0)


def test_rejects_different_batch_sizes() -> None:
    with pytest.raises(ValueError, match="same batch size"):
        make_pad_mask(torch.ones(2, 3), torch.ones(3, 4), 0)


@pytest.mark.parametrize(
    ("query", "key"),
    [
        (torch.empty(2, 0), torch.ones(2, 3)),
        (torch.ones(2, 3), torch.empty(2, 0)),
    ],
)
def test_rejects_empty_token_sequences(
    query: torch.Tensor, key: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="sequence lengths"):
        make_pad_mask(query, key, 0)


@pytest.mark.parametrize(
    ("query_len", "key_len", "message"),
    [(0, None, "query_len"), (-1, None, "query_len"), (2, 0, "key_len")],
)
def test_rejects_invalid_causal_lengths(
    query_len: int, key_len: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_causal_mask(query_len, key_len)
