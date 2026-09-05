"""Unit tests for scaled dot-product attention."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from reconstruct_transformer.attention import scaled_dot_product_attention


def test_output_and_weight_shapes() -> None:
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 6, 8)
    value = torch.randn(2, 6, 10)

    output, weights = scaled_dot_product_attention(query, key, value)

    assert output.shape == (2, 4, 10)
    assert weights.shape == (2, 4, 6)


def test_matches_independent_manual_calculation() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    output, weights = scaled_dot_product_attention(query, key, value)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    expected_weights = torch.softmax(scores, dim=-1)
    expected_output = expected_weights @ value

    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


def test_uses_square_root_scaling() -> None:
    query = torch.tensor([[[2.0, 0.0]]])
    key = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    value = torch.eye(2).unsqueeze(0)

    _, weights = scaled_dot_product_attention(query, key, value)
    dot_products = query @ key.transpose(-2, -1)
    expected = torch.softmax(dot_products / math.sqrt(2), dim=-1)

    torch.testing.assert_close(weights, expected)
    assert not torch.allclose(weights, torch.softmax(dot_products, dim=-1))


def test_unmasked_weights_sum_to_one() -> None:
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 6, 8)
    value = torch.randn(2, 6, 5)

    _, weights = scaled_dot_product_attention(query, key, value)

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4))


def test_boolean_mask_blocks_and_renormalizes() -> None:
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 3, 4)
    value = torch.randn(1, 3, 5)
    mask = torch.tensor([[[True, False, True], [False, True, True]]])

    _, weights = scaled_dot_product_attention(query, key, value, mask)

    assert torch.equal(weights.masked_select(~mask), torch.zeros(2))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))


def test_mask_broadcasts_over_batch() -> None:
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 6, 8)
    value = torch.randn(2, 6, 3)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[:, -1] = False

    _, weights = scaled_dot_product_attention(query, key, value, mask)

    assert torch.equal(weights[..., -1], torch.zeros(2, 4))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4))


def test_arbitrary_leading_dimensions() -> None:
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 6, 10)

    output, weights = scaled_dot_product_attention(query, key, value)

    assert output.shape == (2, 3, 4, 10)
    assert weights.shape == (2, 3, 4, 6)


def test_fully_masked_row_is_finite_and_zero() -> None:
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 3, 4)
    value = torch.randn(1, 3, 5)
    mask = torch.tensor([[[False, False, False], [True, False, True]]])

    output, weights = scaled_dot_product_attention(query, key, value, mask)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.equal(weights[:, 0], torch.zeros(1, 3))
    assert torch.equal(output[:, 0], torch.zeros(1, 5))


def test_gradients_exist_and_are_finite() -> None:
    query = torch.randn(2, 4, 8, requires_grad=True)
    key = torch.randn(2, 6, 8, requires_grad=True)
    value = torch.randn(2, 6, 10, requires_grad=True)

    output, _ = scaled_dot_product_attention(query, key, value)
    output.sum().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    query = torch.randn(2, 4, 8, device="cuda")
    key = torch.randn(2, 6, 8, device="cuda")
    value = torch.randn(2, 6, 10, device="cuda")
    mask = torch.ones(4, 6, dtype=torch.bool)

    output, weights = scaled_dot_product_attention(query, key, value, mask)

    assert output.is_cuda
    assert weights.is_cuda
    assert torch.isfinite(output).all()


def test_rejects_mismatched_key_dimensions() -> None:
    with pytest.raises(ValueError, match="same final dimension"):
        scaled_dot_product_attention(
            torch.randn(2, 4, 7), torch.randn(2, 6, 8), torch.randn(2, 6, 10)
        )


def test_rejects_mismatched_key_value_lengths() -> None:
    with pytest.raises(ValueError, match="same sequence length"):
        scaled_dot_product_attention(
            torch.randn(2, 4, 8), torch.randn(2, 6, 8), torch.randn(2, 5, 10)
        )


def test_rejects_non_boolean_mask() -> None:
    with pytest.raises(TypeError, match="torch.bool"):
        scaled_dot_product_attention(
            torch.randn(2, 4, 8),
            torch.randn(2, 6, 8),
            torch.randn(2, 6, 10),
            torch.ones(4, 6),
        )


@pytest.mark.parametrize("query", [torch.randn(()), torch.randn(3)])
def test_rejects_low_rank_inputs(query: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        scaled_dot_product_attention(query, torch.randn(2, 3), torch.randn(2, 4))


def test_rejects_zero_key_dimension() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        scaled_dot_product_attention(
            torch.empty(2, 4, 0), torch.empty(2, 6, 0), torch.randn(2, 6, 10)
        )


def test_rejects_non_broadcastable_mask() -> None:
    with pytest.raises(ValueError, match="cannot broadcast"):
        scaled_dot_product_attention(
            torch.randn(2, 4, 8),
            torch.randn(2, 6, 8),
            torch.randn(2, 6, 10),
            torch.ones(3, 5, dtype=torch.bool),
        )


def test_dropout_respects_train_and_eval_modes() -> None:
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 3, 4)
    value = torch.randn(1, 3, 5)
    base_output, base_weights = scaled_dot_product_attention(query, key, value)
    dropout = nn.Dropout(p=1.0)

    dropout.eval()
    eval_output, eval_weights = scaled_dot_product_attention(
        query, key, value, dropout=dropout
    )
    torch.testing.assert_close(eval_weights, base_weights)
    torch.testing.assert_close(eval_output, base_output)

    dropout.train()
    train_output, train_weights = scaled_dot_product_attention(
        query, key, value, dropout=dropout
    )
    assert torch.equal(train_weights, torch.zeros_like(train_weights))
    assert torch.equal(train_output, torch.zeros_like(train_output))


def test_rejects_non_module_dropout() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        scaled_dot_product_attention(
            torch.randn(1, 2, 4),
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 5),
            dropout=0.1,  # type: ignore[arg-type]
        )
