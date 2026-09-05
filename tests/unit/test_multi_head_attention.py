"""Unit tests for multi-head attention built on the verified attention core."""

from __future__ import annotations

import inspect

import pytest
import torch

import reconstruct_transformer.attention as attention_module
from reconstruct_transformer.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)


def test_projection_parameter_shapes_and_count() -> None:
    mha = MultiHeadAttention(d_model=8, num_heads=2)

    for projection in (mha.w_q, mha.w_k, mha.w_v, mha.w_o):
        assert projection.weight.shape == (8, 8)
        assert projection.bias.shape == (8,)

    total_params = sum(p.numel() for p in mha.parameters())
    assert total_params == 4 * (8 * 8 + 8)


def test_self_attention_output_shape() -> None:
    mha = MultiHeadAttention(d_model=6, num_heads=3).eval()
    tokens = torch.randn(2, 4, 6)

    output = mha(tokens, tokens, tokens)

    assert output.shape == (2, 4, 6)


def test_cross_attention_with_different_lengths() -> None:
    mha = MultiHeadAttention(d_model=6, num_heads=3).eval()
    query = torch.randn(2, 3, 6)
    key = torch.randn(2, 7, 6)
    value = torch.randn(2, 7, 6)

    output = mha(query, key, value)

    assert output.shape == (2, 3, 6)


def test_need_weights_returns_head_weight_shape() -> None:
    mha = MultiHeadAttention(d_model=6, num_heads=3, dropout=0.0).eval()
    query = torch.randn(2, 3, 6)
    key = torch.randn(2, 5, 6)
    value = torch.randn(2, 5, 6)

    output, weights = mha(query, key, value, need_weights=True)

    assert output.shape == (2, 3, 6)
    assert weights.shape == (2, 3, 3, 5)


def test_matches_core_attention_with_identity_projections() -> None:
    torch.manual_seed(0)
    d_model, num_heads = 4, 2
    head_dim = d_model // num_heads
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0).eval()
    identity = torch.eye(d_model)
    for projection in (mha.w_q, mha.w_k, mha.w_v, mha.w_o):
        with torch.no_grad():
            projection.weight.copy_(identity)
            projection.bias.zero_()

    batch_size, query_len, key_len = 2, 3, 5
    query = torch.randn(batch_size, query_len, d_model)
    key = torch.randn(batch_size, key_len, d_model)
    value = torch.randn(batch_size, key_len, d_model)

    output = mha(query, key, value)

    q = query.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = key.view(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
    v = value.view(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
    expected, _ = scaled_dot_product_attention(q, k, v)
    expected = expected.transpose(1, 2).contiguous().view(
        batch_size, query_len, d_model
    )

    torch.testing.assert_close(output, expected)


def test_mask_broadcasts_over_batch_and_heads() -> None:
    mha = MultiHeadAttention(d_model=4, num_heads=2, dropout=0.0).eval()
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 4, 4)
    value = torch.randn(2, 4, 4)
    mask = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    mask[..., -1] = False

    _, weights = mha(query, key, value, mask=mask, need_weights=True)

    assert weights.shape == (2, 2, 3, 4)
    assert torch.equal(weights[..., -1], torch.zeros(2, 2, 3))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 2, 3))


def test_batch_size_greater_than_one() -> None:
    mha = MultiHeadAttention(d_model=4, num_heads=2).eval()
    query = torch.randn(3, 5, 4)
    key = torch.randn(3, 6, 4)
    value = torch.randn(3, 6, 4)

    output = mha(query, key, value)

    assert output.shape == (3, 5, 4)
    assert torch.isfinite(output).all()


def test_dropout_respects_train_and_eval_modes() -> None:
    mha = MultiHeadAttention(d_model=4, num_heads=2, dropout=1.0)
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 4, 4)
    value = torch.randn(2, 4, 4)

    mha.eval()
    eval_output = mha(query, key, value)
    assert torch.equal(mha(query, key, value), eval_output)

    mha.train()
    train_output = mha(query, key, value)
    assert not torch.allclose(train_output, eval_output)


def test_gradients_exist_and_are_finite() -> None:
    mha = MultiHeadAttention(d_model=4, num_heads=2, dropout=0.0)
    query = torch.randn(2, 3, 4, requires_grad=True)
    key = torch.randn(2, 4, 4, requires_grad=True)
    value = torch.randn(2, 4, 4, requires_grad=True)

    output = mha(query, key, value)
    output.sum().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    for parameter in mha.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mha = MultiHeadAttention(d_model=4, num_heads=2).to("cuda")
    query = torch.randn(2, 3, 4, device="cuda")
    key = torch.randn(2, 4, 4, device="cuda")
    value = torch.randn(2, 4, 4, device="cuda")

    output = mha(query, key, value)

    assert output.is_cuda
    assert torch.isfinite(output).all()


def test_rejects_zero_d_model() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        MultiHeadAttention(0, 2)


def test_rejects_zero_num_heads() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        MultiHeadAttention(4, 0)


def test_rejects_indivisible_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(4, 3)


def test_rejects_wrong_embedding_dimension() -> None:
    mha = MultiHeadAttention(4, 2)
    with pytest.raises(ValueError, match="d_model"):
        mha(torch.randn(2, 3, 5), torch.randn(2, 4, 5), torch.randn(2, 4, 5))


def test_rejects_mismatched_key_value_lengths() -> None:
    mha = MultiHeadAttention(4, 2)
    with pytest.raises(ValueError, match="sequence length"):
        mha(torch.randn(2, 3, 4), torch.randn(2, 4, 4), torch.randn(2, 5, 4))


def test_rejects_mismatched_batch_sizes() -> None:
    mha = MultiHeadAttention(4, 2)
    with pytest.raises(ValueError, match="batch"):
        mha(torch.randn(2, 3, 4), torch.randn(3, 4, 4), torch.randn(3, 4, 4))


def test_rejects_non_3d_inputs() -> None:
    mha = MultiHeadAttention(4, 2)
    with pytest.raises(ValueError, match="batch, seq_len, d_model"):
        mha(torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 4))


def test_rejects_non_boolean_mask() -> None:
    mha = MultiHeadAttention(4, 2)
    with pytest.raises(TypeError, match="torch.bool"):
        mha(
            torch.randn(2, 3, 4),
            torch.randn(2, 4, 4),
            torch.randn(2, 4, 4),
            mask=torch.ones(2, 3, 4),
        )


def test_does_not_use_forbidden_attention_apis() -> None:
    source = inspect.getsource(attention_module)
    assert "nn.MultiheadAttention" not in source
    assert "F.scaled_dot_product_attention" not in source
    assert "functional.scaled_dot_product_attention" not in source
