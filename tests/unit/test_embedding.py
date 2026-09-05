"""Unit tests for Transformer token and positional embeddings."""

from __future__ import annotations

import math

import pytest
import torch

from reconstruct_transformer.embedding import (
    PositionalEmbedding,
    TokenEmbedding,
    TransformerEmbedding,
)


def test_token_embedding_shape() -> None:
    embedding = TokenEmbedding(vocab_size=20, d_model=8)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    output = embedding(tokens)

    assert output.shape == (2, 3, 8)


def test_token_embedding_scales_by_square_root_of_model_dimension() -> None:
    embedding = TokenEmbedding(vocab_size=4, d_model=4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(16, dtype=torch.float32).view(4, 4))
    tokens = torch.tensor([[1, 3]])

    output = embedding(tokens)
    expected = embedding.weight[tokens] * math.sqrt(4)

    torch.testing.assert_close(output, expected)


def test_padding_embedding_is_zero_and_receives_no_gradient() -> None:
    embedding = TokenEmbedding(vocab_size=6, d_model=4, padding_idx=0)
    tokens = torch.tensor([[0, 1, 2, 0]])

    output = embedding(tokens)
    output.sum().backward()

    assert torch.equal(output[:, [0, 3]], torch.zeros(1, 2, 4))
    assert embedding.weight.grad is not None
    assert torch.equal(embedding.weight.grad[0], torch.zeros(4))


def test_position_zero_has_expected_sine_and_cosine_values() -> None:
    positional = PositionalEmbedding(d_model=6, max_len=4)
    tokens = torch.zeros(1, 1, dtype=torch.long)

    position_zero = positional(tokens)[0, 0]

    torch.testing.assert_close(
        position_zero, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    )


def test_positional_encoding_matches_independent_formula() -> None:
    d_model = 6
    positional = PositionalEmbedding(d_model=d_model, max_len=5)
    actual = positional(torch.zeros(1, 5, dtype=torch.long))[0]
    expected = torch.zeros(5, d_model)

    for position in range(5):
        for dimension in range(0, d_model, 2):
            angle = position / (10000 ** (dimension / d_model))
            expected[position, dimension] = math.sin(angle)
            if dimension + 1 < d_model:
                expected[position, dimension + 1] = math.cos(angle)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("d_model", [5, 6])
def test_positional_encoding_supports_odd_and_even_dimensions(d_model: int) -> None:
    positional = PositionalEmbedding(d_model=d_model, max_len=7)

    output = positional(torch.zeros(3, 4, dtype=torch.long))

    assert output.shape == (1, 4, d_model)
    assert torch.isfinite(output).all()


def test_sequence_at_maximum_length_is_allowed() -> None:
    positional = PositionalEmbedding(d_model=8, max_len=5)

    output = positional(torch.zeros(2, 5, dtype=torch.long))

    assert output.shape == (1, 5, 8)


def test_sequence_beyond_maximum_length_is_rejected() -> None:
    positional = PositionalEmbedding(d_model=8, max_len=5)

    with pytest.raises(ValueError, match="exceeds configured max_len"):
        positional(torch.zeros(2, 6, dtype=torch.long))


def test_positional_encoding_is_a_persistent_buffer() -> None:
    positional = PositionalEmbedding(d_model=8, max_len=5)

    buffers = dict(positional.named_buffers())
    state = positional.state_dict()

    assert "encoding" in buffers
    assert "encoding" in state
    assert not buffers["encoding"].requires_grad


def test_transformer_embedding_eval_is_deterministic() -> None:
    embedding = TransformerEmbedding(20, 8, 10, drop_prob=0.9)
    embedding.eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    first = embedding(tokens)
    second = embedding(tokens)
    expected = embedding.tok_emb(tokens) + embedding.pos_emb(tokens)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, expected)


def test_transformer_embedding_output_shape() -> None:
    embedding = TransformerEmbedding(20, 8, 10, drop_prob=0.0)

    output = embedding(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    assert output.shape == (2, 3, 8)


def test_gradient_reaches_token_embedding_weights() -> None:
    embedding = TransformerEmbedding(20, 8, 10, drop_prob=0.0)
    tokens = torch.tensor([[1, 2, 3], [3, 4, 5]])

    embedding(tokens).sum().backward()

    assert embedding.tok_emb.weight.grad is not None
    assert torch.isfinite(embedding.tok_emb.weight.grad).all()
    assert embedding.pos_emb.encoding.grad is None


def test_positional_buffer_moves_with_module_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    positional = PositionalEmbedding(d_model=8, max_len=10).to("cuda")

    output = positional(torch.zeros(2, 4, dtype=torch.long, device="cuda"))

    assert positional.encoding.is_cuda
    assert output.is_cuda


def test_transformer_embedding_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    embedding = TransformerEmbedding(20, 8, 10, drop_prob=0.0).to("cuda")
    tokens = torch.tensor([[1, 2, 3]], device="cuda")

    output = embedding(tokens)

    assert output.is_cuda
    assert output.shape == (1, 3, 8)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: TokenEmbedding(0, 8), "vocab_size"),
        (lambda: TokenEmbedding(10, 0), "d_model"),
        (lambda: PositionalEmbedding(0, 10), "d_model"),
        (lambda: PositionalEmbedding(8, 0), "max_len"),
        (lambda: TransformerEmbedding(10, 8, 10, -0.1), "drop_prob"),
        (lambda: TransformerEmbedding(10, 8, 10, 1.1), "drop_prob"),
    ],
)
def test_rejects_invalid_constructor_arguments(constructor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


@pytest.mark.parametrize("shape", [(4,), (2, 3, 1)])
def test_rejects_non_matrix_token_inputs(shape: tuple[int, ...]) -> None:
    tokens = torch.zeros(shape, dtype=torch.long)

    with pytest.raises(ValueError, match="batch, seq_len"):
        TokenEmbedding(10, 8)(tokens)
    with pytest.raises(ValueError, match="batch, seq_len"):
        PositionalEmbedding(8, 10)(tokens)
