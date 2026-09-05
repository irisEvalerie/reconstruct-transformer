"""Unit tests for the transformer encoder stack."""

from __future__ import annotations

import pytest
import torch

from reconstruct_transformer.encoder import Encoder, EncoderLayer
from reconstruct_transformer.masks import make_pad_mask


def test_encoder_layer_output_shape() -> None:
    layer = EncoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)

    output = layer(x)

    assert output.shape == (2, 5, 8)


def test_encoder_layer_residual_path() -> None:
    layer = EncoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    with torch.no_grad():
        layer.attention.w_o.weight.zero_()
        layer.attention.w_o.bias.zero_()
        layer.ffn.linear2.weight.zero_()
        layer.ffn.linear2.bias.zero_()

    x = torch.randn(2, 5, 8)
    output = layer(x)

    expected = layer.norm2(layer.norm1(x) + layer.ffn(layer.norm1(x)))
    torch.testing.assert_close(output, expected)


def test_dropout_eval_mode_is_deterministic() -> None:
    torch.manual_seed(0)
    encoder = Encoder(
        vocab_size=10,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=2,
        dropout=0.5,
    ).eval()
    tokens = torch.randint(0, 10, (2, 6))

    first = encoder(tokens)
    second = encoder(tokens)

    torch.testing.assert_close(first, second)


def test_mask_changes_output_and_stays_finite() -> None:
    torch.manual_seed(0)
    layer = EncoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[..., 0] = False

    masked = layer(x, mask=mask)
    unmasked = layer(x)

    assert torch.isfinite(masked).all()
    assert not torch.allclose(masked, unmasked)


def test_encoder_applies_source_mask_to_every_layer() -> None:
    torch.manual_seed(0)
    encoder = Encoder(
        vocab_size=20,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=3,
        dropout=0.0,
    ).eval()
    tokens = torch.randint(0, 20, (2, 6))
    fully_blocked = torch.zeros(1, 1, 6, 6, dtype=torch.bool)

    reference = encoder(tokens, src_mask=fully_blocked)

    # With every key blocked, attention weights are zero in every layer, so the
    # input projections have no effect on the output. Zeroing W_Q/W_K/W_V must
    # leave the output unchanged unless some layer silently dropped the mask.
    with torch.no_grad():
        for layer in encoder.layers:
            for projection in (
                layer.attention.w_q,
                layer.attention.w_k,
                layer.attention.w_v,
            ):
                projection.weight.zero_()

    output = encoder(tokens, src_mask=fully_blocked)
    torch.testing.assert_close(output, reference)


def test_num_layers_controls_layer_count() -> None:
    for num_layers in (1, 2, 4):
        encoder = Encoder(
            vocab_size=10,
            max_len=10,
            d_model=8,
            ffn_hidden=16,
            num_heads=2,
            num_layers=num_layers,
        )
        assert len(encoder.layers) == num_layers


def test_encoder_full_embedding_to_output_shape() -> None:
    encoder = Encoder(
        vocab_size=10,
        max_len=12,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=2,
    ).eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    output = encoder(tokens)

    assert output.shape == (2, 3, 8)


def test_gradients_flow_to_embedding_and_every_layer() -> None:
    encoder = Encoder(
        vocab_size=10,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=3,
        dropout=0.0,
    )
    tokens = torch.randint(0, 10, (2, 6))

    output = encoder(tokens)
    output.sum().backward()

    for param in encoder.embedding.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    for layer in encoder.layers:
        for param in layer.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_padding_inputs_produce_no_nan() -> None:
    torch.manual_seed(0)
    encoder = Encoder(
        vocab_size=10,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        padding_idx=0,
    ).eval()
    tokens = torch.tensor([[1, 2, 0, 0], [3, 0, 4, 5]])
    src_mask = make_pad_mask(tokens, tokens, key_pad_idx=0)

    output = encoder(tokens, src_mask=src_mask)

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    encoder = Encoder(
        vocab_size=10,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    ).to("cuda")
    tokens = torch.randint(0, 10, (2, 6), device="cuda")

    output = encoder(tokens)

    assert output.is_cuda
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("num_layers", [0, -1])
def test_rejects_invalid_num_layers(num_layers: int) -> None:
    with pytest.raises(ValueError, match="num_layers"):
        Encoder(
            vocab_size=10,
            max_len=10,
            d_model=8,
            ffn_hidden=16,
            num_heads=2,
            num_layers=num_layers,
        )


def test_encoder_layer_rejects_indivisible_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EncoderLayer(d_model=8, ffn_hidden=16, num_heads=3)
