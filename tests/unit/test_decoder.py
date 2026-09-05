"""Unit tests for the transformer decoder stack."""

from __future__ import annotations

import pytest
import torch

from reconstruct_transformer.decoder import Decoder, DecoderLayer
from reconstruct_transformer.masks import make_decoder_self_attention_mask


def test_decoder_layer_output_shape() -> None:
    layer = DecoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    memory = torch.randn(2, 7, 8)

    output = layer(x, memory)

    assert output.shape == (2, 5, 8)


def test_target_mask_blocks_self_attention_only() -> None:
    torch.manual_seed(0)
    layer = DecoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    memory = torch.randn(2, 4, 8)
    target_mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)

    reference = layer(x, memory, target_mask=target_mask)

    # With self-attention fully blocked, its input projections have no effect.
    with torch.no_grad():
        for projection in (
            layer.self_attention.w_q,
            layer.self_attention.w_k,
            layer.self_attention.w_v,
        ):
            projection.weight.zero_()

    output = layer(x, memory, target_mask=target_mask)
    torch.testing.assert_close(output, reference)


def test_source_mask_blocks_cross_attention_only() -> None:
    torch.manual_seed(0)
    layer = DecoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    memory = torch.randn(2, 4, 8)
    source_mask = torch.zeros(1, 1, 5, 4, dtype=torch.bool)

    reference = layer(x, memory, source_mask=source_mask)

    # With cross-attention fully blocked, its input projections have no effect.
    with torch.no_grad():
        for projection in (
            layer.cross_attention.w_q,
            layer.cross_attention.w_k,
            layer.cross_attention.w_v,
        ):
            projection.weight.zero_()

    output = layer(x, memory, source_mask=source_mask)
    torch.testing.assert_close(output, reference)


def test_self_attention_respects_causal_mask() -> None:
    torch.manual_seed(0)
    layer = DecoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    target_len = 4
    memory = torch.randn(1, 3, 8)
    x = torch.randn(1, target_len, 8)
    causal = torch.tril(
        torch.ones(target_len, target_len, dtype=torch.bool)
    )[None, None, :, :]

    before = layer(x, memory, target_mask=causal)

    perturbed = x.clone()
    perturbed[:, -1, :] += 100.0
    after = layer(perturbed, memory, target_mask=causal)

    # Changing the last token must not affect earlier positions under causality.
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_cross_attention_supports_different_src_tgt_lengths() -> None:
    layer = DecoderLayer(d_model=8, ffn_hidden=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 4, 8)
    memory = torch.randn(2, 9, 8)

    output = layer(x, memory)

    assert output.shape == (2, 4, 8)


def test_num_layers_controls_layer_count() -> None:
    for num_layers in (1, 2, 4):
        decoder = Decoder(
            vocab_size=10,
            max_len=10,
            d_model=8,
            ffn_hidden=16,
            num_heads=2,
            num_layers=num_layers,
        )
        assert len(decoder.layers) == num_layers


def test_decoder_logits_shape() -> None:
    decoder = Decoder(
        vocab_size=20,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    ).eval()
    tokens = torch.randint(0, 20, (2, 5))
    memory = torch.randn(2, 6, 8)

    logits = decoder(tokens, memory)

    assert logits.shape == (2, 5, 20)


def test_teacher_forcing_input_shape() -> None:
    decoder = Decoder(
        vocab_size=20,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    ).eval()
    batch_size, target_len = 2, 4
    decoder_input = torch.randint(0, 20, (batch_size, target_len))
    memory = torch.randn(batch_size, 6, 8)

    logits = decoder(decoder_input, memory)

    assert logits.shape == (batch_size, target_len, 20)


def test_gradients_flow_to_all_parameters() -> None:
    decoder = Decoder(
        vocab_size=10,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=3,
        dropout=0.0,
    )
    tokens = torch.randint(0, 10, (2, 6))
    memory = torch.randn(2, 5, 8)

    logits = decoder(tokens, memory)
    logits.sum().backward()

    for param in decoder.embedding.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    for layer in decoder.layers:
        for param in layer.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    assert decoder.output_projection.weight.grad is not None
    assert torch.isfinite(decoder.output_projection.weight.grad).all()


def test_padding_inputs_produce_no_nan() -> None:
    torch.manual_seed(0)
    decoder = Decoder(
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
    memory = torch.randn(2, 6, 8)
    target_mask = make_decoder_self_attention_mask(tokens, target_pad_idx=0)

    logits = decoder(tokens, memory, target_mask=target_mask)

    assert logits.shape == (2, 4, 10)
    assert torch.isfinite(logits).all()


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    decoder = Decoder(
        vocab_size=10,
        max_len=10,
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    ).to("cuda")
    tokens = torch.randint(0, 10, (2, 6), device="cuda")
    memory = torch.randn(2, 5, 8, device="cuda")

    logits = decoder(tokens, memory)

    assert logits.is_cuda
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("num_layers", [0, -1])
def test_rejects_invalid_num_layers(num_layers: int) -> None:
    with pytest.raises(ValueError, match="num_layers"):
        Decoder(
            vocab_size=10,
            max_len=10,
            d_model=8,
            ffn_hidden=16,
            num_heads=2,
            num_layers=num_layers,
        )


def test_decoder_layer_rejects_indivisible_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DecoderLayer(d_model=8, ffn_hidden=16, num_heads=3)
