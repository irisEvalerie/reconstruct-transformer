'''Top-level encoder-decoder Transformer model.'''

from __future__ import annotations

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .masks import make_causal_mask, make_pad_mask


class Transformer(nn.Module):
    '''Encoder-decoder Transformer from "Attention Is All You Need".

    ``forward`` accepts source tokens ``(batch, src_len)`` and teacher-forced
    target input tokens ``(batch, tgt_len)`` and returns logits shaped
    ``(batch, tgt_len, tgt_vocab_size)``. Four masks are created internally from
    the pad indices -- source padding, target padding, target causality, and
    cross-attention source padding -- so callers only pass token ids and never
    assemble masks by hand.

    Construction applies a unified initialization matching the paper: linear
    projections use Xavier (Glorot) uniform, biases are zero, and embeddings use
    ``N(0, 1)`` (the padding row is kept at zero). Move the model with
    ``model.to(device)``; no device argument is accepted at construction time.
    '''

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        num_layers: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        max_len: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._validate_config(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_heads,
            ffn_hidden,
            num_layers,
            src_pad_idx,
            tgt_pad_idx,
            max_len,
            dropout,
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=src_pad_idx,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=tgt_pad_idx,
        )

        self.init_weights()

    @staticmethod
    def _validate_config(
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        num_layers: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        max_len: int,
        dropout: float,
    ) -> None:
        if src_vocab_size <= 0:
            raise ValueError('src_vocab_size must be greater than zero')
        if tgt_vocab_size <= 0:
            raise ValueError('tgt_vocab_size must be greater than zero')
        if d_model <= 0:
            raise ValueError('d_model must be greater than zero')
        if num_heads <= 0:
            raise ValueError('num_heads must be greater than zero')
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')
        if ffn_hidden <= 0:
            raise ValueError('ffn_hidden must be greater than zero')
        if num_layers <= 0:
            raise ValueError('num_layers must be greater than zero')
        if max_len <= 0:
            raise ValueError('max_len must be greater than zero')
        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be between 0 and 1')
        if not 0 <= src_pad_idx < src_vocab_size:
            raise ValueError('src_pad_idx must index a token in src_vocab_size')
        if not 0 <= tgt_pad_idx < tgt_vocab_size:
            raise ValueError('tgt_pad_idx must index a token in tgt_vocab_size')

    def init_weights(self) -> None:
        '''Re-initialize weights with the paper's scheme.

        Linear layers use Xavier (Glorot) uniform, biases are zeroed, and
        embedding weights are drawn from ``N(0, 1)``. The padding embedding row
        is kept at zero so padded tokens contribute no gradient signal.
        '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx] = 0.0

    def count_parameters(self) -> int:
        '''Return the number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, src_tokens: torch.Tensor, tgt_input_tokens: torch.Tensor
    ) -> torch.Tensor:
        src_mask = make_pad_mask(src_tokens, src_tokens, self.src_pad_idx)
        tgt_pad_mask = make_pad_mask(
            tgt_input_tokens, tgt_input_tokens, self.tgt_pad_idx
        )
        tgt_causal_mask = make_causal_mask(
            tgt_input_tokens.size(1), device=tgt_input_tokens.device
        )
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        cross_mask = make_pad_mask(
            tgt_input_tokens, src_tokens, self.src_pad_idx
        )

        memory = self.encoder(src_tokens, src_mask)
        logits = self.decoder(tgt_input_tokens, memory, tgt_mask, cross_mask)
        return logits
