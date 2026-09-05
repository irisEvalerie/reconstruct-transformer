'''Position-wise feed-forward network.'''

from __future__ import annotations

import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    '''Apply the same two-layer MLP independently to every token.

    This implements ``FFN(x) = W2(ReLU(W1(x)))`` with biases in both linear
    layers and dropout between the activation and second projection. Inputs
    have shape ``(..., d_model)`` and outputs retain the same shape. Because
    linear layers operate only on the final dimension, all leading batch and
    sequence dimensions are processed independently with shared parameters.

    Residual connections and layer normalization intentionally belong to the
    encoder and decoder layers, not this module.
    '''

    def __init__(
        self, d_model: int, ffn_hidden: int, drop_prob: float = 0.1
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError('d_model must be greater than zero')
        if ffn_hidden <= 0:
            raise ValueError('ffn_hidden must be greater than zero')
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError('drop_prob must be between 0 and 1')

        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(ffn_hidden, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''Transform ``inputs`` over the final dimension.'''
        if inputs.ndim == 0 or inputs.size(-1) != self.d_model:
            raise ValueError(
                f'expected final dimension {self.d_model}, '
                f'got shape {tuple(inputs.shape)}'
            )

        hidden = self.linear1(inputs)
        activated = self.activation(hidden)
        dropped = self.dropout(activated)
        return self.linear2(dropped)
