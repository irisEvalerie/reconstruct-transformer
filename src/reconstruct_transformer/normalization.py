'''Normalization layers implemented from first principles.'''

from __future__ import annotations

import torch
from torch import nn


class LayerNorm(nn.Module):
    '''Normalize each input over its final dimension.

    For an input shaped ``(..., normalized_shape)``, the mean and population
    variance are computed independently for every element in the leading
    dimensions. Learnable ``weight`` and ``bias`` then apply an affine
    transformation without changing the input shape.

    The default ``eps=1e-5`` follows common LayerNorm practice and is large
    enough to improve stability for float32 and mixed-precision computation.
    This does not call ``torch.nn.LayerNorm``.
    '''

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError('normalized_shape must be greater than zero')
        if eps <= 0:
            raise ValueError('eps must be greater than zero')

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''Normalize ``inputs`` along the last dimension and apply affine terms.'''
        if inputs.ndim == 0 or inputs.size(-1) != self.normalized_shape:
            raise ValueError(
                f'expected final dimension {self.normalized_shape}, '
                f'got shape {tuple(inputs.shape)}'
            )

        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.weight * normalized + self.bias
