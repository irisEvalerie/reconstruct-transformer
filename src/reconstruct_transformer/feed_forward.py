"""Position-wise feed-forward network."""

from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Apply the same two-layer MLP independently at every token position."""

    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(ffn_hidden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
