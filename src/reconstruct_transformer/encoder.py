from torch import nn

from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding
from .feed_forward import PositionwiseFeedForward
from .normalization import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(
            d_model,
            ffn_hidden,
            dropout,
        )
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        residual = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        device=None,
        dropout=0.1,
        padding_idx=None,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            enc_voc_size,
            d_model,
            max_len,
            dropout,
            device,
            padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    ffn_hidden,
                    n_head,
                    dropout,
                )
                for _ in range(n_layer)
            ]
        )
    def forward(self, x, src_mask=None):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
