from torch import nn

from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding
from .feed_forward import PositionwiseFeedForward
from .normalization import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        n_head,
        drop_prob=0.1,
    ):
        super().__init__()

        # Masked self-attention
        self.attention1 = MultiHeadAttention(d_model, n_head, drop_prob)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        # Encoder-decoder cross-attention
        self.cross_attention = MultiHeadAttention(d_model, n_head, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        # Feed-forward network
        self.ffn = PositionwiseFeedForward(
            d_model,
            ffn_hidden,
            drop_prob,
        )
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
    def forward(
        self,
        dec,
        enc,
        target_mask=None,
        source_mask=None,
    ):
        residual = dec
        x = self.attention1(
            dec,
            dec,
            dec,
            target_mask,
        )
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.cross_attention(
            x,
            enc,
            enc,
            source_mask,
        )
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + residual)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        drop_prob=0.1,
        device=None,
        padding_idx=None,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            dec_voc_size,
            d_model,
            max_len,
            drop_prob,
            device,
            padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    ffn_hidden,
                    n_head,
                    drop_prob,
                )
                for _ in range(n_layer)
            ]
        )
        self.fc = nn.Linear(d_model, dec_voc_size)
    def forward(
        self,
        dec,
        enc,
        target_mask=None,
        source_mask=None,
    ):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(
                dec,
                enc,
                target_mask,
                source_mask,
            )
        output = self.fc(dec)
        return output
