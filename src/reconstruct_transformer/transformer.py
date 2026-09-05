from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .masks import make_causal_mask, make_pad_mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_heads,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
        max_len=100,
    ):
        super().__init__()
        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layer=n_layers,
            device=device,
            dropout=drop_prob,
            padding_idx=src_pad_idx,
        )
        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layer=n_layers,
            drop_prob=drop_prob,
            device=device,
            padding_idx=trg_pad_idx,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def forward(self, src, trg):
        src_mask = make_pad_mask(src, src, self.src_pad_idx)
        trg_pad_mask = make_pad_mask(trg, trg, self.trg_pad_idx)
        trg_causal_mask = make_causal_mask(
            trg.size(1), device=trg.device
        )
        trg_mask = trg_pad_mask & trg_causal_mask
        cross_mask = make_pad_mask(trg, src, self.src_pad_idx)
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(
            trg,
            enc_output,
            trg_mask,
            cross_mask,
        )
        return output
