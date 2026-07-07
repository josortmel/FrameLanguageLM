"""SASRec: transformer causal con weight tying sobre la matriz de items."""

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, causal: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.ln1(h)
        a, _ = self.attn(
            x, x, x, attn_mask=causal, key_padding_mask=pad_mask, need_weights=False
        )
        h = h + self.drop(a)
        h = h + self.drop(self.ffn(self.ln2(h)))
        return h


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        d: int = 256,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = dict(
            n_items=n_items, d=d, n_layers=n_layers, n_heads=n_heads,
            max_len=max_len, dropout=dropout,
        )
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d, n_heads, dropout) for _ in range(n_layers))
        self.final_ln = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B, L) -> (B, L, d)
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device)
        h = self.emb_drop(self.item_emb(seq) + self.pos_emb(pos))
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1
        )
        pad_mask = seq == 0
        # posiciones pad: filas de atencion 100% enmascaradas -> softmax NaN,
        # y 0*NaN contamina el resto via values. Se anulan tras cada bloque.
        keep = (~pad_mask).unsqueeze(-1)
        h = h * keep
        for blk in self.blocks:
            h = blk(h, causal, pad_mask)
            h = torch.nan_to_num(h) * keep
        return self.final_ln(h)

    def score_all(self, h_last: torch.Tensor) -> torch.Tensor:  # (B, d) -> (B, n+1)
        return h_last @ self.item_emb.weight.T
