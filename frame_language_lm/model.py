"""SASRec: transformer causal con weight tying sobre la matriz de items.

Con use_features, la matriz de items es composicional:
  M = E_ID + W_proj(concat(E_director, E_cast, E_genre, E_country,
                           E_language, E_decade, E_budget))
(mean sobre slots multi-valor, slots 0 = desconocido no aportan).
El scoring (weight tying) usa la matriz COMPUESTA.

Con use_rating, un embedding de rating-bucket se suma en las posiciones de
INPUT (contexto "la vi y me encanto/meh"); no interviene en el scoring.
"""

import torch
from torch import nn

D_FEAT = 64
FEATURE_KEYS = ("director", "cast", "genre", "country", "language", "decade", "budget")
N_RATING_BUCKETS = 5  # 0=sin rating/pad, 1=<=2, 2=2.5-3, 3=3.5-4, 4=4.5-5


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


class FeatureTower(nn.Module):
    """Compone el termino de features de la matriz de items: (n+1, d)."""

    def __init__(self, feats: dict[str, torch.Tensor], n_vocab: dict[str, int], d: int):
        super().__init__()
        for key in FEATURE_KEYS:
            arr = feats[key]
            if arr.ndim == 1:
                arr = arr.unsqueeze(1)
            self.register_buffer(f"idx_{key}", arr, persistent=False)
            emb = nn.Embedding(n_vocab[key] + 1, D_FEAT, padding_idx=0)
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()
            setattr(self, f"emb_{key}", emb)
        self.proj = nn.Linear(len(FEATURE_KEYS) * D_FEAT, d)

    def forward(self) -> torch.Tensor:
        parts = []
        for key in FEATURE_KEYS:
            idx = getattr(self, f"idx_{key}")           # (n+1, slots)
            emb = getattr(self, f"emb_{key}")(idx)      # (n+1, slots, D_FEAT)
            valid = (idx > 0).unsqueeze(-1).float()
            denom = valid.sum(1).clamp(min=1.0)
            parts.append((emb * valid).sum(1) / denom)  # mean sobre slots validos
        return self.proj(torch.cat(parts, dim=-1))      # (n+1, d)


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        d: int = 256,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
        features: dict[str, torch.Tensor] | None = None,
        feature_vocab_sizes: dict[str, int] | None = None,
        use_rating: bool = False,
        use_features: bool = False,  # informativo en config; requiere `features`
        id_dropout: float = 0.0,
    ):
        super().__init__()
        if use_features and features is None:
            raise ValueError(
                "use_features=True requiere pasar `features` y "
                "`feature_vocab_sizes` (no van en el checkpoint config)"
            )
        self.id_dropout = id_dropout
        self.config = dict(
            n_items=n_items, d=d, n_layers=n_layers, n_heads=n_heads,
            max_len=max_len, dropout=dropout,
            use_features=features is not None, use_rating=use_rating,
            id_dropout=id_dropout,
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

        self.tower = (
            FeatureTower(features, feature_vocab_sizes, d) if features else None
        )
        self.rate_emb = (
            nn.Embedding(N_RATING_BUCKETS, d, padding_idx=0) if use_rating else None
        )
        if self.rate_emb is not None:
            nn.init.normal_(self.rate_emb.weight, std=0.02)
            with torch.no_grad():
                self.rate_emb.weight[0].zero_()

    def item_matrix(self) -> torch.Tensor:
        """Matriz de items (n+1, d), compuesta si hay features.

        Con id_dropout>0 y en training, una fraccion aleatoria del VOCABULARIO
        pierde su E_ID en este forward: esos items quedan solo-torre, en la
        misma matriz que se usa para input y para scoring — el modelo aprende
        a operar con items frios (cold-start) dentro de distribucion.
        """
        if self.tower is None:
            return self.item_emb.weight
        id_part = self.item_emb.weight
        if self.training and self.id_dropout > 0:
            keep = (
                torch.rand(id_part.size(0), 1, device=id_part.device)
                >= self.id_dropout
            ).to(id_part.dtype)
            id_part = id_part * keep
        m = id_part + self.tower()
        return m * (torch.arange(m.size(0), device=m.device) > 0).unsqueeze(-1)

    def forward(
        self,
        seq: torch.Tensor,                       # (B, L)
        ratings: torch.Tensor | None = None,     # (B, L) buckets, opcional
        matrix: torch.Tensor | None = None,      # matriz precomputada, opcional
    ) -> torch.Tensor:                            # -> (B, L, d)
        B, L = seq.shape
        m = matrix if matrix is not None else self.item_matrix()
        pos = torch.arange(L, device=seq.device)
        h = m[seq] + self.pos_emb(pos)
        if self.rate_emb is not None and ratings is not None:
            h = h + self.rate_emb(ratings)
        h = self.emb_drop(h)
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

    def score_all(
        self, h_last: torch.Tensor, matrix: torch.Tensor | None = None
    ) -> torch.Tensor:  # (B, d) -> (B, n+1)
        m = matrix if matrix is not None else self.item_matrix()
        return h_last @ m.T
