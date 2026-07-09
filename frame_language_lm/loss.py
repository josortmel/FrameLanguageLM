"""gBCE (Petrov & Macdonald, RecSys'23).

Sigmoide generalizada en los positivos: log sigma(s)^beta = -beta*softplus(-s),
con beta = (1-t) + t*alpha y alpha = k/(n_items-1). t=1 -> totalmente calibrado.

Negativos: k muestras uniformes por fila de batch, compartidas entre las
posiciones de la secuencia (ahorra memoria; colision con positivos ~k/n,
despreciable con n~100k).
"""

import torch
import torch.nn.functional as F


def sample_negatives(batch_size: int, k: int, n_items: int, device) -> torch.Tensor:
    return torch.randint(1, n_items + 1, (batch_size, k), device=device)


def gbce_loss(
    h: torch.Tensor,          # (B, L, d) hidden states
    targets: torch.Tensor,    # (B, L) siguiente item, 0 = pad
    item_emb: torch.Tensor,   # (n_items+1, d)
    n_items: int,
    k: int = 256,
    t: float = 0.75,
    target_mask: torch.Tensor | None = None,  # (B, L) bool, targets que puntuan
) -> torch.Tensor:
    B, L, d = h.shape
    alpha = k / (n_items - 1)
    beta = (1.0 - t) + t * alpha

    pos_emb = item_emb[targets]                      # (B, L, d)
    pos_scores = (h * pos_emb).sum(-1)               # (B, L)

    negs = sample_negatives(B, k, n_items, h.device)  # (B, k)
    neg_emb = item_emb[negs]                          # (B, k, d)
    neg_scores = torch.einsum("bld,bkd->blk", h, neg_emb)  # (B, L, k)

    pos_term = beta * F.softplus(-pos_scores)         # (B, L)
    neg_term = F.softplus(neg_scores).sum(-1)         # (B, L)
    per_pos = (pos_term + neg_term) / (1 + k)

    mask = targets != 0
    if target_mask is not None:
        mask = mask & target_mask
    if not mask.any():  # batch sin targets validos (filtro de rating)
        return h.sum() * 0.0
    return per_pos[mask].mean()
