"""Evaluacion full-ranking (sin sampling de candidatos)."""

import numpy as np
import torch

from .data import eval_batches


@torch.no_grad()
def evaluate(
    model,
    seqs: list[np.ndarray],
    mode: str,
    device: str,
    batch_size: int = 256,
    max_len: int = 200,
    ratings: list[np.ndarray] | None = None,
) -> dict[str, float]:
    model.eval()
    if ratings is not None:
        inputs, targets, seen, in_ratings = eval_batches(seqs, mode, max_len, ratings)
    else:
        inputs, targets, seen = eval_batches(seqs, mode, max_len)
        in_ratings = None
    n_users = len(targets)
    ndcg10 = recall10 = recall50 = 0.0
    matrix = model.item_matrix()  # una vez por evaluacion

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        x = torch.from_numpy(inputs[start:end]).to(device)
        tgt = torch.from_numpy(targets[start:end]).to(device)
        r = (
            torch.from_numpy(in_ratings[start:end]).to(device)
            if in_ratings is not None
            else None
        )

        h_last = model(x, ratings=r, matrix=matrix)[:, -1]   # (B, d)
        scores = model.score_all(h_last, matrix=matrix)      # (B, n+1)
        scores[:, 0] = float("-inf")

        rows = np.concatenate(
            [np.full(len(seen[start + i]), i) for i in range(end - start)]
        )
        cols = np.concatenate([seen[start + i] for i in range(end - start)])
        scores[torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)] = (
            float("-inf")
        )

        tgt_scores = scores.gather(1, tgt.unsqueeze(1))          # (B, 1)
        ranks = (scores > tgt_scores).sum(1)                     # (B,) 0-based

        ndcg10 += (1.0 / torch.log2(ranks.float() + 2.0))[ranks < 10].sum().item()
        recall10 += (ranks < 10).sum().item()
        recall50 += (ranks < 50).sum().item()

    return {
        "ndcg@10": ndcg10 / n_users,
        "recall@10": recall10 / n_users,
        "recall@50": recall50 / n_users,
    }
