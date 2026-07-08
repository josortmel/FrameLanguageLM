"""Entrenamiento SASRec con gBCE: solo-ID, +features, +features+rating.

Uso: uv run python -m framelm.train [--use-features --use-rating ...]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import MAX_LEN, TrainDataset, load_sequences, vocab_md5
from .eval import evaluate
from .loss import gbce_loss
from .model import FEATURE_KEYS, SASRec


def load_feature_tensors(npz_path: Path, vocabs_path: Path):
    raw = np.load(npz_path)
    feats = {k: torch.from_numpy(raw[k]) for k in FEATURE_KEYS}
    v = json.loads(vocabs_path.read_text(encoding="utf-8"))
    sizes = {
        "director": len(v["director"]), "cast": len(v["cast"]),
        "genre": len(v["genre"]), "country": len(v["country"]),
        "language": len(v["language"]),
        "decade": v["n_decade"], "budget": v["n_budget"],
    }
    return feats, sizes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sequences.parquet")
    p.add_argument("--vocab", default="data/vocab_map.json")
    p.add_argument("--features", default="data/features.npz")
    p.add_argument("--feature-vocabs", default="data/feature_vocabs.json")
    p.add_argument("--out", default="data/checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--negatives", type=int, default=256)
    p.add_argument("--gbce-t", type=float, default=0.75)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--logdir", default="logs/tb")
    p.add_argument("--use-features", action="store_true")
    p.add_argument("--use-rating", action="store_true")
    p.add_argument("--min-target-rating", type=float, default=3.5)
    args = p.parse_args()

    if args.use_rating:
        seqs, ratings, n_items = load_sequences(
            Path(args.data), Path(args.vocab), args.max_users, with_ratings=True
        )
    else:
        seqs, n_items = load_sequences(Path(args.data), Path(args.vocab), args.max_users)
        ratings = None

    variant = "sasrec" + ("_feat" if args.use_features else "") + (
        "_rating" if args.use_rating else ""
    )
    if variant == "sasrec":
        variant = "sasrec_baseline"
    print(
        f"variante: {variant} | usuarios: {len(seqs):,} | items: {n_items:,} "
        f"| device: {args.device}"
    )

    feats, sizes = (None, None)
    if args.use_features:
        feats, sizes = load_feature_tensors(
            Path(args.features), Path(args.feature_vocabs)
        )

    ds = TrainDataset(
        seqs, MAX_LEN, ratings=ratings, min_target_rating=args.min_target_rating
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SASRec(
        n_items,
        features=feats,
        feature_vocab_sizes=sizes,
        use_rating=args.use_rating,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{variant}.pt"
    best_ndcg, best_epoch = -1.0, -1
    writer = SummaryWriter(log_dir=args.logdir)
    global_step = 0
    epoch_seconds: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, steps = 0.0, 0
        running_loss = 0.0
        pbar = tqdm(dl, desc=f"epoca {epoch}/{args.epochs}", unit="batch")
        for batch in pbar:
            if ratings is not None:
                x, y, rx, y_ok = (b.to(args.device) for b in batch)
            else:
                x, y = (b.to(args.device) for b in batch)
                rx, y_ok = None, None
            matrix = model.item_matrix()
            h = model(x, ratings=rx, matrix=matrix)
            loss = gbce_loss(
                h, y, matrix, n_items, args.negatives, args.gbce_t, target_mask=y_ok
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
            global_step += 1
            running_loss = loss.item() if steps == 1 else 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_postfix(loss=f"{running_loss:.4f}")

            if global_step % 100 == 0:
                writer.add_scalar("train/loss", running_loss, global_step)

        metrics = evaluate(
            model, seqs, "valid", args.device, max_len=MAX_LEN, ratings=ratings
        )
        dt = time.time() - t0
        epoch_seconds.append(dt)
        avg_epoch_seconds = sum(epoch_seconds[-5:]) / len(epoch_seconds[-5:])
        remaining_epochs = min(
            args.epochs - epoch, args.patience - (epoch - best_epoch)
        )
        eta_minutes = max(remaining_epochs, 0) * avg_epoch_seconds / 60

        writer.add_scalar("valid/ndcg10", metrics["ndcg@10"], epoch)
        writer.add_scalar("valid/recall10", metrics["recall@10"], epoch)
        writer.add_scalar("valid/recall50", metrics["recall@50"], epoch)
        writer.add_scalar("time/epoch_seconds", dt, epoch)
        writer.add_scalar("time/eta_minutes", eta_minutes, epoch)

        print(
            f"epoch {epoch}/{args.epochs} | loss {total / steps:.4f} "
            f"| ndcg10 {metrics['ndcg@10']:.4f} | eta_min {eta_minutes:.1f} ({dt:.0f}s)"
        )

        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg, best_epoch = metrics["ndcg@10"], epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "vocab_md5": vocab_md5(Path(args.vocab)),
                    "epoch": epoch,
                    "valid_ndcg@10": best_ndcg,
                },
                ckpt_path,
            )
        elif epoch - best_epoch >= args.patience:
            print(f"early stop (sin mejora desde epoca {best_epoch})")
            break

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    test = evaluate(
        model, seqs, "test", args.device, max_len=MAX_LEN, ratings=ratings
    )
    print(f"TEST {variant} (mejor epoca {ckpt['epoch']}): {test}")
    print(f"checkpoint: {ckpt_path}")
    writer.close()


if __name__ == "__main__":
    main()
