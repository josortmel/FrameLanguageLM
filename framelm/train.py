"""Entrenamiento del baseline SASRec solo-ID con gBCE.

Uso: uv run python -m framelm.train [--max-users 2000 --epochs 3 ...]
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import MAX_LEN, TrainDataset, load_sequences, vocab_md5
from .eval import evaluate
from .loss import gbce_loss
from .model import SASRec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sequences.parquet")
    p.add_argument("--vocab", default="data/vocab_map.json")
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
    args = p.parse_args()

    seqs, n_items = load_sequences(Path(args.data), Path(args.vocab), args.max_users)
    print(f"usuarios: {len(seqs):,} | items: {n_items:,} | device: {args.device}")

    ds = TrainDataset(seqs, MAX_LEN)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SASRec(n_items).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sasrec_baseline.pt"
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
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            h = model(x)
            loss = gbce_loss(
                h, y, model.item_emb.weight, n_items, args.negatives, args.gbce_t
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

        metrics = evaluate(model, seqs, "valid", args.device, max_len=MAX_LEN)
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
            f"epoca {epoch}: loss={total / steps:.4f} "
            f"valid={metrics} ({dt:.0f}s)"
        )
        print(
            f"epoch {epoch}/{args.epochs} | loss {total / steps:.4f} "
            f"| ndcg10 {metrics['ndcg@10']:.4f} | eta_min {eta_minutes:.1f}"
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
    test = evaluate(model, seqs, "test", args.device, max_len=MAX_LEN)
    print(f"TEST (mejor epoca {ckpt['epoch']}): {test}")
    print(f"checkpoint: {ckpt_path}")
    writer.close()


if __name__ == "__main__":
    main()
