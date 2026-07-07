"""Smoke test de framelm: formas, padding, loss decreciente y checkpoint."""

import json
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framelm.data import PAD, TrainDataset, eval_batches, left_pad, load_sequences
from framelm.eval import evaluate
from framelm.loss import gbce_loss
from framelm.model import SASRec

MAX_LEN = 200


def test_shapes_and_padding() -> None:
    short = np.array([5, 9, 3, 7], dtype=np.int64)
    too_short = np.array([5, 9, 3], dtype=np.int64)  # train de 1 item: excluido
    long = np.arange(1, 252, dtype=np.int64)  # 251 items
    ds = TrainDataset([short, too_short, long], MAX_LEN)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (MAX_LEN,) and y.shape == (MAX_LEN,)
    assert x[-1] == 5 and y[-1] == 9  # train de short = [5, 9]
    assert (x[:-1] == PAD).all()
    x2, y2 = ds[1]
    assert (x2 != PAD).all()  # ventana llena
    # long = 1..251; train = 1..249; ventana = ultimos 201 -> x acaba en 248, y en 249
    assert x2[-1] == 248 and y2[-1] == 249

    model = SASRec(n_items=300, d=32, n_layers=2, n_heads=2, max_len=MAX_LEN)
    h = model(torch.stack([x, x2]))
    assert h.shape == (2, MAX_LEN, 32)
    assert not torch.isnan(h).any()
    print("ok: formas y padding")


def test_loss_decreases() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    n_items = 50
    # patron determinista: cadenas ciclicas i -> i+1
    seqs = []
    for _ in range(200):
        start = int(rng.integers(1, n_items + 1))
        seqs.append((np.arange(start, start + 12) - 1) % n_items + 1)
    ds = TrainDataset(list(seqs), max_len=16)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    model = SASRec(n_items, d=32, n_layers=1, n_heads=2, max_len=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        total = 0.0
        for x, y in dl:
            h = model(x)
            loss = gbce_loss(h, y, model.item_emb.weight, n_items, k=16)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(dl))
    assert losses[-1] < losses[0] * 0.5, f"loss no baja: {losses}"
    metrics = evaluate(model, seqs, "valid", "cpu", max_len=16)
    assert metrics["ndcg@10"] > 0.5, f"patron ciclico deberia ser facil: {metrics}"
    print(f"ok: loss {losses[0]:.3f} -> {losses[-1]:.3f}, ndcg@10={metrics['ndcg@10']:.3f}")


def test_checkpoint_roundtrip() -> None:
    model = SASRec(n_items=100, d=32, n_layers=1, n_heads=2, max_len=16)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.config}, path)
        ckpt = torch.load(path, weights_only=True)
        model2 = SASRec(**ckpt["config"])
        model2.load_state_dict(ckpt["state_dict"])
        x = torch.randint(0, 101, (2, 16))
        model.eval(), model2.eval()
        assert torch.allclose(model(x), model2(x))
    print("ok: checkpoint roundtrip")


def test_eval_batches() -> None:
    s = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    inputs, targets, seen = eval_batches([s], "valid", max_len=4)
    assert targets[0] == 4 and list(inputs[0][-3:]) == [1, 2, 3]
    inputs, targets, seen = eval_batches([s], "test", max_len=4)
    assert targets[0] == 5 and list(inputs[0]) == [1, 2, 3, 4]
    assert list(left_pad(np.array([7]), 3)) == [0, 0, 7]
    print("ok: eval_batches")


def test_load_sequences_uses_persisted_vocab() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet = tmp_path / "seq.parquet"
        vocab = tmp_path / "vocab_map.json"
        duckdb.sql(f"""
            COPY (
                SELECT * FROM (VALUES
                    (1, 'tt_a', 1),
                    (1, 'tt_b', 2),
                    (1, 'tt_a', 3)
                ) AS t(userId, tconst, timestamp)
            ) TO '{parquet.as_posix()}' (FORMAT PARQUET)
        """)
        vocab.write_text(
            json.dumps({"n_items": 2, "tconst_to_idx": {"tt_a": 2, "tt_b": 1}}),
            encoding="utf-8",
        )

        seqs, n_items = load_sequences(parquet, vocab)
        assert n_items == 2
        assert len(seqs) == 1
        assert seqs[0].tolist() == [2, 1, 2]
    print("ok: vocab persistido")


if __name__ == "__main__":
    test_shapes_and_padding()
    test_eval_batches()
    test_load_sequences_uses_persisted_vocab()
    test_loss_decreases()
    test_checkpoint_roundtrip()
    print("\nsmoke test COMPLETO")
