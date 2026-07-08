#!/usr/bin/env bash
set -euo pipefail

# Ablation fase 3: solo-ID (control) -> +features -> +features+rating.
# Requiere en data/: sequences.parquet, vocab_map.json, features.npz,
# feature_vocabs.json. Lanzar con PIP_BREAK_SYSTEM_PACKAGES=1.

cd /workspace/FrameLanguageLM

mkdir -p data/checkpoints logs

python -m pip install --upgrade pip
python -m pip install duckdb numpy httpx tqdm tensorboard

tensorboard --logdir logs/tb --host 0.0.0.0 --port 6006 &

common=(
  --data data/sequences.parquet
  --vocab data/vocab_map.json
  --out data/checkpoints
  --epochs "${EPOCHS:-200}"
  --batch-size "${BATCH_SIZE:-256}"
  --lr "${LR:-0.001}"
  --device "${DEVICE:-cuda}"
  --negatives "${NEGATIVES:-256}"
  --gbce-t "${GBCE_T:-0.75}"
  --patience "${PATIENCE:-5}"
)

stamp="$(date -u +%Y%m%dT%H%M%SZ)"

echo "=== 1/3 solo-ID (control reproducibilidad) ==="
python -u -m framelm.train "${common[@]}" \
  --logdir logs/tb/id 2>&1 | tee "logs/ablation_id_${stamp}.log"

echo "=== 2/3 +features ==="
python -u -m framelm.train "${common[@]}" \
  --use-features \
  --logdir logs/tb/feat 2>&1 | tee "logs/ablation_feat_${stamp}.log"

echo "=== 3/3 +features+rating ==="
python -u -m framelm.train "${common[@]}" \
  --use-features --use-rating \
  --logdir logs/tb/feat_rating 2>&1 | tee "logs/ablation_feat_rating_${stamp}.log"

echo "=== ABLATION COMPLETO ==="
grep -h "^TEST" logs/ablation_*_${stamp}.log
