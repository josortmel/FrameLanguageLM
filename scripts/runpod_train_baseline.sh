#!/usr/bin/env bash
set -euo pipefail

cd /workspace/FrameLanguageLM

mkdir -p data/checkpoints logs

python - <<'PY'
import torch

print("python/torch check")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python -m pip install --upgrade pip
python -m pip install duckdb numpy httpx

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="logs/train_${stamp}.log"
echo "logging to ${log}"

python -u -m framelm.train \
  --data data/sequences.parquet \
  --vocab data/vocab_map.json \
  --out data/checkpoints \
  --epochs "${EPOCHS:-200}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr "${LR:-0.001}" \
  --device "${DEVICE:-cuda}" \
  --negatives "${NEGATIVES:-256}" \
  --gbce-t "${GBCE_T:-0.75}" \
  --patience "${PATIENCE:-5}" \
  2>&1 | tee "${log}"
