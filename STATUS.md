# FrameLanguageLM - Estado actual

Fecha: 2026-07-07

## Resumen

FrameLanguageLM ya tiene completadas las fases de corpus/catalogo y un baseline SASRec solo-ID funcional en local. El siguiente hito es entrenar ese baseline completo en GPU para obtener una metrica fiable antes de pasar a embeddings composicionales y rating-buckets.

## Datos y artefactos locales

- Corpus crudo descargado en `data/raw/`:
  - MovieLens 32M.
  - IMDb TSVs.
  - TMDB daily exports.
- Catalogo ensamblado:
  - `data/catalog.sqlite`: 100.000 items.
  - `data/interim/vocab.parquet`: vocabulario top-100k por `numVotes`.
  - `data/sequences.parquet`: 31.605.848 interacciones de 200.947 usuarios.
  - `data/vocab_map.json`: 54.053 items con señal colaborativa real de MovieLens, con indices 1..n y 0 reservado a `<pad>`.
- Checkpoint local existente:
  - `data/checkpoints/sasrec_baseline.pt`.
  - Es una prueba temprana, no el resultado final de fase 2.

## Codigo implementado

- `framelm/data.py`: carga secuencias, split temporal leave-one-out, padding y datasets.
- `framelm/model.py`: SASRec causal con padding seguro para evitar NaNs en atencion.
- `framelm/loss.py`: gBCE con negativos uniformes.
- `framelm/eval.py`: evaluacion full-ranking sin sampling de candidatos.
- `framelm/train.py`: CLI de entrenamiento con early stopping y checkpoint.
- `scripts/smoke_test.py`: smoke test de padding, eval batches, loss decreciente, checkpoint y contrato de vocabulario.
- `scripts/runpod_train_baseline.sh`: lanzador de entrenamiento para RunPod.

## Cambios recientes

- `load_sequences()` ahora usa `data/vocab_map.json` como fuente de verdad para indices. Antes lo cargaba, pero reconstruia una tabla temporal desde `sequences.parquet`, lo que podia desalinear checkpoints si el vocabulario persistido cambiaba.
- `scripts/vocab_analysis.py` usa `printf('tt%07d', ...)` en lugar de `lpad(...)` para mapear IMDb IDs. `lpad` truncaba IDs recientes de mas de 7 digitos.
- `scripts/smoke_test.py` cubre explicitamente que `load_sequences()` respeta el vocabulario persistido.

## Verificacion local

Comandos ejecutados:

```bash
uv run python scripts/smoke_test.py
uv run python scripts/vocab_analysis.py
```

Resultado:

- `smoke_test.py` pasa completo.
- `vocab_analysis.py` pasa y reporta 98.8% de ratings retenidos para el corte de 100k.

## RunPod

Se configuro `runpodctl` localmente y se creo un pod de prueba:

- Pod: `23tzovukqp7om0`
- Nombre: `framelm-baseline`
- GPU: RTX 4090
- Coste reportado: 0.69 USD/h
- Auto-stop solicitado: 6 horas

El pod fue detenido por Pepe antes de completar la subida del paquete y antes de lanzar entrenamiento. No hay nuevo checkpoint de GPU ni logs de entrenamiento de esa ejecucion.

Para reanudar:

1. Crear pod GPU con `runpodctl`.
2. Subir paquete minimo con codigo, `data/sequences.parquet` y `data/vocab_map.json`.
3. Ejecutar:

```bash
cd /workspace/FrameLanguageLM
bash scripts/runpod_train_baseline.sh
```

El script escribe logs en `logs/train_<timestamp>.log` y checkpoints en `data/checkpoints/`.

## Siguiente paso recomendado

Entrenar el baseline SASRec completo en GPU y guardar:

- Log de entrenamiento.
- Mejor checkpoint.
- Metricas valid/test full-ranking.
- Tiempo por epoca.

No pasar a fase 3 hasta tener ese baseline, porque sera el control para saber si las features composicionales y rating-buckets aportan o rompen algo.
