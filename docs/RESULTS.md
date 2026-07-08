# Resultados — sesiones de entrenamiento 2026-07-08

> Actualización (tarde): quinto run `feat_iddrop` (ID-dropout 0.2): TEST ndcg@10 0.1110 / recall@10 0.2066 / recall@50 0.4511, mejor época 39, ~66 min. Peaje caliente -5,5% vs feat; a cambio, validación cold-start: overlap vecinos 0.076→0.697, NDCG ítems descabezados 0.0000→0.0846 (63,7% del caliente, 420× popularidad). Sirve el path frío del montaje dual (ver STATUS.md y SPEC D12). Coste GPU total del proyecto: ~1,25$.

Baseline + ablation de features/rating sobre SASRec, en RunPod (RTX 3090, 0,22 $/h).

## Tabla resumen

| # | Run | Config (flags) | Mejor época | Épocas totales | TEST ndcg@10 | TEST recall@10 | TEST recall@50 | Tiempo aprox. | GPU |
|---|-----|-----------------|:-----------:|:---------------:|:------------:|:---------------:|:---------------:|:-------------:|-----|
| 1 | baseline (mañana) | sin `--use-features` / sin `--use-rating` | 18 | 23 | 0.1028 | 0.1928 | 0.4333 | ~28 min | RTX 3090, 0,22 $/h |
| 2 | id (control ablation) | sin `--use-features` / sin `--use-rating` (mismo config que #1, re-run) | 29 | 34 | 0.1083 | 0.2021 | 0.4474 | ~46 min | RTX 3090, 0,22 $/h |
| 3 | feat | `--use-features` | 50 | 55 | 0.1174 | 0.2164 | 0.4630 | ~84 min | RTX 3090, 0,22 $/h |
| 4 | feat_rating | `--use-features --use-rating` | 56 | 61 | 0.1165 | 0.2133 | 0.4585 | ~102 min | RTX 3090, 0,22 $/h |

Flags comunes a los 4 runs: `--epochs 200 --batch-size 256 --lr 0.001 --negatives 256 --gbce-t 0.75 --patience 5`.

"Épocas totales" = época en la que saltó early stopping (mejor época + patience 5), no el tope de 200.

## Lecturas

- **Control reproduce baseline**: id (0.1083) vs baseline de la mañana (0.1028) — mismo config, diferencia atribuible a variación de seed/shuffle, no a un cambio real. Confirma que el pipeline es estable entre runs.
- **Features suman**: feat (0.1174) vs control (0.1083) → **+8,4% ndcg@10 TEST**. Mejora clara y consistente también en recall@10 (+7,1%) y recall@50 (+3,5%).
- **Rating no suma sobre features**: feat_rating (0.1165) vs feat (0.1174) → **-0,8% ndcg@10**, ligeramente peor. Sigue por delante del control (+7,6% sobre id), pero añadir la señal de rating encima de features no mejora — probablemente el rating ya está correlacionado con la señal que aportan las features, o necesita más regularización/tuning para aportar algo neto. No es una señal fuerte de daño, es empate técnico a la baja frente a feat solo.
- **Conclusión operativa**: para fase 3, quedarnos con `--use-features` sin `--use-rating` como config candidata a producción, salvo que se quiera investigar por qué rating no aporta (posible saturación de capacidad del modelo, o necesidad de un learning rate distinto para esa rama).

## Coste de la sesión

Duración real medida por timestamps de los logs en el pod (no estimada):

- baseline: 08:59 → 09:27 UTC (~28 min)
- id (control): 09:39 → 10:25 UTC (~46 min)
- feat: 10:25 → 11:50 UTC (~84 min)
- feat_rating: 11:50 → 13:31 UTC (~102 min)

Total cómputo: ~260 min (~4h20min) a 0,22 $/h → **~0,95 $**.

Nota: el encargo original estimaba "~2-3h" para la sesión completa; la medición real con timestamps de los propios logs da ~4h20min de cómputo efectivo (más tiempo de pod ocioso entre fases, no incluido aquí). Corrijo la cifra porque tenía datos exactos a mano — el coste sigue siendo marginal (<1 $).

## Anomalía detectada

Los 4 runs de esta mañana y de la ablation usan la misma convención de nombre de checkpoint: cuando ni `--use-features` ni `--use-rating` están activos, `variant = "sasrec_baseline"` y el checkpoint se guarda siempre como `data/checkpoints/sasrec_baseline.pt`.

Como el run "id" (control) de la ablation usa exactamente esa combinación de flags, **su checkpoint sobrescribió en el pod** el `sasrec_baseline.pt` original de esta mañana (época 18) con el de la época 29 del control. En el pod ya no existe el checkpoint original de las 08:59 — solo sus métricas, que sobreviven en `logs/runpod/train_20260708T085902Z.log` y en el mensaje de encargo.

No hay pérdida real: localmente ya existía una copia de seguridad del checkpoint de la mañana como `data/checkpoints/sasrec_baseline_gpu.pt` (descargada antes de la ablation). Pero a partir de ahora, en este repo:

- `data/checkpoints/sasrec_baseline.pt` = checkpoint del run **id/control** de la ablation (época 29, ndcg@10=0.1083).
- `data/checkpoints/sasrec_baseline_gpu.pt` = checkpoint del baseline **original** de la mañana (época 18, ndcg@10=0.1028).

Si se reutiliza más ablation con la config "sin features", conviene pasar un `--name`/sufijo explícito en `train.py` para no seguir pisando `sasrec_baseline.pt` en cada corrida.
