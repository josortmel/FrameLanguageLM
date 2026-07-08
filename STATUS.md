# FrameLanguageLM - Estado actual

Fecha: 2026-07-08 (cierre de sesión)

## Resumen

Fases 0-5 completas + cold-start resuelto vía ID-dropout + servido dual en producción local. El modelo funciona con los datos reales de Pepe (validación cualitativa: recomienda cosas que ya vio y le gustaron sin haberlas visto en su perfil). Pendiente: Pepe está completando su perfil de FilmAffinity para purificar los huecos; después fase 6 (CLI producto) y la capa web.

## Arquitectura de producción (dual)

| Población | Checkpoint | Matriz | Uso |
|---|---|---|---|
| Cine + series con señal (54.053) | `sasrec_feat.pt` (TEST ndcg@10 0.1174) | `item_embeddings.npy` | `gaps_movies`, worth/neighbors warm |
| Catálogo completo incl. 46k fríos | `sasrec_feat_iddrop.pt` (TEST 0.1110; frío 0.0846 = 63,7% del caliente, 420× popularidad) | `item_embeddings_full.npy` | `gaps_series`, `gaps_docs`, cold |

Scores NUNCA se mezclan entre poblaciones. Ranking frío: coseno + suelo de ficha (director o ≥2 cast conocidos) + prior popularidad (α=0.5) + writers-as-creators para series (F4, 19,1% cobertura).

## Lecciones clave de la sesión (2026-07-08)

1. **Composición post-hoc SIN entrenamiento falla** (overlap 0.076, NDCG frío 0.0000): E_ID domina la geometría; vectores solo-torre quedan fuera de distribución. Fix: **ID-dropout p=0.2** (Bernoulli sobre vocabulario, misma matriz para input y scoring) — un solo reentrenamiento (~66 min, 0,30$).
2. **Cuantización int8 dinámica de ORT rompe este modelo** (-70% NDCG): modelo diminuto dominado por embeddings. Producción = fp32 (126MB, 5,5ms de latencia — sobra).
3. **Rutas de checkpoint fijas causaron 2 artefactos stale** en un día. Todo script acepta ya `FRAMELM_CKPT`. Verificar procedencia de artefactos antes de validar.
4. Netflix colapsa 5.603 reproducciones → 401 títulos; FA export oficial RGPD existe (HTML). Matching global 97,0%.

## Artefactos y datos

- `data/checkpoints/`: 4 checkpoints (baseline×2, feat, feat_iddrop) + `sasrec_feat_rating.pt` (descartado, -0,8% vs feat)
- `data/artifacts/`: ONNX fp32 warm + full (verificados contra FRAMELM_CKPT correcto), matrices, `full_aux.npz` (flags F2/F3)
- `data/user/pepe_sequence.parquet`: 705 eventos (456 warm + 241 cold-mapeables), `import_report.json`
- `logs/runpod/`: logs + TensorBoard de los 5 entrenamientos
- `docs/`: SPEC, PLAN, VISION (extraída con vision-extraction), RESULTS, CURSO, TENSORBOARD, HANDOFF

## Coste GPU acumulado del proyecto

~1,25$ (5 entrenamientos, RTX 3090 0,22$/h). Cero pods vivos.

## Siguiente paso

1. Pepe amplía votos en FilmAffinity → re-import (`scripts/import_user.py` + `scripts/dual_report.py`) → re-juicio de huecos
2. Fase 6: CLI `framelm` empaquetado (`gaps`/`worth`/`similar`/`why`)
3. Capa producto (ver docs/VISION.md): web sin instalación, HuggingFace, GitHub público, caso de estudio
