# FrameLanguageLM - Estado actual

Fecha: 2026-07-09

## Resumen

**MODELO VALIDADO POR PEPE (2026-07-09) — fase de modelo cerrada.** Con el perfil FA ampliado (re-import: 1.254 eventos, matching 96,5%, 834 en vocab warm), el reporte dual pasó el juicio cualitativo: cine leyó el perfil con precisión (animación japonesa de autor, saga Ip Man, thriller coreano) y las series frías salieron coherentes (Fallout, Silo, Boba Fett, Solo Leveling). Fases 0-5 completas + cold-start resuelto vía ID-dropout + servido dual. Pendiente: fase 6 (CLI producto) y la capa web/producto (ver docs/VISION.md).

Fix 2026-07-09 en `scripts/dual_report.py`: etiquetas de bloque dinámicas (estaban hardcodeadas con los conteos del perfil viejo) y `gaps_docs` ahora recibe la secuencia completa (antes solo warm → los fríos ya vistos no se excluían del bloque docs).

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

1. ~~Re-import y re-juicio de huecos~~ → HECHO Y VALIDADO (2026-07-09)
2. **Capa de producto EN MARCHA (2026-07-09)**: diseño delegado a sesión relay code-3 (borradores en docs/product/ cuando lleguen) — webapp client-side (onnxruntime-web), lectura de perfil FA público si su ToS lo permite (verificación previa obligatoria), filtros de búsqueda (década/país/director/género), empaquetado fácil de los scripts, HuggingFace (con análisis de licencias: MovieLens/IMDb/TMDB restrictivas), README público + post + portfolio.
3. Nota naming: el nombre de paquete/CLI `framelm` fue un error (decisión de Pepe 2026-07-09: nombres completos, sin abreviaturas) — renombrado pendiente, el nombre del comando CLI se decidirá con la capa de producto.
