# Handoff — FrameLanguageLM

Para: próxima sesión (Fable/Eco/quien retome). Fecha: 2026-07-08.

## Qué es esto

Language model cuyo vocabulario son películas/series (gSASRec + features composicionales), entrenado sobre MovieLens 32M, sirviendo recomendaciones personales desde exports de FilmAffinity/Netflix. Proyecto hijo de LittleMeLLM. Visión completa en `docs/VISION.md` (extraída con vision-extraction, criterios de Pepe verbatim). Curso técnico completo en `docs/CURSO.md`.

## Estado exacto

- **Fases 0-5 del PLAN: completas.** Fase 6 (CLI producto) NO empezada. Capa web NO empezada.
- **Arquitectura dual en producción local** — detalles y números en `STATUS.md`. Regla de oro: cine=`feat`, series/fríos=`iddrop`, scores jamás mezclados.
- **Pepe está ampliando sus votos de FilmAffinity** (su perfil estaba incompleto → los huecos salían "ya vistos", que es validación positiva pero utilidad baja). Cuando traiga export nuevo: `uv run python scripts/import_user.py --filmaffinity <zip> --netflix <csv> --profile "Jose Antonio"` y después `uv run python scripts/dual_report.py`.
- Git limpio, 7 commits, autor `Fable 5 <josortmel@gmail.com>` (config local).

## Cómo correr las cosas

- Smokes: `scripts/smoke_test.py`, `smoke_features.py`, `smoke_dual.py` — los 3 deben estar verdes antes de tocar nada.
- Reporte de huecos de Pepe: `scripts/dual_report.py`.
- Reentrenar (si hiciera falta): `scripts/runpod_train_iddropout.sh` en pod RunPod 3090 (~0,22$/h, patrón completo en STATUS histórico: runpodctl en `~\.local\bin`, clave ssh en `~\.runpod\ssh\`, SIEMPRE `PIP_BREAK_SYSTEM_PACKAGES=1`, puerto 6006/http para TensorBoard, auto-terminate como red).
- `FRAMELM_CKPT` controla qué checkpoint usan build_full_matrix/export_onnx — **verificar SIEMPRE procedencia de artefactos** (2 incidentes de artefactos stale en un día por rutas fijas).

## Trampas conocidas (no repetir)

1. CRLF de Windows rompe scripts bash en el pod → `.gitattributes` ya fija LF; si subes ficheros sueltos, `sed -i 's/\r$//'`.
2. int8 dinámico de ONNX Runtime destruye este modelo (-70% NDCG). fp32 y punto.
3. Composición cold-start post-hoc sin ID-dropout NO funciona — no "arreglarlo" recomponiendo: está medido (0.0000).
4. tqdm ensucia los logs del pod al leerlos por SSH — grep `'^epoch'` para líneas limpias.
5. El eval "épocas totales" = mejor época + patience 5. eta_min=0 significa early-stop inminente, no error.
6. Vecinos de series individuales: flojos, decisión de NO exponer en producto (rankings sí, vecinos no).

## Decisiones pendientes (de Pepe)

- Re-juicio de huecos tras ampliar perfil FA (criterio VISION: >5 de top-10).
- Checkpoint `feat_rating` descartado por métrica pero guardado — candidato a A/B cualitativo si algún día sobra tiempo.
- Producto web: onnxruntime-web apunta bien (modelo fp32 ~60MB navegable, matriz aparte), coste servidor ~0 — alinea con las 3 líneas rojas de la visión.
- Monetización: bloqueada por licencias no-comerciales (MovieLens/IMDb/TMDB) hasta corpus propio — camino documentado en conversación 2026-07-08: gratis → usuarios donan historial con consentimiento → reentrenar corpus propio → vendible.

## Relay / delegación

Sesiones code-2 y code-3 usadas para trabajo mecánico (docs, archivado, instrumentación) — patrón que a Pepe le gusta para ahorrar uso. Los agentes reportan por relay; citar sus mensajes verbatim al usuario.
