# FrameLanguageLM — Plan de ejecución

> Compañero de `SPEC.md`. Cada fase tiene entregable y **verificación concreta** — no se pasa a la siguiente sin cumplirla. Las fases 0-4 son el core; 5-7 son producto. Stack: Python 3.12 + uv, PyTorch (entrenamiento), ONNX Runtime (inferencia), SQLite + Parquet (datos).

## Estado actual — 2026-07-07

- **Fase 0 completada**: corpus crudo descargado y verificado.
- **Fase 1 completada**: catalogo de 100k items, secuencias MovieLens filtradas y spot-check manual.
- **Fase 2 en curso**: baseline SASRec solo-ID implementado y verificado en local; falta entrenamiento completo en GPU.
- **RunPod preparado pero pausado**: se creo un pod RTX 4090 (`23tzovukqp7om0`, `framelm-baseline`) y Pepe lo paro antes de subir/lanzar el entrenamiento. No hay checkpoint nuevo de GPU.
- Ver detalles operativos en `STATUS.md`.

## Fase 0 — Corpus crudo (½ día)

1. Descargar MovieLens 32M, los 7 TSV de IMDb, y los daily ID exports de TMDB.
2. Script de descarga reproducible (`scripts/download_data.py`) con checksums.

**Verificar**: `ratings.csv` = 32.000.204 filas; `links.csv` presente; `title.basics` descomprime y parsea.

## Fase 1 — Catálogo y secuencias (2-3 días)

1. Seleccionar vocabulario: top 50k-100k títulos por `numVotes` (tipos movie/tvSeries/tvMiniSeries). Decidir corte exacto mirando la distribución (¿dónde caen los ratings de MovieLens? ¿qué % del catálogo de un cinéfilo real cubre?).
2. Construir `catalog.sqlite`: joins de IMDb (director, cast top-6, géneros, año) + enriquecimiento TMDB (`/movie/{id}?append_to_response=keywords,credits`): país, idioma, budget, poster path. ~100k requests, <1h.
3. Construir `sequences.parquet`: ratings de ML-32M ordenados por timestamp, filtrados al vocabulario, con rating-bucket.

**Verificar**:
- Cobertura de campos en el catálogo: director >95%, país/idioma >90%, budget informado (se espera solo 20-40% — documentar la cifra real).
- % de ratings de ML-32M retenidos tras filtrar al vocabulario (esperado >90%; si es menor, revisar el corte del vocabulario).
- Spot-check manual de 20 títulos conocidos (¿el join IMDb↔TMDB vía links.csv es correcto?).

## Fase 2 — Baseline: SASRec solo-ID (3-5 días, la fase de más riesgo técnico)

1. Implementación propia limpia (~500 líneas) con `asash/gSASRec-pytorch` como referencia: transformer causal 2 capas, d=256, max_len 200, loss gBCE 256 negativos.
2. Split leave-one-out temporal. Eval full-ranking (sin sampling de candidatos): NDCG@10, Recall@10/@50.
3. Entrenar en GPU (1-4h esperadas). Si no hay GPU local disponible, alquilar spot (RunPod/Lambda, <5€ el experimento).
4. Reproducir el mismo experimento en RecBole como contraste de métricas.

**Verificar**: nuestras métricas ≥ SASRec de RecBole en el mismo split (±5%). Si no, hay bug — no seguir hasta resolverlo. Sanity check adicional: vecinos por embedding de 10 películas conocidas deben ser coherentes a ojo.

## Fase 3 — Embeddings composicionales + rating (3-5 días)

1. Añadir la torre de features: `E(item) = E_ID + proj(concat(director, cast, géneros, país, idioma, década, budget_bucket))`.
2. Añadir rating-bucket embedding en input + filtro de targets (positivos = rating ≥3.5).
3. Ablation: {solo-ID} vs {+features} vs {+features+rating}.

**Verificar**:
- Métricas globales no empeoran vs Fase 2 (las features pueden no mejorar el caso caliente — su valor es cold-start y búsqueda).
- **Test cold-start**: retirar 500 películas del entrenamiento, darles embedding solo composicional, medir si sus vecinos son razonables y si aparecen en recomendaciones pertinentes.
- Series (que entran solo por metadata): vecinos de 10 series conocidas coherentes a ojo — este es el test de la decisión D7.
- Aritmética del espacio: 5 analogías tipo `film - director_A + director_B` con resultado interpretable.

## Fase 4 — Export ONNX + inferencia CPU (2-3 días)

1. Export a ONNX, optimización con el Transformer Optimization Tool de ORT, cuantización int8 dinámica; empaquetar fallback fp32.
2. Módulo de inferencia: secuencia → distribución → huecos (top-k ∖ vistas). Vecinos y búsqueda paramétrica por matmul numpy contra la matriz de embeddings int8. API de scoring puntual: `score(título) → percentil sobre el catálogo no visto` (mismo forward pass, un solo logit + ranking contra la distribución completa).
3. Búsqueda paramétrica: filtro SQL sobre catalog.sqlite + ranking opcional por query-vector composicional.

**Verificar**:
- Paridad de métricas ONNX-int8 vs PyTorch-fp32 (degradación NDCG@10 <2%).
- Latencia <100ms end-to-end en un portátil SIN GPU con historial de 500 ítems.
- Tamaño del bundle de inferencia (modelo+embeddings+sqlite) <300MB.

## Fase 5 — Importadores (2-3 días)

1. **Letterboxd** (P0): parser del ZIP oficial (diary/ratings/watched) + matching título+año→TMDB (search API, retry sin año, rapidfuzz, título original). Reporte explícito de no-matcheados.
2. **IMDb** (P1): parser del CSV de ratings — tconst directo, sin matching.
3. **Trakt** (P1): vía `traktexport` — IDs incluidos, única fuente con series y timestamps.
4. **FilmAffinity** (P0 — plataforma de Pepe): parser del **export oficial RGPD** (`movie-ratings.html`: filas con nota, título ES + año, fecha de voto). Matching por título → TMDB buscando también por título original. `fa-scraper` como fallback para quien no pida el export.
5. **Netflix** (P1): parser de `ViewingActivity.csv` del export RGPD — filtrar por perfil, descartar previews/reproducciones cortas (<15 min), colapsar episodios a nivel de serie, matching título ES → TMDB. Única fuente personal con series en orden real.

**Verificar**: con los datos reales de Pepe (FA: 326 votos en `LittleMeLLM\data\filmaffinity\raw`; Netflix: 5.603 eventos del perfil "Jose Antonio" en `LittleMeLLM\data\Netflix\raw`): matching >97%, secuencia ordenada, ítems fuera de catálogo manejados como `<unk>` sin romper.

## Fase 6 — CLI producto (2-3 días)

```
filmrag import letterboxd export.zip     # → perfil local
filmrag gaps --top 50 [--genre x --decade 1970s]
filmrag similar "Mulholland Drive" --country FR
filmrag search --director "Villeneuve" --language en --min-year 2010
filmrag why <película>                    # explica la recomendación por features
filmrag worth "Twin Peaks"                # ¿me merece la pena? → percentil de afinidad + explicación
```

Instalable con `uv tool install filmrag`. Todo local: los datos del usuario nunca salen de su máquina.

**Verificar**: flujo E2E con el export de Pepe: import → gaps en <2 min total. Evaluación cualitativa del top-50 (criterio de éxito #2 del SPEC). Sanity check de `worth`: una película que Pepe sabe que le encantaría debe salir en percentil alto, y una que sabe que odiaría, en percentil bajo.

## Fase 7 — Extensiones (opcional, tras validar v1)

- **Fine-tune local "deep"**: últimas capas en PyTorch CPU con el historial del usuario (LR bajo, early stopping), 2-3 min. Comparar A/B contra in-context puro — solo se queda si mejora perceptiblemente.
- **App de escritorio**: Tauri v2 + sidecar PyInstaller.
- **Demo web sin instalación**: onnxruntime-web (modelo int8 ~30MB) — el import de Letterboxd se procesa client-side.
- **Corpus de series**: explorar consentimiento de usuarios de Trakt para donar historiales.

## Calendario y dependencias

```
F0 ─ F1 ─┬─ F2 ─ F3 ─ F4 ─┬─ F6 ─ F7
         └─ F5 ────────────┘
```

F5 (importadores) solo depende de F1 (catálogo) — puede hacerse en paralelo con F2-F4. Total estimado del core (F0-F6): **3-4 semanas de trabajo efectivo** a ritmo de proyecto personal.

## Decisiones

1. **Corte del vocabulario**: decidido el 2026-07-07. Se usa 100k: ~79k peliculas + ~21k series, reteniendo ~98.8% de ratings MovieLens.
2. **GPU de entrenamiento**: usar RunPod para baseline y fase 3. La primera prueba de pod quedo pausada antes de entrenar.
3. **Nombre**: FrameLanguageLM es el nombre de proyecto; paquete/CLI actual: `framelm`.
