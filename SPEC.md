# FrameLanguageLM — Especificación

> **Fecha**: 2026-07-07 · **Estado**: draft para revisión de Pepe
> **Nombre**: FrameLanguageLM (paquete/CLI: `framelm`). Nombre de trabajo anterior: FilmRAG.

## 1. Visión

Un **language model donde los tokens son películas y series**, no palabras. Un historial de visionado (Letterboxd, FilmAffinity, Trakt) es una "frase": una secuencia ordenada con estructura estadística. Un transformer pequeño entrenado sobre millones de historiales reales aprende a predecir "la siguiente película", igual que un LM predice la siguiente palabra.

Esta familia de modelos existe y está validada académicamente (SASRec, BERT4Rec y sucesores). No es un LLM: es un modelo de **5-30M de parámetros** que corre en CPU en cualquier ordenador.

### Requisitos de partida (fijados por Pepe)

- Modelo construido por nosotros — nada de indexar en un RAG genérico de terceros.
- LM, no LLM.
- Inferencia optimizada para correr en cualquier ordenador (sin GPU).
- Catálogo: **50k-100k títulos**, películas y series.

### Casos de uso

1. **Recomendación personalizada** (el core): el usuario exporta sus datos de Letterboxd/FilmAffinity/Trakt/IMDb, el sistema los convierte en su secuencia, y el modelo le muestra los "huecos" — películas con alta probabilidad bajo su distribución que no ha visto: `top-k(P(película | historial)) \ vistas`.
2. **Búsqueda paramétrica** (secundario, cae gratis del diseño): consulta por director/género/país/época/presupuesto → filtro SQL + ranking por similitud en el espacio de embeddings ("como X pero francesa").
3. **Consulta puntual — "¿me merece la pena ver X?"**: el usuario pregunta por una película/serie/documental concreta y el sistema devuelve su afinidad con ese título. Técnicamente es el mismo forward pass del caso 1 leyendo un solo logit: `score(X) = P(X | historial)`. Presentación: la probabilidad cruda no es interpretable — se reporta como **percentil dentro del catálogo no visto** ("está en el top 3% de tu distribución") + explicación por features (qué comparte con lo que has valorado alto: director, cast, género, país...). Matiz honesto que la UI debe transmitir: el modelo estima "probabilidad de que esto encaje en tu trayectoria y te guste" (targets filtrados por rating ≥3.5), no calidad objetiva de la obra.

## 2. Decisiones de diseño

| # | Decisión | Elección | Alternativas descartadas y por qué |
|---|----------|----------|-----------------------------------|
| D1 | Arquitectura | **gSASRec** (SASRec 2-4 capas + loss gBCE) | BERT4Rec: sus ventajas originales estaban infladas por malas reproducciones; 10-50× más lento de entrenar. HSTU/TIGER: escala industrial, overkill para 100k ítems. |
| D2 | Embedding de ítem | **Composicional**: `E(item) = E_ID + proj(concat(E_director, mean(E_cast), E_géneros, E_país, E_idioma, E_década, E_budget_bucket))` | Solo ID: sin cold-start ni búsqueda paramétrica. DIF-SR (atención desacoplada): mejor calidad pero complejidad media — reservado como upgrade v2. |
| D3 | Loss | **gBCE con 256 negativos uniformes** | Full softmax sobre 100k: viable pero no siempre gana y usa más memoria; se probará como baseline si sobra VRAM. Sampled softmax con logQ: alternativa válida, segunda opción. |
| D4 | Rating explícito | **Embedding de rating-bucket sumado al ítem en la secuencia de entrada** + filtro suave en targets (solo ítems bien valorados, ≥3.5/5, como positivos) | Ignorarlo (práctica mainstream): desperdicia la señal explícita, que es la gracia de importar Letterboxd. Peso en la loss: menos expresivo que verlo en el input. |
| D5 | Personalización | **v1: inferencia in-context pura** — SASRec no tiene user-embedding; el usuario ES su secuencia. **v2 opcional**: fine-tune ligero de últimas capas (LR bajo, early stopping) como modo "deep" | Fine-tune completo con <2000 ítems: riesgo de overfitting y drift del espacio compartido. Meta-learning (MAML): +5-15% pero complejidad injustificada. |
| D6 | Catálogo | **Top 50k-100k títulos por `numVotes` de IMDb** (películas + series) | Todo IMDb (~10M títulos): ruido masivo, embeddings inservibles para la cola larga. |
| D7 | Series | **Solo por metadata (cold-start composicional) en v1** — sin señal colaborativa | No existe ningún dataset público bueno de historiales de series con timestamps (ver §6, Riesgo R1). Netflix Prize: viejo, sin metadata. Scraping de Trakt/Letterboxd: zona gris de ToS. |
| D8 | Vecinos / ANN | **Brute-force numpy** (matmul contra la matriz de embeddings) | hnswlib/Faiss: innecesarios — 100k×256 int8 = 25MB, una query son ~5-15ms con recall 1.0. Menos dependencias, cero tuning. Reconsiderar solo si >1M vectores. |
| D9 | Runtime de inferencia | **ONNX Runtime fp32** (revisado 2026-07-08: la cuantización int8 dinámica degrada NDCG@10 un ~70% en este modelo — diminuto y dominado por embeddings; la ganancia de tamaño era 2,5MB de 126MB) | int8 dinámica: rota en la práctica pese a la literatura. PyTorch en producción: artefacto mucho mayor. |
| D12 | Cold-start (añadida 2026-07-08) | **ID-dropout p=0.2 en entrenamiento + servido DUAL**: cine → checkpoint `feat` (calidad intacta), series/fríos → checkpoint `iddrop`; scores nunca mezclados entre poblaciones. Ranking frío: coseno + suelo de ficha + prior popularidad + writers-as-creators | Composición post-hoc sin ID-dropout: probada y FALLA (overlap 0.076, NDCG frío 0.0000 — vectores solo-torre fuera de distribución). Un solo modelo iddrop para todo: -5,5% en cine sin necesidad. |
| D10 | Base de código | **Reescritura limpia propia (~500 líneas) usando `asash/gSASRec-pytorch` como referencia**; RecBole solo para validar métricas contra baselines | RecBole como base de producto: framework pesado, obliga a subclasear. |
| D11 | Distribución | **Fase 1: CLI instalable con `uv`. Fase 2: Tauri v2 + sidecar PyInstaller. Opción demo: onnxruntime-web en navegador** | Solo PyInstaller --onefile: problemas conocidos con DLLs nativas. Electron: pesado frente a Tauri. |

## 3. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│ CAPA DE DATOS (offline, una vez + refrescos)                 │
│  MovieLens 32M ──┐                                           │
│  IMDb datasets ──┼──► pipeline ──► catalog.sqlite            │
│  TMDB API      ──┘                 sequences.parquet         │
├─────────────────────────────────────────────────────────────┤
│ ENTRENAMIENTO (offline, GPU de consumo, 1-4h)                │
│  gSASRec + embeddings composicionales + rating-embedding     │
│  ──► export ONNX int8 ──► model.onnx + item_emb.int8.npy     │
├─────────────────────────────────────────────────────────────┤
│ IMPORTADORES DE USUARIO (local, en su máquina)               │
│  Letterboxd CSV │ IMDb CSV │ Trakt API │ FilmAffinity scraper│
│  ──► matching a ID canónico ──► secuencia del usuario        │
├─────────────────────────────────────────────────────────────┤
│ INFERENCIA (local, CPU, <20ms)                               │
│  secuencia ──► ONNX ──► distribución sobre catálogo          │
│  ──► huecos = top-k \ vistas   ──► explicación por features  │
│  búsqueda paramétrica: SQL sobre catalog.sqlite + matmul     │
└─────────────────────────────────────────────────────────────┘
```

## 4. El modelo en detalle

### 4.1 Vocabulario

- 50k-100k ítems seleccionados por `title.ratings.numVotes` de IMDb, tipos `movie`, `tvSeries`, `tvMiniSeries`.
- Tokens especiales: `<pad>`, `<mask>` (reservado), `<unk>` (película fuera de catálogo en el historial del usuario).

### 4.2 Embedding composicional de ítem (d=256)

| Feature | Fuente | Cardinalidad aprox. | Tratamiento |
|---------|--------|--------------------|-------------|
| ID del ítem | — | 100k | embedding libre |
| Director | IMDb `title.crew` | ~30k | embedding compartido; multi-director → mean |
| Cast principal | IMDb `title.principals` (top 4-6 por ordering) | ~100k | mean de embeddings de actor |
| Géneros | IMDb `title.basics` | ~28 | mean/sum multi-hot |
| País de producción | **TMDB** (IMDb no lo tiene) | ~150 | embedding |
| Idioma original | **TMDB** (IMDb no lo tiene) | ~100 | embedding |
| Década | IMDb `startYear` | ~13 | embedding |
| Presupuesto | TMDB `budget`, bucketizado (log) | 6-8 buckets + **"desconocido"** | embedding — cobertura real solo 20-40%, el bucket desconocido es obligatorio |

`E(item) = E_ID + W · concat(features)`. La suma (no reemplazo) permite que el ID capture lo que la metadata no explica, y la parte composicional da cold-start: una serie o película sin señal colaborativa obtiene embedding solo con su ficha. La literatura reciente (RecSys'25) valida la inicialización content-based para ítems fríos.

### 4.3 Secuencia de entrada

Cada posición: `E(item) + E(rating_bucket) + E(posición)`. Los rating-buckets (p.ej. {≤2, 2.5-3, 3.5-4, 4.5-5, sin-rating}) permiten al modelo distinguir "lo vi y me encantó" de "lo vi y meh". El target en cada posición es el siguiente ítem **bien valorado** (filtro suave: no queremos recomendar lo que el usuario vería y odiaría).

### 4.4 Entrenamiento

- Corpus: secuencias de MovieLens 32M ordenadas por timestamp (32M ratings, 200k usuarios, 87.585 películas — intersectadas con nuestro catálogo).
- Loss: gBCE, 256 negativos uniformes; baseline full cross-entropy si la VRAM lo permite.
- Hiperparámetros de partida (de la literatura): 2 capas, 2 heads, d=256, dropout 0.2-0.5 (ML es denso), max_len 200, Adam lr 1e-3.
- Coste estimado: **30-60s/época en RTX 3080/4090 → 1-4 horas total** (50-200 épocas con early stopping sobre NDCG@10 en validación leave-one-out).

### 4.5 Evaluación

- Protocolo estándar: leave-one-out temporal, **métricas sin sampling de candidatos** (full ranking — el sampled evaluation está desacreditado): NDCG@10, Recall@10/@50.
- Referencia a batir: SASRec vanilla reproducido en RecBole sobre el mismo split.
- Evaluación cualitativa de cold-start: coherencia de vecinos de series (que solo tienen embedding composicional).
- Test de aritmética del espacio: `E(Mulholland Drive) - E_director(Lynch) + E_director(Almodóvar) ≈ ?` — sanity checks interpretables.

## 5. Datos: fuentes y licencias

| Fuente | Qué aporta | Licencia / condiciones |
|--------|-----------|------------------------|
| **MovieLens 32M** (`files.grouplens.org/datasets/movielens/ml-32m.zip`, 239MB) | 32M ratings con timestamp; `links.csv` mapea movieId→IMDb/TMDB | Investigación OK; **no comercial sin permiso**; cita a Harper & Konstan 2015 |
| **IMDb Non-Commercial Datasets** (`datasets.imdbws.com`, actualización diaria) | basics, crew, principals, ratings, akas, episode | Solo uso personal/no comercial |
| **TMDB API** (gratuita; daily ID exports para sync) | budget, país, idioma original, keywords, posters. Poblar 100k títulos ≈ <1h de API | No comercial con atribución obligatoria: "This product uses the TMDB API but is not endorsed or certified by TMDB" |

> ⚠️ **Restricción estructural**: todo el corpus es no-comercial. FilmRAG v1 es un proyecto personal/de investigación. Si algún día se comercializa, hay que renegociar (MovieLens/IMDb) o reconstruir el corpus con datos propios/licenciados. Está decidido conscientemente: validar primero, licenciar después si procede.

## 6. Importadores de usuario

| Plataforma | Método | Fechas de visionado | IDs canónicos | Prioridad |
|-----------|--------|--------------------:|--------------|-----------|
| **Letterboxd** | Export oficial ZIP (`diary.csv`, `ratings.csv`, `watched.csv`) | ✅ diary | ❌ solo URI propia → matching título+año | **P0** |
| **IMDb** | Export oficial "Your Ratings" CSV | solo fecha de voto | ✅ tconst directo | P1 (matching gratis) |
| **Trakt** | API gratuita (`traktexport` en PyPI) | ✅ timestamps exactos, **incluye episodios de series** | ✅ imdb/tmdb/tvdb | P1 (única fuente rica en series) |
| **FilmAffinity** | **Export oficial RGPD** (ZIP de HTML; `movie-ratings.html`: nota 1-10, título ES + año, fecha de voto) — verificado con el export real de Pepe, 2026-07. `fa-scraper` queda como fallback | ❌ solo fecha de voto | ❌ título en español → matching por título original | **P0** (es la plataforma de Pepe) |
| **Netflix** | Export oficial RGPD (`CONTENT_INTERACTION/ViewingActivity.csv`: timestamp exacto por reproducción, por perfil; `Ratings.csv` con thumbs) | ✅ timestamps reales | ❌ título ES a nivel de episodio → colapsar a serie + matching | P1 (única fuente personal con series ordenadas) |
| Rotten Tomatoes | Solo userscripts de terceros | ❌ | ❌ | descartado v1 |

**Matching título+año → TMDB ID** (para Letterboxd/FA): `/search/movie` con título+año, retry sin año, fuzzy scoring con rapidfuzz, búsqueda también por título original para FA. Métrica de calidad: % de matcheo sobre exports reales (objetivo >97%); los no matcheados se reportan al usuario, nunca se descartan en silencio.

## 7. Inferencia y distribución

- **Artefacto total estimado: 150-270MB** — embeddings 100k×256 int8 (25.6MB) + transformer int8 (5-20MB) + catalog.sqlite (60-120MB) + runtime (60-100MB; 0 en navegador).
- Latencia: <10-20ms por inferencia con historial de 500 ítems en CPU moderna (referencia: BERT-base 110M cuantizado hace 20ms; nuestro modelo es ~50× menor).
- Vecinos y búsqueda paramétrica: matmul brute-force contra la matriz de embeddings (~5-15ms, recall exacto).
- Fine-tune local v2: entrenar en PyTorch CPU las últimas capas con el historial del usuario — segundos a 2-3 minutos.
- Distribución: **CLI vía `uv tool install` primero** (early adopters, valida el core); **Tauri v2 + sidecar PyInstaller** como app de escritorio final; **onnxruntime-web** como demo pública sin instalación (el modelo int8 cabe en ~30MB de pesos).

## 8. Riesgos y limitaciones

| ID | Riesgo | Impacto | Mitigación |
|----|--------|---------|------------|
| R1 | **Series sin señal colaborativa** — no existe dataset público de historiales de TV | Recomendación de series peor que la de cine en v1 | **Parcialmente mitigado (2026-07-08)** con D12: ítems fríos a 63,7% de la calidad caliente (420× popularidad); rankings de series validados cualitativamente por Pepe; vecinos de series individuales siguen flojos (no exponer en producto). Solución definitiva: corpus propio vía usuarios que consientan |
| R2 | MovieLens acaba en oct-2023 → películas 2024-2026 sin señal colaborativa | Ítems recientes solo cold-start | Mismo mecanismo que R1; refresco periódico si aparece ML-nuevo |
| R3 | Licencias no comerciales de todo el corpus | Bloquea comercialización | Asumido para v1 (ver §5) |
| R4 | Presupuesto: 20-40% de cobertura real en TMDB | Feature débil | Bucket "desconocido"; nunca campo obligatorio en búsqueda |
| R5 | Matching título+año falla (remakes, títulos traducidos) | Secuencias de usuario corruptas | Fuzzy + título original + reporte explícito de no-matcheados |
| R6 | Sesgo de popularidad (MovieLens sobre-representa mainstream USA) | Recomendaciones poco diversas | Medir coverage/diversidad en eval; re-ranking opcional con penalización por popularidad |
| R7 | int8 no acelera en CPUs sin VNNI | Latencia degradada en HW viejo | Fallback fp32 empaquetado (issue conocido onnxruntime#6732) |

## 9. Criterios de éxito de v1

1. NDCG@10 y Recall@10 (full ranking, leave-one-out) ≥ SASRec vanilla de RecBole en el mismo split.
2. Con el export real de Letterboxd de Pepe: top-50 de huecos juzgado cualitativamente como "esto me lo vería" en mayoría.
3. Inferencia CPU <100ms end-to-end (carga excluida) en un portátil sin GPU.
4. Import Letterboxd → recomendaciones en <2 minutos, matching >97%.
5. Artefacto instalable <300MB.
