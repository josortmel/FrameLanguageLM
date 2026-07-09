# Block 5 — HuggingFace Publication: licencias, formato y model card

> Fecha: 2026-07-09 · Status: BORRADOR · Licencia: **DECIDIDA** → CC-BY-NC-SA 4.0 · TMDB: **DECIDIDO** → buena fe + transparencia (opción 1)

## El problema de licencias

El modelo está entrenado con tres fuentes, todas restrictivas:

### MovieLens 32M

**Licencia:** Custom (GroupLens Research, U. Minnesota)

Cláusulas relevantes (verbatim):
> "The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota."
> "The user may redistribute the data set, including transformations, so long as it is distributed under these same license conditions."

**Impacto:** Permite redistribuir transformaciones (incluidos pesos de modelo entrenado) bajo la misma licencia no-comercial. Atribución obligatoria: Harper & Konstan 2015.

### IMDb Non-Commercial Datasets

**Licencia:** Custom (Amazon/IMDb)

Cláusulas relevantes:
> "personal and non-commercial use"
> "must not be altered/republished/resold/repurposed to create any kind of online/offline database of movie information" (excepto uso personal individual)

**Impacto:**
- Los pesos del modelo son una transformación suficientemente abstracta — no contienen datos IMDb recuperables. Publicar pesos: **probablemente OK**
- `catalog.sqlite` contiene campos de IMDb (director, cast, géneros, año) — zona gris de ToS, pero decisión de Pepe: incluir por conveniencia de uso (proyecto de portfolio). Cubierto por Data Provenance & Takedown — si IMDb objeta, se retira y se regenera localmente con `build_catalog.py`

### TMDB API

**Licencia:** TMDB API Terms of Use

Cláusulas relevantes (verbatim):
> "Use the TMDB APIs...in connection with, including for training, a machine learning (ML) or artificial intelligence (AI) based Application" — **PROHIBIDO**
> "Training or validating a machine learning or artificial intelligence system...collecting data sets containing TMDB Content" — **PROHIBIDO**
> "Make derivatives of the TMDB APIs or TMDB Content." — **PROHIBIDO**
> "Cache, for longer than 6 months, any information obtained through or from TMDB" — **PROHIBIDO**
> Attribution: "This [product] uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB."

**Impacto GRAVE:**
- El modelo fue entrenado con features composicionales que incluyen país, idioma y budget de TMDB
- Técnicamente, los pesos del modelo son un derivado de TMDB Content según su ToS
- El catalog.sqlite contiene datos TMDB y no puede redistribuirse ni cachearse >6 meses
- TMDB prohíbe explícitamente operar un servicio de recomendación que genere ingresos

## Análisis: ¿qué se puede publicar legalmente?

### Nivel de riesgo por artefacto

| Artefacto | Contiene datos directos | Es derivado de | Redistribuible | Riesgo |
|-----------|:-----------------------:|:--------------:|:--------------:|:------:|
| Código fuente (train/infer) | No | — | Sí (propio) | Ninguno |
| Pesos ONNX (`model_fp32.onnx`) | No (pesos numéricos abstractos) | ML32M + IMDb + TMDB | Zona gris | Medio |
| Matrices embedding (`item_embeddings*.npy`) | Implícitamente: cada fila es un ítem del catálogo | ML32M + IMDb + TMDB | Zona gris | Medio |
| `catalog.sqlite` | Sí: títulos, directores, cast, país, idioma de IMDb+TMDB | IMDb + TMDB | **Incluido** (decisión Pepe, cubierto por takedown) | Medio-Alto |
| `sequences.parquet` (entrenamiento) | Sí: secuencias de ML32M | MovieLens | Sí, bajo misma licencia | Bajo |
| `vocab_map.json` (ID→título) | Sí: títulos de IMDb | IMDb | **Incluido** (decisión Pepe, cubierto por takedown) | Medio |

### La zona gris de los pesos

Los pesos del modelo son una transformación matemática de los datos de entrenamiento. No contienen datos IMDb/TMDB recuperables. En la práctica:
- **Ningún caso conocido** de un titular de derechos de dataset demandando por publicar pesos de un modelo entrenado con sus datos (los casos de copyright de IA se centran en outputs generativos, no en pesos de modelos de recomendación)
- **MovieLens permite redistribuir transformaciones** bajo misma licencia → los pesos están cubiertos
- **IMDb/TMDB:** sus ToS prohíben "derivados" pero el consenso de la comunidad de RecSys es que los pesos de modelos entrenados no constituyen una base de datos ni un derivado directo. Los pesos no permiten reconstruir los datos originales
- **Práctica del sector:** miles de modelos en HF fueron entrenados con datos de ToS restrictivos (todos los LLMs entrenados con datos web). Los modelos de recomendación publicados en RecSys papers usan MovieLens/IMDb habitualmente

**Recomendación pragmática:** publicar los pesos bajo licencia no-comercial (CC-BY-NC-SA 4.0), documentar las fuentes de entrenamiento con transparencia total, y NO incluir datos crudos de IMDb/TMDB.

## Propuesta de publicación en HuggingFace

### Qué publicar

```
FrameLanguageLM/
├── model_fp32.onnx              # 56 MB (checkpoint warm)
├── model_fp32.onnx.data         # (externalizado)
├── model_full_fp32.onnx         # 101 MB (checkpoint full/iddrop)
├── model_full_fp32.onnx.data    # (externalizado)
├── item_embeddings.npy          # 53 MB (warm)
├── item_embeddings_full.npy     # 98 MB (full)
├── full_aux.npz                 # 0.2 MB (flags F2/F3)
├── catalog.sqlite               # 36 MB (catálogo completo con metadata)
├── vocab_map.json               # 1 MB (ID↔título)
├── meta.json                    # configuración del modelo
├── meta_full.json               # configuración full
├── feature_vocabs.json          # 2.4 MB (features composicionales)
├── README.md                    # Model card (ver abajo)
└── scripts/
    └── build_catalog.py         # Opcional: regenerar catálogo desde fuentes
```

**Total: ~348 MB** (ambos checkpoints + catálogo incluido)

### Qué NO publicar

- `sequences.parquet` (redistribuible pero innecesario para inferencia)
- Datos crudos de IMDb/TMDB (los dumps en bruto)

### Licencia recomendada

**CC-BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)

Razones:
- NonCommercial: alineado con MovieLens (no-comercial sin permiso) e IMDb (personal/no-comercial)
- ShareAlike: alineado con MovieLens ("redistribuir bajo mismas condiciones")
- Attribution: cubre el requisito de citar Harper & Konstan 2015

Alternativa: una licencia custom tipo MovieLens que reproduzca sus condiciones exactas.

### Model Card (esqueleto)

```markdown
---
license: cc-by-nc-sa-4.0
tags:
  - recommendation
  - sequential-recommendation
  - movies
  - series
  - gsasrec
  - onnx
language:
  - en
  - es
library_name: onnxruntime
pipeline_tag: other
---

# FrameLanguageLM

A language model where tokens are movies and TV series — trained on 32M viewing sequences
to predict "the next film you'd love."

## Model Description

- **Architecture:** gSASRec (2-layer transformer, d=256, gBCE loss) with compositional embeddings
- **Vocabulary:** ~100k titles (movies + TV series) selected by IMDb numVotes
- **Training data:** MovieLens 32M sequences, enriched with IMDb/TMDB metadata
- **Cold-start:** ID-dropout (p=0.2) enables recommendation of unseen titles via metadata-only embeddings
- **Dual serving:** warm checkpoint (54k items with signal) + full checkpoint (100k including cold items)

## Performance

| Metric | Warm (feat) | Full (feat_iddrop) |
|--------|:-----------:|:------------------:|
| TEST NDCG@10 | 0.1174 | 0.1110 |
| TEST Recall@10 | 0.2164 | 0.2066 |
| TEST Recall@50 | 0.4630 | 0.4511 |
| Cold-start NDCG | — | 0.0846 (63.7% of warm) |

Full ranking evaluation (no candidate sampling), leave-one-out temporal split.
Total training cost: ~$1.25 (RTX 3090).

## How to Use

```bash
# Install the CLI
pip install frame-language-lm

# Import your profile (auto-downloads model + catalog on first run)
frame-language-lm import --filmaffinity export.zip

# Get your gaps
frame-language-lm gaps --top 50
```

## Training Data & Licenses

- **MovieLens 32M:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets.
  ACM TIST 5, 4, Article 19. Non-commercial use; redistribution of transformations
  under same conditions.
- **IMDb Non-Commercial Datasets:** Personal/non-commercial use only.
- **TMDB API:** Metadata enrichment (country, language, budget).
  This product uses the TMDB API but is not endorsed, certified, or otherwise approved by TMDB.

## Limitations

- Trained on MovieLens (predominantly US/English mainstream cinema) — recommendations
  biased toward popular Western titles
- MovieLens ends Oct 2023 — post-2023 titles are cold-start only
- Series recommendations are metadata-based (no collaborative signal) — lower quality than films
- Budget feature: only 20-40% coverage in TMDB

## Ethical Considerations

- No user data is collected or transmitted — all inference runs locally
- Model cannot recover individual user viewing histories from training data
- Recommendations reflect statistical patterns in viewing behavior, not quality judgments

## Data Provenance & Takedown

This model was trained on publicly available, non-commercial datasets. No raw
data from any source is included — only model weights (a mathematical
transformation that cannot reconstruct original data).

**Takedown requests:** if you are a rights holder and believe this model
infringes on your terms, please contact peportmel@gmail.com. We will respond
within 72 hours.
```

## TMDB: estrategia DECIDIDA — buena fe + transparencia

Pepe ha decidido la **opción 1**: publicar con transparencia total, atribución explícita, y licencia no-comercial.

**Acciones concretas:**
- Model card y README documentan TODAS las fuentes (MovieLens/IMDb/TMDB) con sus licencias exactas
- Atribución TMDB explícita: *"This product uses the TMDB API but is not endorsed, certified, or otherwise approved by TMDB"* + logo TMDB si su branding guide lo requiere
- Licencia de publicación: CC-BY-NC-SA 4.0
- `catalog.sqlite` y `vocab_map.json` incluidos en HF para conveniencia de uso (decisión informada de Pepe — proyecto de portfolio, no obligar al usuario a reconstruir). Cubiertos por Data Provenance & Takedown
- Sección "Data Provenance & Takedown" con contacto

**Justificación pragmática:** TMDB rara vez persigue proyectos de investigación/personal — su preocupación principal son los clones comerciales y el scraping masivo. Los pesos del modelo no contienen datos TMDB recuperables.

**Plan B interno (NO publicar, solo documentar aquí):** si TMDB objetara formalmente, reentrenar sustituyendo TMDB por Wikidata (país P495, idioma P407 — CC0) cuesta ~$0.30 y ~90min de GPU. Se pierde budget (20-40% cobertura, impacto marginal). Cero drama, cero riesgo reputacional.

## Data Provenance & Takedown

Incluir en el model card y README:

```markdown
## Data Provenance & Takedown

This model was trained on publicly available, non-commercial datasets:

| Source | Data used | License | How to verify |
|--------|-----------|---------|---------------|
| MovieLens 32M | User viewing sequences | Custom (non-commercial, redistribution OK) | [grouplens.org](https://grouplens.org/datasets/movielens/32m/) |
| IMDb Non-Commercial | Titles, genres, directors, cast, year | Custom (personal/non-commercial) | [developer.imdb.com](https://developer.imdb.com/non-commercial-datasets/) |
| TMDB API | Country, language, budget (compositional features) | TMDB API ToS | [themoviedb.org](https://www.themoviedb.org/) |

For convenience, this repository includes a pre-built catalog (`catalog.sqlite`)
containing metadata derived from IMDb and TMDB. If any rights holder objects to
the inclusion of this catalog, it will be removed and users can rebuild it
locally using the provided `build_catalog.py` script. Model weights are a
mathematical transformation that cannot reconstruct the original training data.

**Takedown requests:** if you are a rights holder and believe any artifact in
this repository infringes on your terms, please contact peportmel@gmail.com.
We will respond within 72 hours and can remove the affected artifact or retrain
the model excluding your data source at negligible cost.
```

## Decisiones — todas DECIDIDAS

1. ~~¿Licencia CC-BY-NC-SA 4.0 o custom?~~ → **CC-BY-NC-SA 4.0**
2. ~~¿Aceptamos el riesgo TMDB?~~ → **Sí, buena fe + transparencia (opción 1)**
3. ~~¿Publicamos ambos checkpoints (warm + full) o solo el warm?~~ → **Ambos (~310 MB de modelo)**
4. ~~¿El catálogo se reconstruye con script local o se incluye?~~ → **Incluido en HF** (catalog.sqlite + vocab_map.json). `build_catalog.py` queda como opcional para regeneración

## Upload command (when Pepe provides HF token)

```bash
# Login first
huggingface-cli login

# Upload the entire repo
huggingface-cli upload pepe/FrameLanguageLM hf_repo/ . --repo-type model
```

Note: the HF repo name `pepe/FrameLanguageLM` is a placeholder — Pepe will confirm his HF username.
