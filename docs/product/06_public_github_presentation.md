# Block 6 — GitHub público + presentación

> Fecha: 2026-07-09 · Status: BORRADORES · CLI: **DECIDIDO** → `frame-language-lm` · TMDB: **buena fe + transparencia**

---

## A. README público (borrador)

```markdown
# FrameLanguageLM

**A language model where tokens are movies and TV series.**

Your viewing history is a sentence — a sequence of titles with statistical structure.
FrameLanguageLM is a small transformer (5M parameters, CPU-only) trained on 32M real
viewing sequences that predicts "the next film you'd love."

Give it your FilmAffinity/Letterboxd/Netflix export, and it shows you the gaps in your
film culture — the movies you'd love but haven't discovered yet.

## How It Works

FrameLanguageLM treats movie recommendation as next-token prediction:

1. **Vocabulary:** ~100,000 movies and TV series (selected by IMDb popularity)
2. **Sequences:** 32M viewing histories from MovieLens, ordered by timestamp
3. **Architecture:** gSASRec — a 2-layer causal transformer with gBCE loss and
   compositional embeddings (director + cast + genre + country + language + decade)
4. **Cold-start:** ID-dropout (p=0.2) enables recommendations for titles without
   collaborative signal (new releases, TV series) using metadata-only embeddings
5. **Inference:** ONNX Runtime fp32, <20ms per query on CPU

Your history is the "prompt." The model's next-token distribution over the catalog,
minus what you've already seen, is your gap list.

## Results

Validated with real user data (1,254 viewing events, 96.5% catalog matching):

| Metric | Warm (54k items) | Full (100k incl. cold) |
|--------|:----------------:|:----------------------:|
| NDCG@10 | 0.1174 | 0.1110 |
| Recall@10 | 0.2164 | 0.2066 |
| Recall@50 | 0.4630 | 0.4511 |
| Cold-start NDCG | — | 0.0846 (63.7% of warm) |

Full-ranking evaluation (no candidate sampling), leave-one-out temporal split.
Features improve NDCG@10 by +8.4% over ID-only baseline.

Total training cost: **$1.25** (5 runs on RTX 3090, ~4h20min total compute).

## Quick Start

### Use the webapp (no installation)

Visit [framelanguagelm.github.io](https://framelanguagelm.github.io) — upload your
FilmAffinity export and get your gaps. Everything runs in your browser; your data
never leaves your machine.

### Use the CLI

```bash
# Install (auto-downloads model + catalog on first run, ~348 MB)
pip install frame-language-lm

# Import your profile
frame-language-lm import --filmaffinity export.zip

# Get your gaps
frame-language-lm gaps --top 50

# Filter by genre, decade, country
frame-language-lm gaps --genre "Thriller" --decade 1990s --country KR

# "Is this worth watching?"
frame-language-lm worth "Twin Peaks"

# "Movies like X but from Y"
frame-language-lm similar "Mulholland Drive" --country FR
```

### Use your own data

FrameLanguageLM supports multiple import formats:

| Platform | Method | Priority |
|----------|--------|----------|
| FilmAffinity | GDPR export (HTML) | Primary |
| Letterboxd | Official ZIP export | Primary |
| Netflix | GDPR export (CSV) | Supported |
| IMDb | Ratings CSV export | Supported |
| Trakt | API (via traktexport) | Supported |

## Architecture

```
Viewing history → [Compositional Embeddings] → [2-layer Transformer] → Distribution over catalog
                   E_ID + W·concat(director,     gSASRec, d=256         → top-k \ seen = your gaps
                   cast, genre, country,          gBCE, 256 negatives
                   language, decade, budget)
```

Dual serving: warm checkpoint for cinema (full collaborative signal), ID-dropout
checkpoint for TV series and cold items (metadata-only). Scores are never mixed
between populations.

## Training Your Own Model

```bash
# 1. Build catalog (requires IMDb datasets + optional TMDB API key for country/language/budget)
python scripts/build_catalog.py

# 2. Build sequences from MovieLens 32M
python scripts/build_sequences.py

# 3. Train (GPU recommended, ~1-2h on RTX 3090)
python scripts/train.py --use-features --epochs 200 --patience 5

# 4. Export to ONNX
python scripts/export_onnx.py
```

## Privacy

- All inference runs locally (CPU, <20ms)
- Your viewing history never leaves your machine
- The webapp processes everything client-side in WebAssembly
- No accounts, no tracking, no telemetry

## Limitations

- Trained on MovieLens (predominantly US/English mainstream) — recommendations
  biased toward popular Western titles
- MovieLens data ends Oct 2023 — post-2023 titles are cold-start only
- TV series quality is lower (metadata-based, no collaborative signal)
- This is a research/personal project — non-commercial use only

## Citation

If you use FrameLanguageLM in your research:

```bibtex
@misc{framelanguagelm2026,
  title={FrameLanguageLM: A Sequence Model for Movie Recommendation},
  author={José Antonio Portillo},
  year={2026},
  url={https://github.com/pepe/FrameLanguageLM}
}
```

## Acknowledgments

- Training data: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
  (Harper & Konstan, 2015)
- Metadata: [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/),
  [TMDB API](https://www.themoviedb.org/) (This product uses the TMDB API but is not
  endorsed, certified, or otherwise approved by TMDB)
- Architecture: Based on [gSASRec](https://github.com/asash/gSASRec-pytorch)
  (Petrov & Macdonald, 2023)

## Data Provenance & Takedown

This model was trained on publicly available, non-commercial datasets. For
convenience, a pre-built catalog (`catalog.sqlite`) with metadata from IMDb and
TMDB is included. If any rights holder objects, it will be removed and users can
rebuild it locally via `build_catalog.py`. Model weights are a mathematical
transformation that cannot reconstruct original training data.

**Takedown requests:** if you are a rights holder and believe any artifact in
this repository infringes on your terms, please contact peportmel@gmail.com.
We will respond within 72 hours and can remove the affected artifact or retrain
at negligible cost.

## License

CC-BY-NC-SA 4.0 — Non-commercial use only.
See [LICENSE](LICENSE) for details.
```

---

## B. Post de presentación (borrador LinkedIn)

```
I built a language model where the tokens are movies.

Your viewing history is a "sentence" — a sequence of titles with statistical
structure. A small transformer trained on 32 million real viewing sequences
learns to predict "the next film you'd enjoy," just like a language model
predicts the next word.

How it works:
- ~100,000 movies and TV series in the vocabulary
- Compositional embeddings: each title = ID + director + cast + genre + country
  + language + decade
- gSASRec architecture (2 layers, 5M parameters) — runs on any CPU in <20ms
- ID-dropout for cold-start: new releases and TV series get recommendations
  from metadata alone

I trained it on MovieLens 32M, validated it with my own FilmAffinity profile
(1,254 titles), and the gap list actually reads my taste — picks up my
preference for Japanese animation, Korean thrillers, and auteur cinema.

Total training cost: $1.25.

The interesting bits:
- int8 quantization destroys this model (-70% NDCG) because it's tiny and
  embedding-dominated. Sometimes smaller models need full precision.
- Compositional cold-start without ID-dropout fails completely (NDCG 0.0000).
  You need to train with dropout, not compose post-hoc.
- Features add +8.4% NDCG over ID-only, but explicit ratings add nothing on
  top of features.

Try it: [link to webapp] — upload your FilmAffinity or Letterboxd export and
get your gaps. Everything runs in your browser; your data never leaves your
machine.

Code and model: [GitHub link] | [HuggingFace link]

#MachineLearning #RecSys #MovieRecommendation #Transformers
```

---

## C. Post de presentación (borrador Reddit r/MachineLearning)

```
Title: [P] FrameLanguageLM: a next-token language model where tokens are movies

I built a small transformer (gSASRec, 5M params) trained on MovieLens 32M that
treats your viewing history as a "sentence" and predicts the next film you'd
enjoy.

**Architecture:**
- Vocabulary: ~100k movies/series (IMDb top by numVotes)
- Embedding: E_ID + proj(director, cast, genre, country, language, decade, budget)
- gBCE loss, 256 uniform negatives, 2 layers, d=256
- Cold-start via ID-dropout (p=0.2): unseen titles get metadata-only embeddings

**Results (full ranking, leave-one-out):**
- NDCG@10: 0.1174 (warm), 0.1110 (full catalog incl. cold)
- Cold items reach 63.7% of warm performance while being 420x less popular
- Features: +8.4% NDCG over ID-only
- Rating signal: adds nothing over features alone

**Surprises:**
1. int8 dynamic quantization kills this model (-70% NDCG). Tiny models
   dominated by embeddings need fp32.
2. Post-hoc cold-start composition (removing ID from trained embeddings)
   completely fails (NDCG=0.0000). ID-dropout during training is essential.
3. Total training cost: $1.25 (5 runs, RTX 3090, 4h20min).

**Demo:** [webapp link] — upload FilmAffinity/Letterboxd export, get gap
recommendations. Runs entirely client-side (onnxruntime-web, WASM). Your data
never leaves your browser.

**Code:** [github] | **Model:** [huggingface]

Happy to discuss the architecture, cold-start strategy, or the gBCE loss
if anyone's interested.
```

---

## D. Esqueleto de caso de estudio (portfolio)

```markdown
# Case Study: FrameLanguageLM

## Challenge
Build a movie recommendation system that:
- Runs entirely on CPU (<20ms inference)
- Handles cold-start for new releases and TV series
- Costs under $5 to train
- Requires zero infrastructure to operate

## Approach
Reframed movie recommendation as next-token prediction: a viewing history is a
sequence, a recommendation is the next likely token. Used gSASRec with
compositional embeddings for cold-start via metadata.

## Key Decisions
1. **Dual serving** (warm + ID-dropout checkpoints) instead of single degraded
   model — preserved cinema quality while enabling series
2. **fp32 over int8** — measured that quantization destroys small
   embedding-dominated models
3. **Client-side webapp** (onnxruntime-web) — zero server cost, maximum privacy

## Results
- NDCG@10: 0.1174 (8.4% above ID-only baseline)
- Cold-start: 63.7% of warm performance
- Validated qualitatively with real user profile (1,254 titles)
- Training cost: $1.25 | Inference: <20ms | Operating cost: $0/year

## Technical Stack
Python, PyTorch (training), ONNX Runtime (inference), gSASRec, MovieLens 32M,
IMDb, TMDB. Webapp: vanilla JS + onnxruntime-web (WASM).

## Impact
- [X users / Y gap lists generated] (to be filled after launch)
- Published model on HuggingFace
- Open-source on GitHub

## What I'd Do Differently
- Start with ID-dropout from the first run (wasted a training cycle discovering
  post-hoc composition fails)
- Use Wikidata instead of TMDB for metadata to avoid licensing complications
- [To be expanded after launch feedback]
```

---

## Notas para Pepe

1. **README:** CLI name actualizado a `frame-language-lm`, TMDB atribución y Data Provenance & Takedown incluidos
2. **Posts:** el de LinkedIn es más narrativo/personal, el de Reddit más técnico/datos. Ambos son borradores para que Pepe los reescriba con su voz
3. **Portfolio:** esqueleto mínimo — se rellena con métricas de uso reales post-launch
4. **GitHub URL y HF URL** son placeholders — dependen de cómo Pepe organice sus repos públicos
5. **Cifras:** todas sacadas de STATUS.md y RESULTS.md, verificadas
