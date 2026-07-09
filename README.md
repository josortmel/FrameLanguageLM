# FrameLanguageLM

**A language model where tokens are movies and TV series.**

Your viewing history is a sentence — a sequence of titles with statistical structure.
FrameLanguageLM is a small transformer (5M parameters, CPU-only) trained on 32M real
viewing sequences that predicts "the next film you'd love."

Give it your FilmAffinity export, and it shows you the gaps in your
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

| Platform | Method |
|----------|--------|
| FilmAffinity | GDPR export (HTML) |
| Netflix | GDPR export (CSV) |

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
# 1. Build catalog (requires IMDb datasets + optional TMDB API key)
python scripts/build_catalog.py

# 2. Build sequences from MovieLens 32M
python scripts/build_vocab.py

# 3. Train (GPU recommended, ~1-2h on RTX 3090)
# See scripts/ for training scripts

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

## Acknowledgments

- Training data: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
  (Harper & Konstan, 2015)
- Metadata: [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/),
  [TMDB API](https://www.themoviedb.org/)
- Architecture: Based on [gSASRec](https://github.com/asash/gSASRec-pytorch)
  (Petrov & Macdonald, 2023)

This product uses the TMDB API but is not endorsed, certified, or otherwise approved by TMDB.

## Data Provenance & Takedown

This model was trained on publicly available, non-commercial datasets. For
convenience, a pre-built catalog with metadata from IMDb and TMDB is included
on HuggingFace. If any rights holder objects, it will be removed and users can
rebuild it locally via `scripts/build_catalog.py`. Model weights are a mathematical
transformation that cannot reconstruct original training data.

**Takedown requests:** peportmel@gmail.com — we will respond within 72 hours.

## License

Code: MIT. Model weights and catalog: CC-BY-NC-SA 4.0 (non-commercial).
See [LICENSE](LICENSE) for details.
