<h1 align="center">reels-trend-intel</h1>

<p align="center">
  <em>A production-grade, high-efficiency Instagram Reels trend-intelligence engine.</em><br>
  Collect Reels + engagement-over-time → extract fine-grained features → cluster into trends →
  model each trend with real statistics → emit a calibrated, versioned, downstream-agnostic
  <code>TrendReport</code>.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="typed" src="https://img.shields.io/badge/typing-pydantic%20v2%20%C2%B7%20mypy--clean-brightgreen">
  <img alt="tests" src="https://img.shields.io/badge/tests-25%20passing-brightgreen">
  <img alt="storage" src="https://img.shields.io/badge/storage-SQLite%20%7C%20Postgres%2BTimescale%2Bpgvector-orange">
</p>

> **Honesty over hype.** Virality is heavy-tailed and partly irreducible (Salganik, Dodds &
> Watts 2006; Cheng et al. 2014 — see [References](#references)). This system **never** claims
> guaranteed virality. It outputs **calibrated probabilities with confidence intervals**, backed
> by a temporal backtest (Brier score + Expected Calibration Error + reliability diagram).

It stands alone. The output is deliberately generic so any content generator could consume it —
but nothing here is coupled to a specific downstream app.

---

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Data model (ERD)](#data-model-erd)
- [Pipeline & the math](#pipeline--the-math)
- [Efficiency design](#efficiency-design)
- [Output: the TrendReport](#output-the-trendreport)
- [Compliance posture](#compliance-posture)
- [Configuration knobs](#configuration-knobs)
- [Testing & reproducibility](#testing--reproducibility)
- [Project structure](#project-structure)
- [Status & limitations](#status--limitations)
- [References](#references)

---

## Highlights

- **Adaptive engagement re-sampling** — the biggest efficiency lever. Young/accelerating reels
  are sampled often, decaying ones rarely, and dead ones are *retired*. Measured **~75–79% fewer
  samples** than naive fixed-interval polling.
- **Tiered, cached compute** — a cheap gate (perceptual-hash dedupe + quality) runs before any
  expensive model; embeddings are cached by content hash and never recomputed.
- **Real statistics, calibrated** — Hawkes/SEISMIC reproduction number `R(t)`, Cox/Kaplan-Meier
  survival, SpikeM rise/peak/decay, fused and **isotonic/Platt-calibrated** against a temporal
  backtest. Every probability carries a confidence interval.
- **Early / emerging detection** — velocity+acceleration, Kleinberg burst, CUSUM, Bayesian Online
  Change-Point Detection, Bass diffusion, and an early gradient-boosted virality classifier.
- **Whitespace / opportunity ranking** — trends accelerating globally but under-adopted in your
  configured niche (default `food/restaurant`).
- **4 GB-GPU aware** — real CLIP + MiniLM embeddings run on a GTX 1650 (verified: peak VRAM
  ~729 MiB) with automatic CPU fallback, a VRAM cap, and batched inference. Offline mode uses
  deterministic, similarity-preserving embeddings so nothing needs downloading.
- **Swappable storage** — SQLite + `sqlite-vec` (zero-infra default) or PostgreSQL 16 +
  TimescaleDB (hypertables + continuous aggregates) + pgvector.
- **Offline-first** — a synthetic `sample` adapter runs the *entire* pipeline with **zero network**;
  it's the golden-path integration test.

---

## Quick start

### Offline (zero network, no GPU, no model downloads)

```powershell
# Windows
./run.ps1 setup      # venv + install (SQLite backend, deterministic embeddings)
./run.ps1 golden     # collect → extract → cluster → model → report + throughput report
./run.ps1 serve      # http://127.0.0.1:8000  (dashboard + read-only API)
./run.ps1 test       # the test suite
```

```bash
# any OS
pip install -e ".[dev]"
rti run-once         # offline golden path
rti serve            # API + dashboard
```

The **`sample` adapter** generates a realistic synthetic dataset (heavy-tailed engagement,
per-trend adoption profiles, trend-correlated imagery), so the whole pipeline runs with no
Instagram access and no multi-GB downloads.

### Real ML stack on GPU

```powershell
./run.ps1 setup-gpu   # torch(CUDA) + CLIP + audio + OCR + UMAP/HDBSCAN extras
$env:RTI_FEATURES__EMBEDDING_BACKEND="real"; $env:RTI_FEATURES__DEVICE="cuda"
rti run-once
```

Real models (CLIP-ViT-B/32 images, all-MiniLM-L6-v2 captions, Chromaprint audio, RapidOCR) are
**lazily imported** with **automatic CPU fallback** and a **VRAM cap**. On Windows the default
`pip install torch` is CPU-only — install the CUDA wheel explicitly, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.

### Production (Postgres + Timescale + pgvector)

```bash
docker compose up     # brings up Postgres + API + worker
```

---

## Architecture

```
collectors/   pluggable adapters (sample, graph_api, licensed_provider, owned_session)
              + token-bucket polite scheduler + adaptive resampler + raw-payload sink
features/     gate (dedupe/quality) → colour, scene, audio, caption, format + embedding cache
trends/       joint embedding → UMAP/PCA → HDBSCAN; c-TF-IDF topics; audio-fingerprint clusters
models/       hawkes/SEISMIC, survival (Cox/KM), spikem, emerging, whitespace, fusion+calibration
report/       versioned TrendReport schema, ranking, exports (JSONL/Parquet/CSV/Excel), link resolver
storage/      swappable StorageBackend: SQLite+sqlite-vec | Postgres+Timescale+pgvector
api/          read-only FastAPI (GET /trends, /report, /health, /metrics) + HTML dashboard
orchestration/ offline run-once pipeline + resilient APScheduler production loop
observability/ structured logging + Prometheus metrics + VRAM/RAM guards
```

**Data flow**

```
ADAPTERS → RAW SINK → CHEAP GATE → FEATURE EXTRACT → EMBED+CLUSTER → TREND MODELS → REPORT
 (polite,   (verbatim,  (dedupe/     (gated+cached,     (UMAP/HDBSCAN,   (Hawkes/Survival/  (DB + API
  budgeted)  replayable) quality)     batched)           topics, audio)   SpikeM→fuse+cal)   + exports)
     ^______________ adaptive resampler (per-reel next_sample_at) _______________|
```

---

## Data model (ERD)

```
reels ─1─┬─* engagement_samples       (TimescaleDB hypertable; continuous aggregate: hourly rollup)
         ├─1 reel_features            (colour/scene/audio/caption/format + embedding-hash keys)
         ├─* processing_state         (per-stage checkpoints; restart-safe)
         └─1 schedules                (adaptive resampler state: next_sample_at, tier, retired)
audio ───1─* reels                    (audio_id, instagram_audio_url, title/artist, usage_count, links)
embeddings  (content_hash, kind) → vector    (cache keyed by content hash; sqlite-vec / pgvector kNN)
trends ─1─* trend_members ─* reels
trends ─1─1 trend_models              (hawkes/survival/spikem snapshots, health, persistence + CI)
raw_payloads                          (verbatim source payloads, landed BEFORE any transform)
reports                               (versioned TrendReport JSON)
budgets     (adapter, day) → requests_used / cap   (hard daily cap, restart-safe)
```

Raw payloads land verbatim (replayable); all writes are idempotent upserts; dedupe is by reel id
and perceptual hash. Postgres uses hypertables + a continuous aggregate + pgvector + COPY bulk
writes; SQLite uses a NumPy brute-force cosine kNN fallback.

---

## Pipeline & the math

**1. Collect.** Async adapters behind one polite scheduler (global token bucket, randomized
human-scale delays, exponential backoff with jitter, hard daily request budget). Every reel
persists its canonical permalink `https://www.instagram.com/reel/{shortcode}/`, author, posted
time, and audio id; every audio persists `audio_id`, the audio page
`https://www.instagram.com/reels/audio/{audio_id}/`, title/artist, `is_original_audio`,
`usage_count`, and best-effort external links.

**2. Features (gated + cached).**
- **Colour** — k-means in CIELAB (k=5) → dominant/accent/background hex + proportions, luminance,
  saturation, warm/cool ratio, contrast, high-/low-key.
- **Scene** — CLIP zero-shot (real) or keyword+colour heuristic (offline).
- **Audio** — Chromaprint/AcoustID fingerprint + tempo/energy (real) or deterministic per-`audio_id`
  fingerprint (offline) so the same song clusters.
- **Caption** — sentence-transformer embedding, hashtags, length, emoji density, CTA, language;
  OCR on a *sampled* keyframe only.
- **Format/timing** — duration, cut-rate, has-face, TZ-normalized hour-of-day & day-of-week.

**3. Cluster.** Joint embedding `[w_img·img ‖ w_aud·audio ‖ w_cap·caption]` (each L2-normalized)
→ UMAP (or PCA fallback) → HDBSCAN for emergent reel *types*; c-TF-IDF (BERTopic-style) labels;
audio-fingerprint clusters group songs. Each trend has an adoption curve `N(t)` = new adopting
reels per time bucket + aggregate engagement.

**4. Model each trend (all three, then fuse + calibrate).**

| Model | What it computes | Source |
|---|---|---|
| **Hawkes / SEISMIC** | Self-exciting point process, power-law memory kernel `φ(s)=θ·cᶿ/(s+c)^(1+θ)`. Nonparametric branching ratio `R(t) = observed_rate / Σⱼ φ(t−tⱼ)`; R>1 growing, R<1 decaying. Infectiousness `p(t)=R(t)/n*`. Subcritical final size `N∞ ≈ N/(1−R)`. | [\[3\]](#references)[\[4\]](#references) |
| **Survival** | Kaplan-Meier `S(t)` + ridge Cox PH `h(t\|x)=h₀(t)·e^{βᵀx}`. Death = adoption drops >X% from peak for K buckets. **Persists ≥ N days = S(t₀+N\|x)/S(t₀\|x)**. | [\[5\]](#references)[\[6\]](#references) |
| **SpikeM** | Rise-peak-decay fit to `N(t)` → current **phase** + **time-to-peak**. | [\[7\]](#references) |

**Fuse + calibrate.** A **monotone aliveness score** (fixed positive weights on the Hawkes vote,
survival, phase, and velocity — so a rising R>1 pre-peak trend always outranks a dead R≈0 decaying
one) is mapped to a probability by **isotonic (or Platt) calibration**, fit on a **bucket-level
temporal backtest**: for every trend we slide a cutoff across its history, compute the as-of score,
and label whether adoption actually persisted afterwards. We report **Brier score, ECE, and a
reliability diagram**. The confidence interval combines sampling uncertainty (Wald, widening as the
calibration set shrinks) with **model disagreement** — honestly wide when data is thin.

**5. Detect emerging early.** Velocity + acceleration of `N(t)` (seasonally normalized); Kleinberg
2-state burst onset; CUSUM; Bayesian Online Change-Point Detection; Bass diffusion `(p,q,M)` →
peak size + time-to-peak; an early gradient-boosted classifier → `P(crosses virality threshold)`.
**Whitespace** = trends accelerating globally but under-adopted in the configured niche, ranked by
`opportunity = momentum·(1 − niche_saturation)`.

---

## Efficiency design

| Lever | Implementation | Measured (offline, 170 reels) |
|---|---|---|
| **Adaptive re-sampling** | `interval = clamp(base·2^(age/halflife)/(1+w·\|accel\|), min, max)`; tiers hot→warm→cold; **dead reels retired**; budget-aware degradation. | **~75–79% fewer** engagement samples vs naive; ~40 retired |
| **Tiered compute** | perceptual-hash dedupe + quality floor before any CV/audio/OCR model | duplicates/low-quality filtered pre-inference |
| **Cache, never recompute** | embeddings cached by content hash; processed media skipped | each unique media embedded once |
| **Batched inference** | embeddings in batches, bounded concurrency | flat memory under load |
| **Bulk DB + Timescale** | COPY/executemany, pooling, hypertables, continuous aggregates, chunk compression | rollups not recomputed |
| **Incremental modelling** | refit only trends with new data; skip dead trends | — |
| **Columnar analytics** | Polars/DuckDB over Parquet (no pandas) | — |

Per-stage throughput, queue depth, API-budget usage and VRAM/RAM are exposed at `/metrics` and on
the dashboard. All knobs live in one typed settings file (`config/settings.py`).

**Example throughput/cost report**

```
collect : 170 discovered · 8,879 adaptive samples vs 42,521 naive  → 79.1% saved · 40 retired
extract : 170 reels → 164 featurized survivors (6 gated by dedupe/quality)
cluster : 8 trends, 0 noise (100% pure to seeded trends)  |  real CLIP on GPU → 10 finer trends
model   : isotonic calibration on ~90 backtest points → Brier ≈ 0.21, ECE ≈ 0.002
GPU     : real CLIP+MiniLM on GTX 1650, peak VRAM 729 MiB / 4096
```

---

## Output: the TrendReport

A ranked, versioned record (default sort = Trend Health Score; alt = opportunity/whitespace). Per
trend: **A** headline links (top exemplar reel + audio with external links) · **B** identity ·
**C** dynamics (phase, velocity, R(t), health, persistence + CI, forecast peak, post-before) ·
**D** creative recipe (palette, mood/lighting, best hour/day, caption template + hashtags) ·
**E** all aggregated features + full member drill-down (per-reel engagement time-series + features
— nothing dropped) · **F** provenance (model + schema versions, sample size, freshness).

Exposed three ways:
- **Database** (`reports` table)
- **Read-only API** — `GET /trends?niche=&phase=&min_persistence=&limit=`, `GET /trends/{id}`,
  `GET /report`, `GET /health`, `GET /metrics`, and a small HTML dashboard at `/`
- **File exports** — JSONL + Parquet carry the full record incl. member drill-down; CSV/Excel carry
  one row per trend with the two clickable links (`top_reel.url`, `audio.instagram_audio_url`)

---

## Compliance posture

- Collection sources are **pluggable adapters behind one rate-limited, polite, budgeted scheduler**.
  The active set is config-driven (`RTI_COLLECTION__ENABLED_ADAPTERS`).
- Adapters: **Graph API** (Business/Creator — the compliant, recommended path), **licensed
  provider** (lowest risk), **owned-session** (gray-area, moderate ban risk). Each is inert without
  credentials.
- The **owned-session** adapter uses one/few accounts you own, obeys a strict token bucket +
  randomized human-scale delays + backoff + a hard daily cap, collects **only public Reels**, and
  **HALTS immediately on any login challenge/checkpoint** — surfacing it, never routing around it.
- **Explicitly NOT implemented** (and won't be): fingerprint spoofing, User-Agent/device rotation,
  CAPTCHA solving, ban-evasion proxy rotation. If blocked, the adapter stops.
- Secrets come from `.env` / environment only and are never committed (`.env` is gitignored).

> ⚠️ The owned-session adapter is gray-area under Meta's Terms of Service and can get the account
> used flagged or banned. Use the Graph API or a licensed provider for anything you care about.

---

## Configuration knobs (selected)

| Env | Meaning | Default |
|---|---|---|
| `RTI_STORAGE__BACKEND` | `sqlite` \| `postgres` | `sqlite` |
| `RTI_COLLECTION__ENABLED_ADAPTERS` | active source set | `["sample"]` |
| `RTI_COLLECTION__REQUESTS_PER_MINUTE` / `__DAILY_REQUEST_CAP` | polite limits | 30 / 5000 |
| `RTI_SAMPLING__BASE_INTERVAL_S` / `__MIN_` / `__MAX_` | adaptive cadence | 900 / 120 / 86400 |
| `RTI_SAMPLING__RETIRE_AFTER_H` | retirement window | 168 |
| `RTI_FEATURES__EMBEDDING_BACKEND` | `deterministic` \| `real` | `deterministic` |
| `RTI_FEATURES__DEVICE` / `__BATCH_SIZE` / `__VRAM_CAP_MB` | GPU knobs | auto / 16 / 3200 |
| `RTI_MODELS__PERSISTENCE_HORIZON_DAYS` | N in "persists ≥ N days" | 3 |
| `RTI_MODELS__CALIBRATION` | `isotonic` \| `platt` \| `none` | isotonic |
| `RTI_APP__NICHE` | whitespace niche | `food/restaurant` |
| `RTI_REPORT__DEFAULT_RANKING` | `health` \| `opportunity` | health |

See `.env.example` for the full list.

---

## Testing & reproducibility

- Unit tests for each feature extractor and each model on synthetic fixtures.
- A **golden-path integration test** runs collect→extract→cluster→model→report with **no network**
  and asserts calibration-in-range, ranked output, headline links, member drill-down, export
  integrity, and that adaptive sampling beat the naive baseline.
- API + export tests. Everything is **seedable**; model + feature-schema versions are stamped onto
  every record.

```bash
pytest -q          # 25 tests
ruff check . && mypy reels_trend_intel
```

---

## Project structure

```
reels_trend_intel/
├── config/          typed settings (pydantic-settings) — the "knobs"
├── storage/         StorageBackend interface + SQLite and Postgres implementations
├── collectors/      adapters, polite scheduler, token bucket, adaptive resampler
├── features/        gate, colour, scene, audio, caption, format, embeddings, hashing
├── trends/          joint-embedding clustering, topics, trend assembly
├── models/          hawkes, survival, spikem, emerging, whitespace, fusion, calibration
├── report/          schema, ranking, builder, exports, link resolver
├── api/             FastAPI app + dashboard template
├── orchestration/   offline pipeline + resilient loop + modeling glue
├── observability/   logging + metrics
└── fixtures/        synthetic dataset generator (the offline 'sample' source)
tests/               unit + golden-path integration tests
deploy/ · docker-compose.yml · Dockerfile · run.ps1
```

---

## Status & limitations

- ✅ **Offline golden path**, **real GPU models**, **API/dashboard**, and **exports** are all
  verified working. 25 tests pass; ruff + mypy clean.
- ⚠️ **Live collection is credential-gated.** The three live adapters are inert without credentials
  and only collect when you provide them. The owned-session web endpoints are undocumented and
  change; parsing is best-effort and halts on challenge.
- ⚠️ **Live value accrues over time.** The adaptive resampler builds engagement time-series over
  hours/days; a single snapshot is not enough to model a trend. Meaningful, calibrated trends
  typically need on the order of a few hundred reels tracked over a couple of days.
- The persistence probability is calibrated for a configurable horizon (default 3 days); short
  horizons have naturally high base rates. Tune `RTI_MODELS__PERSISTENCE_HORIZON_DAYS`.

---

## References

Papers and methods the logic is built on, grouped by where they're used. (`[n]` markers above link
here.)

**Honesty / predictability of virality**
- **[1]** Salganik, M. J., Dodds, P. S., & Watts, D. J. (2006). *Experimental Study of Inequality
  and Unpredictability in an Artificial Cultural Market.* **Science**, 311(5762), 854–856.
- **[2]** Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J., & Leskovec, J. (2014). *Can Cascades Be
  Predicted?* **WWW 2014**.

**Self-exciting point process (Hawkes / SEISMIC)**
- **[3]** Hawkes, A. G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point
  Processes.* **Biometrika**, 58(1), 83–90.
- **[4]** Zhao, Q., Erdogdu, M. A., He, H. Y., Rajaraman, A., & Leskovec, J. (2015). *SEISMIC: A
  Self-Exciting Point Process Model for Predicting Tweet Popularity.* **KDD 2015**.

**Survival analysis**
- **[5]** Cox, D. R. (1972). *Regression Models and Life-Tables.* **J. Royal Statistical Society,
  Series B**, 34(2), 187–220. (Cox proportional hazards)
- **[6]** Kaplan, E. L., & Meier, P. (1958). *Nonparametric Estimation from Incomplete
  Observations.* **JASA**, 53(282), 457–481. (Kaplan-Meier)

**Rise-and-fall diffusion**
- **[7]** Matsubara, Y., Sakurai, Y., Prakash, B. A., Li, L., & Faloutsos, C. (2012). *Rise and Fall
  Patterns of Information Diffusion: Model and Implications.* **KDD 2012**. (SpikeM)
- **[8]** Bass, F. M. (1969). *A New Product Growth for Model Consumer Durables.* **Management
  Science**, 15(5), 215–227. (Bass diffusion)

**Burst / change-point detection**
- **[9]** Kleinberg, J. (2003). *Bursty and Hierarchical Structure in Streams.* **Data Mining and
  Knowledge Discovery**, 7, 373–397.
- **[10]** Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection.*
  arXiv:0710.3742.
- **[11]** Page, E. S. (1954). *Continuous Inspection Schemes.* **Biometrika**, 41(1/2), 100–115.
  (CUSUM)

**Dimensionality reduction, clustering & topics**
- **[12]** McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction.* arXiv:1802.03426.
- **[13]** Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based
  on Hierarchical Density Estimates.* **PAKDD 2013**. (HDBSCAN)
- **[14]** Grootendorst, M. (2022). *BERTopic: Neural Topic Modeling with a Class-based TF-IDF
  Procedure.* arXiv:2203.05794.

**Feature extractors (ML)**
- **[15]** Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language
  Supervision.* **ICML 2021**. (CLIP)
- **[16]** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks.* **EMNLP 2019**. (sentence-transformers / all-MiniLM)
- **[17]** Zauner, C. (2010). *Implementation and Benchmarking of Perceptual Image Hash Functions.*
  MSc thesis, Upper Austria Univ. of Applied Sciences. (pHash / dedupe)
- **[18]** Lončarić, S., & the AcoustID/Chromaprint project. *Chromaprint acoustic fingerprinting.*
  <https://acoustid.org/chromaprint> (audio fingerprint)
- **[19]** CIE (1976). *CIELAB (L\*a\*b\*) color space.* Commission Internationale de l'Éclairage.
  (colour features)

**Probability calibration & scoring**
- **[20]** Zadrozny, B., & Elkan, C. (2002). *Transforming Classifier Scores into Accurate
  Multiclass Probability Estimates.* **KDD 2002**. (isotonic calibration)
- **[21]** Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods.* **Advances in Large Margin Classifiers**. (Platt scaling)
- **[22]** Brier, G. W. (1950). *Verification of Forecasts Expressed in Terms of Probability.*
  **Monthly Weather Review**, 78(1), 1–3. (Brier score)
- **[23]** Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). *Obtaining Well Calibrated
  Probabilities Using Bayesian Binning.* **AAAI 2015**. (Expected Calibration Error)

**Early classifier**
- **[24]** Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*
  **Annals of Statistics**, 29(5), 1189–1232. (gradient boosting)

---

<p align="center"><sub>
Built as a standalone, downstream-agnostic trend-intelligence engine. Calibrated over confident.
</sub></p>
