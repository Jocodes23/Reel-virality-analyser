# reels-trend-intel — repository layout

This repository holds **two independent versions** of the Reels trend-intelligence
engine. They are kept side by side deliberately: V1 is a frozen reference, V2 is
where all development happens.

```
reel-trend-intelligence-v1/   FROZEN — never edited again (tag: v1-freeze)
reel-trend-intelligence-v2/   ACTIVE — all V2 work happens here
```

## `reel-trend-intelligence-v1/` — frozen reference

The complete, working V1 engine as of the `v1-freeze` tag. It collects Reels
engagement on an adaptive schedule, extracts multimodal features (CLIP visual,
sentence-transformer caption, audio fingerprint, perceptual hash), clusters them
into trends with UMAP + HDBSCAN and c-TF-IDF labelling, and models trend dynamics
with Hawkes / survival / rise–peak–decay fits fused into **calibrated** persistence
probabilities with confidence intervals (evaluated by Brier score and ECE on a
temporal backtest). Ships a FastAPI service + dashboard over either
SQLite/sqlite-vec or PostgreSQL + TimescaleDB + pgvector.

**Nothing in this directory is ever modified.** It exists so V2's changes — most
importantly the redefinition of what a "trend" is — can be compared against a
known-good baseline. Its test suite must continue to pass from this path.

## `reel-trend-intelligence-v2/` — active development

Starts as a byte-for-byte duplicate of V1 and then extends it with:

1. **Reel Analyser** — per-reel video analysis (cinematography, audio identity,
   colour grade, hook, archetype, pacing, text template, loop/structure) driven by
   a keyframe montage plus one VLM call, behind a swappable adapter.
2. **Redefined trend formation** — a trend becomes a *(audio identity × format)*
   pair rather than a topical cluster, fixing V1's over-grouping.
3. **Engagement velocity + lifecycle staging** — normalised velocity/acceleration
   and an `emerging → pre_peak → peaking → post_peak → dormant` classifier, fused
   from V1's existing change-point, Hawkes and rise–peak–decay machinery.
4. **Saved-search Library** — persistent, scheduled collectors (not one-off
   filters), each with its own pool and trend view.

## Compliance posture (unchanged across both versions)

The Instagram Graph API path is the documented, compliant default. The
owned-session adapter is gray-area, is documented openly as such, stays inert
without credentials, and **halts on any login challenge rather than evading it**.
There is no rate-limit evasion, proxy rotation, or fingerprint spoofing in either
version, and none will be added. Where a compliant path cannot supply video for a
given reel, the analyser degrades to metadata-only and records that it did.

## Epistemics (unchanged across both versions)

Every predictive output ships with a confidence value and appears in the temporal
backtest. Virality is heavy-tailed and partly irreducible; this project reports
calibrated probabilities with intervals and does not claim to predict hits. Any
classifier that fails to calibrate is shipped *labelled as uncalibrated* rather
than quietly hidden.

## Running either version

Each directory is self-contained — `cd` into it and follow its own `README.md`.

```bash
cd reel-trend-intelligence-v2
pip install -e ".[dev]"
rti run-once        # offline golden path, no network
rti serve           # dashboard + API on :8000
pytest -q
```

> The large `torch-*.whl` at the repository root is a local CUDA install artifact
> (gitignored). It is kept at root so it is not duplicated into either version.
