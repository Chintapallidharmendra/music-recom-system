# Dataset reconciliation decision (FROZEN, Day 1)

The spec (§3.1) locks FMA-small for audio/features and Last.fm-1K for
interactions/rewards, with an explicit fallback: "fall back to the synthetic
simulator if FMA↔Last.fm matching is too messy." This was the one dataset-dependent
decision left open before Track A could start `reconcile_datasets.py` for real, so it
was resolved here against the actual files rather than left for a track owner to guess.

## What was checked

- `datasets/archive/fma_metadata/fma_metadata/raw_artists.csv` and `raw_tracks.csv`:
  **no MusicBrainz ID field on either** — so there is no reliable ID-based join, only
  fuzzy artist-name / track-title string matching against Last.fm's `artname` /
  `traname` columns.
- Artist-name overlap (case-insensitive exact match): **434 of 2,306** FMA-small
  artists (18.8%) appear anywhere in Last.fm's ~174k unique artist names.
- Full join test: exact-match `(artist, title)` pairs from FMA-small (7,921 pairs)
  against all **19,150,868** Last.fm play events:
  - **9,005 matched events (0.047% of all events)**
  - Only **260 of 8,000** FMA-small tracks (3.25%) are ever reached by a real play
  - Artists that do overlap by name still only account for **215,736 events (1.1%)**
    of total volume before track-title matching narrows it further

## Decision: NO-GO on the real join

Match density is far too sparse and too concentrated (a few hundred tracks) to
support building per-user context or bandit reward signal from real Last.fm↔FMA
joins. `data/reconcile_datasets.py` should print this match rate and this
recommendation, not attempt to build the primary interaction dataset from it.

**Use the synthetic interaction simulator** for `(user, track, action, reward)`
data. Real FMA-small audio still drives `features.parquet` (§3.2) — only the
*interaction/reward* side is simulated. Last.fm remains available as a reference
for realistic session/recency shape (e.g. genre affinity decay patterns) when
tuning the simulator, but is not used as a literal join key.
