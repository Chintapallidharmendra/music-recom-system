# Song feature store — Parquet schema (FROZEN, Day 1)

Source: `Claude_Code_Execution_Spec.docx` §3.2, validated against the actual datasets
in `datasets/` on 2026-08-02. Do not rename/retype columns without a cross-track check-in.

| Column          | Type        | Notes                                                |
|-----------------|-------------|-------------------------------------------------------|
| `track_id`      | string      | FMA track id, primary key                             |
| `genre`         | string      | one of the 8 FMA top-level genres (see list below)     |
| `mfcc_mean`     | float32[20] | librosa MFCC, mean over frames                         |
| `mfcc_var`      | float32[20] | librosa MFCC, variance over frames                     |
| `chroma_mean`   | float32[12] | librosa `chroma_stft`, mean over frames                |
| `tempo`         | float32     | librosa `beat_track` estimate                          |
| `contrast_mean` | float32[7]  | librosa `spectral_contrast`, mean over frames           |

One row per `track_id`. File: `data/features.parquet` (generated, gitignored).

## Dataset validation (done against the real files, not assumed)

- `datasets/archive/fma_metadata/fma_metadata/tracks.csv`, `set,subset == "small"` →
  **8,000 tracks**, matching the spec's locked FMA-small assumption.
- Top-level genres actually present in that subset (**8**, confirmed, canonical order
  used for `genre_vec` indexing everywhere — Track A's `build_user_context.py` and
  Track B's context vectors must use this exact order):
  1. Electronic
  2. Experimental
  3. Folk
  4. Hip-Hop
  5. Instrumental
  6. International
  7. Pop
  8. Rock
- Audio source for feature extraction: `datasets/archive/fma_small/fma_small/**` (mp3s,
  bucketed into `NNN/` subfolders by the standard FMA layout).

See `dataset_reconciliation.md` for the Last.fm↔FMA join decision that
`data/reconcile_datasets.py` depends on.
