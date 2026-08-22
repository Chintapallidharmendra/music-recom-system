"""Reconcile Last.fm-1K plays against FMA-small tracks and print a go/no-go recommendation.

This reproduces the analysis already frozen in contracts/dataset_reconciliation.md rather than
re-deciding it: no MusicBrainz IDs exist in FMA metadata, so the only join key is fuzzy
artist-name/track-title string matching. That join was already measured as far too sparse to
build on (0.047% of Last.fm events, 3.25% of FMA-small tracks) and the frozen decision is
NO-GO — use the synthetic interaction layer (data/synth_user_profiles.py,
data/generate_synthetic_logs.py) instead. This script exists so the finding is reproducible,
not so a fresh judgment call gets made at runtime.
"""

import argparse

import pandas as pd

FMA_METADATA_DIR = "datasets/archive/fma_metadata/fma_metadata"
LASTFM_PLAYS = "datasets/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"

MATCH_RATE_THRESHOLD = 0.01  # 1% of events — well above what was actually measured (0.047%)


def load_fma_small_pairs() -> dict:
    tracks = pd.read_csv(f"{FMA_METADATA_DIR}/tracks.csv", index_col=0, header=[0, 1, 2])
    tracks.columns = tracks.columns.droplevel(2)
    small = tracks[tracks[("set", "subset")] == "small"]
    artist = small[("artist", "name")].astype(str).str.strip().str.lower()
    title = small[("track", "title")].astype(str).str.strip().str.lower()
    return {(a, t): str(tid) for tid, a, t in zip(small.index, artist, title)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="only scan the first N Last.fm rows (for a fast smoke test)",
    )
    args = parser.parse_args()

    pairs = load_fma_small_pairs()
    print(f"FMA-small (artist,title) pairs: {len(pairs)}")

    total = 0
    matched = 0
    matched_tracks = set()
    with open(LASTFM_PLAYS, encoding="utf-8", errors="replace") as f:
        for line in f:
            if args.limit_rows and total >= args.limit_rows:
                break
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            key = (parts[3].strip().lower(), parts[5].strip().lower())
            tid = pairs.get(key)
            if tid:
                matched += 1
                matched_tracks.add(tid)

    match_rate = matched / total if total else 0.0
    print(f"total Last.fm play events scanned: {total}")
    print(f"matched play events (exact artist+title): {matched}")
    print(f"match rate: {match_rate:.4%}")
    print(f"unique FMA-small tracks reached: {len(matched_tracks)} / {len(pairs)}")

    if match_rate >= MATCH_RATE_THRESHOLD:
        print(f"GO: match rate >= {MATCH_RATE_THRESHOLD:.2%} threshold — real join is viable.")
    else:
        print(
            f"NO-GO: match rate < {MATCH_RATE_THRESHOLD:.2%} threshold — too sparse to build on. "
            "Use the synthetic interaction layer (contracts/synthetic_data.md) instead. "
            "See contracts/dataset_reconciliation.md for the frozen decision and rationale."
        )


if __name__ == "__main__":
    main()
