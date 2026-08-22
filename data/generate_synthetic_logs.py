"""Generate synthetic historical listening logs over FMA-small tracks.

Drop-in replacement for what would have been real Last.fm-joined plays (NO-GO per
contracts/dataset_reconciliation.md). Reads data/user_profiles.parquet for genre-affinity
ground truth (does not recompute its own) and data/features.parquet for the track/genre
universe. Writes data/synthetic_logs.parquet: (user_id, timestamp, track_id).

Deliberately injects novelty_bias-weighted off-affinity plays so history isn't perfectly
predictable from genre_affinity alone -- otherwise the bandit problem in Track B becomes
trivial (see contracts/synthetic_data.md).

Optionally injects a time-dependent preference drift (--inject-drift): without it, every
user's effective genre_affinity is constant across the whole history_days window, so
mlops/drift_report.py's reference/current split (older half vs newer half) is just two
random samples of the *same* distribution -- no real drift exists to detect, which is why
the drift check "legitimately almost never fire[d]" (see mlops/dags/retrain_policy.py).
With --inject-drift, plays inside the most recent `drift_start_frac` of the window are
pulled toward DRIFT_TARGET_AFFINITY (ramping in linearly, strongest at the most recent
day), simulating a platform-wide genre trend. That shifts genre and its correlated audio
features (tempo, mfcc/chroma/contrast means), which is what DRIFT_COLUMNS in
drift_report.py actually measures.
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from data.synth_user_profiles import GENRES

# A synthetic "platform trend" toward Electronic/Pop, used as the drift target when
# --inject-drift is set. Order must match data.synth_user_profiles.GENRES.
DRIFT_TARGET_AFFINITY = np.array([0.35, 0.02, 0.02, 0.05, 0.03, 0.03, 0.35, 0.15])
DRIFT_TARGET_AFFINITY = DRIFT_TARGET_AFFINITY / DRIFT_TARGET_AFFINITY.sum()


def _tracks_by_genre(features: pd.DataFrame) -> dict:
    return {g: sub["track_id"].to_numpy() for g, sub in features.groupby("genre")}


def _effective_affinity(
    affinity: np.ndarray,
    day_offset: int,
    history_days: int,
    drift_start_frac: float,
    drift_strength: float,
) -> np.ndarray:
    """Blend a user's base affinity toward DRIFT_TARGET_AFFINITY as day_offset -> 0
    (i.e. as plays get more recent). No-op (returns affinity unchanged) before the
    drift window starts."""
    drift_start_day = drift_start_frac * history_days
    if day_offset >= drift_start_day or drift_start_day <= 0:
        return affinity
    w = drift_strength * (drift_start_day - day_offset) / drift_start_day
    blended = (1 - w) * affinity + w * DRIFT_TARGET_AFFINITY
    return blended / blended.sum()


def generate_logs(
    profiles: pd.DataFrame,
    features: pd.DataFrame,
    history_days: int,
    seed: int,
    inject_drift: bool = False,
    drift_start_frac: float = 0.5,
    drift_strength: float = 0.7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tracks_by_genre = _tracks_by_genre(features)
    now = pd.Timestamp.utcnow().tz_localize(None)

    rows = []
    for _, user in profiles.iterrows():
        affinity = np.asarray(user["genre_affinity"], dtype=np.float64)
        affinity = affinity / affinity.sum()
        novelty = float(user["novelty_bias"])

        n_active_days = rng.integers(5, min(60, history_days) + 1)
        active_day_offsets = rng.choice(history_days, size=n_active_days, replace=False)

        for day_offset in active_day_offsets:
            n_plays = max(1, rng.poisson(lam=5))
            day_ts = now - timedelta(days=int(day_offset))

            if inject_drift:
                eff_affinity = _effective_affinity(
                    affinity, int(day_offset), history_days, drift_start_frac, drift_strength
                )
            else:
                eff_affinity = affinity

            explore_mask = rng.random(n_plays) < novelty
            for is_explore in explore_mask:
                if is_explore:
                    genre = rng.choice(GENRES)
                else:
                    genre = rng.choice(GENRES, p=eff_affinity)
                candidates = tracks_by_genre.get(genre)
                if candidates is None or len(candidates) == 0:
                    continue
                track_id = rng.choice(candidates)
                play_ts = day_ts - timedelta(seconds=int(rng.integers(0, 86400)))
                rows.append((user["user_id"], play_ts, track_id))

    df = pd.DataFrame(rows, columns=["user_id", "timestamp", "track_id"])
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="data/user_profiles.parquet")
    parser.add_argument("--features", default="data/features.parquet")
    parser.add_argument("--history-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic_logs.parquet")
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Shift genre preference toward DRIFT_TARGET_AFFINITY over the most recent "
        "portion of history, so reference/current windows actually differ.",
    )
    parser.add_argument(
        "--drift-start-frac",
        type=float,
        default=0.5,
        help="Fraction of history_days (counting back from today) over which drift ramps "
        "in. 0.5 = drift begins at the window's midpoint, matching drift_report.py's "
        "median-timestamp split.",
    )
    parser.add_argument(
        "--drift-strength",
        type=float,
        default=0.7,
        help="Max blend weight (0-1) toward DRIFT_TARGET_AFFINITY, reached at day_offset=0.",
    )
    args = parser.parse_args()

    profiles = pd.read_parquet(args.profiles)
    features = pd.read_parquet(args.features)
    df = generate_logs(
        profiles,
        features,
        args.history_days,
        args.seed,
        inject_drift=args.inject_drift,
        drift_start_frac=args.drift_start_frac,
        drift_strength=args.drift_strength,
    )
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} plays across {df['user_id'].nunique()} users to {args.out}")
    print(f"timestamp range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(
        f"drift injected: {args.inject_drift}"
        + (
            f" (starts {args.drift_start_frac:.0%} into history, "
            f"strength {args.drift_strength:.2f})"
            if args.inject_drift
            else ""
        )
    )


if __name__ == "__main__":
    main()
