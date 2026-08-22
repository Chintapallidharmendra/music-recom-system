"""Build per-user context vectors from synthetic listening history.

context = [recency-weighted genre affinity (8)] + [mean audio features of last N plays].
Cold-start users (no history) get a uniform genre vector + zero audio vector -- never raises
on an unknown user_id.
"""

import numpy as np
import pandas as pd

from data.synth_user_profiles import GENRES

HALF_LIFE_DAYS = 30.0
SESSION_N = 10

# audio feature vector length: mfcc_mean(20) + mfcc_var(20) + chroma_mean(12) + tempo(1)
# + contrast_mean(7)
AUDIO_FEATURE_DIM = 20 + 20 + 12 + 1 + 7
CONTEXT_DIM = len(GENRES) + AUDIO_FEATURE_DIM


def _audio_vector(row: pd.Series) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(row["mfcc_mean"], dtype=np.float64),
            np.asarray(row["mfcc_var"], dtype=np.float64),
            np.asarray(row["chroma_mean"], dtype=np.float64),
            [float(row["tempo"])],
            np.asarray(row["contrast_mean"], dtype=np.float64),
        ]
    )


def default_context() -> np.ndarray:
    """Neutral vector for cold-start users: uniform genre prior, zero audio signal."""
    genre_vec = np.full(len(GENRES), 1.0 / len(GENRES))
    audio_vec = np.zeros(AUDIO_FEATURE_DIM)
    return np.concatenate([genre_vec, audio_vec]).astype(np.float32)


def build_user_context(
    user_id: str,
    plays: pd.DataFrame,
    features: pd.DataFrame,
    half_life_days: float = HALF_LIFE_DAYS,
    session_n: int = SESSION_N,
    live_plays: dict | pd.DataFrame | None = None,
) -> np.ndarray:
    """Recency-weighted genre affinity + session audio average. Never raises on an
    unknown/cold-start user_id -- returns default_context() instead."""
    # Start from immutable historical data, then overlay interactions produced by the
    # live simulator. This keeps the original dataset unchanged while allowing the
    # next recommendation to reflect recent behavior.
    user_plays = plays[plays["user_id"] == user_id].copy()

    if isinstance(live_plays, dict):
        rows = live_plays.get(user_id, [])
        if rows:
            live_df = pd.DataFrame(rows)
            live_df["timestamp"] = pd.to_datetime(live_df["timestamp"], utc=True).dt.tz_localize(
                None
            )
            live_df = live_df[["user_id", "timestamp", "track_id"]]
            user_plays = pd.concat([user_plays, live_df], ignore_index=True)
    elif isinstance(live_plays, pd.DataFrame) and not live_plays.empty:
        live_df = live_plays[live_plays["user_id"] == user_id].copy()
        if not live_df.empty:
            live_df["timestamp"] = pd.to_datetime(live_df["timestamp"], utc=True).dt.tz_localize(
                None
            )
            user_plays = pd.concat(
                [user_plays, live_df[["user_id", "timestamp", "track_id"]]], ignore_index=True
            )

    if user_plays.empty:
        return default_context()

    user_plays["timestamp"] = pd.to_datetime(user_plays["timestamp"], utc=True).dt.tz_localize(None)

    features_by_id = features.set_index("track_id")
    genre_by_track = features_by_id["genre"]

    now = user_plays["timestamp"].max()
    age_days = (now - user_plays["timestamp"]).dt.total_seconds() / 86400.0
    weight = 0.5 ** (age_days / half_life_days)

    genre = user_plays["track_id"].map(genre_by_track)
    valid = genre.notna()
    genre_weight = weight[valid].groupby(genre[valid]).sum()
    genre_vec = np.array([genre_weight.get(g, 0.0) for g in GENRES], dtype=np.float64)
    total = genre_vec.sum()
    genre_vec = genre_vec / total if total > 0 else np.full(len(GENRES), 1.0 / len(GENRES))

    recent_ids = user_plays.sort_values("timestamp")["track_id"].tail(session_n)
    recent_feats = features_by_id.reindex(recent_ids).dropna(subset=["mfcc_mean"])
    if len(recent_feats) == 0:
        audio_vec = np.zeros(AUDIO_FEATURE_DIM)
    else:
        audio_vec = np.mean([_audio_vector(r) for _, r in recent_feats.iterrows()], axis=0)

    return np.concatenate([genre_vec, audio_vec]).astype(np.float32)


if __name__ == "__main__":
    plays = pd.read_parquet("data/synthetic_logs.parquet")
    features = pd.read_parquet("data/features.parquet")

    warm_user = plays["user_id"].iloc[0]
    ctx_warm = build_user_context(warm_user, plays, features)
    ctx_cold = build_user_context("user_does_not_exist", plays, features)

    assert ctx_warm.shape == (CONTEXT_DIM,), ctx_warm.shape
    assert ctx_cold.shape == (CONTEXT_DIM,), ctx_cold.shape
    print(f"context dim: {CONTEXT_DIM}")
    print(f"warm user ({warm_user}) context[:8] (genre affinity): {ctx_warm[:8]}")
    print(f"cold-start context[:8] (genre affinity): {ctx_cold[:8]}")
