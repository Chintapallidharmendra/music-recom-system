"""Continuous synthetic play-event producer: a live-streaming twin of
data/generate_synthetic_logs.py's --inject-drift mode, publishing one Kafka event per
play instead of writing a static parquet file. Lets mlops/drift_stream_monitor.py (or
any Evidently-based consumer) observe the same genre-affinity drift toward
DRIFT_TARGET_AFFINITY *live*, ramping over wall-clock seconds instead of synthetic days.

NOTE: publishes to its own topic (SYNTHETIC_PLAYS_TOPIC below), separate from
recommendation-events/user-feedback (contracts/kafka_topics.md, not included in this
upload) since it's simulating raw listening history, not actual policy-served
recommendations. Check that file / with your team before reusing this topic name if the
topic list is meant to be frozen.
"""
import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

from data.generate_synthetic_logs import DRIFT_TARGET_AFFINITY
from data.synth_user_profiles import GENRES

logger = logging.getLogger(__name__)

SYNTHETIC_PLAYS_TOPIC = "synthetic-plays"


def _effective_affinity(
    affinity: np.ndarray, elapsed_seconds: float, ramp_seconds: float, drift_strength: float
) -> np.ndarray:
    """Blend toward DRIFT_TARGET_AFFINITY as elapsed_seconds -> ramp_seconds."""
    w = drift_strength if ramp_seconds <= 0 else drift_strength * min(1.0, elapsed_seconds / ramp_seconds)
    blended = (1 - w) * affinity + w * DRIFT_TARGET_AFFINITY
    return blended / blended.sum()


def run(
    bootstrap_servers: str,
    interval_seconds: float,
    ramp_seconds: float,
    drift_strength: float,
    inject_drift: bool,
    seed: int,
) -> None:
    profiles = pd.read_parquet("data/user_profiles.parquet")
    features = pd.read_parquet("data/features.parquet")
    tracks_by_genre = {g: sub["track_id"].to_numpy() for g, sub in features.groupby("genre")}

    rng = np.random.default_rng(seed)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        request_timeout_ms=3000,
    )

    start = time.monotonic()
    n_sent = 0
    drift_desc = f"on, ramping over {ramp_seconds:.0f}s" if inject_drift else "off"
    print(f"streaming synthetic plays to '{SYNTHETIC_PLAYS_TOPIC}' (drift={drift_desc})")

    try:
        while True:
            elapsed = time.monotonic() - start
            user = profiles.iloc[rng.integers(len(profiles))]
            affinity = np.asarray(user["genre_affinity"], dtype=np.float64)
            affinity = affinity / affinity.sum()
            novelty = float(user["novelty_bias"])

            eff_affinity = (
                _effective_affinity(affinity, elapsed, ramp_seconds, drift_strength)
                if inject_drift
                else affinity
            )

            if rng.random() < novelty:
                genre = rng.choice(GENRES)
            else:
                genre = rng.choice(GENRES, p=eff_affinity)

            candidates = tracks_by_genre.get(genre)
            if candidates is None or len(candidates) == 0:
                continue
            track_id = str(rng.choice(candidates))

            event = {
                "user_id": user["user_id"],
                "track_id": track_id,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
            try:
                producer.send(SYNTHETIC_PLAYS_TOPIC, event)
            except KafkaError as exc:
                logger.warning("send failed (continuing): %s", exc)

            n_sent += 1
            if n_sent % 50 == 0:
                producer.flush(timeout=3)
                print(f"sent {n_sent} plays (elapsed {elapsed:.0f}s)")

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush(timeout=3)
        print(f"stopped after {n_sent} plays")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument(
        "--interval-seconds", type=float, default=0.2,
        help="Sleep between events -- lower = higher throughput.",
    )
    parser.add_argument("--inject-drift", action="store_true")
    parser.add_argument(
        "--ramp-seconds", type=float, default=120.0,
        help="Wall-clock seconds for drift to reach full strength (default 2 min, "
        "so you can watch it happen during a demo).",
    )
    parser.add_argument("--drift-strength", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(
        args.bootstrap_servers,
        args.interval_seconds,
        args.ramp_seconds,
        args.drift_strength,
        args.inject_drift,
        args.seed,
    )


if __name__ == "__main__":
    main()
