"""Live drift monitor: consumes service/synthetic_play_producer.py's synthetic-plays
stream and periodically re-runs the same Evidently check as mlops/drift_report.py --
but against a reference window frozen early in the stream vs. a sliding current window
of the most recent plays, instead of a one-shot median split of a static file. This is
what actually makes drift observable in real time rather than requiring a batch
regenerate-then-rerun cycle.

Run alongside service/synthetic_play_producer.py --inject-drift:
    python -m mlops.drift_stream_monitor
"""
import argparse
import json
from collections import deque

import pandas as pd
from kafka import KafkaConsumer

from mlops.drift_report import DRIFT_COLUMNS, _flatten_features, compute_drift
from mlops.tracking import log_drift_summary
from service.synthetic_play_producer import SYNTHETIC_PLAYS_TOPIC


def run(bootstrap_servers: str, reference_size: int, window_size: int, check_every: int) -> None:
    features = pd.read_parquet("data/features.parquet")
    flat = _flatten_features(features).set_index("track_id")

    consumer = KafkaConsumer(
        SYNTHETIC_PLAYS_TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        # earliest + no group_id: each run replays the topic's current backlog (fast)
        # then continues live, so starting the monitor after the producer still catches
        # the full reference window instead of only whatever arrives from now on.
        auto_offset_reset="earliest",
    )

    reference_rows = []
    current_window = deque(maxlen=window_size)
    since_last_check = 0

    print(
        f"listening on '{SYNTHETIC_PLAYS_TOPIC}' -- filling reference window "
        f"({reference_size} plays) before checks start"
    )

    for message in consumer:
        event = message.value
        track_id = event["track_id"]
        if track_id not in flat.index:
            continue
        row = flat.loc[track_id, DRIFT_COLUMNS].to_dict()

        if len(reference_rows) < reference_size:
            reference_rows.append(row)
            if len(reference_rows) == reference_size:
                print(f"reference window filled ({reference_size} plays) -- checks starting")
            continue

        current_window.append(row)
        since_last_check += 1

        if len(current_window) < window_size or since_last_check < check_every:
            continue
        since_last_check = 0

        reference_df = pd.DataFrame(reference_rows)
        current_df = pd.DataFrame(list(current_window))
        summary, _ = compute_drift(reference_df, current_df)
        print(summary)
        log_drift_summary(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument(
        "--reference-size", type=int, default=300,
        help="Plays frozen as the reference distribution before checks start.",
    )
    parser.add_argument("--window-size", type=int, default=300, help="Sliding 'current' window size.")
    parser.add_argument(
        "--check-every", type=int, default=50,
        help="Re-run the drift check every N new plays once both windows are full.",
    )
    args = parser.parse_args()

    run(args.bootstrap_servers, args.reference_size, args.window_size, args.check_every)


if __name__ == "__main__":
    main()
