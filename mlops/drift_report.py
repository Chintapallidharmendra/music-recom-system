"""Drift monitoring for the live recommendation stream.

Reference data = the original synthetic listening history.
Current data = feedback events produced by the live simulator.
The monitor therefore compares the system's historical population with what is
actually happening online instead of splitting one static file in half.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

DRIFT_COLUMNS = [
    "genre",
    "tempo",
    "mfcc_mean_avg",
    "chroma_mean_avg",
    "contrast_mean_avg",
    "reward",
]
MIN_CURRENT_EVENTS = 100
RECENT_WINDOW_MINUTES = 3


def _flatten_features(features: pd.DataFrame) -> pd.DataFrame:
    flat = features[["track_id", "genre", "tempo"]].copy()
    flat["mfcc_mean_avg"] = features["mfcc_mean"].apply(np.mean)
    flat["chroma_mean_avg"] = features["chroma_mean"].apply(np.mean)
    flat["contrast_mean_avg"] = features["contrast_mean"].apply(np.mean)
    return flat


def _load_live_feedback(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["user_id", "timestamp", "track_id", "action", "reward"])

    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df


def _reference_frame(plays: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Historical baseline plus ground-truth expected reward."""
    from bandit.reward_simulator import RewardSimulator

    merged = plays.merge(_flatten_features(features), on="track_id", how="inner")
    sim = RewardSimulator()
    merged["reward"] = [
        sim.expected_reward(row.user_id, row.track_id) for row in merged.itertuples(index=False)
    ]
    return merged[DRIFT_COLUMNS]


def _current_frame(live: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = live.merge(_flatten_features(features), on="track_id", how="inner")
    return merged[DRIFT_COLUMNS]


# def build_windows(
#     plays: pd.DataFrame,
#     features: pd.DataFrame,
#     live_feedback: pd.DataFrame | None = None,
#     current_window: int = 2000,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """Return a fixed historical reference and the newest live interaction window."""
#     reference = _reference_frame(plays, features)

#     if live_feedback is not None and not live_feedback.empty:
#         live_feedback = live_feedback.sort_values("timestamp").tail(current_window)
#         current = _current_frame(live_feedback, features)
#         return reference, current

#     # Backward-compatible standalone behavior when no live event store exists yet.
#     merged = plays.merge(_flatten_features(features), on="track_id", how="inner")
#     merged = merged.sort_values("timestamp")
#     midpoint = len(merged) // 2
#     return merged.iloc[:midpoint][DRIFT_COLUMNS], merged.iloc[midpoint:][DRIFT_COLUMNS]


def build_windows(
    plays: pd.DataFrame,
    features: pd.DataFrame,
    live_feedback: pd.DataFrame | None = None,
    current_window: int = 2000,
    recent_window_minutes: int = RECENT_WINDOW_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a fixed historical reference and only the recent live interaction window.

    This intentionally ignores stale events from earlier simulator runs, so the DAG does not
    keep reporting drift after the stream has stopped.
    """

    reference = _reference_frame(plays, features)

    if live_feedback is not None and not live_feedback.empty:
        live_feedback = live_feedback.sort_values("timestamp").copy()

        if "timestamp" in live_feedback.columns:
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(
                minutes=recent_window_minutes
            )
            live_feedback = live_feedback[live_feedback["timestamp"] >= cutoff]

        live_feedback = live_feedback.tail(current_window)
        current = _current_frame(live_feedback, features)
        return reference, current

    return reference, pd.DataFrame(columns=DRIFT_COLUMNS)


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict, Report]:
    if len(current) < MIN_CURRENT_EVENTS:
        summary = {
            "dataset_drift": False,
            "number_of_drifted_columns": 0,
            "share_of_drifted_columns": 0.0,
            "monitoring_ready": False,
            "current_events": int(len(current)),
        }
        return summary, Report(metrics=[DataDriftPreset()])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()["metrics"][0]["result"]
    summary = {
        "dataset_drift": bool(result["dataset_drift"]),
        "number_of_drifted_columns": int(result["number_of_drifted_columns"]),
        "share_of_drifted_columns": float(result["share_of_drifted_columns"]),
        "monitoring_ready": True,
        "current_events": int(len(current)),
        "reference_events": int(len(reference)),
        "current_avg_reward": float(current["reward"].mean()),
        "reference_avg_reward": float(reference["reward"].mean()),
    }
    return summary, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plays", default="data/synthetic_logs.parquet")
    parser.add_argument("--features", default="data/features.parquet")
    parser.add_argument("--live", default="data/live_feedback.jsonl")
    parser.add_argument("--out", default="mlops/drift_report.html")
    parser.add_argument("--current-window", type=int, default=2000)
    parser.add_argument("--recent-window-minutes", type=int, default=RECENT_WINDOW_MINUTES)
    args = parser.parse_args()

    plays = pd.read_parquet(args.plays)
    features = pd.read_parquet(args.features)
    live = _load_live_feedback(args.live)
    reference, current = build_windows(
        plays,
        features,
        live,
        args.current_window,
        recent_window_minutes=args.recent_window_minutes,
    )

    summary, report = compute_drift(reference, current)
    print(f"reference window: {len(reference)} events, current window: {len(current)} events")
    print(summary)

    from mlops.tracking import log_drift_summary

    log_drift_summary(summary)

    if summary["monitoring_ready"]:
        report.save_html(args.out)
        print(f"saved HTML report to {args.out}")
    else:
        print(f"waiting for {MIN_CURRENT_EVENTS} live events before drift monitoring")


if __name__ == "__main__":
    main()
