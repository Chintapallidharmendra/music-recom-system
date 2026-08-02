"""Evidently drift report: compares two time-windows of synthetic_logs.parquet's
listening feature distribution. Used standalone (prints + saves an HTML report) and by
mlops/dags/retrain_policy.py's check_drift task to decide whether retraining is warranted
(see PROJECT_PLAN.md's "Policy swap on drift" lifecycle).
"""
import argparse

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

DRIFT_COLUMNS = ["genre", "tempo", "mfcc_mean_avg", "chroma_mean_avg", "contrast_mean_avg"]


def _flatten_features(features: pd.DataFrame) -> pd.DataFrame:
    """Reduce array-valued feature columns to single scalar summaries -- Evidently's
    drift metrics operate on flat tabular columns, not array cells."""
    flat = features[["track_id", "genre", "tempo"]].copy()
    flat["mfcc_mean_avg"] = features["mfcc_mean"].apply(np.mean)
    flat["chroma_mean_avg"] = features["chroma_mean"].apply(np.mean)
    flat["contrast_mean_avg"] = features["contrast_mean"].apply(np.mean)
    return flat


def build_windows(
    plays: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split at the median play timestamp: older half = reference,
    newer half = current."""
    merged = plays.merge(_flatten_features(features), on="track_id", how="inner")
    merged = merged.sort_values("timestamp")
    midpoint = len(merged) // 2
    reference = merged.iloc[:midpoint][DRIFT_COLUMNS]
    current = merged.iloc[midpoint:][DRIFT_COLUMNS]
    return reference, current


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()["metrics"][0]["result"]
    return {
        "dataset_drift": bool(result["dataset_drift"]),
        "number_of_drifted_columns": int(result["number_of_drifted_columns"]),
        "share_of_drifted_columns": float(result["share_of_drifted_columns"]),
    }, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plays", default="data/synthetic_logs.parquet")
    parser.add_argument("--features", default="data/features.parquet")
    parser.add_argument("--out", default="mlops/drift_report.html")
    args = parser.parse_args()

    plays = pd.read_parquet(args.plays)
    features = pd.read_parquet(args.features)
    reference, current = build_windows(plays, features)

    summary, report = compute_drift(reference, current)
    print(f"reference window: {len(reference)} plays, current window: {len(current)} plays")
    print(summary)

    from mlops.tracking import log_drift_summary

    log_drift_summary(summary)

    report.save_html(args.out)
    print(f"saved HTML report to {args.out}")


if __name__ == "__main__":
    main()
