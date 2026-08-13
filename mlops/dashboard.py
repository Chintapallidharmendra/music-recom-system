"""Streamlit dashboard: pure consumer of the live service's /metrics endpoint and
MLflow's tracked runs -- built last since nothing downstream depends on it.

Run with: streamlit run mlops/dashboard.py
"""
import os

import httpx
import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

from mlops.tracking import EXPERIMENT_NAME, init_tracking

SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")

st.set_page_config(page_title="music-bandit dashboard", layout="wide")
st.title("Music-Bandit Dashboard")


def _fetch_live_metrics() -> dict | None:
    try:
        resp = httpx.get(f"{SERVICE_URL}/metrics", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001 -- service may not be running during offline review
        return None


def _fetch_runs(stage_tag: str) -> pd.DataFrame:
    init_tracking()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.stage = '{stage_tag}'",
        order_by=["start_time DESC"],
    )
    return runs


st.header("Live service metrics")
live = _fetch_live_metrics()
if live is None:
    st.warning(f"Service not reachable at {SERVICE_URL} -- start it and refresh.")
else:
    cols = st.columns(4)
    cols[0].metric("CTR", f"{live['ctr']:.3f}")
    cols[1].metric("Avg reward", f"{live['avg_reward']:.3f}")
    cols[2].metric("Cumulative regret", f"{live['cumulative_regret']:.2f}")
    cols[3].metric("Total recommendations", live["total_recommendations"])

st.header("Offline policy comparison (bandit/compare_policies.py)")
replay_runs = _fetch_runs("offline_replay")
if replay_runs.empty:
    st.info("No offline replay runs logged yet -- run `python -m bandit.compare_policies`.")
else:
    latest_run_time = replay_runs["start_time"].max()
    latest = replay_runs[replay_runs["start_time"] == latest_run_time]
    table = latest[[
        "tags.policy", "metrics.ctr", "metrics.avg_reward", "metrics.cumulative_regret",
    ]].rename(columns={
        "tags.policy": "policy", "metrics.ctr": "ctr",
        "metrics.avg_reward": "avg_reward", "metrics.cumulative_regret": "cumulative_regret",
    })
    st.dataframe(table, use_container_width=True)
    fig = px.bar(table, x="policy", y="cumulative_regret", title="Cumulative regret by policy")
    st.plotly_chart(fig, use_container_width=True)

st.header("Drift monitoring (mlops/drift_report.py)")
drift_runs = _fetch_runs("drift_monitoring")
if drift_runs.empty:
    st.info("No drift checks logged yet -- run `python -m mlops.drift_report`.")
else:
    drift_table = drift_runs[[
        "start_time", "metrics.dataset_drift", "metrics.number_of_drifted_columns",
    ]].rename(columns={
        "metrics.dataset_drift": "dataset_drift",
        "metrics.number_of_drifted_columns": "number_of_drifted_columns",
    })
    st.dataframe(drift_table, use_container_width=True)
