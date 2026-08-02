"""Drift-triggered retrain -> canary -> resolve lifecycle (PROJECT_PLAN.md's "Policy
swap on drift"). Runs hourly. The swap is always a full-population replacement of the
one serving policy -- service/main.py's registry poller is what actually flips traffic,
this DAG only decides Staging vs Production vs Archived in the MLflow Model Registry.

FORCE_RETRAIN=true bypasses the drift check for demo purposes -- our synthetic
interaction data is stationary (no simulated preference shift over time), so the real
drift check will legitimately almost never fire; this lets the retrain->canary->resolve
path still be exercised end-to-end without waiting for organic drift.
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from mlops.tracking import POLICY_MODEL_NAME as MODEL_NAME

CANARY_MARGIN = 0.05  # candidate must beat Production's avg_reward by this much to be promoted
CANDIDATE_ARTIFACT_DIR = "/tmp/music_bandit_artifacts"


def _collect(**context) -> None:
    # Production would pull fresh interaction data from the user-feedback topic here.
    # Our interaction data is the static synthetic dataset from data/generate_synthetic_logs.py,
    # so "collect" just confirms the expected inputs exist before the rest of the DAG runs.
    assert os.path.exists("data/synthetic_logs.parquet"), "synthetic_logs.parquet missing"
    assert os.path.exists("data/features.parquet"), "features.parquet missing"


def _check_drift(**context) -> str:
    from mlops.drift_report import build_windows, compute_drift
    from mlops.tracking import log_drift_summary

    plays = pd.read_parquet("data/synthetic_logs.parquet")
    features = pd.read_parquet("data/features.parquet")
    reference, current = build_windows(plays, features)
    summary, _ = compute_drift(reference, current)
    log_drift_summary(summary)
    context["ti"].xcom_push(key="drift_summary", value=summary)

    force = os.environ.get("FORCE_RETRAIN", "false").lower() == "true"
    return "update_features" if (summary["dataset_drift"] or force) else "no_drift"


def _update_features(**context) -> None:
    # Full re-extraction (data/extract_features.py) is the plan's flagged multi-hour step;
    # re-running it on every DAG firing isn't the point of this task. A real deployment
    # would re-run extraction only when new tracks are ingested, not on every retrain.
    features = pd.read_parquet("data/features.parquet")
    assert len(features) > 0


def _retrain(**context) -> None:
    from bandit.policies.linucb import LinUCBPolicy
    from bandit.replay_evaluator import generate_replay_log, replay_evaluate
    from bandit.reward_simulator import RewardSimulator

    seed = int(datetime.utcnow().timestamp())
    log = generate_replay_log(n_events=8000, pool_size=15, seed=seed)
    policy = LinUCBPolicy(alpha=1.0, context_dim=8)
    metrics = replay_evaluate(policy, log, RewardSimulator())

    os.makedirs(CANDIDATE_ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(CANDIDATE_ARTIFACT_DIR, f"candidate_{context['run_id']}.pkl")
    with open(path, "wb") as f:
        pickle.dump(policy, f)

    context["ti"].xcom_push(key="candidate_metrics", value=metrics)
    context["ti"].xcom_push(key="candidate_path", value=path)


def _evaluate(**context) -> None:
    metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="retrain")
    if metrics["avg_reward"] < -1.0:  # sanity floor before it's even allowed into Staging
        raise ValueError(f"candidate policy failed sanity check: {metrics}")


def _register(**context) -> None:
    from mlops.tracking import register_policy

    path = context["ti"].xcom_pull(key="candidate_path", task_ids="retrain")
    metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="retrain")
    version = register_policy(path, metrics, MODEL_NAME)
    context["ti"].xcom_push(key="candidate_version", value=version)


def _evaluate_canary(**context) -> str:
    from mlops.tracking import get_latest_metrics_by_stage

    candidate_metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="retrain")
    production_metrics = get_latest_metrics_by_stage(MODEL_NAME, "Production")
    context["ti"].xcom_push(key="production_metrics", value=production_metrics)

    if production_metrics is None:
        return "promote"  # first run ever -- nothing to compare against, promote unconditionally
    if candidate_metrics["avg_reward"] > production_metrics["avg_reward"] + CANARY_MARGIN:
        return "promote"
    return "rollback"


def _promote(**context) -> None:
    from mlops.tracking import promote_to_production

    version = context["ti"].xcom_pull(key="candidate_version", task_ids="register")
    promote_to_production(MODEL_NAME, version)


def _rollback(**context) -> None:
    from mlops.tracking import archive_version

    version = context["ti"].xcom_pull(key="candidate_version", task_ids="register")
    archive_version(MODEL_NAME, version)


default_args = {"owner": "music-bandit", "retries": 0}

with DAG(
    dag_id="retrain_policy",
    default_args=default_args,
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["music-bandit"],
) as dag:
    collect = PythonOperator(task_id="collect", python_callable=_collect)
    check_drift = BranchPythonOperator(task_id="check_drift", python_callable=_check_drift)
    no_drift = EmptyOperator(task_id="no_drift")

    update_features = PythonOperator(task_id="update_features", python_callable=_update_features)
    retrain = PythonOperator(task_id="retrain", python_callable=_retrain)
    evaluate = PythonOperator(task_id="evaluate", python_callable=_evaluate)
    register = PythonOperator(task_id="register", python_callable=_register)
    evaluate_canary = BranchPythonOperator(
        task_id="evaluate_canary", python_callable=_evaluate_canary
    )
    promote = PythonOperator(task_id="promote", python_callable=_promote)
    rollback = PythonOperator(task_id="rollback", python_callable=_rollback)

    collect >> check_drift >> [update_features, no_drift]
    update_features >> retrain >> evaluate >> register >> evaluate_canary >> [promote, rollback]
