"""Live-data drift -> candidate training -> canary -> Production/Archive.

Unlike the original DAG, this version never creates a fresh synthetic replay log when
retraining. It learns from the feedback events produced by the running service.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from mlops.tracking import POLICY_MODEL_NAME as MODEL_NAME

LIVE_FEEDBACK_PATH = os.environ.get("LIVE_FEEDBACK_PATH", "/mlruns/live_feedback.jsonl")
CANARY_MARGIN = 0.05
CANDIDATE_ARTIFACT_DIR = "/tmp/music_bandit_artifacts"
MIN_LIVE_EVENTS = 100
EVAL_USERS = 100
POOL_SIZE = 15


def _load_live_feedback() -> pd.DataFrame:
    if not os.path.exists(LIVE_FEEDBACK_PATH):
        return pd.DataFrame(columns=["user_id", "timestamp", "track_id", "action", "reward"])

    rows = []
    with open(LIVE_FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def _collect(**context) -> None:
    live = _load_live_feedback()
    assert os.path.exists("data/features.parquet"), "features.parquet missing"
    assert os.path.exists("data/synthetic_logs.parquet"), "synthetic_logs.parquet missing"
    context["ti"].xcom_push(key="live_event_count", value=len(live))


# def _check_drift(**context) -> str:
#     from mlops.drift_report import build_windows, compute_drift
#     from mlops.tracking import log_drift_summary

#     plays = pd.read_parquet("data/synthetic_logs.parquet")
#     features = pd.read_parquet("data/features.parquet")
#     live = _load_live_feedback()

#     reference, current = build_windows(plays, features, live_feedback=live)
#     summary, _ = compute_drift(reference, current)
#     log_drift_summary(summary)
#     context["ti"].xcom_push(key="drift_summary", value=summary)

#     force = os.environ.get("FORCE_RETRAIN", "false").lower() == "true"
#     drift = summary.get("dataset_drift", False)
#     ready = summary.get("monitoring_ready", False)
#     return "update_features" if (force or (ready and drift)) else "no_drift"

def _check_drift(**context) -> str:
    from mlops.drift_report import build_windows, compute_drift
    from mlops.tracking import log_drift_summary

    plays = pd.read_parquet("data/synthetic_logs.parquet")
    features = pd.read_parquet("data/features.parquet")
    live = _load_live_feedback()

    reference, current = build_windows(
        plays,
        features,
        live_feedback=live,
    )

    summary, _ = compute_drift(reference, current)

    # ---------------------------------------------------------
    # Print drift information explicitly into Airflow logs
    # ---------------------------------------------------------

    print("=" * 70)
    print("DRIFT MONITORING RESULT")
    print("=" * 70)

    print(f"Live feedback events : {len(live)}")
    print(f"Reference rows       : {len(reference)}")
    print(f"Current rows         : {len(current)}")

    print(f"Monitoring ready     : {summary.get('monitoring_ready')}")
    print(f"Dataset drift        : {summary.get('dataset_drift')}")
    print(
        f"Drifted columns      : "
        f"{summary.get('number_of_drifted_columns')}"
    )
    print(
        f"Drift share           : "
        f"{summary.get('share_of_drifted_columns')}"
    )

    print("-" * 70)
    print("Complete drift summary:")
    print(json.dumps(summary, indent=2, default=str))

    force = os.environ.get("FORCE_RETRAIN", "false").lower() == "true"

    drift = bool(summary.get("dataset_drift", False))
    ready = bool(summary.get("monitoring_ready", False))

    print("-" * 70)
    print(f"FORCE_RETRAIN         : {force}")
    print(f"DRIFT DETECTED        : {drift}")
    print(f"MONITORING READY      : {ready}")

    if force:
        decision = "update_features"
        reason = "FORCE_RETRAIN=true"
    elif ready and drift:
        decision = "update_features"
        reason = "monitoring_ready=true AND dataset_drift=true"
    else:
        decision = "no_drift"
        reason = "No retraining condition satisfied"

    print(f"BRANCH DECISION       : {decision}")
    print(f"REASON                : {reason}")
    print("=" * 70)

    # Save in Airflow XCom
    context["ti"].xcom_push(
        key="drift_summary",
        value=summary,
    )

    context["ti"].xcom_push(
        key="drift_detected",
        value=drift,
    )

    context["ti"].xcom_push(
        key="monitoring_ready",
        value=ready,
    )

    context["ti"].xcom_push(
        key="drifted_columns",
        value=summary.get("number_of_drifted_columns", 0),
    )

    # MLflow
    log_drift_summary(summary)

    return decision


def _update_features(**context) -> None:
    # Features are already materialized. New tracks would trigger extraction separately;
    # user-behavior drift does not require re-extracting audio features.
    features = pd.read_parquet("data/features.parquet")
    assert len(features) > 0


def _train_from_live_feedback(policy, live, plays, features) -> int:
    """Reconstruct the online learning sequence exactly as it happened in production."""
    from data.build_user_context import build_user_context

    live_history: dict[str, list[dict]] = {}
    updates = 0

    for row in live.itertuples(index=False):
        user_id = str(row.user_id)
        user_snapshot = {k: list(v) for k, v in live_history.items()}
        context = build_user_context(
            user_id,
            plays,
            features,
            live_plays=user_snapshot,
        )
        policy.update(str(row.track_id), context, float(row.reward))
        live_history.setdefault(user_id, []).append(
            {
                "user_id": user_id,
                "timestamp": row.timestamp.isoformat(),
                "track_id": str(row.track_id),
            }
        )
        updates += 1

    return updates


def _evaluate_policy(policy, users, candidate_pool, plays, features) -> dict:
    from bandit.reward_simulator import RewardSimulator
    from data.build_user_context import build_user_context

    reward_sim = RewardSimulator()
    rewards = []
    for user_id in users:
        context = build_user_context(user_id, plays, features)
        chosen = policy.select_action(context, candidate_pool)
        rewards.append(reward_sim.expected_reward(user_id, chosen))

    rewards = np.asarray(rewards, dtype=float)
    return {
        "eval_users": int(len(rewards)),
        "avg_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "ctr": float(np.mean(rewards > 0)) if len(rewards) else 0.0,
        "cumulative_reward": float(rewards.sum()),
        "cumulative_regret": 0.0,
    }


def _retrain_policy(policy_name: str, **context) -> None:
    if policy_name == "LinUCB":
        from bandit.policies.linucb import LinUCBPolicy
        policy = LinUCBPolicy(alpha=1.0, context_dim=8)
    elif policy_name == "LinTS":
        from bandit.policies.linear_thompson_sampling import LinearThompsonSamplingPolicy
        policy = LinearThompsonSamplingPolicy(v=0.3, seed=42, context_dim=8)
    else:
        raise ValueError(f"unsupported drift retrain policy: {policy_name}")

    live = _load_live_feedback()
    if len(live) < MIN_LIVE_EVENTS:
        raise ValueError(f"only {len(live)} live events; need {MIN_LIVE_EVENTS}")

    plays = pd.read_parquet("data/synthetic_logs.parquet")
    features = pd.read_parquet("data/features.parquet")

    updates = _train_from_live_feedback(policy, live, plays, features)

    rng = np.random.default_rng(42)
    users = pd.read_parquet("data/user_profiles.parquet")["user_id"].to_numpy()
    users = rng.choice(users, size=min(EVAL_USERS, len(users)), replace=False)
    candidate_pool = list(
        rng.choice(features["track_id"].to_numpy(), size=POOL_SIZE, replace=False)
    )

    metrics = _evaluate_policy(policy, users, candidate_pool, plays, features)
    metrics["training_events"] = updates

    os.makedirs(CANDIDATE_ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(CANDIDATE_ARTIFACT_DIR, f"candidate_{policy_name}_{context['run_id']}.pkl")
    with open(path, "wb") as f:
        pickle.dump(policy, f)

    context["ti"].xcom_push(key="candidate_metrics", value=metrics)
    context["ti"].xcom_push(key="candidate_path", value=path)
    context["ti"].xcom_push(key="candidate_policy_name", value=policy_name)


def _select_best_candidate(**context) -> None:
    linucb_metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="retrain_linucb")
    lints_metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="retrain_lints")
    linucb_path = context["ti"].xcom_pull(key="candidate_path", task_ids="retrain_linucb")
    lints_path = context["ti"].xcom_pull(key="candidate_path", task_ids="retrain_lints")

    if linucb_metrics["avg_reward"] >= lints_metrics["avg_reward"]:
        selected_policy = "LinUCB"
        selected_metrics = linucb_metrics
        selected_path = linucb_path
    else:
        selected_policy = "LinTS"
        selected_metrics = lints_metrics
        selected_path = lints_path

    context["ti"].xcom_push(key="candidate_metrics", value=selected_metrics)
    context["ti"].xcom_push(key="candidate_path", value=selected_path)
    context["ti"].xcom_push(key="candidate_policy_name", value=selected_policy)


def _evaluate(**context) -> None:
    metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="select_best_candidate")
    if metrics["avg_reward"] < -1.0:
        raise ValueError(f"candidate policy failed sanity check: {metrics}")


def _register(**context) -> None:
    from mlops.tracking import register_policy

    path = context["ti"].xcom_pull(key="candidate_path", task_ids="select_best_candidate")
    metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="select_best_candidate")
    policy_name = context["ti"].xcom_pull(key="candidate_policy_name", task_ids="select_best_candidate")
    version = register_policy(path, metrics, MODEL_NAME, policy_name=policy_name)
    context["ti"].xcom_push(key="candidate_version", value=version)


def _evaluate_canary(**context) -> str:
    from mlops.tracking import get_latest_metrics_by_stage

    candidate_metrics = context["ti"].xcom_pull(key="candidate_metrics", task_ids="retrain")
    production_metrics = get_latest_metrics_by_stage(MODEL_NAME, "Production")
    context["ti"].xcom_push(key="production_metrics", value=production_metrics)

    if production_metrics is None:
        return "promote"
    if candidate_metrics["avg_reward"] > production_metrics.get("avg_reward", -999.0) + CANARY_MARGIN:
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
    schedule_interval=timedelta(minutes=15),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["music-bandit", "live-drift"],
) as dag:
    collect = PythonOperator(task_id="collect", python_callable=_collect)
    check_drift = BranchPythonOperator(task_id="check_drift", python_callable=_check_drift)
    no_drift = EmptyOperator(task_id="no_drift")
    update_features = PythonOperator(task_id="update_features", python_callable=_update_features)
    retrain_linucb = PythonOperator(
        task_id="retrain_linucb",
        python_callable=_retrain_policy,
        op_kwargs={"policy_name": "LinUCB"},
    )
    retrain_lints = PythonOperator(
        task_id="retrain_lints",
        python_callable=_retrain_policy,
        op_kwargs={"policy_name": "LinTS"},
    )
    select_best_candidate = PythonOperator(task_id="select_best_candidate", python_callable=_select_best_candidate)
    evaluate = PythonOperator(task_id="evaluate", python_callable=_evaluate)
    register = PythonOperator(task_id="register", python_callable=_register)
    evaluate_canary = BranchPythonOperator(task_id="evaluate_canary", python_callable=_evaluate_canary)
    promote = PythonOperator(task_id="promote", python_callable=_promote)
    rollback = PythonOperator(task_id="rollback", python_callable=_rollback)

    collect >> check_drift >> [update_features, no_drift]
    update_features >> [retrain_linucb, retrain_lints] >> select_best_candidate >> evaluate >> register >> evaluate_canary >> [promote, rollback]
