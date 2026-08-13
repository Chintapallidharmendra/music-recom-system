"""MLflow helpers. Wired into the two things that actually produce runs worth logging:
bandit/compare_policies.py (one run per offline-replay policy evaluation) and
service/main.py (one run per service lifetime, with periodic live-metric snapshots) --
plus the Model Registry helpers mlops/dags/retrain_policy.py and service/main.py's policy
poller use to drive the drift-triggered canary lifecycle (see PROJECT_PLAN.md).
"""
import os
import pickle
import threading

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "music-bandit"
POLICY_MODEL_NAME = "music-bandit-policy"


def init_tracking() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_drift_summary(summary: dict) -> None:
    """One MLflow run per drift check (standalone mlops/drift_report.py run or the
    DAG's check_drift task) -- lets mlops/dashboard.py read drift history purely from
    MLflow rather than recomputing or re-reading the HTML report."""
    init_tracking()
    with mlflow.start_run(run_name="drift_check"):
        mlflow.set_tag("stage", "drift_monitoring")
        mlflow.log_metric("dataset_drift", float(summary["dataset_drift"]))
        mlflow.log_metric("number_of_drifted_columns", summary["number_of_drifted_columns"])
        mlflow.log_metric("share_of_drifted_columns", summary["share_of_drifted_columns"])


def log_replay_evaluation(policy_name: str, params: dict, metrics: dict) -> None:
    """One MLflow run per policy evaluated in bandit/compare_policies.py."""
    init_tracking()
    with mlflow.start_run(run_name=f"replay_{policy_name}"):
        mlflow.set_tag("policy", policy_name)
        mlflow.set_tag("stage", "offline_replay")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


class ServiceMetricsLogger:
    """Keeps one MLflow run open for the service's lifetime and periodically logs a
    snapshot of the running CTR/reward/regret metrics as a time series (mlflow's
    log_metrics with an increasing `step` gives a curve over the service's uptime)."""

    def __init__(self, policy_name: str, snapshot_fn, interval_seconds: float = 30.0):
        self._policy_name = policy_name
        self._snapshot_fn = snapshot_fn
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None
        self._run = None
        self._step = 0

    def start(self) -> None:
        init_tracking()
        self._run = mlflow.start_run(run_name=f"service_{self._policy_name}")
        mlflow.set_tag("policy", self._policy_name)
        mlflow.set_tag("stage", "online_serving")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._run is not None:
            mlflow.end_run()

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                metrics = self._snapshot_fn()
                mlflow.log_metrics(metrics, step=self._step)
                self._step += 1
            except Exception:  # noqa: BLE001 -- logging must never take the service down
                pass


# --- Model Registry: drift-triggered retrain -> canary -> resolve lifecycle ---
# Policies are plain Python objects (not sklearn/pytorch), so they're pickled and logged
# as a run artifact, then registered against that artifact path -- simpler than writing a
# custom mlflow.pyfunc flavor for something only our own code ever loads back.
# Uses the classic Staging/Production/Archived stage model (mirrors PROJECT_PLAN.md's own
# vocabulary); mlflow 2.14 deprecates stages in favor of aliases but they remain functional.


def register_policy(artifact_path: str, metrics: dict, model_name: str) -> int:
    """Logs the pickled policy + its offline-replay metrics as a run, registers that run's
    artifact as a new model version, and stages it as Staging (it hasn't earned
    Production yet -- see PROJECT_PLAN.md step 2)."""
    init_tracking()
    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        mlflow.set_tag("stage", "candidate")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(artifact_path, artifact_path="policy")
        model_uri = f"runs:/{run.info.run_id}/policy"
        result = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.transition_model_version_stage(model_name, result.version, stage="Staging")
    return int(result.version)


def get_latest_metrics_by_stage(model_name: str, stage: str) -> dict | None:
    """Metrics logged on the run behind the latest model version in `stage`, or None if
    no version currently holds that stage (e.g. no Production model registered yet)."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
    except Exception:  # noqa: BLE001 -- model_name may not exist yet on first run
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return dict(run.data.metrics)


def promote_to_production(model_name: str, version: int) -> None:
    """Archives whatever currently holds Production, then promotes `version` to it --
    always a full swap of the single serving policy, never a partial one."""
    client = MlflowClient()
    for v in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(model_name, v.version, stage="Archived")
    client.transition_model_version_stage(model_name, version, stage="Production")


def archive_version(model_name: str, version: int) -> None:
    """Rollback path: the candidate didn't beat Production, discard it."""
    client = MlflowClient()
    client.transition_model_version_stage(model_name, version, stage="Archived")


def load_policy_by_stage(model_name: str, stage: str):
    """Returns (policy, version) for the latest model version in `stage`, or
    (None, None) if nothing currently holds that stage."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
    except Exception:  # noqa: BLE001 -- model_name may not exist yet on first run
        return None, None
    if not versions:
        return None, None

    version = versions[0]
    local_dir = mlflow.artifacts.download_artifacts(f"runs:/{version.run_id}/policy")
    pkl_name = next(f for f in os.listdir(local_dir) if f.endswith(".pkl"))
    with open(os.path.join(local_dir, pkl_name), "rb") as f:
        policy = pickle.load(f)
    return policy, int(version.version)
