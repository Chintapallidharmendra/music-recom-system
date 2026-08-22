"""Central Prometheus metric definitions.

Two kinds of processes need metrics here:

1. The long-running FastAPI service (service/main.py) -- these metrics are scraped
   straight off its /metrics/prometheus endpoint (see monitoring.metrics.metrics_asgi_app).
   The existing `/metrics` route in service/main.py is part of the frozen JSON API
   contract (contracts/openapi_notes.md) and is left untouched; Prometheus gets its own
   path instead of overloading that one.

2. Short-lived batch jobs (mlops/drift_report.py, mlops/drift_stream_monitor.py's periodic
   checks) -- these processes exit before Prometheus could ever scrape them, so they push
   their metrics to a Prometheus Pushgateway instead (see `push_drift_metrics`).

Import this module and use the metric objects directly; don't redefine metrics elsewhere,
duplicate metric names crash on process start (prometheus_client raises on double
registration).
"""
import logging
import os
import time
from contextlib import contextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)

PUSHGATEWAY_URL = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", "localhost:9091")

# --- HTTP layer -------------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the service",
    ["method", "path", "status"],
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

# --- Bandit / recommendation business metrics --------------------------------------
RECOMMENDATIONS_TOTAL = Counter(
    "recommendations_total",
    "Recommendations served, by which policy served them",
    ["policy"],
)
FEEDBACK_EVENTS_TOTAL = Counter(
    "feedback_events_total",
    "Feedback events ingested, by user action",
    ["action"],
)
CTR_RATIO = Gauge("bandit_ctr_ratio", "Running positive-outcome ratio over all feedback")
AVG_REWARD = Gauge("bandit_avg_reward", "Running average reward over all feedback")
CUMULATIVE_REGRET = Gauge("bandit_cumulative_regret", "Cumulative regret vs. the oracle policy")
ACTIVE_POLICY_VERSION = Gauge(
    "bandit_active_policy_version", "Currently serving policy version", ["policy_name"]
)
CANARY_ACTIVE = Gauge("bandit_canary_active", "1 if a challenger canary policy is live, else 0")

# --- Dependency / infra health -------------------------------------------------------
KAFKA_REACHABLE = Gauge("kafka_reachable", "1 if the Kafka producer can reach the broker, else 0")

# --- Parquet / data loading ----------------------------------------------------------
PARQUET_LOAD_SECONDS = Histogram(
    "parquet_load_duration_seconds",
    "Time spent reading a parquet file into memory",
    ["file"],
)
PARQUET_ROWS_LOADED = Gauge(
    "parquet_rows_loaded",
    "Row count of the most recently loaded parquet file",
    ["file"],
)

# --- Drift monitoring (mlops/drift_report.py, mlops/drift_stream_monitor.py) --------
DRIFT_CHECKS_TOTAL = Counter(
    "drift_checks_total", "Number of drift checks run, by outcome", ["result"]
)
DRIFT_DATASET_DRIFT = Gauge(
    "drift_dataset_drift", "1 if the last drift check flagged dataset-level drift, else 0"
)
DRIFT_DRIFTED_COLUMNS = Gauge(
    "drift_drifted_columns_count", "Number of columns flagged as drifted in the last check"
)
DRIFT_SHARE_DRIFTED_COLUMNS = Gauge(
    "drift_share_drifted_columns", "Share of columns flagged as drifted in the last check"
)
DRIFT_CURRENT_WINDOW_EVENTS = Gauge(
    "drift_current_window_events", "Size of the current (live) window used in the last drift check"
)
DRIFT_LAST_CHECK_UNIXTIME = Gauge(
    "drift_last_check_unixtime", "Unix timestamp of the last completed drift check"
)


@contextmanager
def track_parquet_load(file_label: str):
    """Times a parquet read and records row count.

    Usage:
        with track_parquet_load("features.parquet") as record_rows:
            df = pd.read_parquet(path)
            record_rows(len(df))
    """
    start = time.perf_counter()
    rows_holder = {"rows": None}

    def record_rows(n_rows: int) -> None:
        rows_holder["rows"] = n_rows

    try:
        yield record_rows
    finally:
        elapsed = time.perf_counter() - start
        PARQUET_LOAD_SECONDS.labels(file=file_label).observe(elapsed)
        if rows_holder["rows"] is not None:
            PARQUET_ROWS_LOADED.labels(file=file_label).set(rows_holder["rows"])
        logger.info(
            "loaded parquet file=%s rows=%s elapsed_seconds=%.4f",
            file_label,
            rows_holder["rows"],
            elapsed,
        )


def push_drift_metrics(summary: dict, job: str = "drift_monitor") -> None:
    """Push a drift-check summary to the Prometheus Pushgateway.

    Batch jobs like mlops/drift_report.py exit right after computing a result, so they
    can't be scraped -- pushing is the standard Prometheus pattern for this case.
    Failure to reach the Pushgateway is logged and swallowed: a monitoring job should
    never fail the pipeline it's monitoring.
    """
    registry = CollectorRegistry()
    dataset_drift = Gauge(
        "drift_dataset_drift", "1 if the drift check flagged dataset-level drift, else 0",
        registry=registry,
    )
    drifted_columns = Gauge(
        "drift_drifted_columns_count", "Number of columns flagged as drifted", registry=registry
    )
    share_drifted = Gauge(
        "drift_share_drifted_columns", "Share of columns flagged as drifted", registry=registry
    )
    current_events = Gauge(
        "drift_current_window_events", "Size of the current (live) window", registry=registry
    )
    monitoring_ready = Gauge(
        "drift_monitoring_ready", "1 if enough live events existed to run a real check",
        registry=registry,
    )
    last_check = Gauge(
        "drift_last_check_unixtime", "Unix timestamp this drift check completed", registry=registry
    )

    dataset_drift.set(1 if summary.get("dataset_drift") else 0)
    drifted_columns.set(summary.get("number_of_drifted_columns", 0))
    share_drifted.set(summary.get("share_of_drifted_columns", 0.0))
    current_events.set(summary.get("current_events", 0))
    monitoring_ready.set(1 if summary.get("monitoring_ready") else 0)
    last_check.set(time.time())

    result_label = "drift" if summary.get("dataset_drift") else "no_drift"
    if not summary.get("monitoring_ready", True):
        result_label = "insufficient_data"

    # Also reflect into the in-process gauges, in case this is called from a
    # long-running loop (mlops/drift_stream_monitor.py) rather than a one-shot script.
    DRIFT_CHECKS_TOTAL.labels(result=result_label).inc()
    DRIFT_DATASET_DRIFT.set(1 if summary.get("dataset_drift") else 0)
    DRIFT_DRIFTED_COLUMNS.set(summary.get("number_of_drifted_columns", 0))
    DRIFT_SHARE_DRIFTED_COLUMNS.set(summary.get("share_of_drifted_columns", 0.0))
    DRIFT_CURRENT_WINDOW_EVENTS.set(summary.get("current_events", 0))
    DRIFT_LAST_CHECK_UNIXTIME.set(time.time())

    try:
        push_to_gateway(PUSHGATEWAY_URL, job=job, registry=registry)
        logger.info("pushed drift metrics to pushgateway=%s job=%s", PUSHGATEWAY_URL, job)
    except Exception:  # noqa: BLE001 -- monitoring must never fail the pipeline it watches
        logger.warning(
            "could not push drift metrics to pushgateway=%s (is it running?)",
            PUSHGATEWAY_URL,
            exc_info=True,
        )
