"""FastAPI service: the four routes frozen in contracts/openapi_notes.md, wired to the
feature store + context builder + one bandit policy (config flag: BANDIT_POLICY env var --
see contracts/kafka_topics.md and PROJECT_PLAN.md's "swap policies with zero code changes"
acceptance criterion). Kafka wiring is soft-fail: the service still serves recommendations
if Kafka is down, and /health reports the degraded state rather than crashing.
"""
import json
import logging
import os
import threading
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from bandit.policies.epsilon_greedy import EpsilonGreedyPolicy
from bandit.policies.linear_thompson_sampling import LinearThompsonSamplingPolicy
from bandit.policies.linucb import LinUCBPolicy
from bandit.policies.random_policy import RandomPolicy
from bandit.policies.thompson_sampling import ThompsonSamplingPolicy
from bandit.policies.ucb1 import UCB1Policy
from bandit.reward_simulator import REWARD_MAP, RewardSimulator
from data.build_user_context import build_user_context
from data.feature_store import FeatureStore
from mlops.tracking import POLICY_MODEL_NAME, ServiceMetricsLogger, load_policy_by_stage
from service.kafka_consumer import FeedbackConsumer
from service.kafka_producer import RecommendationEventProducer
from service.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    RecommendRequest,
    RecommendResponse,
)

logger = logging.getLogger(__name__)

# Swap the active policy via the BANDIT_POLICY env var -- zero code changes required.
POLICY_FACTORIES = {
    "random": lambda: RandomPolicy(seed=0),
    "epsilon_greedy": lambda: EpsilonGreedyPolicy(epsilon=0.1, seed=0),
    "ucb1": lambda: UCB1Policy(),
    "thompson_sampling": lambda: ThompsonSamplingPolicy(seed=0),
    "linear_thompson_sampling": lambda: LinearThompsonSamplingPolicy(
        v=0.3, seed=0, context_dim=8
    ),
    "linucb": lambda: LinUCBPolicy(alpha=1.0, context_dim=8),
}
POLICY_NAME = os.environ.get("BANDIT_POLICY", "linucb")
CANDIDATE_POOL_SIZE = int(os.environ.get("CANDIDATE_POOL_SIZE", "15"))
CANARY_POLL_INTERVAL_SECONDS = float(os.environ.get("CANARY_POLL_INTERVAL_SECONDS", "30"))
LIVE_FEEDBACK_PATH = Path(os.environ.get("LIVE_FEEDBACK_PATH", "data/live_feedback.jsonl"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


app = FastAPI(title="music-bandit", lifespan=lifespan)


class AppState:
    feature_store: FeatureStore = None
    plays: pd.DataFrame = None
    features_df: pd.DataFrame = None
    live_history: dict = None
    live_history_lock: threading.Lock = None
    policy = None
    policy_name: str = POLICY_NAME
    policy_version: int = None
    challenger = None
    challenger_version: int = None
    candidate_pool: list = None
    producer: RecommendationEventProducer = None
    consumer: FeedbackConsumer = None
    reward_sim: RewardSimulator = None
    metrics_logger: ServiceMetricsLogger = None
    registry_poll_stop: threading.Event = None
    total_recommendations: int = 0
    total_feedback: int = 0
    cumulative_reward: float = 0.0
    positive_outcomes: int = 0
    cumulative_regret: float = 0.0


state = AppState()


def _select_serving_policy(user_id: str):
    """The champion/challenger split is temporary and population-wide, not a permanent
    per-user assignment (see PROJECT_PLAN.md's "Policy swap on drift" lifecycle): the
    hash bucket only matters while a canary (state.challenger) is active at all, and a
    user's bucket is deterministic so the same user isn't flip-flopped mid-evaluation."""
    if state.challenger is not None and hash(user_id) % 10 == 0:
        return state.challenger, f"challenger_v{state.challenger_version}"
    return state.policy, state.policy_name


def _poll_registry_once() -> None:
    """Adopts a new Production version if one appeared (a full swap for all users), and
    starts/stops the temporary 10% canary split based on whether a Staging version
    currently exists. Called on a background timer and by /admin/reload-policy."""
    prod_policy, prod_version = load_policy_by_stage(POLICY_MODEL_NAME, "Production")
    if prod_policy is not None and prod_version != state.policy_version:
        state.policy = prod_policy
        state.policy_version = prod_version
        state.policy_name = f"registry_v{prod_version}"
        logger.info("adopted new Production policy version %d", prod_version)

    staging_policy, staging_version = load_policy_by_stage(POLICY_MODEL_NAME, "Staging")
    if staging_policy is not None and staging_version != state.challenger_version:
        state.challenger = staging_policy
        state.challenger_version = staging_version
        logger.info("started canary with Staging version %d", staging_version)
    elif staging_policy is None and state.challenger is not None:
        logger.info("Staging cleared -- collapsing canary back to a single policy")
        state.challenger = None
        state.challenger_version = None


def _registry_poll_loop() -> None:
    while not state.registry_poll_stop.wait(CANARY_POLL_INTERVAL_SECONDS):
        try:
            _poll_registry_once()
        except Exception:  # noqa: BLE001 -- polling must never take the service down
            logger.exception("registry poll failed")


def _append_live_feedback(event: dict) -> None:
    """Keep recent interactions available to the next recommendation and persist a
    JSONL event store that Airflow/Evidently can read from the shared volume."""
    user_id = event["user_id"]
    live_row = {
        "user_id": user_id,
        "timestamp": event["timestamp"],
        "track_id": event["track_id"],
        "action": event["action"],
        "reward": float(event["reward"]),
    }
    with state.live_history_lock:
        state.live_history[user_id].append(live_row)
        LIVE_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LIVE_FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(live_row) + "\n")


def _handle_feedback(event: dict) -> None:
    # IMPORTANT: build the update context BEFORE adding this feedback event, so the
    # reward does not leak into the context used to train on the same interaction.
    with state.live_history_lock:
        live_history = {k: list(v) for k, v in state.live_history.items()}
    context = build_user_context(
        event["user_id"], state.plays, state.features_df, live_plays=live_history
    )
    policy, _ = _select_serving_policy(event["user_id"])
    policy.update(event["track_id"], context, event["reward"])
    _append_live_feedback(event)

    state.total_feedback += 1
    state.cumulative_reward += event["reward"]
    if event["reward"] > 0:
        state.positive_outcomes += 1

    optimal = max(
        state.reward_sim.expected_reward(event["user_id"], c) for c in state.candidate_pool
    )
    state.cumulative_regret += optimal - event["reward"]


def startup() -> None:
    state.feature_store = FeatureStore()
    state.plays = pd.read_parquet("data/synthetic_logs.parquet")
    state.live_history = defaultdict(list)
    state.live_history_lock = threading.Lock()
    if LIVE_FEEDBACK_PATH.exists():
        try:
            with LIVE_FEEDBACK_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        state.live_history[event["user_id"]].append(event)
            logger.info("loaded live history from %s", LIVE_FEEDBACK_PATH)
        except Exception:
            logger.exception("failed to restore live history from %s", LIVE_FEEDBACK_PATH)
    state.features_df = pd.read_parquet("data/features.parquet")
    state.reward_sim = RewardSimulator()

    state.policy_name = POLICY_NAME
    state.policy = POLICY_FACTORIES[POLICY_NAME]()

    rng = np.random.default_rng(42)
    state.candidate_pool = list(
        rng.choice(state.feature_store.track_ids, size=CANDIDATE_POOL_SIZE, replace=False)
    )

    state.producer = RecommendationEventProducer()
    state.consumer = FeedbackConsumer(_handle_feedback)
    state.consumer.start()

    state.metrics_logger = ServiceMetricsLogger(POLICY_NAME, _current_metrics)
    state.metrics_logger.start()

    try:
        _poll_registry_once()  # adopt an already-registered Production policy, if any
    except Exception:  # noqa: BLE001 -- no registered model yet is a normal first-run state
        logger.info("no registered policy found at startup, using default %s", POLICY_NAME)
    state.registry_poll_stop = threading.Event()
    threading.Thread(target=_registry_poll_loop, daemon=True).start()

    logger.info("started with policy=%s, candidate_pool_size=%d", POLICY_NAME, CANDIDATE_POOL_SIZE)


def shutdown() -> None:
    if state.consumer:
        state.consumer.stop()
    if state.metrics_logger:
        state.metrics_logger.stop()
    if state.registry_poll_stop:
        state.registry_poll_stop.set()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store_ok = state.feature_store is not None and len(state.feature_store) > 0
    if not store_ok:
        raise HTTPException(status_code=503, detail="feature store not loaded")
    kafka_ok = state.producer.is_reachable() if state.producer else False
    return HealthResponse(status="ok" if kafka_ok else "degraded: kafka unreachable")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    with state.live_history_lock:
        live_history = {k: list(v) for k, v in state.live_history.items()}
    context = build_user_context(
        req.user_id, state.plays, state.features_df, live_plays=live_history
    )
    policy, policy_label = _select_serving_policy(req.user_id)
    track_id = policy.select_action(context, state.candidate_pool)

    timestamp = datetime.now(timezone.utc).isoformat()
    state.producer.send_recommendation_event(req.user_id, track_id, policy_label, timestamp)
    state.total_recommendations += 1

    return RecommendResponse(track_id=track_id)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    reward = REWARD_MAP[req.action]
    timestamp = datetime.now(timezone.utc).isoformat()
    state.producer.send_user_feedback(req.user_id, req.track_id, req.action, reward, timestamp)
    return FeedbackResponse()


@app.post("/feedback/direct", response_model=FeedbackResponse)
def feedback_direct(req: FeedbackRequest) -> FeedbackResponse:
    """Synchronous feedback ingestion for demos.

    This endpoint bypasses Kafka and directly applies the feedback to the in-memory
    policy (calls the same handler used by the background consumer). Use this from
    Swagger to demonstrate: call `/recommend` -> then POST `/feedback/direct` with a
    `skip` (or other) action -> call `/recommend` again to see the updated policy's
    recommendation for the same user.
    """
    reward = REWARD_MAP[req.action]
    timestamp = datetime.now(timezone.utc).isoformat()
    event = {
        "user_id": req.user_id,
        "track_id": req.track_id,
        "action": req.action,
        "reward": float(reward),
        "timestamp": timestamp,
    }
    try:
        _handle_feedback(event)
    except Exception:
        logger.exception("direct feedback handling failed")
        raise
    return FeedbackResponse()


def _current_metrics() -> dict:
    n = state.total_feedback
    return {
        "ctr": state.positive_outcomes / n if n else 0.0,
        "avg_reward": state.cumulative_reward / n if n else 0.0,
        "cumulative_regret": state.cumulative_regret,
        "total_recommendations": state.total_recommendations,
    }


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**_current_metrics())


@app.post("/admin/reload-policy")
def reload_policy() -> dict:
    """Operational endpoint (not part of the frozen contracts/openapi_notes.md API) to
    force/inspect the canary lifecycle during a demo, per PROJECT_PLAN.md step 5 -- the
    swap itself never requires this, the background poller does it automatically."""
    _poll_registry_once()
    return {
        "policy_name": state.policy_name,
        "policy_version": state.policy_version,
        "challenger_active": state.challenger is not None,
        "challenger_version": state.challenger_version,
    }
