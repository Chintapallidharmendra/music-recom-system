# Music-Bandit: Full Build Plan (Solo Execution)

## Context

Day-1 alignment is done: `contracts/{parquet_schema,kafka_topics,openapi_notes,dataset_reconciliation}.md`
freeze the feature schema, Kafka topics, REST API, and the data-driven decision to use a
**synthetic interaction/reward layer** instead of a real Last.fm↔FMA join (real join match rate
measured at 0.047% of events, touching 260/8,000 tracks — too sparse to build on).

This plan turns the two source documents (`Claude_Code_Execution_Spec.docx`,
`MLOps_Workplan.docx`) — written for a 4-person parallel team — into a sequenced solo build.
Confirmed with the user: **solo execution** (tracks run A→B→C→D, not in parallel) and
**full infra scope** (Kafka, Docker, MLflow, Evidently, Airflow, CI/CD, canary — not descoped
up front; the docs' descope order is a fallback only, used if time runs short, never the plan).

The one design gap neither source doc resolves: since the real Last.fm join is NO-GO, Track A's
`build_user_context.py` (needs historical plays) and Track B's `replay_evaluator.py` (needs
ground-truth reward) both need synthetic data that doesn't exist yet. This plan adds that layer
explicitly, owned by a shared ground-truth model so the two don't drift apart.

## Synthetic-data architecture (new, resolves the join gap)

One shared ground-truth model, two consumers:

- **`data/synth_user_profiles.py`** (Phase 0, build first) — defines ~1,000 synthetic users,
  each with a Dirichlet-sampled genre-affinity vector over the 8 FMA-small genres + a novelty
  bias scalar. Writes `data/user_profiles.parquet`. This is the *only* place affinity ground
  truth is defined.
- **`data/generate_synthetic_logs.py`** (Track A) — samples historical `(user_id, timestamp,
  track_id)` plays from `user_profiles.parquet` + `features.parquet`, weighted by genre
  affinity with noise/novelty so history isn't perfectly predictable. Writes
  `data/synthetic_logs.parquet`. This is the drop-in replacement for what would have been
  real Last.fm-joined plays — `build_user_context.py`'s interface doesn't change.
- **`bandit/reward_simulator.py`** (Track B) — imports the *same* `user_profiles.parquet`
  affinity vectors (does not recompute its own), adds a content-similarity term against track
  audio features, and exposes `sample_action(user_id, track_id) -> str` and
  `expected_reward(user_id, track_id) -> float`. Used by `replay_evaluator.py` for offline
  replay AND imported live by `service/main.py` / a demo load-generator, since there are no
  real users to click `/feedback` in a solo academic build.

Add **`contracts/synthetic_data.md`** at the start of Phase 0, documenting
`user_profiles.parquet`'s schema and the reward-simulator's two function signatures — same
role as `parquet_schema.md` plays for `features.parquet`, so later steps code against a frozen
shape.

**Watch for triviality:** if `generate_synthetic_logs.py` only samples tracks a user already
loves, there's nothing left to explore and every bandit policy will look the same in
`compare_policies.py`. Deliberately inject noise/off-affinity plays via the novelty bias.

## Sequencing (solo — linear, not parallel)

No concurrency benefit to interleaving tracks solo, so build in dependency order:

1. **Phase 0 — shared foundation:** `synth_user_profiles.py` → `contracts/synthetic_data.md`.
2. **Track A — Data & Features:** `extract_features.py` (librosa, multiprocessing over
   `datasets/archive/fma_small/`, → `data/features.parquet` matching `parquet_schema.md`) →
   `reconcile_datasets.py` (thin script reproducing/printing the already-frozen match-rate +
   NO-GO finding from `dataset_reconciliation.md` — keep it, don't skip it, don't re-run the
   full 19M-row join at runtime) → `generate_synthetic_logs.py` → `build_user_context.py`
   (recency-weighted genre affinity, 30-day half-life, + session audio average, + cold-start
   default) → `feature_store.py` (`get_features(track_id)` in-memory dict loader).
3. **Track B — Bandit:** `reward_simulator.py` → policies in increasing complexity, each with
   a unit test before the next: `random_policy.py` → `epsilon_greedy.py` → `ucb1.py` →
   `thompson_sampling.py` → `linucb.py` (ridge-regularized, guard matrix inversion) →
   `replay_evaluator.py` → `compare_policies.py` (must show LinUCB beating Random on
   cumulative regret).
4. **Track C — Serving & Streaming:** `service/schemas.py` → `service/main.py` (`/health`
   first, then `/recommend` wired to feature_store + build_user_context + one bandit policy,
   then `/feedback` with the reward map from `contracts/kafka_topics.md`) →
   `kafka_producer.py`/`kafka_consumer.py` (validate against Kafka via
   `kafka-console-producer`/`consumer` in Docker *before* wiring the Python client) →
   `service/demo_loadgen.py` (drives synthetic `/recommend`→`reward_simulator`→`/feedback`
   traffic, since there are no real users) → `Dockerfile`.
5. **Track D — MLOps:** `mlops/tracking.py` (MLflow — wire into `replay_evaluator.py` and
   `service/main.py` first, the two things that actually produce runs worth logging) →
   `mlops/drift_report.py` (Evidently, compare two time-windows of `synthetic_logs.parquet`) →
   `mlops/dags/retrain_policy.py` (Airflow: collect → update_features → retrain → evaluate →
   register, orchestrating steps already built in A/B) → `mlops/dashboard.py` (Streamlit,
   last — pure consumer of MLflow + `/metrics`).

## Phase boundaries ("Build" / "Integrate" / "Automate & polish")

Keep the workplan's 3-phase framing, but as effort-phases with concrete done-criteria, not
calendar weeks (solo wall-clock will run longer on phases 1–2 than the team estimate):

- **Phase 1 done when:** each track's pieces run and pass their own tests in isolation, no
  cross-track wiring — `features.parquet` complete for all 8,000 tracks; context vectors
  produced for warm + cold-start users; each policy passes its interface test; `uvicorn
  service.main:app` serves `/health`, `/recommend`, `/feedback` against an in-memory policy
  with no Kafka/MLflow yet.
- **Phase 2 done when:** `docker-compose up` brings up Kafka (KRaft) + MLflow + FastAPI
  together; `/recommend` → `recommendation-events` → `/feedback` → `user-feedback` → consumer
  calls `bandit.update()` round-trips live; MLflow logs a run per batch; the Airflow DAG
  completes end-to-end on a manual trigger.
- **Phase 3 done when:** the Airflow DAG runs on its own hourly schedule (not manual); canary
  routing (`hash(user_id) % 10`, 90/10 split) serves two live policy instances simultaneously
  and logs which policy served each request; CI runs pytest + lint on push; Evidently drift
  report and the Streamlit dashboard render; README has a 5-minute demo walkthrough.

Run the full end-to-end smoke sequence (below) at every phase boundary, not just at the end.

## Verification

**Per-file, before moving to the next** (spec's own discipline):
```bash
pytest tests/test_bandit_policies.py -k random -v
pytest tests/test_bandit_policies.py -k epsilon_greedy -v
pytest tests/test_bandit_policies.py -k ucb1 -v
pytest tests/test_bandit_policies.py -k thompson -v
pytest tests/test_bandit_policies.py -k linucb -v
pytest tests/test_api_contract.py -v
```
Each policy test asserts the shared interface directly (`select_action` returns a valid arm,
`update` doesn't raise, LinUCB's ridge coefficients change after `update()` and never NaN).

**Data-layer sanity checks:**
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/features.parquet'); \
  assert len(df) == 8000; print(df.shape, df.dtypes)"
python -c "import pandas as pd; df = pd.read_parquet('data/synthetic_logs.parquet'); \
  print(df.user_id.nunique(), 'users,', len(df), 'plays')"
```

**End-to-end smoke sequence (Phase 2/3 gate):**
```bash
docker-compose up -d
curl -s localhost:8000/health
curl -s -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{"user_id":"user_000001"}'
curl -s -X POST localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"user_id":"user_000001","track_id":"...","action":"liked"}'
docker exec -it <kafka-container> kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic recommendation-events --from-beginning --max-messages 5
python service/demo_loadgen.py --n 200
airflow dags trigger retrain_policy && airflow dags list-runs -d retrain_policy
curl -s localhost:8000/metrics
```

## Risks to plan around (not fixes to make now, but things to expect)

- **librosa on 8,000 files is the single longest step** (hours if serial) — parallelize with
  `multiprocessing.Pool`, set `OMP_NUM_THREADS=1` per worker to avoid oversubscription, confirm
  arm64 wheels for `librosa`/`numba`/`llvmlite` before other code depends on the output. Run
  once, cache `features.parquet`, never rerun unless extraction logic changes.
- **Kafka KRaft first-run failures** (cluster ID / volume permissions) are common — validate
  one topic round-trip via `kafka-console-producer`/`consumer` before wiring the Python client,
  so Kafka failures and client-code failures aren't conflated. Pin an image tag, not `latest`.
- **MLflow/Evidently version drift** — Evidently's report API has had breaking changes across
  versions; pin both in a requirements file before writing `drift_report.py`, don't install
  latest mid-project.
- **Airflow is heavy on a 16GB machine already running Kafka+MLflow+FastAPI in Docker** — do a
  throwaway SQLite/LocalExecutor manual-trigger test first; only stand up the full
  docker-compose Airflow (Postgres+scheduler+webserver) stack in Phase 3.
- **Canary logic is more than a config flag** — `service/main.py` must hold two live policy
  instances simultaneously, route by `hash(user_id) % 10`, log which policy served each
  request (the `policy` field already exists in `kafka_topics.md`'s schema), and reconcile
  which object is "current" against MLflow registry stage transitions. Plan a small in-process
  policy reload mechanism (poll interval or `/admin/reload-policy`), don't assume it's free.
- **CI/CD "auto-deploy" needs a concrete target** — default to pushing the Docker image to
  GHCR/tag on push, not an actual remote server, unless one is specifically wanted.

## Fallback order (only if solo time runs out — not the default plan)

From `MLOps_Workplan.docx`, in order: (1) shrink the Airflow DAG to 3 tasks, (2) collapse Kafka
to 1 topic, (3) ship 1 bandit policy instead of the full comparison, (4) drop the custom
dashboard for MLflow/Airflow's built-in UIs. Kafka end-to-end, Airflow end-to-end, and the core
recommend/feedback loop are the graded/never-drop items — cut in this order, never skip ahead.

## Critical files (new, to be created)

- `contracts/synthetic_data.md` — new frozen contract for the synthetic layer
- `data/synth_user_profiles.py`, `data/generate_synthetic_logs.py`, `data/extract_features.py`,
  `data/reconcile_datasets.py`, `data/build_user_context.py`, `data/feature_store.py`
- `bandit/reward_simulator.py`, `bandit/policies/{random_policy,epsilon_greedy,ucb1,
  thompson_sampling,linucb}.py`, `bandit/replay_evaluator.py`, `bandit/compare_policies.py`
- `service/schemas.py`, `service/main.py`, `service/kafka_producer.py`,
  `service/kafka_consumer.py`, `service/demo_loadgen.py`, `service/Dockerfile`
- `mlops/tracking.py`, `mlops/drift_report.py`, `mlops/dashboard.py`, `mlops/dags/retrain_policy.py`
- `tests/test_bandit_policies.py`, `tests/test_api_contract.py`
- `docker-compose.yml`, `.github/workflows/ci.yml`, top-level `requirements.txt`
