# music-bandit

Music recommendation with contextual bandits — an MLOps project that recommends tracks
from FMA-small using online learning (LinUCB, Linear Thompson Sampling, UCB1, ε-greedy,
Thompson Sampling, Random), served behind FastAPI, streamed through Kafka, tracked in
MLflow, monitored for drift with Evidently, and retrained/redeployed via Airflow.

## Architecture

```
FMA audio ──▶ librosa feature extraction ──▶ Parquet feature store
                                                     │
synthetic user profiles ──▶ synthetic listening logs ──▶ user context (genre affinity + audio)
                                                     │
                                    Bandit policy picks a track
                                                     │
                              FastAPI /recommend ──▶ Kafka recommendation-events
                                                     │
                              FastAPI /feedback  ──▶ Kafka user-feedback ──▶ consumer ──▶ policy.update()
                                                     │
                    MLflow tracks runs + Model Registry (Staging/Production/Archived)
                                                     │
              Evidently checks drift ──▶ Airflow retrains ──▶ canary ──▶ promote or rollback
```

**Why synthetic interaction data?** The original plan called for Last.fm-1K as the
reward/interaction dataset. We tested the real join against FMA-small directly: no
MusicBrainz IDs in FMA metadata, and an exact artist+title match against all 19.15M
Last.fm play events found only 9,005 matches (0.047%), touching 260/8,000 tracks — too
sparse to build on. That decision, and the synthetic layer that replaces it, are
documented in [`contracts/dataset_reconciliation.md`](contracts/dataset_reconciliation.md)
and [`contracts/synthetic_data.md`](contracts/synthetic_data.md). Real FMA audio still
drives the feature store; only the interaction/reward side is simulated.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full build plan, sequencing rationale,
and the drift-triggered canary lifecycle design, and [`contracts/`](contracts/) for the
frozen data/API contracts every component was built against.

## Repo layout

```
contracts/    frozen schemas: Parquet feature store, Kafka topics, REST API, synthetic data
data/         feature extraction, synthetic user/interaction generation, context builder
bandit/       6 bandit policies, reward simulator, offline replay evaluator, comparison
service/      FastAPI app, Kafka producer/consumer, demo load generator, Dockerfile
mlops/        MLflow tracking + registry, Evidently drift report, Airflow DAG, dashboard
tests/        pytest suite (bandit policy interface + API contract)
```

## Setup

Requires Python 3.10, Docker, and the datasets under `datasets/` (FMA-small +
`fma_metadata`, and `lastfm-dataset-1K` — not committed; see `.gitignore`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Fetching the datasets (DVC)

`datasets/` is tracked with [DVC](https://dvc.org) (`datasets.dvc`) and stored in a
shared Google Drive remote, not in git. After installing dependencies:

```bash
dvc pull
```

This uses a project-specific Google OAuth client (`gdrive_client_id`/`gdrive_client_secret`
in `.dvc/config`), not DVC's shared default — Google's shared-client OAuth now gets hard
blocked ("This app is blocked") on many personal Gmail accounts, and service accounts
can't be used either since they have no storage quota outside a Workspace Shared Drive.
The app is left in Google's "Testing" publish status, so **only allowlisted Google
accounts can authorize it** — ask a maintainer to add your Gmail as a test user in the
GCP project's OAuth consent screen before your first `dvc pull`.

The first `dvc pull`/`dvc push` on a machine triggers a one-time interactive Google
login: it opens a browser to a Google consent screen, shows a "Google hasn't verified
this app" warning (expected, since it's a small Testing-mode app, not a public one),
click **Advanced → Go to \<app name\> (unsafe)** to continue. The token is then cached
locally and reused on future runs.

Note: the login flow needs a local callback server on `localhost:8080` — if something
else on your machine is already bound to that port (e.g. Airflow from
[Docker Compose](#docker-compose) below), stop it first or the browser login will hang
with no error.

Large files are split into `*.part-*` chunks (~500MB–1GB each) before being tracked —
Google Drive's API is too slow/unreliable per-file for thousands of small files, and
proved unreliable uploading any single multi-hundred-MB+ blob in one shot too. Affected:

- `fma_small`'s ~8000 individual mp3s, tarred then split: `fma_small.tar.part-*`
- `lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv.part-*`
- `fma_metadata/features.csv.part-*`

After `dvc pull`, reassemble before running the pipeline:

```bash
mkdir -p datasets/archive/fma_small/fma_small
cat datasets/archive/fma_small.tar.part-* | tar -xf - -C datasets/archive/fma_small/fma_small

cat datasets/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv.part-* \
  > datasets/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv

cat datasets/archive/fma_metadata/fma_metadata/features.csv.part-* \
  > datasets/archive/fma_metadata/fma_metadata/features.csv
```

After modifying anything under `datasets/` (re-split any large file you touched first —
see the `split -b 500m <file> <file>.part-` pattern above), run
`dvc add datasets && dvc push` and commit the updated `datasets.dvc`.

`requirements-prod.txt` covers the runtime libraries the app actually needs (librosa,
bandits, FastAPI, MLflow, Evidently, Streamlit, Airflow). `requirements-dev.txt` layers
local-only tooling on top (`dvc[gdrive]`, `pytest`, `ruff`) via `-r requirements-prod.txt`,
so `pip install -r requirements-dev.txt` above gets you everything. `service/requirements.txt`
and `mlops/requirements.txt` are separate, further-trimmed subsets actually installed inside
the Docker images (see [Docker Compose](#docker-compose) below).

## Running the pipeline locally

Run once, in order, to produce the artifacts everything else depends on:

```bash
python data/synth_user_profiles.py                # data/user_profiles.parquet
python data/extract_features.py                   # data/features.parquet (~8000 tracks, run once)
python data/reconcile_datasets.py                  # reproduces the NO-GO join finding
python -m data.generate_synthetic_logs             # data/synthetic_logs.parquet
python -m data.build_user_context                  # smoke test: warm + cold-start context
python -m data.feature_store                       # smoke test: get_features() latency
```

### Reproducing with DVC

The commands above that write `data/user_profiles.parquet`, `data/features.parquet`,
and `data/synthetic_logs.parquet` are also wired up as a [DVC](https://dvc.org)
pipeline (`dvc.yaml` / `params.yaml`), so you don't have to remember the order or
which ones are already up to date:

```bash
dvc dag        # show the stage graph
dvc repro      # (re)generate any of the three parquet files whose deps changed
```

`dvc repro` skips a stage if its script(s), params, and input files/dirs haven't
changed since the last run — most useful for `extract_features`, the expensive one
(~8000 ffmpeg+librosa extractions). The three outputs stay committed to git as
regular files (`cache: false` in `dvc.yaml`), so no `dvc push`/`dvc pull` is needed
for them — only `datasets.dvc` uses the DVC cache/remote.

The `extract_features` stage needs the `fma_small` audio reassembled locally first.
So on a fresh clone, the full path from zero to reproduced artifacts is: `dvc pull`,
then the `fma_small` reassembly command from
[Fetching the datasets](#fetching-the-datasets-dvc) above, then `dvc repro` — the
`lastfm`/`features.csv` reassembly from that section isn't needed here, only for
`reconcile_datasets.py`.

`ffmpeg` must also be installed and on your `PATH` (`brew install ffmpeg` on macOS) —
`extract_features.py` shells out to it per track and fails silently (writes an empty
`features.parquet` and exits 0) if it's missing, so `dvc repro` won't know to
re-run that stage once you do install it; force it with `dvc repro -f extract_features`
in that case.

Use the manual commands above instead of `dvc repro` for one-off runs with
non-default flags (e.g. `--inject-drift`), or for `reconcile_datasets.py` /
`build_user_context.py` / `feature_store.py`, which aren't part of the DVC pipeline.

Compare the bandit policies offline (replay evaluation against a fixed candidate pool):

```bash
python -m bandit.compare_policies --n-events 8000 --pool-size 15
```

Run the service locally (Kafka/MLflow optional — the service degrades gracefully without
them, see `/health`):

```bash
uvicorn service.main:app --reload
```

Check for drift and log it to MLflow:

```bash
python -m mlops.drift_report
```

## Docker Compose

Brings up Kafka (KRaft, single broker), MLflow (backed by `./mlruns`), the FastAPI
service, and Airflow (`standalone` — webserver + scheduler + triggerer in one container):

```bash
docker compose up -d --build
```

| Service | URL                   |
| ------- | --------------------- |
| Service | http://localhost:8000 |
| MLflow  | http://localhost:5001 |
| Airflow | http://localhost:8081 |
| Kafka   | localhost:9092        |

Smoke test:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{"user_id":"user_000001"}'
curl -X POST localhost:8000/feedback  -H 'content-type: application/json' \
  -d '{"user_id":"user_000001","track_id":"<track_id from above>","action":"liked"}'
curl localhost:8000/metrics
python -m service.demo_loadgen --n 200          # simulate traffic end-to-end
```

Trigger the retrain → canary → promote/rollback DAG manually (our synthetic data is
stationary, so `FORCE_RETRAIN` bypasses the drift gate to exercise the full lifecycle on
demand — see `mlops/dags/retrain_policy.py`):

```bash
docker exec -e FORCE_RETRAIN=true bandit-airflow airflow dags test retrain_policy 2026-01-01
```

Dashboard (reads live `/metrics` + MLflow-tracked runs):

```bash
streamlit run mlops/dashboard.py
```

## API

Frozen in [`contracts/openapi_notes.md`](contracts/openapi_notes.md):

| Method & path                 | Purpose                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| `POST /recommend`           | `{user_id}` → `{track_id}`                                                          |
| `POST /feedback`            | `{user_id, track_id, action}` → `{status}`                                          |
| `GET /metrics`              | running CTR, reward, regret                                                              |
| `GET /health`               | service + Kafka reachability                                                             |
| `POST /admin/reload-policy` | operational addition — forces the MLflow registry poll that drives the canary lifecycle |

## Testing

```bash
pytest tests/ -v      # 34 tests: bandit policy interface conformance + API contract
ruff check .           # lint
```

`tests/test_bandit_policies.py` is fully hermetic (no fixture data needed) and is what CI
runs on every push. `tests/test_api_contract.py` needs the generated parquet artifacts
from the pipeline above, so it's run locally rather than in CI.

## Add User to aiflow
- To create a new admin user on airflow, run the following command in airflow docker bash shell:

```bash
airflow users create \
  --username newadmin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin1@example.com \
  --password 'admin'
```
