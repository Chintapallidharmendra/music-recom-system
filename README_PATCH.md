# Live drift workflow patch

Replace the corresponding files in the original project with the files in this patch.

Files changed:
- service/live_simulator.py (new)
- service/main.py
- data/build_user_context.py
- mlops/drift_report.py
- mlops/dags/retrain_policy.py
- docker-compose.yml (shared LIVE_FEEDBACK_PATH)

Run:
1. docker compose up --build
2. python service/live_simulator.py --scenario normal --n 500 --interval 0.02
3. python service/live_simulator.py --scenario preference_shift --n 3000 --start-after 500 --magnitude 0.8 --interval 0.02
4. python mlops/drift_report.py --live /mlruns/live_feedback.jsonl

The service persists feedback to /mlruns/live_feedback.jsonl, updates online policy state,
and includes live interactions in the next user context. Airflow reads the same event store,
checks drift every 15 minutes, trains a candidate from live events, evaluates it, and
promotes or archives it through MLflow.
