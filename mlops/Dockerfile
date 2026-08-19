FROM apache/airflow:2.9.2-python3.10

# DAG task code (mlops/dags/retrain_policy.py) imports bandit/, data/, mlops/ modules,
# which need numpy/pandas/mlflow/evidently on top of Airflow's own bundled deps.
COPY mlops/requirements.txt /tmp/mlops-requirements.txt
RUN pip install --no-cache-dir -r /tmp/mlops-requirements.txt

# Project code is volume-mounted at /app in docker-compose.yml (not copied at build
# time) so DAG edits don't require an image rebuild -- see PYTHONPATH in docker-compose.
