FROM python:3.10-slim

WORKDIR /app

# Only the serving-time dependencies (no librosa/mlflow/airflow -- those belong to the
# offline data/training pipeline, not the FastAPI image) -- keeps the image small.
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# data/ holds generated parquet artifacts + the (small) Python read/context modules --
# the raw audio in datasets/ is a separate top-level directory and is never copied in.
# mlops/ is needed for tracking.py (MLflow client + the canary registry poller); its
# heavier submodules (dashboard.py, drift_report.py, dags/) are never imported here, so
# their extra deps (streamlit, evidently, airflow) don't need to be installed.
COPY data/ ./data/
COPY bandit/ ./bandit/
COPY mlops/ ./mlops/
COPY monitoring/ ./monitoring/
COPY service/ ./service/

ENV BANDIT_POLICY=linucb
ENV CANDIDATE_POOL_SIZE=15
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092

EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
