FROM python:3.10-slim
RUN pip install --no-cache-dir mlflow==2.14.1
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "/mlruns", "--default-artifact-root", "/mlruns"]
