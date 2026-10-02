FROM python:3.11-slim
# Server version matches the client pinned in uv.lock (mlflow 3.x).
RUN pip install --no-cache-dir "mlflow==3.16.1" psycopg2-binary
EXPOSE 5000
