FROM ghcr.io/mlflow/mlflow:v2.13.0

RUN pip install pymysql cryptography
