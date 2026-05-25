import os
from pathlib import Path
from dotenv import load_dotenv

ENV_NAME = os.getenv("ENV_NAME", "dev")
PROJECT_DIR = Path(__file__).parent.parent.parent

load_dotenv(PROJECT_DIR / f".env.{ENV_NAME}")


if ENV_NAME == "dev":
    DATA_DIR = PROJECT_DIR / "data"
else:
    from upath import UPath

    DATA_DIR = UPath(f"s3://{os.getenv('S3_BUCKET_NAME')}")

PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_DATA_DIR = DATA_DIR / "feature_store"
PREDICTIONS_DIR = DATA_DIR / "predictions"

STORAGE_OPTIONS = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {
        "endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
        "region_name": os.getenv("AWS_REGION"),
    },
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_REGISTRY_NAME = "store-sales-forecaster"

FH = 16
STATIC_FEATURES = ["store_nbr", "family"]

EVIDENTLY_PROJECT_NAME = "Store Sales Demand Forecasting"
EVIDENTLY_API_KEY = os.getenv("EVIDENTLY_API_KEY")
EVIDENTLY_ORG_ID = os.getenv("EVIDENTLY_ORG_ID")
if ENV_NAME == "dev":
    EVIDENTLY_WORKSPACE = PROJECT_DIR / ".evidently_workspace"
