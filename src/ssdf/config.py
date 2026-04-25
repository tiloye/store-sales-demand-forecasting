import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_DATA_DIR = DATA_DIR / "feature_store"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_REGISTRY_NAME = "store-sales-forecaster"

FH = 16
