from __future__ import annotations

from typing import TYPE_CHECKING
import pandas as pd
from mlforecast import flavor
from ssdf.config import (
    MLFLOW_MODEL_REGISTRY_NAME,
    PREDICTIONS_DIR,
    FH,
    FEATURES_DATA_DIR,
    STATIC_FEATURES,
)

if TYPE_CHECKING:
    from pathlib import Path
    from mlforecast import MLForecast


def get_model(model_uri: str | None = None) -> MLForecast:
    if model_uri is None:
        model_uri = f"models:/{MLFLOW_MODEL_REGISTRY_NAME}@production"
    return flavor.load_model(model_uri=model_uri)


def get_features(future_df: pd.DataFrame) -> pd.DataFrame:
    min_date = future_df["date"].min()
    features = pd.read_parquet(
        FEATURES_DATA_DIR / "features.parquet", filters=[("date", ">=", min_date)]
    )
    features = features.merge(future_df, on=["unique_id", "date"]).drop(
        columns=STATIC_FEATURES
    )
    return features


def generate_forecasts(model_uri: str | None = None, fh: int = FH) -> pd.DataFrame:
    forecaster = get_model(model_uri)
    X_df = get_features(forecaster.make_future_dataframe(h=fh))
    return forecaster.predict(h=fh, X_df=X_df)


def save_forecasts(forecasts: pd.DataFrame, path: Path) -> None:
    forecasts = forecasts.copy()

    # Extract store_nbr and family from unique_id
    forecasts[["store_nbr", "family"]] = forecasts["unique_id"].str.split(
        "_", expand=True
    )
    forecasts["store_nbr"] = forecasts["store_nbr"].astype(int)
    forecasts = forecasts.drop(columns=["unique_id"]).rename(
        columns={"forecaster": "sales"}
    )
    forecasts = forecasts[["date", "store_nbr", "family", "sales"]].sort_values(
        ["date", "store_nbr", "family"]
    )
    forecasts.to_parquet(path / "sales_forecasts.parquet", index=False)


def run(model_uri: str | None = None):
    print("Loading model and generating forecasts...")
    predictions = generate_forecasts(model_uri=model_uri)
    print("Successfully generated forecasts")

    print("Saving forecasts...")
    save_forecasts(predictions, PREDICTIONS_DIR)
    print("Successfully saved forecasts")


if __name__ == "__main__":
    model_uri = "models:/m-b5dcb1ca35a547c4b370764cbe5e0274"
    run(model_uri=model_uri)
