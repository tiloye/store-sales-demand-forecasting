from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import mlflow
from mlforecast import MLForecast, flavor
from sklearn.base import BaseEstimator, RegressorMixin

from ssdf.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, FEATURES_DATA_DIR

if TYPE_CHECKING:
    from pathlib import Path


def get_data():
    path = FEATURES_DATA_DIR / "target.parquet"
    return pd.read_parquet(path)


class SeasonalNaiveRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, sp: int = 7):
        self.sp = sp

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if hasattr(X, "columns") and f"lag{self.sp}" in X.columns:
            return X[f"lag{self.sp}"].values
        else:
            raise ValueError(f"No lag{self.sp} feature found in X")


def get_model():
    forecaster = MLForecast(
        models={"forecaster": SeasonalNaiveRegressor()},
        freq="D",
        lags=[7],
    )
    return forecaster


def run(
    df,
    static_features: list[str] | None = None,
    data_source: str | Path | None = None,
    model_name: str | None = None,
    exp_run_id: str | None = None,
    exp_run_name: str | None = None,
) -> tuple[MLForecast, mlflow.entities.Run]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    forecaster = get_model()
    with mlflow.start_run(run_id=exp_run_id, run_name=exp_run_name) as run_env:
        model_name = (
            forecaster.models["forecaster"].__class__.__name__
            if model_name is None
            else model_name
        )
        model_params = forecaster.models["forecaster"].get_params()
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(model_params)

        print("Logging training data to MLflow")
        dataset = mlflow.data.from_pandas(df, targets="sales")
        mlflow.log_input(dataset, context="training")

        print("Training the forecaster")
        forecaster.fit(
            df,
            id_col="unique_id",
            time_col="date",
            target_col="sales",
            static_features=static_features,
        )
        print("Training complete")

        print("Logging the trained model to MLflow")
        flavor.log_model(
            forecaster, artifact_path=model_name
        )  # artifact_path is now name in new mlflow version

    return forecaster, mlflow.get_run(run_env.info.run_id)


if __name__ == "__main__":
    df = get_data()
    run(df)
