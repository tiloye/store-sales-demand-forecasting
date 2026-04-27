import pandas as pd
import mlflow
from mlforecast import MLForecast, flavor

from ssdf.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, FEATURES_DATA_DIR
from ssdf.training.model import get_model


def get_data() -> pd.DataFrame:
    target = pd.read_parquet(FEATURES_DATA_DIR / "target.parquet")
    features = pd.read_parquet(FEATURES_DATA_DIR / "features.parquet")
    df = pd.merge(target, features, on=["unique_id", "date"])
    cols = ["unique_id", "date", "sales"] + [
        f for f in features.columns if f not in ["unique_id", "date"]
    ]
    return df[cols]


def run(
    df,
    static_features: list[str] | None = None,
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
    from ssdf.config import STATIC_FEATURES

    df = get_data()
    run(df, static_features=STATIC_FEATURES)
