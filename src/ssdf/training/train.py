import pickle

import mlflow
import pandas as pd
from mlforecast import MLForecast, flavor

from ssdf.config import (
    ENV_NAME,
    FEATURES_DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_REGISTRY_NAME,
)
from ssdf.data_io import read_data_from_storage
from ssdf.training.model import get_model
from ssdf.training.utils import get_best_model_run_id_from_mlflow, mlflow_run


def get_data() -> pd.DataFrame:
    target = read_data_from_storage(FEATURES_DATA_DIR / "target.parquet")
    features = read_data_from_storage(FEATURES_DATA_DIR / "features.parquet")
    df = pd.merge(target, features, on=["unique_id", "date"])
    cols = ["unique_id", "date", "sales"] + [
        f for f in features.columns if f not in ["unique_id", "date"]
    ]
    return df[cols]


def run(
    df: pd.DataFrame,
    static_features: list[str] | None = None,
    model_name: str | None = None,
    exp_run_id: str | None = None,
    exp_run_name: str | None = None,
    pull_best_model_artifact: bool = False,
    register_model: bool = False,
) -> tuple[MLForecast, mlflow.entities.Run]:
    if pull_best_model_artifact:
        print("Pulling best model artifact from MLflow...")
        best_model_run_id = get_best_model_run_id_from_mlflow(MLFLOW_EXPERIMENT_NAME)
        if best_model_run_id:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=best_model_run_id, artifact_path="model/model.pkl"
            )
            print(f"Loading best model from {local_path}")
            with open(local_path, "rb") as f:
                forecaster = pickle.load(f)
        else:
            print("No best model found in MLflow. Using defaults.")
            forecaster = get_model()
    else:
        forecaster = get_model()

    with mlflow_run(
        forecaster,
        model_name=model_name,
        run_id=exp_run_id,
        run_name=exp_run_name,
        datasets=[(df, "training")],
    ) as run_env:
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
        model_info = flavor.log_model(
            forecaster, artifact_path=model_name
        )  # artifact_path is now name in new mlflow version

        if register_model:
            print("Registering the trained model to MLflow...")
            model_version = mlflow.register_model(
                model_info.model_uri,
                name=MLFLOW_MODEL_REGISTRY_NAME,
            )
            mlflow.MlflowClient().set_registered_model_alias(
                name=MLFLOW_MODEL_REGISTRY_NAME,
                alias=ENV_NAME,
                version=model_version.version,
            )
            print(
                f"Successfully registered the trained model to MLflow as version {model_version.version}"
            )

    return forecaster, mlflow.get_run(run_env.info.run_id)


if __name__ == "__main__":
    from ssdf.config import STATIC_FEATURES
    from ssdf.training.model import MODEL_NAME

    df = get_data()
    run(
        df,
        static_features=STATIC_FEATURES,
        model_name=MODEL_NAME,
        pull_best_model_artifact=True,
        register_model=True,
    )
