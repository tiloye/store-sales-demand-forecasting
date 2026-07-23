from __future__ import annotations

import copy

import mlflow
import pandas as pd
from joblib import Parallel, delayed
from mlforecast import MLForecast
from sklearn.model_selection import ParameterGrid

from ssdf.config import FH
from ssdf.training.eval import cross_validate
from ssdf.training.utils import (
    get_train_test_sets,
    log_mlflow_figures,
    log_model_artifact,
    mlflow_run,
)


def _eval_param_set(
    params: dict,
    forecaster: MLForecast,
    train: pd.DataFrame,
    fh: int,
    k: int,
    static_features: list[str] | None,
    refit: bool,
    parent_run_id: str,
    tracking_uri: str,
    experiment_id: str,
):
    mlflow.set_tracking_uri(tracking_uri)

    forecaster_copy = copy.deepcopy(forecaster)
    model = forecaster_copy.models["forecaster"]
    model.set_params(**params)

    with mlflow.start_run(
        experiment_id=experiment_id,
        nested=True,
        tags={"mlflow.parentRunId": parent_run_id},
    ):
        mlflow.log_params(model.get_params())
        print(f"Evaluating parameters: {params}")
        metrics, cv_plots = cross_validate(
            forecaster_copy,
            train,
            fh=fh,
            k=k,
            static_features=static_features,
            refit=refit,
        )

        mlflow.log_metrics(metrics)
        log_mlflow_figures(cv_plots, plot_dir="plots/cv/")

        score = metrics["avg_cv_rmsle"]

    return params, score


def run_tuning(
    forecaster: MLForecast,
    df: pd.DataFrame,
    param_grid: dict | list[dict],
    static_features: list[str] | None = None,
    fh: int = FH,
    k: int = 5,
    refit: bool = False,
    n_jobs: int = 1,
    model_name: str | None = None,
    exp_run_name: str | None = None,
) -> tuple[dict, mlflow.entities.Run]:
    print("Splitting data into train and test sets...")
    train, test = get_train_test_sets(df, test_size=fh * k)

    best_score = float("inf")
    best_params = None

    with mlflow_run(
        forecaster,
        model_name=model_name,
        run_name=exp_run_name,
        datasets=[(train, "train"), (test, "test")],
        log_model_metadata=False,
    ) as parent_run:
        tracking_uri = mlflow.get_tracking_uri()
        experiment_id = parent_run.info.experiment_id
        parent_run_id = parent_run.info.run_id

        results = Parallel(n_jobs=n_jobs)(
            delayed(_eval_param_set)(
                params,
                forecaster,
                train,
                fh,
                k,
                static_features,
                refit,
                parent_run_id,
                tracking_uri,
                experiment_id,
            )
            for params in ParameterGrid(param_grid)
        )

        for params, score in results:
            if score < best_score:
                best_score = score
                best_params = params

        print(f"Best params: {best_params} with CV RMSLE: {best_score}")

        if best_params is not None:
            # Format the best parameters to avoid dictionary key conflicts in mlflow
            best_params_log = {
                f"best_{key}": value for key, value in best_params.items()
            }
            mlflow.log_params(best_params_log)

        mlflow.log_metric("best_avg_cv_rmsle", best_score)

        print("Evaluating best model on test set...")
        forecaster.models["forecaster"].set_params(**best_params)
        mlflow.log_params(forecaster.models["forecaster"].get_params())

        test_rmsle, test_plots = cross_validate(
            forecaster,
            df,
            fh=fh,
            k=k,
            static_features=static_features,
            refit=refit,
            backtest=True,
        )
        mlflow.log_metrics(test_rmsle)
        log_mlflow_figures(test_plots, plot_dir="plots/test/")

        print("Logging best model artifact to MLflow...")
        log_model_artifact(forecaster)

    return best_params, mlflow.get_run(parent_run.info.run_id)


if __name__ == "__main__":
    from ssdf.config import STATIC_FEATURES
    from ssdf.training.model import MODEL_NAME, PARAM_GRID, get_model
    from ssdf.training.train import get_data

    df = get_data()
    run_tuning(
        forecaster=get_model(),
        df=df,
        param_grid=PARAM_GRID,
        static_features=STATIC_FEATURES,
        n_jobs=2,
        model_name=MODEL_NAME,
    )
