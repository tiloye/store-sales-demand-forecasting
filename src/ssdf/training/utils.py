from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import log_artifact

from ssdf.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

if TYPE_CHECKING:
    from mlforecast import MLForecast


@contextmanager
def mlflow_run(
    model: MLForecast,
    model_name: str | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    datasets: list[tuple[pd.DataFrame, str]] | None = None,
    log_model_metadata: bool = True,
) -> Iterator[mlflow.entities.Run]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_id=run_id, run_name=run_name, tags=tags) as run:
        if log_model_metadata:
            resolved_name = model_name or model.models["forecaster"].__class__.__name__
            model_params = model.models["forecaster"].get_params()
            mlflow.set_tag("model_name", resolved_name)
            mlflow.log_params(model_params)

        if datasets:
            for df, context in datasets:
                dataset = mlflow.data.from_pandas(df, targets="sales")
                dataset_tags = {
                    "start_date": str(df["date"].min()),
                    "end_date": str(df["date"].max()),
                }
                mlflow.log_input(dataset, context=context, tags=dataset_tags)

        yield run


def log_mlflow_figures(plots: dict[str, plt.Figure], plot_dir: str = "plots/") -> None:
    for fig_name, fig in plots.items():
        mlflow.log_figure(fig, f"{plot_dir}{fig_name}.png")


def get_best_model_run_id_from_mlflow(experiment_name: str) -> str | None:
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.avg_test_rmsle >= 0",
        order_by=["metrics.avg_test_rmsle ASC"],
        max_results=1,
    )

    if not runs:
        return None

    return runs[0].info.run_id


def get_train_test_sets(
    df: pd.DataFrame, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_size - 1)

    train = df[df["date"] < test_start]
    test = df[df["date"] >= test_start]
    return train, test


def log_model_artifact(forecaster: MLForecast) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(forecaster, f)
        log_artifact(model_path, artifact_path="model")
