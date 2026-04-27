import numpy as np
import mlflow
from mlforecast import MLForecast
from sklearn.dummy import DummyRegressor

from ssdf.config import STATIC_FEATURES
from ssdf.training.tune import run_tuning


def test_run_tuning(monkeypatch, training_data, mlflow_configs):
    monkeypatch.setattr(
        "ssdf.training.eval.np.random.choice", lambda *args, **kwargs: np.array([1, 2])
    )
    monkeypatch.setattr(
        "ssdf.training.tune.MLFLOW_TRACKING_URI", mlflow_configs["tracking_uri"]
    )

    forecaster = MLForecast(
        models={"forecaster": DummyRegressor()},
        freq="D",
        lags=[3],
    )

    param_grid = {"strategy": ["mean", "median"]}

    best_params, mlflow_run = run_tuning(
        forecaster,
        training_data,
        param_grid=param_grid,
        fh=3,
        static_features=STATIC_FEATURES,
        k=2,
    )

    assert isinstance(mlflow_run, mlflow.entities.Run)
    assert isinstance(best_params, dict)
    assert best_params != {}

    assert len(mlflow_run.data.metrics) > 0
    assert len(mlflow_run.data.params) > 0
    assert len(mlflow_run.inputs.dataset_inputs) > 0

    client = mlflow.MlflowClient(tracking_uri=mlflow_configs["tracking_uri"])

    # Check if there are nested runs for the two parameters
    query = f"tags.mlflow.parentRunId = '{mlflow_run.info.run_id}'"
    child_runs = client.search_runs(
        experiment_ids=[mlflow_run.info.experiment_id], filter_string=query
    )

    assert len(child_runs) > 1
    for child_run in child_runs:
        assert len(child_run.data.metrics) > 0
        assert len(child_run.data.params) > 0

    # Verify artifacts were logged for the parent test run
    test_artifacts = [
        a.path for a in client.list_artifacts(mlflow_run.info.run_id, "plots/test")
    ]
    assert "plots/test/avg_sales_store_1.png" in test_artifacts
    assert "plots/test/avg_sales_store_2.png" in test_artifacts

    # Verify artifacts were logged for the nested cv runs
    for child_run in child_runs:
        cv_artifacts = [
            a.path for a in client.list_artifacts(child_run.info.run_id, "plots/cv")
        ]
        assert "plots/cv/avg_sales_store_1.png" in cv_artifacts
        assert "plots/cv/avg_sales_store_2.png" in cv_artifacts
