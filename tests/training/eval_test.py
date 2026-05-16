import mlflow
import pandas as pd
from mlforecast import MLForecast
from sklearn.dummy import DummyRegressor

from ssdf.config import STATIC_FEATURES
from ssdf.training.eval import rmsle, get_cv_avg_predictions, run


def test_rmsle():
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, -5.0])  # Negative prediction to test handling
    score = rmsle(y_true, y_pred)
    assert isinstance(score, float)
    assert score >= 0


def test_get_cv_avg_predictions(training_data):
    # Create mock cv dataframe
    cv_df = pd.DataFrame(
        {
            "unique_id": ["1_A", "1_A", "2_B", "2_B"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]
            ),
            "cutoff": pd.to_datetime(
                ["2019-12-31", "2019-12-31", "2019-12-31", "2019-12-31"]
            ),
            "sales": [10.0, 20.0, 30.0, 40.0],
            "forecaster": [12.0, 18.0, 28.0, 42.0],
        }
    )

    comparison_list = get_cv_avg_predictions(training_data, cv_df)

    assert len(comparison_list) == 2  # true + 1 fold
    assert "store_1" in comparison_list[0].columns


def test_run(monkeypatch, training_data, mlflow_configs):
    monkeypatch.setattr(
        "ssdf.training.eval.MLFLOW_TRACKING_URI", mlflow_configs["tracking_uri"]
    )

    forecaster = MLForecast(
        models={"forecaster": DummyRegressor()},
        freq="D",
        lags=[3],
    )

    mlflow_run = run(
        forecaster, training_data, fh=3, static_features=STATIC_FEATURES, k=2
    )

    assert isinstance(mlflow_run, mlflow.entities.Run)
    assert "avg_cv_rmsle" in mlflow_run.data.metrics
    assert "std_cv_rmsle" in mlflow_run.data.metrics
    assert "avg_test_rmsle" in mlflow_run.data.metrics
    assert "std_test_rmsle" in mlflow_run.data.metrics
    assert "model_name" in mlflow_run.data.tags
    assert len(mlflow_run.inputs.dataset_inputs) == 2

    client = mlflow.MlflowClient()
    cv_artifacts = [
        a.path for a in client.list_artifacts(mlflow_run.info.run_id, "plots/cv")
    ]
    test_artifacts = [
        a.path for a in client.list_artifacts(mlflow_run.info.run_id, "plots/test")
    ]
    assert "plots/cv/avg_daily_sales_across_stores.png" in cv_artifacts
    assert "plots/test/avg_daily_sales_across_stores.png" in test_artifacts
