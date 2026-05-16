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
    forecaster = MLForecast(
        models={"forecaster": DummyRegressor()},
        freq="D",
        lags=[3],
    )
    cv_df = forecaster.cross_validation(
        df=training_data,
        n_windows=2,
        h=5,
        id_col="unique_id",
        time_col="date",
        target_col="sales",
        static_features=STATIC_FEATURES,
        refit=False,
    )
    exp_daily_averages_true = (
        training_data.groupby("date")["sales"].mean().to_frame("avg_sales")
    )
    start_date = cv_df["cutoff"].min() - pd.Timedelta(days=16 - 1)
    exp_daily_averages_true = exp_daily_averages_true.loc[start_date:]
    exp_daily_averages_predicted = (
        cv_df.groupby("date")["forecaster"].mean().to_frame("avg_sales")
    )
    exp_result = [
        exp_daily_averages_true,
        exp_daily_averages_predicted.loc["2023-01-21":"2023-01-25"],
        exp_daily_averages_predicted["2023-01-26":"2023-01-30"],
    ]

    comparison_list = get_cv_avg_predictions(training_data, cv_df)

    pd.testing.assert_frame_equal(
        comparison_list[0], exp_result[0], check_index_type=False
    )
    pd.testing.assert_frame_equal(
        comparison_list[1], exp_result[1], check_index_type=False
    )
    pd.testing.assert_frame_equal(
        comparison_list[2], exp_result[2], check_index_type=False
    )


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

    model_artifacts = client.list_artifacts(mlflow_run.info.run_id, "model")
    assert model_artifacts[0].path == "model/model.pkl"
