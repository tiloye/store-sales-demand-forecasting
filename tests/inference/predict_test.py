import mlflow
import pytest
import pandas as pd
from mlforecast import MLForecast, flavor
from sklearn.dummy import DummyRegressor
from ssdf.config import ENV_NAME, MLFLOW_MODEL_REGISTRY_NAME, STATIC_FEATURES
from ssdf.inference.predict import (
    get_model,
    get_series_update,
    get_features,
    generate_forecasts,
    save_forecasts,
)


@pytest.fixture
def train_model(training_data, mlflow_configs):
    mlflow.set_tracking_uri(mlflow_configs["tracking_uri"])
    mlflow.set_experiment(mlflow_configs["experiment_name"])

    data = training_data

    with mlflow.start_run():
        forecaster = MLForecast(
            models={"forecaster": DummyRegressor()},
            freq="D",
            lags=[3],
        )
        forecaster.fit(
            data,
            id_col="unique_id",
            time_col="date",
            target_col="sales",
            static_features=STATIC_FEATURES,
        )
        model_info = flavor.log_model(forecaster, forecaster.__class__.__name__)

    model_version = mlflow.register_model(
        model_uri=model_info.model_uri, name=MLFLOW_MODEL_REGISTRY_NAME
    )
    mlflow.MlflowClient().set_registered_model_alias(
        name=MLFLOW_MODEL_REGISTRY_NAME,
        alias=ENV_NAME,
        version=model_version.version,
    )
    return model_info.model_id


@pytest.fixture
def features_data(training_data):
    # Create feature data (X_df) with future values
    future_df = training_data.loc[training_data["date"] > "2023-01-27"].reset_index(
        drop=True
    )
    future_df["date"] = future_df["date"] + pd.Timedelta(days=3)
    future_df = future_df.drop(columns=["sales"])
    features_data = pd.concat(
        [training_data.drop("sales", axis=1), future_df]
    ).reset_index(drop=True)
    return features_data


def test_get_model(train_model, monkeypatch):
    model_uri = [None, f"models:/{train_model}"]
    for uri in model_uri:
        model = get_model(uri)
        assert isinstance(model, MLForecast)
        assert "forecaster" in model.models


def test_get_features(features_data, tmp_path, monkeypatch):
    features_data.to_parquet(tmp_path / "features.parquet", index=False)
    future_df = features_data.drop(["store_nbr", "family", "onpromotion"], axis=1)
    future_df = future_df.loc[future_df["date"] > "2023-01-30"].reset_index(drop=True)
    monkeypatch.setattr("ssdf.inference.predict.FEATURES_DATA_DIR", tmp_path)

    features = get_features(future_df)
    expected_df = features_data.loc[
        features_data["date"] > "2023-01-30",
        ["unique_id", "date", "onpromotion"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(features, expected_df)


def test_get_series_update(training_data, features_data, tmp_path, monkeypatch):
    target_data = training_data[["unique_id", "date", "sales"]].copy()
    target_data.to_parquet(tmp_path / "target.parquet", index=False)
    features_data.to_parquet(tmp_path / "features.parquet", index=False)
    monkeypatch.setattr("ssdf.inference.predict.FEATURES_DATA_DIR", tmp_path)

    last_date = pd.Timestamp("2024-01-01")
    assert get_series_update(last_date) is None

    last_date = pd.Timestamp("2023-01-20")
    result = get_series_update(last_date)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(training_data.columns)

    expected_df = training_data.loc[
        (training_data["date"] > last_date) & (training_data["date"] <= "2023-01-30")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_df)


def test_generate_forecast(
    train_model, training_data, features_data, tmp_path, monkeypatch
):
    target_data = training_data[["unique_id", "date", "sales"]].copy()
    target_data.to_parquet(tmp_path / "target.parquet", index=False)
    features_data.to_parquet(tmp_path / "features.parquet", index=False)
    monkeypatch.setattr("ssdf.inference.predict.FEATURES_DATA_DIR", tmp_path)

    forecasts = generate_forecasts(fh=3)

    assert isinstance(forecasts, pd.DataFrame)
    assert set(["unique_id", "date", "forecaster"]).issubset(forecasts.columns)
    assert forecasts["date"].min() == pd.Timestamp("2023-01-31")
    assert forecasts["date"].max() == pd.Timestamp("2023-02-02")
    assert forecasts.shape == (12, 3)


def test_generate_forecast_updates_model(
    train_model, training_data, features_data, tmp_path, monkeypatch
):
    target_data = training_data[["unique_id", "date", "sales"]].copy()

    # Create new targets after training period (which ends on 2023-01-30)
    # This will act as new_df for updating the model (3 days: 31st, 1st, 2nd)
    new_targets = target_data.loc[target_data["date"] > "2023-01-27"].copy()
    new_targets["date"] = new_targets["date"] + pd.Timedelta(days=3)
    target_data = pd.concat([target_data, new_targets]).reset_index(drop=True)

    # Extend features data by 3 days (2023-02-03 to 2023-02-05)
    new_features = features_data.loc[features_data["date"] > "2023-01-30"].copy()
    new_features["date"] = new_features["date"] + pd.Timedelta(days=3)

    features_data = pd.concat([features_data, new_features]).reset_index(drop=True)

    target_data.to_parquet(tmp_path / "target.parquet", index=False)
    features_data.to_parquet(tmp_path / "features.parquet", index=False)
    monkeypatch.setattr("ssdf.inference.predict.FEATURES_DATA_DIR", tmp_path)

    forecasts = generate_forecasts(fh=3)

    assert isinstance(forecasts, pd.DataFrame)
    assert set(["unique_id", "date", "forecaster"]).issubset(forecasts.columns)
    assert forecasts["date"].min() == pd.Timestamp("2023-02-03")
    assert forecasts["date"].max() == pd.Timestamp("2023-02-05")
    assert forecasts.shape == (12, 3)


def test_save_forecasts(tmp_path):
    forecasts = pd.DataFrame(
        {
            "unique_id": ["1_a", "1_a", "2_b", "2_b"],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]
            ),
            "forecaster": [10.0, 15.0, 20.0, 25.0],
        }
    )
    expected_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
            ),
            "store_nbr": [1, 2, 1, 2],
            "family": ["a", "b", "a", "b"],
            "sales": [10.0, 20.0, 15.0, 25.0],
        }
    )

    save_forecasts(forecasts, tmp_path)
    saved_df = pd.read_parquet(
        tmp_path / "sales_forecasts.parquet",
    )

    pd.testing.assert_frame_equal(saved_df, expected_df)
