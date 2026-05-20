import pytest
import pandas as pd


from ssdf.monitoring.metrics import get_data, get_ref_curr_data


@pytest.fixture
def dummy_data():
    dates = pd.date_range(start="2023-01-01", end="2023-01-10")
    data = pd.DataFrame({"date": dates, "value": range(len(dates))})
    return data


def test_get_data_unsupported_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset: invalid"):
        get_data("invalid")


@pytest.mark.parametrize("dataset", ["predictions", "features"])
def test_get_data_no_dates(dataset, dummy_data, tmp_path, monkeypatch):
    if dataset == "predictions":
        dummy_data = dummy_data.rename(columns={"value": "sales"})
        dummy_data.to_parquet(tmp_path / "sales_forecasts.parquet")
        monkeypatch.setattr("ssdf.monitoring.metrics.PREDICTIONS_DIR", tmp_path)
    elif dataset == "features":
        dummy_data.to_parquet(tmp_path / "features.parquet")
        monkeypatch.setattr("ssdf.monitoring.metrics.FEATURES_DATA_DIR", tmp_path)
    monkeypatch.setattr("ssdf.monitoring.metrics.FH", 3)

    ref_data, curr_data = get_data(dataset)
    expected_curr_start = "2023-01-08"

    expected_curr = dummy_data[dummy_data["date"] >= expected_curr_start]
    expected_ref = dummy_data[dummy_data["date"] < expected_curr_start]
    if dataset == "predictions":
        expected_curr.rename(columns={"sales": "sales_forecast"}, inplace=True)
        expected_ref.rename(columns={"sales": "sales_forecast"}, inplace=True)

    pd.testing.assert_frame_equal(curr_data, expected_curr)
    pd.testing.assert_frame_equal(ref_data, expected_ref)


@pytest.mark.parametrize("dataset", ["predictions", "features"])
def test_get_data_with_dates(dataset, dummy_data, tmp_path, monkeypatch):
    if dataset == "predictions":
        dummy_data = dummy_data.rename(columns={"value": "sales"})
        dummy_data.to_parquet(tmp_path / "sales_forecasts.parquet")
        monkeypatch.setattr("ssdf.monitoring.metrics.PREDICTIONS_DIR", tmp_path)
    elif dataset == "features":
        dummy_data.to_parquet(tmp_path / "features.parquet")
        monkeypatch.setattr("ssdf.monitoring.metrics.FEATURES_DATA_DIR", tmp_path)

    ref_start = pd.Timestamp("2023-01-02")
    ref_end = pd.Timestamp("2023-01-04")
    curr_start = pd.Timestamp("2023-01-06")
    curr_end = pd.Timestamp("2023-01-08")

    ref_data, curr_data = get_data(
        dataset,
        ref_start_date=ref_start,
        ref_end_date=ref_end,
        curr_start_date=curr_start,
        curr_end_date=curr_end,
    )

    expected_ref = dummy_data[
        (dummy_data["date"] >= ref_start) & (dummy_data["date"] <= ref_end)
    ]
    expected_curr = dummy_data[
        (dummy_data["date"] >= curr_start) & (dummy_data["date"] <= curr_end)
    ]
    if dataset == "predictions":
        expected_curr.rename(columns={"sales": "sales_forecast"}, inplace=True)
        expected_ref.rename(columns={"sales": "sales_forecast"}, inplace=True)

    pd.testing.assert_frame_equal(curr_data, expected_curr)
    pd.testing.assert_frame_equal(ref_data, expected_ref)


@pytest.mark.parametrize("dataset", ["predictions", "features"])
def test_get_data_raises_error_with_unspecified_dates(
    dummy_data, tmp_path, monkeypatch, dataset
):
    if dataset == "predictions":
        dummy_data = dummy_data.rename(columns={"value": "sales"})
        dummy_data.to_parquet(tmp_path / "sales_forecasts.parquet")
        monkeypatch.setattr("ssdf.monitoring.metrics.PREDICTIONS_DIR", tmp_path)
    elif dataset == "features":
        dummy_data.to_parquet(tmp_path / "features.parquet")
        monkeypatch.setattr("ssdf.monitoring.metrics.FEATURES_DATA_DIR", tmp_path)

    with pytest.raises(AssertionError):
        get_data(
            dataset,
            curr_start_date=pd.Timestamp("2023-01-06"),
        )


def test_get_ref_curr_data(training_data, tmp_path, monkeypatch):
    monkeypatch.setattr("ssdf.monitoring.metrics.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("ssdf.monitoring.metrics.FEATURES_DATA_DIR", tmp_path)
    monkeypatch.setattr("ssdf.monitoring.metrics.FH", 3)

    predictions_data = training_data[["date", "store_nbr", "family", "sales"]].copy()
    predictions_data.to_parquet(tmp_path / "sales_forecasts.parquet")

    features_data = training_data[["date", "store_nbr", "family", "onpromotion"]].copy()
    features_data.to_parquet(tmp_path / "features.parquet")

    ref_data, curr_data = get_ref_curr_data()
    ref_data, curr_data = ref_data.as_dataframe(), curr_data.as_dataframe()

    expected_curr_start = pd.Timestamp("2023-01-28")
    expected_cols = ["date", "store_nbr", "family", "onpromotion", "sales_forecast"]

    assert set(ref_data.columns) == set(expected_cols)
    assert set(curr_data.columns) == set(expected_cols)

    assert curr_data["date"].min() == expected_curr_start
    assert ref_data["date"].max() < expected_curr_start
