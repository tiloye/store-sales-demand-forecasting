import pandas as pd
import pytest
from ssdf.features.feature import create_features, create_target


@pytest.fixture
def dummy_sales_data():
    return pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02"],
            "store_nbr": [1, 1],
            "family": ["BEAUTY", "BEAUTY"],
            "sales": [10.0, 15.0],
        }
    )


@pytest.fixture
def dummy_promotions_data():
    return pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-01", "2013-01-02"],
            "store_nbr": [1, 2, 1],
            "family": ["BEAUTY", "BEAUTY", "BEAUTY"],
            "onpromotion": [0, 1, 0],
        }
    )


@pytest.fixture
def dummy_target_data():
    return pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02"],
            "sales": [10.0, 15.0],
            "unique_id": ["1_BEAUTY", "1_BEAUTY"],
        }
    )


def test_create_target(dummy_sales_data, tmp_path, monkeypatch):
    mock_processed_dir = tmp_path / "processed"
    mock_features_dir = tmp_path / "features"
    mock_processed_dir.mkdir()
    mock_features_dir.mkdir()

    dummy_sales_data.to_parquet(mock_processed_dir / "sales.parquet")

    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    create_target()

    saved_target = pd.read_parquet(mock_features_dir / "target.parquet")

    expected_target = pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02"],
            "sales": [10.0, 15.0],
            "unique_id": ["1_BEAUTY", "1_BEAUTY"],
        }
    )

    pd.testing.assert_frame_equal(saved_target, expected_target, check_like=True)


def test_create_features(
    dummy_promotions_data, dummy_target_data, tmp_path, monkeypatch
):
    mock_processed_dir = tmp_path / "processed"
    mock_features_dir = tmp_path / "features"
    mock_processed_dir.mkdir()
    mock_features_dir.mkdir()

    dummy_promotions_data.to_parquet(mock_processed_dir / "promotions.parquet")
    dummy_target_data.to_parquet(mock_features_dir / "target.parquet")

    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    create_features()

    saved_features = pd.read_parquet(mock_features_dir / "features.parquet")

    expected_features = pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02"],
            "store_nbr": [1, 1],
            "family": ["BEAUTY", "BEAUTY"],
            "onpromotion": [0, 0],
            "unique_id": ["1_BEAUTY", "1_BEAUTY"],
        }
    )

    pd.testing.assert_frame_equal(saved_features, expected_features, check_like=True)
