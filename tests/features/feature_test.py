import pandas as pd
import pytest
from ssdf.features.feature import create_features, create_target


def test_create_target(training_data, tmp_path, monkeypatch):
    mock_processed_dir = tmp_path / "processed"
    mock_features_dir = tmp_path / "features"
    mock_processed_dir.mkdir()
    mock_features_dir.mkdir()

    training_data.drop(["unique_id", "onpromotion"], axis=1).to_parquet(
        mock_processed_dir / "sales.parquet"
    )

    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    create_target()

    saved_target = pd.read_parquet(mock_features_dir / "target.parquet")

    expected_target = training_data[["unique_id", "date", "sales"]]

    pd.testing.assert_frame_equal(saved_target, expected_target, check_like=True)


def test_create_features(training_data, tmp_path, monkeypatch):
    mock_processed_dir = tmp_path / "processed"
    mock_features_dir = tmp_path / "features"
    mock_processed_dir.mkdir()
    mock_features_dir.mkdir()

    training_data.drop(["unique_id", "sales"], axis=1).to_parquet(
        mock_processed_dir / "promotions.parquet"
    )
    training_data.drop(["store_nbr", "family", "onpromotion"], axis=1).to_parquet(
        mock_features_dir / "target.parquet"
    )

    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    create_features()

    saved_features = pd.read_parquet(mock_features_dir / "features.parquet")

    expected_features = training_data.drop("sales", axis=1)

    pd.testing.assert_frame_equal(saved_features, expected_features, check_like=True)
