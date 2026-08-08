import numpy as np
import pandas as pd

from ssdf.features.feature import (
    _add_promotions,
    _add_stores,
    _add_transactions,
    _create_feature_ids,
    create_features,
    create_target,
)

np.random.seed(42)

# Setup dummy data

dates = pd.to_datetime(
    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
)
store_nbrs = [1, 2]
family = ["a", "b"]

sales_keys = pd.MultiIndex.from_product(
    (
        dates,
        store_nbrs,
        family,
    ),
    names=["date", "store_nbr", "family"],
)
sales_data = pd.Series(
    np.random.randint(10, 100, len(sales_keys)),
    index=sales_keys,
    dtype=float,
    name="sales",
).reset_index()
sales_data.loc[
    sales_data["date"].eq("2023-01-01") & sales_data["store_nbr"].eq(2), "sales"
] = 0

promotions_keys = pd.MultiIndex.from_product(
    (
        pd.date_range(
            sales_data["date"].min(), sales_data["date"].max() + pd.Timedelta(days=2)
        ),
        store_nbrs,
        family,
    ),
    names=["date", "store_nbr", "family"],
)
promotions_data = pd.Series(
    np.random.randint(0, 10, len(promotions_keys)),
    index=promotions_keys,
    dtype=int,
    name="onpromotion",
).reset_index()

transactions_keys = pd.MultiIndex.from_product(
    (dates, store_nbrs), names=["date", "store_nbr"]
)
transactions_data = pd.Series(
    np.random.randint(100, 300, len(transactions_keys)),
    index=transactions_keys,
    dtype=int,
    name="transactions",
).reset_index()
transactions_data = transactions_data.loc[
    ~(transactions_data["date"].eq("2023-01-01") & transactions_data["store_nbr"].eq(2))
]

stores_data = pd.DataFrame(
    {
        "store_nbr": store_nbrs,
        "city": ["city1", "city2"],
        "state": ["state1", "state2"],
        "type": ["x", "y"],
        "cluster": [5, 6],
    }
)


def test_create_target(tmp_path, monkeypatch):
    mock_processed_dir = tmp_path / "processed"
    mock_processed_dir.mkdir()
    mock_features_dir = tmp_path / "features"
    mock_features_dir.mkdir()
    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    sales_data.to_parquet(mock_processed_dir / "sales.parquet")

    create_target()

    saved_target = pd.read_parquet(mock_features_dir / "target.parquet")

    expected_target = sales_data.copy()
    expected_target["unique_id"] = (
        expected_target["store_nbr"].astype(str) + "_" + expected_target["family"]
    )
    expected_target.drop(columns=["store_nbr", "family"], inplace=True)

    pd.testing.assert_frame_equal(saved_target, expected_target, check_like=True)


def _dummy_a(df):
    df["col_a"] = 1
    return df


def _dummy_b(df):
    df["col_b"] = 2
    return df


def test_create_features(tmp_path, monkeypatch):
    """Test that create_features applies exactly the functions provided."""
    mock_features_dir = tmp_path / "features"
    mock_features_dir.mkdir()
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    target = sales_data.copy()
    target["unique_id"] = target["store_nbr"].astype(str) + "_" + target["family"]
    target[["unique_id", "date"]].to_parquet(mock_features_dir / "target.parquet")

    create_features(2, [_dummy_a])
    result = pd.read_parquet(mock_features_dir / "features.parquet")
    assert "col_a" in result.columns
    assert "col_b" not in result.columns

    create_features(2, [_dummy_a, _dummy_b])
    result = pd.read_parquet(mock_features_dir / "features.parquet")
    assert "col_a" in result.columns
    assert "col_b" in result.columns


def test_create_feature_ids(tmp_path, monkeypatch):
    """Test that _create_feature_ids builds the correct date x unique_id grid."""
    mock_features_dir = tmp_path / "features"
    mock_features_dir.mkdir()
    monkeypatch.setattr("ssdf.features.feature.FEATURES_DATA_DIR", mock_features_dir)

    target = sales_data.copy()
    target["unique_id"] = target["store_nbr"].astype(str) + "_" + target["family"]
    target[["unique_id", "date"]].to_parquet(mock_features_dir / "target.parquet")

    result = _create_feature_ids(fh=2)

    expected_dates = pd.to_datetime(
        [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
            "2023-01-07",
        ]
    )
    expected_unique_ids = ["1_a", "1_b", "2_a", "2_b"]
    expected_rows = len(expected_dates) * len(expected_unique_ids)

    assert len(result) == expected_rows
    assert set(result["date"]) == set(expected_dates)
    assert set(result["unique_id"]) == set(expected_unique_ids)
    assert list(result.columns) == ["date", "unique_id", "store_nbr", "family"]
    assert result["store_nbr"].dtype == int


def test_add_promotions(tmp_path, monkeypatch):
    """Test that _add_promotions merges onpromotion correctly."""
    mock_processed_dir = tmp_path / "processed"
    mock_processed_dir.mkdir()
    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)

    promotions_data.to_parquet(mock_processed_dir / "promotions.parquet")

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "unique_id": ["1_a", "1_a"],
            "store_nbr": [1, 1],
            "family": ["a", "a"],
        }
    )

    result = _add_promotions(features)

    assert "onpromotion" in result.columns
    assert len(result) == 2


def test_add_transactions(tmp_path, monkeypatch):
    """Test that _add_transactions merges and fills NaN with 0."""
    mock_processed_dir = tmp_path / "processed"
    mock_processed_dir.mkdir()
    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)

    transactions_data.to_parquet(mock_processed_dir / "transactions.parquet")

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "unique_id": ["1_a", "1_a"],
            "store_nbr": [1, 1],
            "family": ["a", "a"],
        }
    )

    result = _add_transactions(features)

    assert "transactions" in result.columns
    assert result["transactions"].isna().sum() == 0
    assert len(result) == 2


def test_add_stores(tmp_path, monkeypatch):
    """Test that _add_stores merges and renames columns correctly."""
    mock_processed_dir = tmp_path / "processed"
    mock_processed_dir.mkdir()
    monkeypatch.setattr("ssdf.features.feature.PROCESSED_DATA_DIR", mock_processed_dir)

    stores_data.to_parquet(mock_processed_dir / "stores.parquet")

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "unique_id": ["1_a", "2_b"],
            "store_nbr": [1, 2],
            "family": ["a", "b"],
        }
    )

    result = _add_stores(features)

    assert "store_city" in result.columns
    assert "store_state" in result.columns
    assert "store_type" in result.columns
    assert "store_cluster" in result.columns
    assert "city" not in result.columns
    assert "state" not in result.columns
    assert result.loc[result["store_nbr"] == 1, "store_city"].iloc[0] == "city1"
    assert result.loc[result["store_nbr"] == 2, "store_city"].iloc[0] == "city2"
