from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import MultiIndex, Timedelta, concat, period_range

from ssdf.config import FEATURES_DATA_DIR, PROCESSED_DATA_DIR
from ssdf.data_io import read_data_from_storage, write_data_to_storage
from ssdf.features.utils import prep_mlforecast_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas import DataFrame


def create_target() -> None:
    target = read_data_from_storage(PROCESSED_DATA_DIR / "sales.parquet")
    target = prep_mlforecast_data(target).drop(["store_nbr", "family"], axis=1)
    write_data_to_storage(target, FEATURES_DATA_DIR / "target.parquet")


def _create_feature_ids(fh: int) -> DataFrame:
    """Create a DataFrame of date * unique_id pairs spanning historical and future dates."""
    target_ids = read_data_from_storage(
        FEATURES_DATA_DIR / "target.parquet", columns=["date", "unique_id"]
    )
    future_dates = period_range(
        start=target_ids["date"].max() + Timedelta(days=1), periods=fh, freq="D"
    ).to_timestamp()
    future_ids = (
        MultiIndex.from_product(
            [future_dates, target_ids["unique_id"].unique()],
            names=["date", "unique_id"],
        )
        .to_frame()
        .reset_index(drop=True)
    )
    features = concat([target_ids, future_ids], ignore_index=True)
    features[["store_nbr", "family"]] = features["unique_id"].str.split(
        "_", expand=True
    )
    features["store_nbr"] = features["store_nbr"].astype(int)
    return features


def create_features(
    fh: int, feature_funcs: list[Callable[[DataFrame], DataFrame]]
) -> None:
    features = _create_feature_ids(fh)
    for func in feature_funcs:
        features = func(features)
    write_data_to_storage(features, FEATURES_DATA_DIR / "features.parquet")


def _add_promotions(features: DataFrame) -> DataFrame:
    promotions = read_data_from_storage(PROCESSED_DATA_DIR / "promotions.parquet")
    features = features.merge(
        promotions, on=["date", "store_nbr", "family"], how="left"
    )
    return features


def _add_transactions(features: DataFrame) -> DataFrame:
    transactions = read_data_from_storage(PROCESSED_DATA_DIR / "transactions.parquet")
    features = features.merge(transactions, on=["date", "store_nbr"], how="left")
    features["transactions"] = features["transactions"].fillna(0)
    return features


def _add_stores(features: DataFrame) -> DataFrame:
    stores = read_data_from_storage(PROCESSED_DATA_DIR / "stores.parquet")
    stores = stores.rename(
        columns={
            "city": "store_city",
            "state": "store_state",
            "type": "store_type",
            "cluster": "store_cluster",
        }
    )
    features = features.merge(stores, on="store_nbr", how="left")
    return features


DEFAULT_FEATURE_FUNCS: list[Callable[[DataFrame], DataFrame]] = [
    _add_promotions,
    _add_transactions,
    _add_stores,
]


if __name__ == "__main__":
    create_target()
    create_features(16, DEFAULT_FEATURE_FUNCS)
