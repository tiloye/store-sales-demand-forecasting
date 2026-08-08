from __future__ import annotations

import pandas as pd

from ssdf.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from ssdf.data.ingest import get_source_data
from ssdf.data.transform import (
    wrangle_holidays_events,
    wrangle_stores,
    wrangle_train_test,
    wrangle_transactions,
)
from ssdf.data_io import read_data_from_storage, write_data_to_storage


def run(path: str | None = None, force_download: bool = False) -> None:
    try:
        print("Downloading the source data...")
        _ = get_source_data(path, force_download)
        print("Successfully downloaded the source data")
    except FileExistsError:
        print("Source data already exists, skipping download")

    print("Wrangling train data...")
    df = read_data_from_storage(RAW_DATA_DIR / "train.csv")
    wtrain_df = wrangle_train_test(df)
    print("Successfully wrangled train data")

    print("Wrangling test data...")
    df = read_data_from_storage(RAW_DATA_DIR / "test.csv")
    wtest_df = wrangle_train_test(df)
    print("Successfully wrangled test data")

    print("Merging and splitting train and test data to sales and promotions...")
    sales = wtrain_df.drop("onpromotion", axis=1)
    promotions = wtrain_df.drop("sales", axis=1)
    promotions = pd.concat([promotions, wtest_df], ignore_index=True).reset_index(
        drop=True
    )
    print("Successfully merged and split train and test data to sales and promotions")

    print("Saving sales data...")
    write_data_to_storage(sales, PROCESSED_DATA_DIR / "sales.parquet", index=False)
    print("Successfully saved sales data")

    print("Saving promotions data...")
    write_data_to_storage(
        promotions, PROCESSED_DATA_DIR / "promotions.parquet", index=False
    )
    print("Successfully saved promotions data")

    print("Wrangling transactions data...")
    transactions_df = read_data_from_storage(RAW_DATA_DIR / "transactions.csv")
    transactions_df = wrangle_transactions(transactions_df)
    print("Successfully wrangled transactions data")

    print("Saving transactions data...")
    write_data_to_storage(
        transactions_df, PROCESSED_DATA_DIR / "transactions.parquet", index=False
    )
    print("Successfully saved transactions data")

    print("Wrangling holiday events data...")
    holidays_df = read_data_from_storage(RAW_DATA_DIR / "holidays_events.csv")
    holidays_df = wrangle_holidays_events(holidays_df)
    print("Saving holiday events data...")
    write_data_to_storage(
        holidays_df, PROCESSED_DATA_DIR / "holidays_events.parquet", index=False
    )
    print("Successfully saved holiday events data")

    print("Wrangling stores data...")
    stores_df = read_data_from_storage(RAW_DATA_DIR / "stores.csv")
    stores_df = wrangle_stores(stores_df)
    print("Saving stores data...")
    write_data_to_storage(stores_df, PROCESSED_DATA_DIR / "stores.parquet", index=False)
    print("Successfully stores data")


if __name__ == "__main__":
    run()
