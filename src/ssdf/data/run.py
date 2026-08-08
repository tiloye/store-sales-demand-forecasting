from __future__ import annotations

import pandas as pd

from ssdf.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from ssdf.data.ingest import get_source_data
from ssdf.data.transform import wrangle_train_test
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


if __name__ == "__main__":
    run()
