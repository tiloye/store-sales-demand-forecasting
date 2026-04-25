import kagglehub
import pandas as pd
from itertools import product
from dotenv import load_dotenv
from ssdf.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

load_dotenv()


def get_source_data(path: str | None = None, force_download: bool = False) -> str:
    """Get the source data from the Kaggle competition."""
    return kagglehub.competition_download(
        "store-sales-time-series-forecasting",
        path=path,
        output_dir=RAW_DATA_DIR.as_posix(),
        force_download=force_download,
    )


def wrangle_train_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the sales data (train.csv on kaggle) for analysis and modeling.
    """

    df = df.copy()

    df.drop("id", axis=1, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    # Impute missing dates and store-family combinations with 0 values
    missing_dates = pd.date_range(df["date"].min(), df["date"].max()).difference(
        df["date"]
    )
    if len(missing_dates) > 0:
        missing_data = list(
            product(missing_dates, df["store_nbr"].unique(), df["family"].unique())
        )
        missing_rows = pd.DataFrame(
            missing_data, columns=["date", "store_nbr", "family"]
        )
        df = pd.concat([df, missing_rows], ignore_index=True).fillna(0)

    df["family"] = df["family"].str.lower()
    df["onpromotion"] = df["onpromotion"].astype(int)

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return df


def run(path: str | None = None, force_download: bool = False):
    try:
        print("Downloading the source data...")
        path = get_source_data(path, force_download)
        print("Successfully downloaded the source data")
    except FileExistsError:
        print("Source data already exists, skipping download")

    print("Wrangling train data...")
    df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    wtrain_df = wrangle_train_test(df)
    print("Successfully wrangled train data")

    print("Wrangling test data...")
    df = pd.read_csv(RAW_DATA_DIR / "test.csv")
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
    sales.to_parquet(PROCESSED_DATA_DIR / "sales.parquet", index=False)
    print("Successfully saved sales data")

    print("Saving promotions data...")
    promotions.to_parquet(PROCESSED_DATA_DIR / "promotions.parquet", index=False)
    print("Successfully saved promotions data")


if __name__ == "__main__":
    run()
