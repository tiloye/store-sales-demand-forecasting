from __future__ import annotations

from itertools import product

import pandas as pd


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
