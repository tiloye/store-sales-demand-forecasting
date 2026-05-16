import pandas as pd


def get_train_test_sets(
    df: pd.DataFrame, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_size - 1)

    train = df[df["date"] < test_start]
    test = df[df["date"] >= test_start]
    return train, test


def get_avg_daily_sales(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average daily sales for each store.
    """

    data = data.groupby(["date"])["sales"].mean().to_frame("avg_sales")
    return data
