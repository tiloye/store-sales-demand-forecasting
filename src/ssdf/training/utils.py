import pandas as pd


def get_avg_daily_sales(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average daily sales for each store.
    """

    data = (
        data.groupby(["store_nbr", "date"])["sales"]
        .mean()
        .unstack("store_nbr")
        .add_prefix("store_")
    )
    return data
