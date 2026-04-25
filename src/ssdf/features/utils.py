import pandas as pd


def prep_mlforecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the data for MLForecast by adding a unique_id column.
    """
    df = df.copy()
    df["unique_id"] = df["store_nbr"].astype(str) + "_" + df["family"]
    return df
