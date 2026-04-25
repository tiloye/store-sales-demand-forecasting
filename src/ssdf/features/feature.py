import pandas as pd
from ssdf.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR
from ssdf.features.utils import prep_mlforecast_data


def create_target():
    print("Creating target variable...")
    target = pd.read_parquet(PROCESSED_DATA_DIR / "sales.parquet")
    target = prep_mlforecast_data(target).drop(["store_nbr", "family"], axis=1)
    target.to_parquet(FEATURES_DATA_DIR / "target.parquet")
    print("Successfully created target variable")


if __name__ == "__main__":
    create_target()
