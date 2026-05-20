import pandas as pd

from ssdf.config import PREDICTIONS_DIR, RAW_DATA_DIR
from ssdf.data_io import read_data_from_storage, write_data_to_storage


def create_submission_file():
    """
    Creates a submission file in the required format.
    """
    print("Loading test and forecasts data...")
    test_data = read_data_from_storage(RAW_DATA_DIR / "test.csv")
    test_data["date"] = pd.to_datetime(test_data["date"])
    forecasts = read_data_from_storage(PREDICTIONS_DIR / "sales_forecasts.parquet")
    forecasts["family"] = forecasts["family"].str.upper()
    print("Successfully loaded test and forecasts data")

    print("Creating submission file...")
    submission = test_data.merge(forecasts, on=["date", "store_nbr", "family"])
    submission = submission[["id", "sales"]]
    write_data_to_storage(submission, PREDICTIONS_DIR / "submission.csv", index=False)
    print("Successfully created submission file")


if __name__ == "__main__":
    create_submission_file()
