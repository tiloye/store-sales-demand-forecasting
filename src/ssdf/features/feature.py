from ssdf.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR
from ssdf.features.utils import prep_mlforecast_data
from ssdf.data_io import read_data_from_storage, write_data_to_storage


def create_target():
    print("Creating target variable...")
    target = read_data_from_storage(PROCESSED_DATA_DIR / "sales.parquet")
    target = prep_mlforecast_data(target).drop(["store_nbr", "family"], axis=1)
    write_data_to_storage(target, FEATURES_DATA_DIR / "target.parquet")
    print("Successfully created target variable")


def create_features():
    print("Creating features...")
    promotions = read_data_from_storage(PROCESSED_DATA_DIR / "promotions.parquet")

    features = prep_mlforecast_data(promotions)

    write_data_to_storage(features, FEATURES_DATA_DIR / "features.parquet")
    print("Successfully created features")


if __name__ == "__main__":
    create_target()
    create_features()
