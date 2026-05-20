from __future__ import annotations

import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from ssdf.config import (
    FH,
    FEATURES_DATA_DIR,
    PREDICTIONS_DIR,
    STATIC_FEATURES,
)
from ssdf.monitoring.utils import get_project, log_snapshot
from ssdf.data_io import read_data_from_storage


def generate_drift_snapshot(
    current_data: pd.DataFrame | Dataset,
    reference_data: pd.DataFrame | Dataset | None = None,
    timestamp: pd.Timestamp | None = None,
):
    drift_report = Report([DataDriftPreset(columns=["sales_forecast", "onpromotion"])])
    snapshot = drift_report.run(
        reference_data=reference_data, current_data=current_data, timestamp=timestamp
    )
    return snapshot


def get_ref_curr_data(
    ref_start_date: pd.Timestamp | None = None,
    ref_end_date: pd.Timestamp | None = None,
    curr_start_date: pd.Timestamp | None = None,
    curr_end_date: pd.Timestamp | None = None,
) -> tuple[Dataset, Dataset]:
    pred_drift_ref, pred_drift_curr = get_pred_data(
        ref_start_date, ref_end_date, curr_start_date, curr_end_date
    )
    feature_drift_ref, feature_drift_curr = get_feature_data(
        ref_start_date, ref_end_date, curr_start_date, curr_end_date
    )

    ref_data = pred_drift_ref.merge(
        feature_drift_ref, on=["date", "store_nbr", "family"]
    )
    curr_data = pred_drift_curr.merge(
        feature_drift_curr, on=["date", "store_nbr", "family"]
    )

    schema = DataDefinition(
        numerical_columns=["sales_forecast", "onpromotion"],
        categorical_columns=STATIC_FEATURES,
        timestamp="date",
    )
    ref_dataset = Dataset.from_pandas(ref_data, data_definition=schema)
    curr_dataset = Dataset.from_pandas(curr_data, data_definition=schema)
    return ref_dataset, curr_dataset


def get_pred_data(
    ref_start_date: pd.Timestamp | None = None,
    ref_end_date: pd.Timestamp | None = None,
    curr_start_date: pd.Timestamp | None = None,
    curr_end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return get_data(
        "predictions", ref_start_date, ref_end_date, curr_start_date, curr_end_date
    )


def get_feature_data(
    ref_start_date: pd.Timestamp | None = None,
    ref_end_date: pd.Timestamp | None = None,
    curr_start_date: pd.Timestamp | None = None,
    curr_end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return get_data(
        "features", ref_start_date, ref_end_date, curr_start_date, curr_end_date
    )


def get_data(
    dataset: str,
    ref_start_date: pd.Timestamp | None = None,
    ref_end_date: pd.Timestamp | None = None,
    curr_start_date: pd.Timestamp | None = None,
    curr_end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "predictions":
        data_path = PREDICTIONS_DIR / "sales_forecasts.parquet"
    elif dataset == "features":
        data_path = FEATURES_DATA_DIR / "features.parquet"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    data = read_data_from_storage(data_path)

    if dataset == "predictions":
        data.rename(columns={"sales": "sales_forecast"}, inplace=True)

    if (ref_start_date is None and ref_end_date is None) and (
        curr_start_date is None and curr_end_date is None
    ):
        max_date = data["date"].max()
        curr_start_date = max_date - pd.Timedelta(days=FH - 1)
        curr_data = data.loc[data["date"] >= curr_start_date]
        ref_data = data.loc[data["date"] < curr_start_date]
        return ref_data, curr_data

    assert curr_start_date is not None
    assert curr_end_date is not None
    assert ref_start_date is not None
    assert ref_end_date is not None

    curr_data = data.loc[
        (data["date"] >= curr_start_date) & (data["date"] <= curr_end_date)
    ]
    ref_data = data.loc[
        (data["date"] >= ref_start_date) & (data["date"] <= ref_end_date)
    ]
    return ref_data, curr_data


if __name__ == "__main__":
    from ssdf.config import EVIDENTLY_PROJECT_NAME

    print("Getting/Creating Evidently project...")
    project = get_project(EVIDENTLY_PROJECT_NAME)

    print("Getting reference and current data...")
    ref_data, curr_data = get_ref_curr_data(
        ref_start_date=pd.Timestamp("2017-08-16"),
        ref_end_date=pd.Timestamp("2017-08-23"),
        curr_start_date=pd.Timestamp("2017-08-24"),
        curr_end_date=pd.Timestamp("2017-08-31"),
    )

    print("Generating drift snapshot...")
    snapshot = generate_drift_snapshot(ref_data, curr_data)

    print("Logging drift snapshot to Evidently workspace...")
    log_snapshot(snapshot, project)
    print("Snapshot logged successfully!")
