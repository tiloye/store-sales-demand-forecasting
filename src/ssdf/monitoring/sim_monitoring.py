import pandas as pd

from evidently import Dataset, DataDefinition

from ssdf.config import EVIDENTLY_PROJECT_NAME, FH, STATIC_FEATURES
from ssdf.monitoring.metrics import get_project, generate_drift_snapshot, log_snapshot
from ssdf.training.model import get_model
from ssdf.training.train import get_data

print("Instantiating model and loading data...")
model = get_model()
df = get_data()

print("Generating backtest results...")
val_start_date = "2017-01-01"
val_end_date = "2017-08-15"
n_windows = (
    (pd.to_datetime(val_end_date) - pd.to_datetime(val_start_date)).days + 1
) // FH

val_results = model.cross_validation(
    df,
    h=FH,
    n_windows=n_windows,
    time_col="date",
    target_col="sales",
    refit=False,
    static_features=STATIC_FEATURES,
)
data = (
    val_results[["unique_id", "date", "forecaster", "cutoff"]]
    .copy()
    .merge(df, on=["unique_id", "date"])
    .drop("sales", axis=1)
).rename(columns={"forecaster": "sales_forecast"})

print("Creating evidently reports...")
project = get_project(EVIDENTLY_PROJECT_NAME)
cutoffs = data["cutoff"].unique()

for cutoff in cutoffs[1:]:
    ref_data = data.loc[data["cutoff"] < cutoff]
    curr_data = data.loc[data["cutoff"] == cutoff]
    schema = DataDefinition(
        numerical_columns=["sales_forecast", "onpromotion"],
        categorical_columns=STATIC_FEATURES,
        timestamp="date",
    )
    ref_dataset = Dataset.from_pandas(ref_data, data_definition=schema)
    curr_dataset = Dataset.from_pandas(curr_data, data_definition=schema)

    snapshot = generate_drift_snapshot(curr_dataset, curr_dataset, timestamp=cutoff)
    log_snapshot(snapshot, project)
print("Successfully created dummy evidently reports.")
