import numpy as np
import pytest
import pandas as pd
from ssdf.features.utils import prep_mlforecast_data


@pytest.fixture
def training_data():
    dates = pd.date_range("2023-01-01", "2023-01-30", freq="D")
    stores = [1, 2]
    families = ["a", "b"]
    records = []
    for d in dates:
        for s in stores:
            for f in families:
                records.append(
                    {
                        "date": d,
                        "store_nbr": s,
                        "family": f,
                        "sales": float(np.random.randint(10, 100)),
                        "onpromotion": np.random.randint(0, 2),
                    }
                )
    df = pd.DataFrame(records)
    df = prep_mlforecast_data(df)
    cols = ["unique_id", "date", "sales"] + [
        c for c in df.columns if c not in ["unique_id", "date", "sales"]
    ]
    return df[cols]


@pytest.fixture
def mlflow_configs():
    return {
        "tracking_uri": "file:///tmp/mlflow_tracking",
        "experiment_name": "Store Sales Demand Forecasting",
    }
