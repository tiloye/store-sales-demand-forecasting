import numpy as np
import pytest
import pandas as pd
from mlforecast import MLForecast
from ssdf.features.utils import prep_mlforecast_data
from ssdf.training.train import SeasonalNaiveRegressor


@pytest.fixture
def forecaster():
    forecaster_ = MLForecast(
        models={"forecaster": SeasonalNaiveRegressor(sp=3)},
        freq="D",
        lags=[3],
    )
    return forecaster_


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
                    }
                )
    df = pd.DataFrame(records)
    df = prep_mlforecast_data(df).drop(["store_nbr", "family"], axis=1)
    return df


@pytest.fixture
def mlflow_configs():
    return {
        "tracking_uri": "file:///tmp/mlflow_tracking",
        "experiment_name": "Store Sales Demand Forecasting",
    }
