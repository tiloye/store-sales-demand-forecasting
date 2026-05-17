from mlforecast import MLForecast
from sklearn.base import BaseEstimator


class SeasonalNaiveRegressor(BaseEstimator):
    def __init__(self, lag: int = 7):
        self.lag = lag

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[f"lag{self.lag}"]


MODEL_NAME = "SeasonalNaiveRegressor"
PARAM_GRID = {"lag": [7, 14, 16]}


def get_model() -> MLForecast:
    regressor = SeasonalNaiveRegressor(lag=7)
    forecaster = MLForecast(
        models={"forecaster": regressor},
        freq="D",
        lags=list(range(1, 17)),
        date_features=["dayofweek"],
        num_threads=4,
    )
    return forecaster
