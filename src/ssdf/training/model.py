from mlforecast import MLForecast

# from sklearn.compose import ColumnTransformer
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator


class SeasonalNaiveRegressor(BaseEstimator):
    def __init__(self, lag: int = 7):
        self.lag = lag

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[f"lag{self.lag}"]


# MODEL_NAME = "DecisionTreeRegressor"
# PARAM_GRID = {"decisiontreeregressor__max_depth": list(range(2, 6, 2))}


def get_model() -> MLForecast:
    # regressor = DecisionTreeRegressor(max_depth=10, random_state=42)
    # encoder = OrdinalEncoder()
    # ctransformer = ColumnTransformer(
    #     [("encoder", encoder, ["store_nbr", "family"])],
    #     remainder="passthrough",
    # )
    # pipeline = make_pipeline(ctransformer, regressor)
    regressor = SeasonalNaiveRegressor(lag=7)
    forecaster = MLForecast(
        models={"forecaster": regressor},
        freq="D",
        lags=list(range(1, 17)),
        date_features=["dayofweek"],
        num_threads=4,
    )
    return forecaster
