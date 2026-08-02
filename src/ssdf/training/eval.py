from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from sklearn.metrics import root_mean_squared_log_error

from ssdf.config import FH
from ssdf.training.utils import (
    get_train_test_sets,
    log_mlflow_figures,
    log_model_artifact,
    mlflow_run,
)


def rmsle(y_true, y_pred):
    y_pred = np.where(y_pred < 0, 0, y_pred)
    return root_mean_squared_log_error(y_true, y_pred)


def get_avg_daily_sales(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average daily sales for each store.
    """

    data = data.groupby(["date"])["sales"].mean().to_frame("avg_sales")
    return data


def get_cv_avg_predictions(
    train: pd.DataFrame, cv_df: pd.DataFrame
) -> list[pd.DataFrame]:

    cv_df = cv_df.copy()
    cv_df[["store_nbr", "family"]] = cv_df["unique_id"].str.split("_", expand=True)
    cv_df["store_nbr"] = cv_df["store_nbr"].astype(int)

    cutoffs = cv_df["cutoff"].unique()
    cutoffs = np.sort(cutoffs)

    forecast_list = []
    for cutoff in cutoffs:
        fold_df = cv_df[cv_df["cutoff"] == cutoff]
        pred_df = fold_df[["store_nbr", "date", "forecaster"]].rename(
            columns={"forecaster": "sales"}
        )
        forecast_list.append(get_avg_daily_sales(pred_df))

    train = train.copy()
    train[["store_nbr", "family"]] = train["unique_id"].str.split("_", expand=True)
    true_df = train[["store_nbr", "family", "date", "sales"]].drop_duplicates()
    true_avg_sales = get_avg_daily_sales(true_df)

    start_date = forecast_list[0].index[0] - pd.Timedelta(days=16)
    true_avg_sales = true_avg_sales.loc[start_date:]

    comparison_list = [true_avg_sales] + forecast_list
    return comparison_list


def plot_avg_sales(
    train_df: pd.DataFrame,
    cv_result: pd.DataFrame | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if cv_result is not None:
        data_list = get_cv_avg_predictions(train_df, cv_result)
    else:
        data_list = [get_avg_daily_sales(train_df)]
    n_plots = len(data_list)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    labels = ["True"] + [f"Predicted F{fold}" for fold in range(1, n_plots)]
    for data, label in zip(data_list, labels):
        ax.plot(data.index, data.values, marker="o", label=label)

    ax.set_title("Average Daily Sales Across Stores [True vs Predicted]")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return fig, ax


def compute_cv_metrics(
    cv_result: pd.DataFrame,
    backtest: bool = False,
) -> dict[str, float]:

    # Calculate RMSLE for each fold
    fold_metrics = []
    for cutoff in cv_result["cutoff"].unique():
        fold_df = cv_result[cv_result["cutoff"] == cutoff]
        score = rmsle(fold_df["sales"], fold_df["forecaster"])
        fold_metrics.append(score)

    mean_rmsle = max(0, np.mean(fold_metrics))
    std_rmsle = max(0, np.std(fold_metrics))
    print("Average RMSLE across all folds:", mean_rmsle)
    print("Standard deviation of RMSLE across all folds:", std_rmsle)
    metrics = {
        f"avg_{'test' if backtest else 'cv'}_rmsle": mean_rmsle,
        f"std_{'test' if backtest else 'cv'}_rmsle": std_rmsle,
    }
    return metrics


def cross_validate(
    forecaster: MLForecast,
    df: pd.DataFrame,
    fh: int = FH,
    k: int = 5,
    static_features: list[str] | None = None,
    refit: bool = False,
    backtest: bool = False,
) -> tuple[dict[str, float], dict[str, plt.Figure]]:
    cv_res = forecaster.cross_validation(
        df=df,
        n_windows=k,
        step_size=fh,
        h=fh,
        id_col="unique_id",
        time_col="date",
        target_col="sales",
        static_features=static_features,
        refit=refit,
    )
    eval_metrics = compute_cv_metrics(cv_res, backtest=backtest)
    eval_plot = {"avg_daily_sales_across_stores": plot_avg_sales(df, cv_res)[0]}

    return eval_metrics, eval_plot


def run(
    forecaster: MLForecast,
    df: pd.DataFrame,
    static_features: list[str] | None = None,
    fh: int = FH,
    k: int = 5,
    model_name: str | None = None,
    refit: bool = False,
    exp_run_id: str | None = None,
    exp_run_name: str | None = None,
) -> mlflow.entities.Run:
    forecaster_copy = copy.deepcopy(forecaster)

    print("Logging the data to MLflow...")
    train_df, test_df = get_train_test_sets(df, fh * k)
    datasets = [
        (train_df, "training"),
        (test_df, "testing"),
    ]

    with mlflow_run(
        forecaster_copy,
        model_name=model_name,
        run_id=exp_run_id,
        run_name=exp_run_name,
        datasets=datasets,
    ) as eval_run:
        print("Evaluating (cross-validation) forecaster on train set...")

        eval_metrics, eval_plot = cross_validate(
            forecaster_copy,
            train_df,
            fh=fh,
            k=k,
            static_features=static_features,
            refit=refit,
            backtest=False,
        )
        mlflow.log_metrics(eval_metrics)
        log_mlflow_figures(eval_plot, plot_dir="plots/cv/")

        print("Evaluating (backtesting) forecaster on test set...")
        test_metrics, test_plots = cross_validate(
            forecaster_copy,
            df,
            fh=fh,
            k=k,
            static_features=static_features,
            refit=refit,
            backtest=True,
        )
        mlflow.log_metrics(test_metrics)
        log_mlflow_figures(test_plots, plot_dir="plots/test/")

        log_model_artifact(forecaster)

    return mlflow.get_run(eval_run.info.run_id)


if __name__ == "__main__":
    from ssdf.config import STATIC_FEATURES
    from ssdf.training.train import get_data, get_model

    df = get_data()
    forecaster = get_model()
    run(
        forecaster,
        df,
        static_features=STATIC_FEATURES,
        model_name="SeasonalNaiveRegressor",
    )
