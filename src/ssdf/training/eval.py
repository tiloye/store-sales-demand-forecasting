from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

from mlforecast import MLForecast
from ssdf.config import FH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from ssdf.training.utils import get_avg_daily_sales, get_train_test_sets


def rmsle(y_true, y_pred):
    y_pred = np.where(y_pred < 0, 0, y_pred)
    return root_mean_squared_log_error(y_true, y_pred)


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


def plot_avg_sales(data_list: list[pd.DataFrame]):
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

    # Calculate RMSLE for each fold
    fold_metrics = []
    for cutoff in cv_res["cutoff"].unique():
        fold_df = cv_res[cv_res["cutoff"] == cutoff]
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

    print(
        f"Plotting average daily sales across stores for {'test' if backtest else 'cross validation'} result"
    )
    comparison_list = get_cv_avg_predictions(df, cv_res)
    fig, ax = plot_avg_sales(comparison_list)
    plots = {"avg_daily_sales_across_stores": fig}
    return metrics, plots


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
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_id=exp_run_id, run_name=exp_run_name) as eval_run:
        model_name = (
            forecaster.models["forecaster"].__class__.__name__
            if model_name is None
            else model_name
        )
        model_params = forecaster.models["forecaster"].get_params()
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(model_params)

        print("Logging the data to MLflow...")
        train_df, test_df = get_train_test_sets(df, fh * k)
        train_dataset = mlflow.data.from_pandas(train_df, targets="sales")
        test_dataset = mlflow.data.from_pandas(test_df, targets="sales")
        train_dataset_tags = {
            "start_date": str(train_df["date"].min()),
            "end_date": str(train_df["date"].max()),
        }
        test_dataset_tags = {
            "start_date": str(test_df["date"].min()),
            "end_date": str(test_df["date"].max()),
        }
        mlflow.log_input(train_dataset, context="training", tags=train_dataset_tags)
        mlflow.log_input(test_dataset, context="testing", tags=test_dataset_tags)

        print("Evaluating (cross-validation) forecaster on train set...")
        metrics, cv_plots = cross_validate(
            forecaster,
            train_df,
            fh=fh,
            k=k,
            static_features=static_features,
            refit=refit,
            backtest=False,
        )
        mlflow.log_metrics(metrics)
        plot_dir = "plots/cv/"
        for fig_name, fig in cv_plots.items():
            mlflow.log_figure(fig, f"{plot_dir}{fig_name}.png")

        print("Evaluating (backtesting) forecaster on test set...")
        metrics, cv_plots = cross_validate(
            forecaster,
            df,
            fh=fh,
            k=k,
            static_features=static_features,
            refit=refit,
            backtest=True,
        )
        mlflow.log_metrics(metrics)
        plot_dir = "plots/test/"
        for fig_name, fig in cv_plots.items():
            mlflow.log_figure(fig, f"{plot_dir}{fig_name}.png")

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
        model_name="DecisionTreeRegressor",
    )
