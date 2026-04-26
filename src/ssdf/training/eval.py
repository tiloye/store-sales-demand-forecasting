from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

from mlforecast import MLForecast
from ssdf.training.utils import get_avg_daily_sales
from ssdf.config import FH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def get_train_test_sets(
    df: pd.DataFrame, fh: int = FH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=fh - 1)

    train = df[df["date"] < test_start].copy()
    test = df[df["date"] >= test_start].copy()
    return train, test


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


def plot_avg_sales(data_list: list[pd.DataFrame], store_nbr: int):
    data_list = [data.loc[:, f"store_{store_nbr}"] for data in data_list]
    n_plots = len(data_list)

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = ["True"] + [f"Predicted F{fold}" for fold in range(1, n_plots)]
    for data, label in zip(data_list, labels):
        ax.plot(data.index, data.values, marker="o", label=label)

    ax.set_title(f"Average Daily Sales at Store {store_nbr} [True vs Predicted]")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return fig, ax


def run(
    forecaster: MLForecast,
    df: pd.DataFrame,
    static_features: list[str] | None = None,
    fh: int = FH,
    k: int = 5,
    model_name: str | None = None,
    exp_run_id: str | None = None,
    exp_run_name: str | None = None,
) -> tuple[pd.DataFrame, mlflow.entities.Run]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Splitting data into train and test sets...")
    train, test = get_train_test_sets(df, fh=fh)

    with mlflow.start_run(run_id=exp_run_id, run_name=exp_run_name) as eval_run:
        model_name = (
            forecaster.models["forecaster"].__class__.__name__
            if model_name is None
            else model_name
        )
        model_params = forecaster.models["forecaster"].get_params()
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(model_params)

        print("Logging the training data to MLflow...")
        dataset = mlflow.data.from_pandas(df, targets="sales")
        mlflow.log_input(dataset, context="training")

        print(f"Evaluating forecaster on {k} validation set(s)...")
        cv_res = forecaster.cross_validation(
            df=train,
            n_windows=k,
            step_size=fh,
            h=fh,
            id_col="unique_id",
            time_col="date",
            target_col="sales",
            static_features=static_features,
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
        mlflow.log_metric("avg_cv_rmsle", mean_rmsle)
        mlflow.log_metric("std_cv_rmsle", std_rmsle)

        print(
            "Plotting average daily sales for random stores from cross validation result"
        )
        comparison_list = get_cv_avg_predictions(train, cv_res)
        stores = np.random.choice(np.arange(1, 55), size=5)
        for store_nbr in stores:
            fig, ax = plot_avg_sales(comparison_list, store_nbr)
            mlflow.log_figure(fig, f"plots/cv/avg_sales_store_{store_nbr}.png")
            plt.close(fig)

        print("Evaluating forecaster on test set...")
        forecaster.fit(
            train,
            id_col="unique_id",
            time_col="date",
            target_col="sales",
            static_features=static_features,
        )
        y_pred = forecaster.predict(
            h=fh, X_df=test.drop(["sales"] + static_features, axis=1)
        )

        test_merged = test.merge(y_pred, on=["unique_id", "date"], how="inner")
        test_rmsle = rmsle(test_merged["sales"], test_merged["forecaster"])
        print("Test RMSLE:", test_rmsle)
        mlflow.log_metric("test_rmsle", test_rmsle)

        print("Plotting average daily sales for random stores from test set...")

        test_true_pred = test_merged.copy()
        test_true_pred[["store_nbr", "family"]] = test_true_pred["unique_id"].str.split(
            "_", expand=True
        )
        test_true_pred["store_nbr"] = test_true_pred["store_nbr"].astype(int)

        true_test_avg = get_avg_daily_sales(
            test_true_pred[["store_nbr", "date", "sales"]]
        )
        pred_test_avg = get_avg_daily_sales(
            test_true_pred[["store_nbr", "date", "forecaster"]].rename(
                columns={"forecaster": "sales"}
            )
        )

        train_hist = train[
            train["date"] >= test_merged["date"].min() - pd.Timedelta(days=16)
        ].copy()
        train_hist[["store_nbr", "family"]] = train_hist["unique_id"].str.split(
            "_", expand=True
        )
        train_hist_avg = get_avg_daily_sales(train_hist[["store_nbr", "date", "sales"]])

        test_comparison_list = [
            pd.concat([train_hist_avg, true_test_avg]),
            pred_test_avg,
        ]

        for store_nbr in stores:
            fig, ax = plot_avg_sales(test_comparison_list, store_nbr)
            mlflow.log_figure(fig, f"plots/test/avg_sales_store_{store_nbr}.png")
            plt.close(fig)

    return cv_res, mlflow.get_run(eval_run.info.run_id)


if __name__ == "__main__":
    from ssdf.training.train import get_data, get_model

    df = get_data()
    forecaster = get_model()
    run(forecaster, df)
