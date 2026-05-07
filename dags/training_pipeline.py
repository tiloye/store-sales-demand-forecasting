from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowSkipException

from ssdf.config import STATIC_FEATURES
from ssdf.training import train, tune
from ssdf.training.model import MODEL_NAME, PARAM_GRID, get_model


@dag(
    catchup=False,
    tags=["ssdf", "training"],
    params={
        "tune": False,
    },
)
def training_pipeline():

    @task
    def get_train_data():
        return train.get_data()

    @task
    def tune_model(df, **context):
        if not context["params"]["tune"]:
            raise AirflowSkipException

        print("Starting hyperparameter tuning")
        forecaster = get_model()
        _, mlflow_run = tune.run_tuning(
            forecaster,
            df,
            param_grid=PARAM_GRID,
            static_features=STATIC_FEATURES,
            model_name=MODEL_NAME,
        )
        print(
            f"Finished hyperparameter tuning with MLFlow Run ID: {mlflow_run.info.run_id}"
        )

    @task(trigger_rule="none_failed")
    def train_model(df):
        print("Starting model training")
        _, mlflow_run_id = train.run(
            df,
            static_features=STATIC_FEATURES,
            model_name=MODEL_NAME,
            pull_best_model_artifact=True,
            register_model=True,
        )
        print(
            f"Finished model training with MLFlow Run ID: {mlflow_run_id.info.run_id}"
        )

    df = get_train_data()
    tune_model(df) >> train_model(df)


training_pipeline()
