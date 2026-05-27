import os

from airflow.sdk import dag, task

PYTHON_ENVIRONMENT = os.getenv("PYTHON_ENVIRONMENT", "/usr/python/bin")


@dag(
    catchup=False,
    tags=["ssdf", "training"],
    params={
        "tune": False,
    },
)
def training_pipeline():

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def get_train_data():
        from ssdf.training.train import get_data

        return get_data()

    @task.short_circuit(ignore_downstream_trigger_rules=False)
    def tune_or_skip_tunning(params):
        return params["tune"]

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def tune_model(df):
        from ssdf.config import STATIC_FEATURES
        from ssdf.training import tune
        from ssdf.training.model import MODEL_NAME, PARAM_GRID, get_model

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

    @task.external_python(python=PYTHON_ENVIRONMENT, trigger_rule="none_failed")
    def train_model(df):
        from ssdf.config import STATIC_FEATURES
        from ssdf.training import train
        from ssdf.training.model import MODEL_NAME

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
    df >> tune_or_skip_tunning() >> tune_model(df) >> train_model(df)


training_pipeline()
