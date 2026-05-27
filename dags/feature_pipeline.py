import os
from airflow.sdk import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime, duration

PYTHON_ENVIRONMENT = os.getenv("PYTHON_ENVIRONMENT", "/usr/local/bin/python")


@dag(
    schedule=duration(days=16),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ssdf", "feature"],
    params={
        "path": None,
        "force_download": False,
    },
)
def feature_pipeline():
    """
    Feature Pipeline: Pulls the processed data from the data pipeline and creates features and target parquet file.
    """

    @task
    def get_dag_params(params):
        return params

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def process_raw_data(dag_params):
        from ssdf import data

        path = dag_params["path"]
        force_download = dag_params["force_download"]
        data.run(path=path, force_download=force_download)

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def generate_target():
        from ssdf.features.feature import create_target

        create_target()

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def generate_features():
        from ssdf.features.feature import create_features

        create_features()

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference_pipeline",
        trigger_dag_id="inference_pipeline",
    )

    p = get_dag_params()
    process_raw_data(p) >> [generate_target(), generate_features()] >> trigger_inference


feature_pipeline()
