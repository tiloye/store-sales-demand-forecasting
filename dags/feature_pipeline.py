from airflow.sdk import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime, duration

import ssdf.data
from ssdf.features.feature import create_features, create_target


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
    def process_raw_data(params):
        path = params["path"]
        force_download = params["force_download"]
        ssdf.data.run(path=path, force_download=force_download)

    @task
    def generate_target():
        create_target()

    @task
    def generate_features():
        create_features()

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference_pipeline",
        trigger_dag_id="inference_pipeline",
    )

    process_raw_data() >> [generate_target(), generate_features()] >> trigger_inference


feature_pipeline()
