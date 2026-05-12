from airflow.sdk import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime, duration
from ssdf.config import FH
from ssdf.inference import predict, submission


@dag(
    schedule=duration(days=16),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ssdf", "inference"],
    params={
        "model_uri": None,
        "fh": FH,
    },
)
def inference_pipeline():
    """
    Inference Pipeline: Pulls the latest feature from the feature file, generates forecasts, and saves it to local directory.
    """

    @task
    def generate_forecasts(params):
        return predict.generate_forecasts(
            model_uri=params["model_uri"], fh=params["fh"]
        )

    @task
    def save_forecasts(forecasts):
        predict.save_forecasts(forecasts)

    @task
    def generate_submission():
        submission.generate_submission()

    trigger_monitoring = TriggerDagRunOperator(
        task_id="trigger_monitoring_pipeline",
        trigger_dag_id="monitoring_pipeline",
        conf={
            "ref_start_date": "2017-08-16",
            "ref_end_date": "2017-08-23",
            "curr_start_date": "2017-08-24",
            "curr_end_date": "2017-08-31",
        },
    )

    forecasts = generate_forecasts()
    save_forecasts(forecasts) >> trigger_monitoring


inference_pipeline()
