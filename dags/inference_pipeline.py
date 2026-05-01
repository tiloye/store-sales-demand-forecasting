from airflow.sdk import dag, task
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

    forecasts = generate_forecasts()
    save_forecasts(forecasts)


inference_pipeline()
