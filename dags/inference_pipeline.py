import os
from airflow.sdk import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

PYTHON_ENVIRONMENT = os.getenv("PYTHON_ENVIRONMENT", "/usr/local/bin/python")


@dag(
    catchup=False,
    tags=["ssdf", "inference"],
    params={
        "model_uri": None,
        "fh": 16,
    },
)
def inference_pipeline():
    """
    Inference Pipeline: Pulls the latest feature from the feature file, generates forecasts, and saves it to local directory.
    """

    @task
    def get_dag_params(params):
        return params

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def generate_forecasts(dag_params):
        from ssdf.inference.predict import generate_forecasts

        return generate_forecasts(
            model_uri=dag_params["model_uri"], fh=dag_params["fh"]
        )

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def save_forecasts(forecasts):
        from ssdf.inference.predict import save_forecasts

        save_forecasts(forecasts)

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def generate_submission():
        from ssdf.inference import submission

        submission.generate_submission()

    trigger_monitoring = TriggerDagRunOperator(
        task_id="trigger_monitoring_pipeline",
        trigger_dag_id="monitoring_pipeline",
    )

    p = get_dag_params()
    forecasts = generate_forecasts(p)
    save_forecasts(forecasts) >> trigger_monitoring


inference_pipeline()
