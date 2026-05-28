import os

from airflow.sdk import dag, task, Param
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

PYTHON_ENVIRONMENT = os.getenv("PYTHON_ENVIRONMENT", "/usr/local/bin/python")


@dag(
    catchup=False,
    tags=["ssdf", "monitoring"],
    params={
        "ref_start_date": Param(None, type=["string", "null"], format="date-time"),
        "ref_end_date": Param(None, type=["string", "null"], format="date-time"),
        "curr_start_date": Param(None, type=["string", "null"], format="date-time"),
        "curr_end_date": Param(None, type=["string", "null"], format="date-time"),
        "timestamp": Param(None, type=["string", "null"], format="date-time"),
    },
)
def monitoring_pipeline():
    """
    Monitoring Pipeline: Pulls the predicted data and feature, generates evidently snapshots, and logs the snapshot to evidently workspace.
    """

    @task
    def get_dag_params(params):
        return params

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def get_ref_curr_data(dag_params):
        params = dag_params
        import pickle
        import base64
        from pandas import Timestamp
        from ssdf.monitoring import metrics

        print("Getting reference and current data...")
        ref_start = (
            Timestamp(params["ref_start_date"])
            if params.get("ref_start_date")
            else None
        )
        ref_end = (
            Timestamp(params["ref_end_date"]) if params.get("ref_end_date") else None
        )
        curr_start = (
            Timestamp(params["curr_start_date"])
            if params.get("curr_start_date")
            else None
        )
        curr_end = (
            Timestamp(params["curr_end_date"]) if params.get("curr_end_date") else None
        )

        ref_data, curr_data = metrics.get_ref_curr_data(
            ref_start_date=ref_start,
            ref_end_date=ref_end,
            curr_start_date=curr_start,
            curr_end_date=curr_end,
        )

        # Custom serialization to avoid disk IO and Airflow deserialization restrictions
        serialized_data = base64.b64encode(pickle.dumps((ref_data, curr_data))).decode(
            "utf-8"
        )
        return serialized_data

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def generate_snapshot(serialized_data, dag_params):
        params = dag_params
        import pickle
        import base64
        from pandas import Timestamp
        from ssdf.monitoring import metrics

        # Custom deserialization
        ref_data, curr_data = pickle.loads(base64.b64decode(serialized_data))

        print("Generating drift snapshot...")
        timestamp = Timestamp(params["timestamp"]) if params.get("timestamp") else None

        snapshot = metrics.generate_drift_snapshot(
            reference_data=ref_data, current_data=curr_data, timestamp=timestamp
        )

        serialized_snapshot = base64.b64encode(pickle.dumps(snapshot)).decode("utf-8")
        return serialized_snapshot

    @task.external_python(python=PYTHON_ENVIRONMENT)
    def log_snapshot(serialized_snapshot):
        import pickle
        import base64
        from ssdf.config import EVIDENTLY_PROJECT_NAME
        from ssdf.monitoring import utils

        snapshot = pickle.loads(base64.b64decode(serialized_snapshot))

        print("Getting/Creating Evidently project...")
        project = utils.get_project(EVIDENTLY_PROJECT_NAME)

        print("Logging drift snapshot to Evidently workspace...")
        utils.log_snapshot(snapshot, project)
        print("Snapshot logged successfully!")

    @task.short_circuit
    def retrain_or_skip(serialized_snapshot, dag_params):
        import pickle
        import base64
        from ssdf.monitoring import utils

        params = dag_params
        # Skip retraining if drift metrics was generated for custom date range
        if params.get("ref_start_date") is not None:
            return False

        snapshot = pickle.loads(base64.b64decode(serialized_snapshot))

        prediction_drift_detected = utils.check_prediction_drift(snapshot)

        return prediction_drift_detected

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training_pipeline",
        trigger_dag_id="training_pipeline",
    )

    p = get_dag_params()
    serialized_data = get_ref_curr_data(p)
    serialized_snapshot = generate_snapshot(serialized_data, p)
    log_snapshot(serialized_snapshot)
    retrain_or_skip(serialized_snapshot, p) >> trigger_training


monitoring_pipeline()
