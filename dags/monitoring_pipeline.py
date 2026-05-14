import pickle
import base64

from airflow.sdk import dag, task, Param
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pandas import Timestamp

from ssdf.config import EVIDENTLY_PROJECT_NAME
from ssdf.monitoring import utils, metrics


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
    def get_ref_curr_data(params):
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

    @task
    def generate_snapshot(serialized_data, params):

        # Custom deserialization
        ref_data, curr_data = pickle.loads(base64.b64decode(serialized_data))

        print("Generating drift snapshot...")
        timestamp = Timestamp(params["timestamp"]) if params.get("timestamp") else None

        snapshot = metrics.generate_drift_snapshot(
            reference_data=ref_data, current_data=curr_data, timestamp=timestamp
        )

        serialized_snapshot = base64.b64encode(pickle.dumps(snapshot)).decode("utf-8")
        return serialized_snapshot

    @task
    def log_snapshot(serialized_snapshot):
        snapshot = pickle.loads(base64.b64decode(serialized_snapshot))

        print("Getting/Creating Evidently project...")
        project = utils.get_project(EVIDENTLY_PROJECT_NAME)

        print("Logging drift snapshot to Evidently workspace...")
        utils.log_snapshot(snapshot, project)
        print("Snapshot logged successfully!")

    @task.short_circuit
    def check_forecast_drift(serialized_snapshot, params):
        # Skip retraining if drift metrics was generated for custom date range
        if params.get("ref_start_date") is not None:
            return False

        snapshot = pickle.loads(base64.b64decode(serialized_snapshot))

        for _metric in snapshot.dict()["metrics"]:
            if (
                _metric["config"]["type"] == "evidently:metric_v2:ValueDrift"
                and _metric["config"]["column"] == "sales_forecast"
            ):
                prediction_drift_metric = _metric
                break

        if (
            prediction_drift_metric["value"]
            > prediction_drift_metric["config"]["threshold"]
        ):
            print("Prediction drift detected. Triggering training pipeline...")
            return True
        else:
            print("No prediction drift detected.")
            return False

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training_pipeline",
        trigger_dag_id="training_pipeline",
    )

    serialized_data = get_ref_curr_data()
    serialized_snapshot = generate_snapshot(serialized_data)
    log_snapshot(serialized_snapshot)
    forecast_drifted = check_forecast_drift(serialized_snapshot)
    forecast_drifted >> trigger_training


monitoring_pipeline()
