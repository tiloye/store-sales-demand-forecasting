from airflow.sdk import dag, task, Param
import pandas as pd

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
        import pickle
        import base64

        print("Getting reference and current data...")
        ref_start = (
            pd.Timestamp(params["ref_start_date"])
            if params.get("ref_start_date")
            else None
        )
        ref_end = (
            pd.Timestamp(params["ref_end_date"]) if params.get("ref_end_date") else None
        )
        curr_start = (
            pd.Timestamp(params["curr_start_date"])
            if params.get("curr_start_date")
            else None
        )
        curr_end = (
            pd.Timestamp(params["curr_end_date"])
            if params.get("curr_end_date")
            else None
        )

        ref_data, curr_data = metrics.get_ref_curr_data(
            ref_start_date=ref_start,
            ref_end_date=ref_end,
            curr_start_date=curr_start,
            curr_end_date=curr_end,
        )

        # Custom serialization to avoid disk IO and Airflow deserialization restrictions
        serialized_data = base64.b64encode(pickle.dumps((ref_data, curr_data))).decode('utf-8')
        return serialized_data

    @task
    def generate_snapshot(serialized_data, params):
        import pickle
        import base64

        # Custom deserialization
        ref_data, curr_data = pickle.loads(base64.b64decode(serialized_data))

        print("Generating drift snapshot...")
        timestamp = (
            pd.Timestamp(params["timestamp"]) if params.get("timestamp") else None
        )

        snapshot = metrics.generate_drift_snapshot(
            reference_data=ref_data, current_data=curr_data, timestamp=timestamp
        )

        # Custom serialization
        serialized_snapshot = base64.b64encode(pickle.dumps(snapshot)).decode('utf-8')
        return serialized_snapshot

    @task
    def log_snapshot(serialized_snapshot):
        import pickle
        import base64

        # Custom deserialization
        snapshot = pickle.loads(base64.b64decode(serialized_snapshot))

        print("Getting/Creating Evidently project...")
        project = utils.get_project(EVIDENTLY_PROJECT_NAME)

        print("Logging drift snapshot to Evidently workspace...")
        utils.log_snapshot(snapshot, project)
        print("Snapshot logged successfully!")

    serialized_data = get_ref_curr_data()
    serialized_snapshot = generate_snapshot(serialized_data)
    log_snapshot(serialized_snapshot)


monitoring_pipeline()
