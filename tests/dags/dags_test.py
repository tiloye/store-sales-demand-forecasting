import base64
import pickle

from pathlib import Path

import pytest
from airflow.models import DagBag
from airflow.utils.state import State
from pendulum import duration

DAGS_DIR = Path(__file__).parent.parent.parent / "dags"


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=DAGS_DIR, include_examples=False)


def test_all_dags_exist(dagbag):
    expected_dags = [
        "feature_pipeline",
        "inference_pipeline",
        "monitoring_pipeline",
        "training_pipeline",
    ]
    for dag_id in expected_dags:
        assert dag_id in dagbag.dags, f"{dag_id} DAG not found"


def test_no_import_errors(dagbag):
    assert len(dagbag.import_errors) == 0, (
        f"Import errors found: {dagbag.import_errors}"
    )


@pytest.mark.parametrize(
    "dag_id",
    [
        "feature_pipeline",
        "inference_pipeline",
        "monitoring_pipeline",
        "training_pipeline",
    ],
)
def test_dag_arguments(dagbag, dag_id):
    dag = dagbag.get_dag(dag_id=dag_id)

    assert dag.catchup is False
    if dag_id == "feature_pipeline":
        assert dag.schedule == duration(days=16)
        assert dag.start_date is not None
    else:
        assert dag.schedule is None
        assert dag.start_date is None


@pytest.mark.parametrize(
    "dag_id,tags",
    [
        ("feature_pipeline", {"ssdf", "feature"}),
        ("inference_pipeline", {"ssdf", "inference"}),
        ("monitoring_pipeline", {"ssdf", "monitoring"}),
        ("training_pipeline", {"ssdf", "training"}),
    ],
)
def test_dags_have_expected_tags(dagbag, dag_id, tags):
    dag = dagbag.get_dag(dag_id=dag_id)
    assert dag.tags == tags


def test_feature_pipeline_has_trigger_inference_task(dagbag):
    dag = dagbag.get_dag(dag_id="feature_pipeline")

    trigger_task = dag.get_task("trigger_inference_pipeline")
    assert trigger_task is not None
    assert trigger_task.task_type == "TriggerDagRunOperator"
    assert trigger_task.trigger_dag_id == "inference_pipeline"
    assert len(trigger_task.downstream_list) == 0


def test_inference_pipeline_has_trigger_monitoring_task(dagbag):
    dag = dagbag.get_dag(dag_id="inference_pipeline")

    trigger_task = dag.get_task("trigger_monitoring_pipeline")
    assert trigger_task is not None
    assert trigger_task.task_type == "TriggerDagRunOperator"
    assert trigger_task.trigger_dag_id == "monitoring_pipeline"
    assert len(trigger_task.downstream_list) == 0


def test_monitoring_pipeline_has_trigger_training_task(dagbag):
    dag = dagbag.get_dag(dag_id="monitoring_pipeline")

    trigger_task = dag.get_task("trigger_training_pipeline")
    assert trigger_task is not None
    assert trigger_task.task_type == "TriggerDagRunOperator"
    assert trigger_task.trigger_dag_id == "training_pipeline"
    assert len(trigger_task.downstream_list) == 0
    assert len(trigger_task.upstream_list) == 1
    assert trigger_task.upstream_list[0].custom_operator_name == "@task.short_circuit"


class DummySnapshot:
    def __init__(self, drift_score):
        self.drift_score = drift_score

    def dict(self):
        data = {
            "metrics": [
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "sales_forecast",
                        "threshold": 0.1,
                    },
                    "value": self.drift_score,
                }
            ]
        }
        return data


@pytest.mark.parametrize(
    "dag_params,drifted,train_trigger_state",
    [
        (None, True, State.SUCCESS),
        (None, False, State.SKIPPED),
        (
            {"ref_start_date": "dummy_date"},
            False,
            State.SKIPPED,
        ),
        (
            {"ref_start_date": "dummy_date"},
            True,
            State.SKIPPED,
        ),
    ],
    ids=[
        "forecast_drifted_and_no_custom_date_params",
        "forecast_not_drifted_and_no_custom_date_params",
        "custom_date_params_no_forecast_drift",
        "custom_date_params_with_forecast_drift",
    ],
)
def test_monitoring_pipeline_trigger_training_state(
    dagbag, monkeypatch, dag_params, drifted, train_trigger_state
):
    dummy_snapshot = DummySnapshot(drift_score=0.2 if drifted else 0.0)
    serialized_dummy_snapshot = base64.b64encode(pickle.dumps(dummy_snapshot)).decode(
        "utf-8"
    )
    monkeypatch.setattr(
        "airflow.sdk.execution_time.xcom.XCom.get_one",
        lambda *args, **kwargs: serialized_dummy_snapshot,
    )
    dag = dagbag.get_dag(dag_id="monitoring_pipeline")
    dagrun = dag.test(
        mark_success_pattern=r"get_ref_curr_data|generate_snapshot|log_snapshot",
        run_conf=dag_params,
    )
    trigger_ti = dagrun.get_task_instance("trigger_training_pipeline")
    assert trigger_ti.state == train_trigger_state


def test_training_pipeline_has_no_trigger_operator(dagbag):
    dag = dagbag.get_dag(dag_id="training_pipeline")
    training_task_opertors = [t.operator_name for t in dag.tasks]
    assert "TriggerDagRunOperator" not in training_task_opertors
