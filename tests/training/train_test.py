import mlflow
from mlforecast import MLForecast

from ssdf.training.train import run


def test_start_new_run(monkeypatch, training_data, mlflow_configs):
    monkeypatch.setattr(
        "ssdf.training.train.get_data", lambda *args, **kwargs: training_data
    )
    monkeypatch.setattr(
        "ssdf.training.train.MLFLOW_TRACKING_URI", mlflow_configs["tracking_uri"]
    )

    forecaster, mlflow_run = run(training_data)

    assert isinstance(forecaster, MLForecast)
    assert isinstance(mlflow_run, mlflow.entities.Run)

    assert "model_name" in mlflow_run.data.tags
    assert len(mlflow_run.data.params) > 0
    assert len(mlflow_run.inputs.dataset_inputs) > 0
    assert len(mlflow_run.outputs.model_outputs) > 0


def test_continue_run(monkeypatch, training_data, mlflow_configs):
    monkeypatch.setattr(
        "ssdf.training.train.MLFLOW_TRACKING_URI", mlflow_configs["tracking_uri"]
    )

    mlflow.set_tracking_uri(mlflow_configs["tracking_uri"])
    mlflow.set_experiment(mlflow_configs["experiment_name"])

    with mlflow.start_run() as first_run:
        mlflow.set_tag("model_name", "SeasonalNaive")
        mlflow.log_input(
            mlflow.data.from_pandas(
                training_data,
                source=mlflow.data.DatasetSource.from_dict({"source": "train.parquet"}),
                targets="sales",
            ),
            context="training",
        )
        first_run_id = first_run.info.run_id

    forecaster, second_run = run(training_data, exp_run_id=first_run_id)
    first_run = mlflow.get_run(first_run_id)

    assert second_run.info.run_id == first_run_id
    assert first_run.data.tags == second_run.data.tags
    assert first_run.data.params == second_run.data.params
    assert (
        first_run.inputs.dataset_inputs[0].dataset.digest
        == second_run.inputs.dataset_inputs[0].dataset.digest
    )
    assert len(second_run.outputs.model_outputs) > 0
