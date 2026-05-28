import pytest
from evidently.ui.workspace import Workspace
from ssdf.monitoring.utils import get_project, check_prediction_drift
from unittest.mock import Mock


@pytest.fixture
def mock_workspace(tmp_path):
    return tmp_path / ".evidently_workspace"


def test_get_project_returns_existing_project(mock_workspace, monkeypatch):
    ws = Workspace(mock_workspace)
    expected_project = ws.create_project("Project 1")
    expected_project.save()
    monkeypatch.setattr("ssdf.monitoring.utils.get_workspace", lambda: ws)

    project = get_project("Project 1")

    assert project.id == expected_project.id


def test_get_project_creates_new_project(mock_workspace, monkeypatch):
    ws = Workspace(mock_workspace)
    monkeypatch.setattr("ssdf.monitoring.utils.get_workspace", lambda: ws)

    project = get_project("New Project")

    projects = list(filter(lambda p: p.name == "New Project", ws.list_projects()))
    assert len(projects) == 1
    assert projects[0].id == project.id


def test_get_project_raises_error_if_multiple_projects_found(
    mock_workspace, monkeypatch
):
    ws = Workspace(mock_workspace)
    expected_project = ws.create_project("Project 1")
    expected_project.save()
    ws.create_project("Project 1")
    monkeypatch.setattr("ssdf.monitoring.utils.get_workspace", lambda: ws)

    with pytest.raises(
        ValueError, match="More than one project with the name Project 1 found"
    ):
        get_project("Project 1")


def test_check_prediction_drift_exceeds_threshold():
    mock_snapshot = Mock()
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "sales_forecast",
                    "threshold": 0.5,
                },
                "value": 0.6,
            }
        ]
    }
    assert check_prediction_drift(mock_snapshot) is True


def test_check_prediction_drift_below_threshold():
    mock_snapshot = Mock()
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "sales_forecast",
                    "threshold": 0.5,
                },
                "value": 0.4,
            }
        ]
    }
    assert check_prediction_drift(mock_snapshot) is False
