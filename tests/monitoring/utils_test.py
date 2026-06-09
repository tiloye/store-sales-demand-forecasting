import pytest
from ssdf.monitoring.utils import check_prediction_drift, get_project, get_workspace
from unittest.mock import Mock


def test_get_project_returns_existing_project():
    ws = get_workspace()
    expected_project = ws.create_project("Project 1")
    expected_project.save()

    project = get_project("Project 1")

    assert project.id == expected_project.id


def test_get_project_creates_new_project():
    ws = get_workspace()
    project = get_project("New Project")
    projects = list(filter(lambda p: p.name == "New Project", ws.list_projects()))

    assert len(projects) == 1
    assert projects[0].id == project.id


def test_get_project_raises_error_if_multiple_projects_found():
    ws = get_workspace()
    expected_project = ws.create_project("Project 1")
    expected_project.save()
    ws.create_project("Project 1")

    with pytest.raises(
        ValueError, match="More than one project with the name Project 1 found"
    ):
        get_project("Project 1")


@pytest.mark.parametrize(
    "value, expected",
    [(0.6, True), (0.4, False)],
    ids=["drift_detected", "no_drift_detected"],
)
def test_check_prediction_drift(value, expected):
    mock_snapshot = Mock()
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "sales_forecast",
                    "threshold": 0.5,
                },
                "value": value,
            }
        ]
    }
    assert check_prediction_drift(mock_snapshot) is expected
