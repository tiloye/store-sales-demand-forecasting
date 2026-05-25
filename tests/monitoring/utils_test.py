import pytest
from evidently.ui.workspace import Workspace
from ssdf.monitoring.utils import get_project


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
