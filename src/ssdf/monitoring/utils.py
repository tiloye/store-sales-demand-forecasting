from __future__ import annotations

from typing import TYPE_CHECKING

from evidently.ui.workspace import Workspace, CloudWorkspace
from ssdf.config import ENV_NAME, EVIDENTLY_ORG_ID

if TYPE_CHECKING:
    from evidently.ui.workspace import Project
    from evidently.core.report import Snapshot


def get_workspace() -> Workspace | CloudWorkspace:
    if ENV_NAME == "dev":
        from ssdf.config import EVIDENTLY_WORKSPACE

        return Workspace(EVIDENTLY_WORKSPACE)
    else:
        from ssdf.config import EVIDENTLY_API_KEY

        return CloudWorkspace(token=EVIDENTLY_API_KEY)


def get_project(name: str) -> Project:
    """Get or create a project"""

    ws = get_workspace()
    project = list(filter(lambda p: p.name == name, ws.list_projects()))

    if len(project) == 0:
        project = ws.create_project(name, org_id=EVIDENTLY_ORG_ID)
        project.save()
        return project

    if len(project) > 1:
        raise ValueError(f"More than one project with the name {name} found")

    return project[0]


def log_snapshot(snapshot: Snapshot, project: Project) -> None:
    ws = get_workspace()
    ws.add_run(project.id, snapshot, include_data=False)


def check_prediction_drift(snapshot: Snapshot) -> bool:
    for _metric in snapshot.dict()["metrics"]:
        if (
            _metric["config"]["type"] == "evidently:metric_v2:ValueDrift"
            and _metric["config"]["column"] == "sales_forecast"
        ):
            prediction_drift_metric = _metric
            break

    return (
        prediction_drift_metric["value"]
        > prediction_drift_metric["config"]["threshold"]
    )
