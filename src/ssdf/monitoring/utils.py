from __future__ import annotations

from typing import TYPE_CHECKING

from evidently.ui.workspace import Workspace
from ssdf.config import EVIDENTLY_WORKSPACE

if TYPE_CHECKING:
    from evidently.ui.workspace import Project
    from evidently.core.report import Snapshot

ws = Workspace(EVIDENTLY_WORKSPACE)


def get_project(name: str) -> Project:
    """Get or create a project"""
    project = list(filter(lambda p: p.name == name, ws.list_projects()))

    if len(project) == 0:
        project = ws.create_project(name)
        project.save()
        return project

    if len(project) > 1:
        raise ValueError(f"More than one project with the name {name} found")

    return project[0]


def log_snapshot(snapshot: Snapshot, project: Project):
    ws.add_run(project.id, snapshot, include_data=False)
