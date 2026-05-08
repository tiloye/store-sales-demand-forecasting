from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot

from ssdf.config import EVIDENTLY_PROJECT_NAME
from ssdf.monitoring.utils import get_project

project = get_project(EVIDENTLY_PROJECT_NAME)

project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Dataset column drift",
        subtitle="Share of drifted columns",
        size="half",
        values=[
            PanelMetric(
                legend="Share",
                metric="DriftedColumnsCount",
                metric_labels={"value_type": "share"},
            ),
        ],
        plot_params={"plot_type": "line"},
    ),
    tab="Data Drift",
)
project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Prediction drift",
        subtitle="""Drift in the prediction column (sales_forecast)""",
        size="half",
        values=[
            PanelMetric(
                legend="Drift score",
                metric="evidently:metric_v2:ValueDrift",
                metric_labels={"column": "sales_forecast"},
            ),
        ],
        plot_params={"plot_type": "line"},
    ),
    tab="Data Drift",
)
