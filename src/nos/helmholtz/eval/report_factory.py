import pathlib
from typing import Dict, List

from nos.data.helmholtz import HelmholtzDataset

from .gap_report import GapReport
from .pressure_report import PressureReport
from .report import Report


class ReportFactory:
    """Produces reports about a specific data-set."""

    def __init__(self, reports: List[type(Report)], data_sets: Dict[str, HelmholtzDataset], out_dir: pathlib.Path):
        self.reports = reports
        self.out_dir = out_dir
        self.data_sets = data_sets

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=False)

        for report in self.reports:
            report.run(self.out_dir, self.data_sets)


class AllReportsFactory(ReportFactory):
    def __init__(self, data_sets: Dict[str, HelmholtzDataset], out_dir: pathlib.Path):
        super().__init__([GapReport, PressureReport], data_sets, out_dir)
