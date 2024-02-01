import pathlib
from abc import ABC, abstractmethod
from typing import Dict

from nos.data.helmholtz import HelmholtzDataset


class Report(ABC):
    @staticmethod
    @abstractmethod
    def run(out_dir: pathlib.Path, data_sets: Dict[str, HelmholtzDataset]):
        pass
