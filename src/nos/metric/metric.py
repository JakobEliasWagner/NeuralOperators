from abc import ABC, abstractmethod
from typing import Dict

from continuity.data import OperatorDataset
from continuity.operators import Operator


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, operator: Operator, dataset: OperatorDataset) -> Dict:
        """

        :param operator:
        :param dataset:
        :return:
        """

    def __str__(self):
        return self.name
