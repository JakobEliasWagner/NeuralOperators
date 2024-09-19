from abc import ABC, abstractmethod  # noqa: D100

from continuiti.data import OperatorDataset
from continuiti.operators import Operator


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self, name: str) -> None:  # noqa: D107
        self.name = name

    @abstractmethod
    def __call__(self, operator: Operator, dataset: OperatorDataset) -> dict:
        """Evaluate the metric.

        Args:
            operator: operator for which the metric is evaluated.
            dataset: dataset on which the metric is evaluated.

        Returns:
            dict containing the results of the metric (keys "value" and "unit" should be in the dict).

        """

    def __str__(self) -> str:
        """Return metric name as string."""
        return self.name
