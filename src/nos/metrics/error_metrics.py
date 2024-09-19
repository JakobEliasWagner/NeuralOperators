import torch  # noqa: D100
from continuiti.data import OperatorDataset
from continuiti.operators import Operator

from nos.metrics.metric import Metric


class Loss(Metric):
    """Class for evaluating error metrics.

    Args:
        name: The name of the metric.
        loss: The loss function for calculating the metric.

    """

    def __init__(self, name: str, loss: torch.nn.Module) -> None:  # noqa: D107
        super().__init__(name)
        self.loss = loss

    def __call__(self, operator: Operator, dataset: OperatorDataset) -> dict:
        """Return loss evaluated on entire dataset."""
        operator.eval()
        prediction = operator(dataset.x, dataset.u, dataset.v)
        value = self.loss(prediction, dataset.v).item()
        value /= len(dataset)
        return {
            "Value": value,
            "Unit": "[1]",
        }


class L1Error(Loss):
    """L1 error metric (Mean Absolute Error)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__("L1_error", torch.nn.L1Loss())


class MSError(Loss):
    """Mean square error metric (L2 Error)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__("MS_error", torch.nn.MSELoss())
