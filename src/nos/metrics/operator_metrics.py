import time  # noqa: D100

from continuiti.data import OperatorDataset
from continuiti.operators import Operator

from nos.metrics.metric import Metric


class NumberOfParameters(Metric):
    """Number of parameters in the operator."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__("Number_of_parameters")

    def __call__(self, operator: Operator, dataset: OperatorDataset) -> dict:  # noqa: ARG002
        """Return the number of trainable parameters in the operator."""
        num_params = sum(p.numel() for p in operator.parameters() if p.requires_grad)
        return {"Value": num_params, "Unit": "[1]"}


class SpeedOfEvaluation(Metric):
    """Speed of a single evaluation in milliseconds."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__("Speed_of_evaluation")

    def __call__(self, operator: Operator, dataset: OperatorDataset) -> dict:
        """Return the time per observation of the operator for the entire dataset."""
        operator.eval()
        start_time = time.time_ns()
        _ = operator(dataset.x, dataset.u, dataset.v)
        end_time = time.time_ns()
        delta_time = (end_time - start_time) * 1e-6
        delta_time = delta_time / len(dataset)
        return {"Value": delta_time, "Unit": "[ms]"}
