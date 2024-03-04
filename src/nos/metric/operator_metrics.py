import time
from typing import Dict

from continuity.data import OperatorDataset
from continuity.operators import Operator

from .metric import Metric


class NumberOfParametersMetric(Metric):
    def __init__(self):
        super().__init__("Parameters")

    def calculate(self, operator: Operator, dataset: OperatorDataset) -> Dict:
        num_params = sum(p.numel() for p in operator.parameters() if p.requires_grad)
        return {"Value": num_params}


class SpeedOfEvaluation(Metric):
    def __init__(self):
        super().__init__("Speed_of_evaluation")

    def calculate(self, operator: Operator, dataset: OperatorDataset) -> Dict:
        operator.eval()
        start_time = time.time_ns()
        _ = operator(dataset.x, dataset.u, dataset.v)
        end_time = time.time_ns()
        return {"Value": (end_time - start_time) / 1e9}
