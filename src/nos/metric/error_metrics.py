from typing import Dict

import torch

from continuity.data import OperatorDataset
from continuity.operators import Operator

from .metric import Metric


class ErrorMetric(Metric):
    def __init__(self, name: str, loss):
        super().__init__(name)
        self.loss = loss

    def calculate(self, operator: Operator, dataset: OperatorDataset) -> Dict:
        operator.eval()
        prediction = operator(dataset.x, dataset.u, dataset.v)
        return {"Value": self.loss(prediction, dataset.v).item()}


class L1Metric(ErrorMetric):
    def __init__(self):
        super().__init__("L1_error", torch.nn.L1Loss())


class MSEMetric(ErrorMetric):
    def __init__(self):
        super().__init__("MS_error", torch.nn.MSELoss())
