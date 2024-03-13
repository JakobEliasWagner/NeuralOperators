from typing import List
from dataclasses import dataclass, field

from continuity.data import OperatorDataset
from nos.metrics import Metric, L1Error, MSError, NumberOfParameters, SpeedOfEvaluation


@dataclass
class Benchmark:
    train_set: OperatorDataset
    test_set: OperatorDataset
    metrics: List[Metric] = field(
        default_factory=lambda: [L1Error(), MSError(), NumberOfParameters(), SpeedOfEvaluation()])
