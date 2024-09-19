from dataclasses import dataclass, field  # noqa: D100

from continuiti.data import OperatorDataset

from nos.metrics import L1Error, Metric, MSError, NumberOfParameters, SpeedOfEvaluation


@dataclass
class Benchmark:
    """Benchmarks manage training and test datasets as well as metrics for evaluation."""

    train_set: OperatorDataset
    test_set: OperatorDataset
    metrics: list[Metric] = field(
        default_factory=lambda: [L1Error(), MSError(), NumberOfParameters(), SpeedOfEvaluation()],
    )
