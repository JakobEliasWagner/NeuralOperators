import torch

from continuity.data import (
    OperatorDataset,
)
from nos.benchmarks import (
    Benchmark,
)
from nos.metrics import (
    L1Error,
    SpeedOfEvaluation,
)


def test_can_initialize():
    triv_set = OperatorDataset(*[torch.rand(10, 1, 1)] * 4)
    benchmark = Benchmark(train_set=triv_set, test_set=triv_set, metrics=[L1Error(), SpeedOfEvaluation()])
    assert isinstance(benchmark, Benchmark)
