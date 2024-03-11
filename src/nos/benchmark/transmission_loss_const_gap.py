import pathlib

from continuity.benchmarks import Benchmark
from continuity.benchmarks.metrics import L1Metric, MSEMetric, NumberOfParametersMetric, SpeedOfEvaluationMetric
from nos.data import TLDatasetCompact

# paths for this specific benchmark
TRAIN_PATH = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss", "gw_6e-1", "dset.csv")
TEST_PATH = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss", "gw_6e-1", "dset_test.csv")


class TransmissionLossConstGap(Benchmark):
    def __init__(self):
        super().__init__(
            train_dataset=TLDatasetCompact(TRAIN_PATH),
            test_dataset=TLDatasetCompact(TEST_PATH),
            metrics=[MSEMetric(), L1Metric(), NumberOfParametersMetric(), SpeedOfEvaluationMetric()],
        )
