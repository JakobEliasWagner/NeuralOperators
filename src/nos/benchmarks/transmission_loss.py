import pathlib

from nos.data import (
    TLDatasetCompact,
)

from .benchmark import (
    Benchmark,
)

# paths for this specific benchmark
TRAIN_PATH = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss", "gw_6e-1", "dset.csv")
TEST_PATH = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss", "gw_6e-1", "dset_test.csv")

transmission_loss_const_gap = Benchmark(train_set=TLDatasetCompact(TRAIN_PATH), test_set=TLDatasetCompact(TEST_PATH))
