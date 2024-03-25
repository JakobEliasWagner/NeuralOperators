import pathlib
from typing import (
    List,
)

import pytest

from nos.data import (
    TLDatasetCompact,
)


@pytest.fixture(scope="session")
def tl_dataset_sizes() -> List[int]:
    return [-1, 1, 42]


@pytest.fixture(scope="session")
def tl_dir() -> pathlib.Path:
    return pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_lin")


@pytest.fixture(scope="session")
def tl_csv_file(tl_dir) -> pathlib.Path:
    return next(tl_dir.rglob("*.csv"))


@pytest.fixture(scope="session")
def tl_paths(tl_dir, tl_csv_file) -> List[pathlib.Path]:
    return [tl_dir, tl_csv_file]


@pytest.fixture(scope="session")
def tl_compact_dataset(tl_csv_file):
    return TLDatasetCompact(tl_csv_file, n_samples=9)
