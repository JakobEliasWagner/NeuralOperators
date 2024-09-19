import pathlib

import pytest


@pytest.fixture(scope="session")
def tl_dir() -> pathlib.Path:
    return pathlib.Path.cwd().joinpath("data", "test_data", "transmission_loss")


@pytest.fixture(scope="session")
def tl_csv_file(tl_dir) -> pathlib.Path:
    return next(tl_dir.rglob("*.csv"))


@pytest.fixture(scope="session")
def tl_paths(tl_dir, tl_csv_file) -> list[pathlib.Path]:
    return [tl_dir, tl_csv_file]
