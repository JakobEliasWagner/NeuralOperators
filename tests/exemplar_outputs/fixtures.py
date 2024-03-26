import pathlib

import pytest

from nos.plots import (
    MultiRunData,
)


@pytest.fixture(scope="session")
def exemplar_multirun_path():
    return pathlib.Path.cwd().joinpath("tests", "exemplar_outputs", "some_multirun")


@pytest.fixture(scope="session")
def exemplar_multirun_data(exemplar_multirun_path):
    return MultiRunData.from_dir(exemplar_multirun_path)


@pytest.fixture(scope="session")
def exemplar_run_path(exemplar_multirun_path):
    return exemplar_multirun_path.joinpath("0")


@pytest.fixture(scope="session")
def exemplar_model_path(exemplar_run_path):
    return exemplar_run_path.joinpath("models", "DeepNeuralOperator_2024_03_25_15_21_22")
