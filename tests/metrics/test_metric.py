import pytest
import torch
from continuiti.data import (
    OperatorDataset,
)
from continuiti.operators import (
    Operator,
)

from nos.metrics import (
    Metric,
)


@pytest.fixture(scope="session")
def ones_metric():
    class Ones(Metric):
        def __call__(self, operator: Operator, dataset: OperatorDataset) -> dict:
            num_ones = 0
            for vec in [dataset.x, dataset.u, dataset.v, dataset.x]:
                num_ones += torch.sum(torch.isclose(vec, torch.ones(vec.shape)))
            return {"value": num_ones, "unit": "[1]"}

    return Ones("CountOnes")


def test_can_initialize(ones_metric):
    assert isinstance(ones_metric, Metric)


def test_name_correct(ones_metric):
    assert str(ones_metric) == "CountOnes"


def test_call_correct(ones_metric):
    n_obs = 10
    dataset = OperatorDataset(
        x=torch.zeros(n_obs, 1, 10),
        u=torch.ones(n_obs, 3, 10),  # 300 zeros
        y=torch.ones(n_obs, 2, 5),  # 100 zeros
        v=torch.zeros(n_obs, 1, 5),
    )
    result = ones_metric(None, dataset)

    correct = 300
    assert result["value"] == correct
