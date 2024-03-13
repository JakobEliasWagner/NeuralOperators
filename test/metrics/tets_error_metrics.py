import pytest
import torch

from continuity.operators import Operator
from continuity.data import OperatorDataset
from nos.metrics import L1Error, MSError


@pytest.fixture(scope='session')
def identity_operator() -> Operator:
    class Identity(Operator):
        def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return u

    return Identity()


def test_can_initialize():
    assert isinstance(L1Error(), L1Error)
    assert isinstance(MSError(), MSError)


def test_call_correct(identity_operator):
    square_dataset = OperatorDataset(
        x=torch.zeros(10, 1, 1),
        u=torch.linspace(-1, 1, 100),
        y=torch.zeros(10, 1, 1),
        v=torch.linspace(-1, 1, 100) ** 2
    )

    l1_m = L1Error()
    l1_res = l1_m(identity_operator, square_dataset)
    assert torch.allclose(l1_res["value"], torch.zeros(1))

    l2_m = MSError()
    l2_res = l2_m(identity_operator, square_dataset)
    assert torch.allclose(l2_res["value"], torch.ones(1) / 3, atol=1e-1)
