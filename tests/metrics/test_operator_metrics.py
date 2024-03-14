import time

import pytest
import torch

from continuity.data import (
    OperatorDataset,
)
from continuity.operators import (
    Operator,
)
from nos.metrics import (
    NumberOfParameters,
    SpeedOfEvaluation,
)


@pytest.fixture(scope="session")
def dense() -> Operator:
    class Linear(Operator):
        def __init__(self):
            super().__init__()
            self.d = torch.nn.Linear(3, 5)

        def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.d(u)

    return Linear()


def test_can_initialize():
    assert isinstance(SpeedOfEvaluation(), SpeedOfEvaluation)
    assert isinstance(NumberOfParameters(), NumberOfParameters)


def test_call_correct(dense):
    n_obs = 10
    square_dataset = OperatorDataset(
        x=torch.zeros(n_obs, 1, 1), u=torch.ones(n_obs, 1, 3), y=torch.zeros(n_obs, 1, 1), v=torch.ones(n_obs, 1, 5)
    )

    s_m = SpeedOfEvaluation()
    start = time.time_ns()
    result = s_m(dense, square_dataset)
    end = time.time_ns()
    time_ms = (end - start) * 1e-6
    assert torch.isclose(torch.tensor(result["Value"]), torch.tensor(time_ms), atol=10)
    assert result["Unit"] == "[ms]"

    p_m = NumberOfParameters()
    result = p_m(dense, square_dataset)
    assert result["Value"] == 20
    assert result["Unit"] == "[1]"
