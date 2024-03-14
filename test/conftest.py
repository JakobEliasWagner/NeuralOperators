import random

import numpy as np
import pytest
import torch

pytest_plugins = [
    "test.operators.fixtures",
]


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
