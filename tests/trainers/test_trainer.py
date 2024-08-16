import pathlib
import tempfile

import pytest
import torch

from nos.operators import (
    DeepNeuralOperator,
)
from nos.trainers import (
    Trainer,
)


@pytest.mark.slow
def test_can_run(tl_compact_dataset):
    operator = DeepNeuralOperator(tl_compact_dataset.shapes)
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3, weight_decay=5e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        trainer = Trainer(
            operator=operator,
            criterion=torch.nn.MSELoss(),
            optimizer=optimizer,
            max_epochs=10,
            out_dir=tmp_path,
        )
        trainer(tl_compact_dataset)

    assert True
