import pathlib
import tempfile

import torch

from nos.operators import (
    DeepNeuralOperator,
)
from nos.trainers import (
    Trainer,
)


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
            max_n_saved_models=2,
        )
        trainer(tl_compact_dataset)

    assert True
