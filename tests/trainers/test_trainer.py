import pathlib
import tempfile

import pytest
import torch
from continuiti.benchmarks.sine import (
    SineBenchmark,
)
from continuiti.operators import (
    DeepNeuralOperator,
)

from nos.trainers import (
    Trainer,
)


@pytest.mark.slow
def test_can_run():
    benchmark = SineBenchmark(n_train=1, n_test=1)
    dataset = benchmark.train_dataset

    operator = DeepNeuralOperator(dataset.shapes)
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
        trainer(dataset)

    assert True
