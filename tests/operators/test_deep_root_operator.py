import pytest
from continuity.benchmarks.sine import (
    SineBenchmark,
)
from continuity.operators.losses import (
    MSELoss,
)
from continuity.trainer import (
    Trainer,
)

from nos.operators.deep_root_operator import (
    DeepRootOperator,
)

from .util import (
    get_shape_mismatches,
)


def test_shapes(random_shape_operator_datasets):
    operators = [DeepRootOperator(dataset.shapes) for dataset in random_shape_operator_datasets]
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


@pytest.mark.slow
def test_convergence():
    # Data set
    benchmark = SineBenchmark(n_train=1, n_test=1)
    dataset = benchmark.train_dataset

    # Operator
    operator = DeepRootOperator(dataset.shapes, dot_depth=4, dot_width=48)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3, batch_size=1)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 5e-3
