import pathlib
import tempfile

import pytest
import torch
import torch.nn as nn
from continuity.benchmarks.sine import (
    SineBenchmark,
)
from continuity.operators import (
    OperatorShapes,
)

from nos.operators import (
    NosOperator,
    serialize,
    to_json,
    to_pt,
)
from nos.utils import (
    dataclass_to_dict,
)


@pytest.fixture(scope="module")
def simple_dataset():
    return SineBenchmark(n_train=1, n_test=1).train_dataset


@pytest.fixture(scope="module")
def simple_operator(simple_dataset):
    class Simple(NosOperator):
        def __init__(self, shapes: OperatorShapes):
            super().__init__({"shapes": dataclass_to_dict(shapes), "width": 10})

            self.net = nn.Sequential(nn.Linear(shapes.u.dim, 10), nn.Linear(10, shapes.v.dim))

        def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.net(u)

    return Simple


def test_to_json_correct(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        to_json(operator=operator, out_dir=tmp_path)
        assert len(list(tmp_path.glob("*.json"))) == 1


def test_to_pt_can_save(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        to_pt(operator=operator, out_dir=tmp_path)
        assert len(list(tmp_path.glob("*.pt"))) == 1


def test_to_pt_can_load(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        to_pt(operator=operator, out_dir=tmp_path)

        save_path = next(iter(tmp_path.glob("*.pt")))
        operator2 = simple_operator(simple_dataset.shapes)
        operator2.load_state_dict(torch.load(save_path))

    out1 = operator(simple_dataset.x[:], simple_dataset.u[:], simple_dataset.y[:])
    out2 = operator2(simple_dataset.x[:], simple_dataset.u[:], simple_dataset.y[:])

    assert torch.allclose(out1, out2)


def test_serialize_out_dir_create(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        serialize(operator=operator, out_dir=tmp_path)

        dir_path = next(iter(tmp_path.glob("*")))
        assert dir_path.is_dir()


def test_serialize_all_outputs(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        serialize(operator=operator, out_dir=tmp_path)

        json_paths = list(tmp_path.rglob("*.json"))
        pth_paths = list(tmp_path.rglob("*.pt"))

        assert len(json_paths) == 1
        assert len(pth_paths) == 1
