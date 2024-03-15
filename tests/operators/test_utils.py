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
    deserialize,
    from_json,
    from_pt,
    serialize,
    to_json,
    to_pt,
)


@pytest.fixture(scope="module")
def simple_dataset():
    return SineBenchmark(n_train=1, n_test=1).train_dataset


@pytest.fixture(scope="module")
def simple_operator(simple_dataset):
    class Simple(NosOperator):
        def __init__(self, shapes: OperatorShapes, width: int = 10, act: nn.Module = nn.Tanh):
            super().__init__(properties={"width": width, "act": act.__name__}, shapes=shapes)

            self.net = nn.Sequential(nn.Linear(shapes.u.dim, width), act(), nn.Linear(width, shapes.v.dim))

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


def test_load_json(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        to_json(operator=operator, out_dir=tmp_path)

        param = from_json(model_dir=tmp_path)

    for p in param.keys():
        assert p in operator.properties


def test_pt_can_load(simple_operator, simple_dataset):
    operator = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        to_pt(operator=operator, out_dir=tmp_path)

        param = from_pt(model_dir=tmp_path)

    for p, v in param.items():
        assert p in operator.state_dict()
        assert torch.allclose(v, operator.state_dict()[p])


def test_can_deserialize(simple_operator, simple_dataset):
    operator1 = simple_operator(simple_dataset.shapes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        serialize(operator=operator1, out_dir=tmp_path)

        model_dir = next(iter(tmp_path.glob("*")))

        operator2 = deserialize(model_dir=model_dir, model_base_class=simple_operator)
        assert isinstance(operator2, simple_operator)

    out1 = operator1(simple_dataset.x[:], simple_dataset.u[:], simple_dataset.y[:])
    out2 = operator2(simple_dataset.x[:], simple_dataset.u[:], simple_dataset.y[:])

    assert torch.allclose(out1, out2)
