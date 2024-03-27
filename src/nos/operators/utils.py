import json
import pathlib
from datetime import (
    datetime,
)

import torch
from continuity.operators import (
    Operator,
)
from continuity.operators.shape import (
    OperatorShapes,
    TensorShape,
)

import nos
from nos.utils import (
    dataclass_to_dict,
)

from .operator import (
    NeuralOperator,
)


def to_json(operator: NeuralOperator, out_dir: pathlib.Path, json_handle: str = "model_parameters.json"):
    json_path = out_dir.joinpath(json_handle)
    properties = operator.properties
    properties["shapes"] = dataclass_to_dict(operator.shapes)
    properties["base_class"] = operator.__class__.__name__
    with open(json_path, "w") as file_handle:
        json.dump(operator.properties, file_handle)


def to_pt(operator: NeuralOperator, out_dir: pathlib.Path, pt_handle: str = "model.pt"):
    pt_path = out_dir.joinpath(pt_handle)
    torch.save(operator.state_dict(), pt_path)


def serialize(operator: NeuralOperator, out_dir: pathlib.Path = None) -> pathlib.Path:
    if out_dir is None:
        out_dir = pathlib.Path.cwd().joinpath("out_models")

    time_stamp = datetime.now()
    name = f"{operator.__class__.__name__}_{time_stamp.strftime('%Y_%m_%d_%H_%M_%S')}"
    out_dir = out_dir.joinpath(name)
    i = 0
    while out_dir.is_dir():
        out_dir = out_dir.parent.joinpath(f"{name}-{i}")
        i += 1
    out_dir.mkdir(parents=True, exist_ok=False)

    to_json(operator, out_dir)
    to_pt(operator, out_dir)

    return out_dir


def from_json(model_dir: pathlib.Path, json_handle: str = "model_parameters.json") -> dict:
    json_path = model_dir.joinpath(json_handle)
    with open(json_path, "r") as file_handle:
        return json.load(file_handle)


def from_pt(model_dir: pathlib.Path, pt_handle: str = "model.pt"):
    pt_path = model_dir.joinpath(pt_handle)
    return torch.load(pt_path)


def deserialize(
    model_dir: pathlib.Path,
    model_base_class: type(NeuralOperator) = None,
    json_handle: str = "model_parameters.json",
    pt_handle="model.pt",
) -> Operator:
    parameters = from_json(model_dir=model_dir, json_handle=json_handle)
    shapes = parameters["shapes"]
    parameters["shapes"] = OperatorShapes(
        x=TensorShape(shapes["x"]["num"], shapes["x"]["dim"]),
        u=TensorShape(shapes["u"]["num"], shapes["u"]["dim"]),
        y=TensorShape(shapes["y"]["num"], shapes["y"]["dim"]),
        v=TensorShape(shapes["v"]["num"], shapes["v"]["dim"]),
    )
    if "act" in parameters:
        act = parameters["act"]
        parameters["act"] = getattr(torch.nn, act)()

    if model_base_class is None:
        model_base_class = getattr(nos.operators, parameters["base_class"])
    del parameters["base_class"]

    operator = model_base_class(**parameters)

    state_dict = from_pt(model_dir=model_dir, pt_handle=pt_handle)
    operator.load_state_dict(state_dict)

    return operator
