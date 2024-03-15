import json
import pathlib
from datetime import (
    datetime,
)

import torch

from .operator import (
    NosOperator,
)


def to_json(operator: NosOperator, out_dir: pathlib.Path):
    json_path = out_dir.joinpath("model_parameters.json")
    with open(json_path, "w") as file_handle:
        json.dump(operator.properties, file_handle)


def to_pt(operator: NosOperator, out_dir: pathlib.Path):
    pt_path = out_dir.joinpath("model.pt")
    torch.save(operator.state_dict(), pt_path)


def serialize(operator: NosOperator, out_dir: pathlib.Path = None):
    if out_dir is None:
        out_dir = pathlib.Path.cwd().joinpath("models")

    time_stamp = datetime.now()
    name = f"{operator.__class__.__name__}_{time_stamp.strftime('%Y_%m_%d_%H_%M_%S')}"
    out_dir = out_dir.joinpath(name)
    out_dir.mkdir(parents=True, exist_ok=False)

    to_json(operator, out_dir)
    to_pt(operator, out_dir)
