import json
import pathlib

import torch
from continuity.data import (
    OperatorDataset,
)

from nos.data.utils import (
    xdmf_to_torch,
)
from nos.transforms import (
    MinMaxScale,
    QuantileScaler,
)


class PulsatingSphere(OperatorDataset):
    def __init__(self, data_dir: pathlib.Path, n_samples: int = -1):
        # load required files
        data = xdmf_to_torch(next(data_dir.rglob("*.xdmf")))
        with open(data_dir.joinpath("properties.json"), "r") as file_handle:
            properties = json.load(file_handle)

        v = data["Values"]

        y = data["Geometry"]
        y = y.unsqueeze(0).expand(v.size(0), -1, -1)
        y = y[:, :, :3]

        top_p = torch.tensor(properties["top_samples"]).unsqueeze(1).expand(-1, 1, -1)
        right_p = torch.tensor(properties["right_samples"]).unsqueeze(1).expand(-1, 1, -1)
        frequency = torch.tensor(properties["frequency_samples"]).reshape(-1, 1, 1)
        x = torch.cat([top_p, right_p, frequency], dim=2)
        u = x

        if n_samples != -1:
            indices = torch.randperm(u.size(0))[:n_samples]
            x = x[indices]
            u = u[indices]
            y = y[indices]
            v = v[indices]

        x = x.to(torch.get_default_dtype())
        u = u.to(torch.get_default_dtype())
        y = y.to(torch.get_default_dtype())
        v = v.to(torch.get_default_dtype())

        x_min, _ = torch.min(x.view(-1, x.size(-1)), dim=0)
        x_max, _ = torch.max(x.view(-1, x.size(-1)), dim=0)
        u_min, _ = torch.min(u.view(-1, u.size(-1)), dim=0)
        u_max, _ = torch.max(u.view(-1, u.size(-1)), dim=0)
        y_min, _ = torch.min(y.view(-1, y.size(-1)), dim=0)
        y_max, _ = torch.max(y.view(-1, y.size(-1)), dim=0)

        # transformations
        transformations = {
            "x_transform": MinMaxScale(x_min, x_max),
            "u_transform": MinMaxScale(u_min, u_max),
            "y_transform": MinMaxScale(y_min, y_max),
            "v_transform": QuantileScaler(v),
        }

        super().__init__(x, u, y, v, **transformations)
