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
)


class PressureBoundaryDataset(OperatorDataset):
    def __init__(
        self, data_dir: pathlib.Path, n_samples: int = -1, n_observations: int = 100, do_normalize: bool = True
    ):
        # load required files
        data = xdmf_to_torch(next(data_dir.rglob("*.xdmf")))
        with open(data_dir.joinpath("properties.json"), "r") as file_handle:
            properties = json.load(file_handle)

        u = data["Values"]
        if do_normalize:
            u_abs_max = torch.amax(torch.abs(u), dim=1).reshape(-1, 1, 2)
            u = u / u_abs_max

        x = data["Geometry"]
        x = x.unsqueeze(0).expand(u.size(0), -1, -1)
        x = x[:, :, :3]

        y = torch.linspace(0, 1, n_observations)
        y = y.unsqueeze(0)
        y = y.unsqueeze(-1).expand(u.size(0), -1, -1)

        top_p = torch.tensor(properties["top_samples"]).unsqueeze(1).expand(-1, n_observations, -1)
        right_p = torch.tensor(properties["right_samples"]).unsqueeze(1).expand(-1, n_observations, -1)
        v = torch.cat([top_p, right_p], dim=2)

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

        # transformations
        transformations = {
            "x_transform": MinMaxScale(torch.min(x), torch.max(x)),
            "u_transform": MinMaxScale(torch.min(u), torch.max(u)),
            "y_transform": MinMaxScale(torch.min(y), torch.max(y)),
            "v_transform": MinMaxScale(torch.min(v), torch.max(v)),
        }

        super().__init__(x, u, y, v, **transformations)
