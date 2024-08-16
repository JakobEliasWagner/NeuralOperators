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
    CenterQuantileScaler,
    MinMaxScale,
)


class PressureBoundaryDatasetSampled(OperatorDataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        n_samples: int = -1,
        n_observations: int = 2,
        n_versions: int = 16,
        n_sensors: int = 64,
    ):
        # load required files
        data = xdmf_to_torch(next(data_dir.rglob("*.xdmf")))
        with open(data_dir.joinpath("properties.json"), "r") as file_handle:
            properties = json.load(file_handle)

        u = data["Values"]

        x = data["Geometry"]
        x = x.unsqueeze(0).expand(u.size(0), -1, -1)
        x = x[:, :, :3]

        y = torch.linspace(0, 1, n_observations)
        y = y.unsqueeze(0)
        y = y.unsqueeze(-1).expand(u.size(0), -1, -1)

        self.frequencies = torch.tensor(properties["frequency_samples"])

        top_p = torch.tensor(properties["top_samples"]).unsqueeze(1).expand(-1, n_observations, -1)
        right_p = torch.tensor(properties["right_samples"]).unsqueeze(1).expand(-1, n_observations, -1)
        v = torch.cat([top_p, right_p], dim=2)

        if n_samples != -1:
            indices = torch.randperm(u.size(0))[:n_samples]
            x = x[indices]
            u = u[indices]
            y = y[indices]
            v = v[indices]

        x = x.to(torch.get_default_dtype())[:, :, :2]
        u = u.to(torch.get_default_dtype())
        y = y.to(torch.get_default_dtype())
        v = v.to(torch.get_default_dtype())

        # sample everything
        x_final = []
        u_final = []
        y_final = []
        v_final = []
        for xi, ui, yi, vi in zip(x, u, y, v):
            for _ in range(n_versions):
                perm = torch.randperm(ui.size(0))
                indices = perm[:n_sensors]
                x_final.append(xi[indices])
                u_final.append(ui[indices])
                y_final.append(yi)
                v_final.append(vi)

        x_final = torch.stack(x_final, dim=0)
        u_final = torch.stack(u_final, dim=0)
        y_final = torch.stack(y_final, dim=0)
        v_final = torch.stack(v_final, dim=0)

        # transformations
        transformations = {
            "x_transform": MinMaxScale(torch.min(x_final), torch.max(x_final)),
            "u_transform": CenterQuantileScaler(u_final),
            "y_transform": MinMaxScale(torch.min(y_final), torch.max(y_final)),
            "v_transform": MinMaxScale(torch.min(v_final), torch.max(v_final)),
        }

        super().__init__(x_final, u_final, y_final, v_final, **transformations)
