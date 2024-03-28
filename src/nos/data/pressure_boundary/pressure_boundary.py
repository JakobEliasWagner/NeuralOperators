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
    def __init__(self, data_dir: pathlib.Path, n_samples: int = -1, n_observations: int = 100):
        # load required files
        data = xdmf_to_torch(data_dir.joinpath("solution.xdmf"))
        with open(data_dir.joinpath("properties.json"), "r") as file_handle:
            properties = json.load(file_handle)

        # derive more properties (currently unused)
        prop_frequencies = torch.tensor(properties["frequencies"])
        biggest_frequency = torch.max(prop_frequencies)
        frequencies = data["Encoding"]
        frequencies = frequencies - torch.floor(frequencies)
        frequencies = frequencies * (10 ** (torch.floor(torch.log10(biggest_frequency)) + 1))

        u = data["Values"]

        x = data["Geometry"]
        x = x.unsqueeze(0).expand(u.size(0), -1, -1)
        x = x[:, :, :3]

        y = torch.linspace(0, 1, n_observations)
        y = y.unsqueeze(0)
        y = y.unsqueeze(-1).expand(u.size(0), -1, -1)

        top_p = torch.tensor(properties["top_parameters"]).unsqueeze(1).expand(-1, n_observations, -1)
        right_p = torch.tensor(properties["right_parameters"]).unsqueeze(1).expand(-1, n_observations, -1)
        v = torch.cat([top_p, right_p], dim=2)

        # transformations
        transformations = {
            "x_transform": MinMaxScale(torch.min(x), torch.max(x)),
            "u_transform": MinMaxScale(torch.min(u), torch.max(u)),
            "y_transform": MinMaxScale(torch.min(y), torch.max(y)),
            "v_transform": MinMaxScale(torch.min(v), torch.max(v)),
        }

        super().__init__(x, u, y, v, **transformations)
