import pathlib
from typing import (
    Callable,
)

import numpy as np
import pandas as pd
import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.transforms import (
    Normalize,
)


def simple_sine_encoding(u, x):
    vals = torch.einsum("abc,ade->adc", u, x)  # different modes for the properties
    # min-max scale all values
    min_vals, _ = torch.min(vals.view(-1, vals.size(-1)), dim=0)
    max_vals, _ = torch.max(vals.view(-1, vals.size(-1)), dim=0)
    vals = ((vals - min_vals) / (max_vals - min_vals) + 1) * torch.pi

    x_min = torch.min(x)
    x_max = torch.max(x)
    x_scaled = ((x - x_min) / (x_max - x_min) - 0.5) * 2

    return torch.sin(x_scaled * vals)


def gaussian_modulated_sine_encoding(u, x):
    # leads to massive overfitting
    alpha = 5

    min_u, _ = torch.min(u.view(-1, u.size(-1)), dim=0)
    max_u, _ = torch.max(u.view(-1, u.size(-1)), dim=0)
    u_scaled = ((u - min_u) / (max_u - min_u) + 1) * 2  # [1, 2]

    min_x, _ = torch.min(x.view(-1, x.size(-1)), dim=0)
    max_x, _ = torch.max(x.view(-1, x.size(-1)), dim=0)
    x_scaled = ((x - min_x) / (max_x - min_x) - 0.5) * 2  # [-1, 1]

    return torch.exp(-alpha * x_scaled**2) * torch.sin(4 * np.pi * u_scaled * (x_scaled + 1))


class TLDatasetCompactWave(OperatorDataset):
    """Transmission loss dataset, for use with FNO."""

    def __init__(self, path: pathlib.Path, n_samples: int = -1, wave_encoding: Callable = simple_sine_encoding):
        if path.is_file():
            df = pd.read_csv(path, dtype=np.float32)
        else:
            df = pd.DataFrame()
            for file in path.rglob("*.csv"):
                df_tmp = pd.read_csv(file, dtype=np.float32)
                df = pd.concat([df, df_tmp])

        unique_crystals = df[["radius", "inner_radius", "gap_width"]].drop_duplicates()

        num_evals = len(df) // len(unique_crystals)

        x = torch.empty((len(unique_crystals), 1, 3))
        u = x
        y = torch.empty((len(unique_crystals), num_evals, 1))
        v = torch.empty((len(unique_crystals), num_evals, 1))

        for i, (_, crystal) in enumerate(unique_crystals.iterrows()):
            c_df = df.loc[
                (df["radius"] == crystal["radius"])
                * (df["inner_radius"] == crystal["inner_radius"])
                * (df["gap_width"] == crystal["gap_width"])
            ]

            u[i] = torch.tensor([crystal["radius"], crystal["inner_radius"], crystal["gap_width"]]).reshape(1, 3)
            y[i] = torch.tensor([c_df["frequency"].to_list()]).reshape(num_evals, 1)
            v[i] = torch.tensor([[c_df["transmission_loss"].to_list()]]).reshape(num_evals, 1)

        if n_samples != -1:
            perm = torch.randperm(x.size(0))
            idx = perm[:n_samples]
            u = u[idx]
            y = y[idx]
            v = v[idx]

        x = y  # for FNO x and y need to be sampled on the same grid

        # function heavily influences overfitting
        u = wave_encoding(u, x)

        # find appropriate transformations
        means = df.mean().to_dict()
        stds = df.std().to_dict()

        x_transforms = Normalize(
            torch.tensor(means["frequency"]).reshape(1, 1), torch.tensor(stds["frequency"]).reshape(1, 1)
        )
        u_transforms = None
        y_transforms = Normalize(
            torch.tensor(means["frequency"]).reshape(1, 1), torch.tensor(stds["frequency"]).reshape(1, 1)
        )
        v_transforms = Normalize(
            torch.tensor(means["transmission_loss"]).reshape(1, 1),
            torch.tensor(stds["transmission_loss"]).reshape(1, 1),
        )

        super().__init__(x, u, y, v, x_transforms, u_transforms, y_transforms, v_transforms)
