import pathlib
from typing import (
    Callable,
)

import numpy as np
import torch
from continuity.data import (
    OperatorDataset,
)

from .transmission_loss import (
    get_transformations,
)
from .transmission_loss_compact import (
    get_tl_compact,
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
        df = get_tl_compact(path, n_samples)

        x = torch.tensor(df["frequencies"].tolist()).reshape(len(df), -1, 1)
        u = torch.stack(
            [
                torch.tensor(df["radius"].tolist()),
                torch.tensor(df["inner_radius"].tolist()),
                torch.tensor(df["gap_width"].tolist()),
            ],
            dim=1,
        ).reshape(len(df), 1, 3)
        y = torch.tensor(df["frequencies"].tolist()).reshape(len(df), -1, 1)
        v = torch.tensor(df["transmission_losses"]).unsqueeze(1).reshape(len(df), -1, 1)

        # apply wave encoding
        u = wave_encoding(u, x)

        transformations = get_transformations(x, u, y, v)

        super().__init__(x, u, y, v, **transformations)
