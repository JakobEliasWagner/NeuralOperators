import pathlib
from typing import (
    Callable,
    Dict,
)

import numpy as np
import pandas as pd
import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.transforms import (
    Transform,
)

from nos.transforms import (
    MinMaxScale,
    QuantileScaler,
)

FILTER_COLS = ["radius", "inner_radius", "gap_width"]


def get_tl_from_path(path: pathlib.Path):
    if path.is_file():
        df = pd.read_csv(path, dtype=np.float32)
    else:
        df = pd.DataFrame()
        for file in path.rglob("*.csv"):
            df_tmp = pd.read_csv(file, dtype=np.float32)
            df = pd.concat([df, df_tmp])
    return df


def get_unique_crystals(df: pd.DataFrame) -> pd.DataFrame:
    return df[FILTER_COLS].drop_duplicates()


def get_n_unique(df: pd.DataFrame, n_samples: int = -1):
    if n_samples == -1:
        return df

    unique_crystals = get_unique_crystals(df)
    unique_crystals = unique_crystals.sample(n_samples)

    return pd.merge(df, unique_crystals, on=FILTER_COLS)


def get_tl_frame(path: pathlib.Path, n_samples: int = -1):
    df = get_tl_from_path(path)
    return get_n_unique(df, n_samples)


def get_transformations(x: torch.Tensor, u: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> Dict[str, Transform]:
    x_tmp = x.view(-1, x.size(-1))
    x_min, _ = torch.min(x_tmp, dim=0)
    x_max, _ = torch.max(x_tmp, dim=0)

    u_tmp = u.view(-1, u.size(-1))
    u_min, _ = torch.min(u_tmp, dim=0)
    u_max, _ = torch.max(u_tmp, dim=0)

    y_tmp = y.view(-1, y.size(-1))
    y_min, _ = torch.min(y_tmp, dim=0)
    y_max, _ = torch.max(y_tmp, dim=0)

    v_tmp = v.view(-1, v.size(-1))
    v_min, _ = torch.min(v_tmp, dim=0)
    v_max, _ = torch.max(v_tmp, dim=0)

    return {
        "x_transform": MinMaxScale(x_min, x_max),
        "u_transform": MinMaxScale(u_min, u_max),
        "y_transform": MinMaxScale(y_min, y_max),
        "v_transform": QuantileScaler(v),
    }


class TLDataset(OperatorDataset):
    def __init__(self, path: pathlib.Path, n_samples: int = -1):
        # retrieve data
        df = get_tl_frame(path, n_samples)

        x = torch.stack(
            [
                torch.tensor(df["radius"].tolist()),
                torch.tensor(df["inner_radius"].tolist()),
                torch.tensor(df["gap_width"].tolist()),
            ],
            dim=1,
        ).reshape(-1, 1, 3)
        u = x
        y = torch.tensor(df["frequency"].tolist()).reshape(-1, 1, 1)
        v = torch.tensor(df["transmission_loss"].tolist()).unsqueeze(1).reshape(-1, 1, 1)

        transformations = get_transformations(x, u, y, v)

        super().__init__(x, u, y, v, **transformations)


def get_tl_compact(path: pathlib.Path, n_samples: int = -1):
    df = get_tl_frame(path, n_samples)
    unique_crystals = get_unique_crystals(df)

    compact_df = pd.DataFrame()
    for _, crystal in unique_crystals.iterrows():
        tmp_df = df[
            (df["radius"] == crystal["radius"])
            & (df["inner_radius"] == crystal["inner_radius"])
            & (df["gap_width"] == crystal["gap_width"])
        ]
        c_df = pd.DataFrame(
            {
                "radius": crystal["radius"],
                "inner_radius": crystal["inner_radius"],
                "gap_width": crystal["gap_width"],
                "frequencies": [tmp_df["frequency"].tolist()],
                "min_frequency": min(tmp_df["frequency"]),
                "max_frequency": max(tmp_df["frequency"]),
                "transmission_losses": [tmp_df["transmission_loss"].tolist()],
                "min_transmission_loss": min(tmp_df["transmission_loss"]),
                "max_transmission_loss": max(tmp_df["transmission_loss"]),
            }
        )

        compact_df = pd.concat([compact_df, c_df], ignore_index=True)
    return compact_df


class TLDatasetCompact(OperatorDataset):
    """Transmission loss dataset, with bigger evaluation space."""

    def __init__(self, path: pathlib.Path, n_samples: int = -1):
        df = get_tl_compact(path, n_samples)

        x = torch.stack(
            [
                torch.tensor(df["radius"].tolist()),
                torch.tensor(df["inner_radius"].tolist()),
                torch.tensor(df["gap_width"].tolist()),
            ],
            dim=1,
        ).reshape(len(df), 1, 3)
        u = x
        y = torch.tensor(df["frequencies"].tolist()).reshape(len(df), -1, 1)
        v = torch.tensor(df["transmission_losses"]).unsqueeze(1).reshape(len(df), -1, 1)

        transformations = get_transformations(x, u, y, v)

        super().__init__(x, u, y, v, **transformations)


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


class TLDatasetCompactExp(OperatorDataset):
    """Transmission loss dataset, with bigger evaluation space."""

    def __init__(self, path: pathlib.Path, n_samples: int = -1):
        df = get_tl_compact(path, n_samples)

        x = torch.stack(
            [
                torch.tensor(df["radius"].tolist()),
                torch.tensor(df["inner_radius"].tolist()),
                torch.tensor(df["gap_width"].tolist()),
            ],
            dim=1,
        ).reshape(len(df), 1, 3)
        u = x
        y = torch.tensor(df["frequencies"].tolist()).reshape(len(df), -1, 1)
        v = torch.tensor(df["transmission_losses"]).unsqueeze(1).reshape(len(df), -1, 1)

        # exp part
        v = torch.clamp(v, max=0.0)  # clamp unphysical values over zero to zero
        v = torch.pow(v, 10)

        transformations = get_transformations(x, u, y, v)

        super().__init__(x, u, y, v, **transformations)
