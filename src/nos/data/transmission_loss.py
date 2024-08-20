import pathlib
from typing import (
    Dict,
)

import numpy as np
import pandas as pd
import torch
from continuiti.data import (
    OperatorDataset,
)
from continuiti.transforms import (
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
    x_tmp = x.transpose(0, 1).flatten(1, -1)
    x_min, _ = torch.min(x_tmp, dim=1)
    x_max, _ = torch.max(x_tmp, dim=1)
    x_min = x_min.reshape(x.size(1), *[1] * (x.ndim - 1))  # without observation dimension for dataloader.
    x_max = x_max.reshape(x.size(1), *[1] * (x.ndim - 1))

    u_tmp = u.transpose(0, 1).flatten(1, -1)
    u_min, _ = torch.min(u_tmp, dim=1)
    u_max, _ = torch.max(u_tmp, dim=1)
    u_min = u_min.reshape(u.size(1), *[1] * (u.ndim - 1))  # without observation dimension for dataloader.
    u_max = u_max.reshape(u.size(1), *[1] * (u.ndim - 1))

    y_tmp = y.transpose(0, 1).flatten(1, -1)
    y_min, _ = torch.min(y_tmp, dim=1)
    y_max, _ = torch.max(y_tmp, dim=1)
    y_min = y_min.reshape(y.size(1), *[1] * (y.ndim - 1))  # without observation dimension for dataloader.
    y_max = y_max.reshape(y.size(1), *[1] * (y.ndim - 1))

    v_tmp = v.transpose(0, 1).flatten(1, -1)
    v_min, _ = torch.min(v_tmp, dim=1)
    v_max, _ = torch.max(v_tmp, dim=1)
    v_min = v_min.reshape(v.size(1), *[1] * (v.ndim - 1))  # without observation dimension for dataloader.
    v_max = v_max.reshape(v.size(1), *[1] * (v.ndim - 1))

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
        ).unsqueeze(-1)
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
        ).unsqueeze(-1)
        u = x
        y = torch.tensor(df["frequencies"].tolist()).reshape(len(df), 1, -1)
        v = torch.tensor(df["transmission_losses"]).unsqueeze(1).reshape(len(df), 1, -1)

        transformations = get_transformations(x, u, y, v)

        super().__init__(x, u, y, v, **transformations)
