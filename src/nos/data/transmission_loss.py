import pathlib
from typing import (
    Literal,
)

import numpy as np
import pandas as pd
import torch
from continuiti.data import (
    OperatorDataset,
)
from continuiti.transforms import (
    Normalize,
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


def get_min_max_transform(src: torch.Tensor) -> Transform:
    src_tmp = src.transpose(0, 1).flatten(1, -1)
    src_min, _ = torch.min(src_tmp, dim=1)
    src_max, _ = torch.max(src_tmp, dim=1)
    src_min = src_min.reshape(src.size(1), *[1] * (src.ndim - 2))  # without observation dimension for dataloader.
    src_max = src_max.reshape(src.size(1), *[1] * (src.ndim - 2))

    return MinMaxScale(src_min, src_max)

def get_normalize_transform(src: torch.Tensor) -> Transform:
    src_tmp = src.transpose(0, 1).flatten(1, -1)
    src_mean, _ = torch.mean(src_tmp, dim=1)
    src_std, _ = torch.std(src_tmp, dim=1)
    src_mean = src_mean.reshape(src.size(1), *[1] * (src.ndim - 2))  # without observation dimension for dataloader.
    src_std = src_std.reshape(src.size(1), *[1] * (src.ndim - 2))

    return Normalize(src_mean, src_std)


class TLDataset(OperatorDataset):
    def __init__(
        self,
        path: pathlib.Path,
        n_samples: int = -1,
        v_transform: Literal["quantile", "min_max", "normalize"] = "quantile",
    ):
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

        v_t: Transform
        if v_transform == "quantile":
            v_t = QuantileScaler(v)
        elif v_transform == "min_max":
            v_t = get_min_max_transform(v)
        elif v_transform == "normalize":
            v_t = get_normalize_transform(v)
        else:
            raise ValueError(f"Unknown transformation: {v_transform}.")

        transformations = {
            "x_transform": get_min_max_transform(x),
            "u_transform": get_min_max_transform(u),
            "y_transform": get_min_max_transform(y),
            "v_transform": v_t,
        }

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

    def __init__(
        self,
        path: pathlib.Path,
        n_samples: int = -1,
        v_transform: Literal["quantile", "min_max", "normalize"] = "quantile",
    ):
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

        v_t: Transform
        if v_transform == "quantile":
            v_t = QuantileScaler(v)
        elif v_transform == "min_max":
            v_t = get_min_max_transform(v)
        elif v_transform == "normalize":
            v_t = get_normalize_transform(v)
        else:
            raise ValueError(f"Unknown transformation: {v_transform}.")

        transformations = {
            "x_transform": get_min_max_transform(x),
            "u_transform": get_min_max_transform(u),
            "y_transform": get_min_max_transform(y),
            "v_transform": v_t,
        }

        super().__init__(x, u, y, v, **transformations)
