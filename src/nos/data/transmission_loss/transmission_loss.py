import pathlib
from typing import (
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
