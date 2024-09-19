import pathlib  # noqa: D100
from typing import Literal

import numpy as np
import pandas as pd
import torch
from continuiti.data import OperatorDataset
from continuiti.transforms import Normalize, Transform

from nos.transforms import MinMaxScale, QuantileScaler

FILTER_COLS = ["radius", "inner_radius", "gap_width"]


def get_tl_from_path(path: pathlib.Path) -> pd.DataFrame:
    """Get transmission loss from csv file to pandas DataFrame."""
    if path.is_file():
        tl_data = pd.read_csv(path, dtype=np.float32)
    else:
        tl_data = pd.DataFrame()
        for file in path.rglob("*.csv"):
            tl_tmp = pd.read_csv(file, dtype=np.float32)
            tl_data = pd.concat([tl_data, tl_tmp])
    return tl_data


def get_unique_crystals(df: pd.DataFrame) -> pd.DataFrame:
    """Get unique configurations from pandas data-frame as pandas data-frame."""
    return df[FILTER_COLS].drop_duplicates()


def get_n_unique(df: pd.DataFrame, n_samples: int = -1) -> pd.DataFrame:
    """Get n unique crystal configurations from data-frame."""
    if n_samples == -1:
        return df

    unique_crystals = get_unique_crystals(df)
    unique_crystals = unique_crystals.sample(n_samples)

    return df.merge(unique_crystals, on=FILTER_COLS)


def get_tl_frame(path: pathlib.Path, n_samples: int = -1) -> pd.DataFrame:
    """Retrieve exactly n samples from csv-path."""
    tl_data = get_tl_from_path(path)
    return get_n_unique(tl_data, n_samples)


def get_min_max_transform(src: torch.Tensor) -> Transform:
    """Get min-max transformation for src tensor."""
    src_tmp = src.transpose(0, 1).flatten(1, -1)
    src_min, _ = torch.min(src_tmp, dim=1)
    src_max, _ = torch.max(src_tmp, dim=1)
    src_min = src_min.reshape(src.size(1), *[1] * (src.ndim - 2))  # without observation dimension for dataloader.
    src_max = src_max.reshape(src.size(1), *[1] * (src.ndim - 2))

    return MinMaxScale(src_min, src_max)


def get_normalize_transform(src: torch.Tensor) -> Transform:
    """Get normalize transformation fro src tensor."""
    src_tmp = src.transpose(0, 1).flatten(1, -1)
    src_mean = torch.mean(src_tmp, dim=1)
    src_std = torch.std(src_tmp, dim=1)
    src_mean = src_mean.reshape(src.size(1), *[1] * (src.ndim - 2))  # without observation dimension for dataloader.
    src_std = src_std.reshape(src.size(1), *[1] * (src.ndim - 2))

    return Normalize(src_mean, src_std)

class UnknownTransformationError(Exception):
    """Provided literal transformation is not known."""
class TLDataset(OperatorDataset):
    """Transmission loss dataset with transmission loss on a evaluation basis (no grouping in observations).

    Args:
        OperatorDataset (_type_): _description_

    """

    def __init__(
        self,
        path: pathlib.Path,
        n_samples: int = -1,
        v_transform: Literal["quantile", "min_max", "normalize"] = "quantile",
    ) -> None:
        """Initialize.

        Args:
            path (pathlib.Path): Path to csv file or dir containing multiple csv files.
            n_samples (int, optional): Number of observations in the dataset. Defaults to -1 (all).
            v_transform (Literal[&quot;quantile&quot;, &quot;min_max&quot;, &quot;normalize&quot;], optional):
                Transformation to use for v. Defaults to "quantile".

        Raises:
            UnknownTransformationError: The v_transfrom literal does not match available transformations.

        """
        # retrieve data
        tl_data = get_tl_frame(path, n_samples)

        x = torch.stack(
            [
                torch.tensor(tl_data["radius"].tolist()),
                torch.tensor(tl_data["inner_radius"].tolist()),
                torch.tensor(tl_data["gap_width"].tolist()),
            ],
            dim=1,
        ).unsqueeze(-1)
        u = x
        y = torch.tensor(tl_data["frequency"].tolist()).reshape(-1, 1, 1)
        v = torch.tensor(tl_data["transmission_loss"].tolist()).unsqueeze(1).reshape(-1, 1, 1)

        v_t: Transform
        if v_transform == "quantile":
            v_t = QuantileScaler(v)
        elif v_transform == "min_max":
            v_t = get_min_max_transform(v)
        elif v_transform == "normalize":
            v_t = get_normalize_transform(v)
        else:
            raise UnknownTransformationError

        transformations = {
            "x_transform": get_min_max_transform(x),
            "u_transform": get_min_max_transform(u),
            "y_transform": get_min_max_transform(y),
            "v_transform": v_t,
        }

        super().__init__(x, u, y, v, **transformations)


def get_tl_compact(path: pathlib.Path, n_samples: int = -1) -> pd.DataFrame:
    """Get transmissionloss data-frame in a compacted form."""
    unique_crystals = get_unique_crystals(get_tl_frame(path, n_samples))

    compact_df = pd.DataFrame()
    for _, crystal in unique_crystals.iterrows():
        tmp_df = (get_tl_frame(path, n_samples))[
            ((get_tl_frame(path, n_samples))["radius"] == crystal["radius"])
            & ((get_tl_frame(path, n_samples))["inner_radius"] == crystal["inner_radius"])
            & ((get_tl_frame(path, n_samples))["gap_width"] == crystal["gap_width"])
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
            },
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
    ) -> None:
        """Initialize.

        Args:
            path (pathlib.Path): Path to csv file or dir containing multiple csv files.
            n_samples (int, optional): Number of observations in the dataset. Defaults to -1 (all).
            v_transform (Literal[&quot;quantile&quot;, &quot;min_max&quot;, &quot;normalize&quot;], optional):
                Transformation to use for v. Defaults to "quantile".

        Raises:
            UnknownTransformationError: The v_transfrom literal does not match available transformations.

        """
        tl_data = get_tl_compact(path, n_samples)

        x = torch.stack(
            [
                torch.tensor(tl_data["radius"].tolist()),
                torch.tensor(tl_data["inner_radius"].tolist()),
                torch.tensor(tl_data["gap_width"].tolist()),
            ],
            dim=1,
        ).unsqueeze(-1)
        u = x
        y = torch.tensor(tl_data["frequencies"].tolist()).reshape(len(tl_data), 1, -1)
        v = torch.tensor(tl_data["transmission_losses"]).unsqueeze(1).reshape(len(tl_data), 1, -1)

        v_t: Transform
        if v_transform == "quantile":
            v_t = QuantileScaler(v)
        elif v_transform == "min_max":
            v_t = get_min_max_transform(v)
        elif v_transform == "normalize":
            v_t = get_normalize_transform(v)
        else:
            raise UnknownTransformationError

        transformations = {
            "x_transform": get_min_max_transform(x),
            "u_transform": get_min_max_transform(u),
            "y_transform": get_min_max_transform(y),
            "v_transform": v_t,
        }

        super().__init__(x, u, y, v, **transformations)
