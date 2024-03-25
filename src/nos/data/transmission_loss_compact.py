import pathlib

import pandas as pd
import torch
from continuity.data import (
    OperatorDataset,
)

from .transmission_loss import (
    get_tl_frame,
    get_transformations,
    get_unique_crystals,
)


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
