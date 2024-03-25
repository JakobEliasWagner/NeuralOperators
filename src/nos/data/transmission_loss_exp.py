import pathlib

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
