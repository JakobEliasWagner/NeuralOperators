import pathlib

import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.discrete import (
    UniformBoxSampler,
)
from continuity.discrete.box_sampler import (
    BoxSampler,
)

from nos.data.transmission_loss import (
    get_tl_compact,
    get_transformations,
)

from .params_to_indicator import (
    params_to_indicator,
)


class IndicatorTLDataset(OperatorDataset):
    def __init__(
        self, path: pathlib.Path, n_samples: int = -1, sampler: BoxSampler = None, n_box_samples: int = 2**10
    ):
        # retrieve data
        df = get_tl_compact(path, n_samples)

        # params
        if sampler is None:
            sampler = UniformBoxSampler([0.0, 0.0], [22e-3, 22e-3])
        params = torch.stack(
            [
                torch.tensor(df["radius"].tolist()),
                torch.tensor(df["inner_radius"].tolist()),
                torch.tensor(df["gap_width"].tolist()),
            ],
            dim=1,
        ).reshape(-1, 1, 3)
        x, u = params_to_indicator(params, sampler, n_box_samples)
        y = torch.tensor(df["frequencies"].tolist()).reshape(len(df), -1, 1)
        v = torch.tensor(df["transmission_losses"]).unsqueeze(1).reshape(len(df), -1, 1)

        transformations = get_transformations(x, u, y, v)

        super().__init__(x, u, y, v, **transformations)
