import pathlib

import numpy as np
import pandas as pd
import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.transforms import (
    Normalize,
)

from nos.transforms import (
    MinMaxScale,
)


class TLDataset(OperatorDataset):
    def __init__(self, csv_file: pathlib.Path):
        # retrieve data
        df = pd.read_csv(csv_file, dtype=np.float32)

        x = torch.stack(
            [torch.tensor(df["radius"]), torch.tensor(df["inner_radius"]), torch.tensor(df["gap_width"])], dim=1
        ).reshape(-1, 1, 3)
        u = x
        y = torch.tensor(df["frequency"]).reshape(-1, 1, 1)
        v = torch.tensor(df["transmission_loss"]).unsqueeze(1).reshape(-1, 1, 1)

        # find appropriate transformations
        means = df.mean().to_dict()
        stds = df.std().to_dict()
        min_vals = df.min().to_dict()
        max_vals = df.max().to_dict()

        x_transforms = MinMaxScale(
            torch.tensor([min_vals["radius"], min_vals["inner_radius"], min_vals["gap_width"]]).reshape(1, 3),
            torch.tensor([max_vals["radius"], max_vals["inner_radius"], max_vals["gap_width"]]).reshape(1, 3),
        )
        u_transforms = x_transforms
        y_transforms = Normalize(
            torch.tensor(means["frequency"]).reshape(1, 1), torch.tensor(stds["frequency"]).reshape(1, 1)
        )
        v_transforms = Normalize(
            torch.tensor(means["transmission_loss"]).reshape(1, 1),
            torch.tensor(stds["transmission_loss"]).reshape(1, 1),
        )

        super().__init__(x, u, y, v, x_transforms, u_transforms, y_transforms, v_transforms)
