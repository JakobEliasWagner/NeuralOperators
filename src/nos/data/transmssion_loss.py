import pathlib

import numpy as np
import pandas as pd
import torch

from continuity.data import OperatorDataset
from continuity.transforms import Normalize


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

        x_transforms = Normalize(
            torch.tensor([means["radius"], means["inner_radius"], means["gap_width"]]).reshape(1, 1, 3),
            torch.tensor([stds["radius"], stds["inner_radius"], stds["gap_width"]]).reshape(1, 1, 3),
        )
        u_transforms = x_transforms
        y_transforms = Normalize(torch.tensor(means["frequency"]), torch.tensor(stds["frequency"]))
        v_transforms = Normalize(
            torch.tensor(means["transmission_loss"]), torch.tensor(stds["transmission_loss"])
        )

        super().__init__(x, u, y, v, x_transforms, u_transforms, y_transforms, v_transforms)


class TLDatasetCompact(OperatorDataset):
    """Transmission loss dataset, with bigger evaluation space."""

    def __init__(self, csv_file: pathlib.Path):
        df = pd.read_csv(csv_file, dtype=np.float32)

        unique_crystals = df[["radius", "inner_radius", "gap_width"]].drop_duplicates()

        num_evals = len(df) // len(unique_crystals)

        x = torch.empty((len(unique_crystals), 1, 3))
        u = x
        y = torch.empty((len(unique_crystals), num_evals, 1))
        v = torch.empty((len(unique_crystals), num_evals, 1))

        for i, (_, crystal) in enumerate(unique_crystals.iterrows()):
            c_df = df.loc[
                (df["radius"] == crystal["radius"])
                * (df["inner_radius"] == crystal["inner_radius"])
                * (df["gap_width"] == crystal["gap_width"])
            ]

            x[i] = torch.tensor([crystal["radius"], crystal["inner_radius"], crystal["gap_width"]]).reshape(1, 3)
            y[i] = torch.tensor([c_df["frequency"].to_list()]).reshape(num_evals, 1)
            v[i] = torch.tensor([[c_df["transmission_loss"].to_list()]]).reshape(num_evals, 1)

        # find appropriate transformations
        means = df.mean().to_dict()
        stds = df.std().to_dict()

        x_transforms = Normalize(
            torch.tensor([means["radius"], means["inner_radius"], means["gap_width"]]).reshape(1, 1, 3),
            torch.tensor([stds["radius"], stds["inner_radius"], stds["gap_width"]]).reshape(1, 1, 3),
        )
        u_transforms = x_transforms
        y_transforms = Normalize(torch.tensor(means["frequency"]), torch.tensor(stds["frequency"]))
        v_transforms = Normalize(
            torch.tensor(means["transmission_loss"]), torch.tensor(stds["transmission_loss"])
        )

        super().__init__(x, u, y, v, x_transforms, u_transforms, y_transforms, v_transforms)
