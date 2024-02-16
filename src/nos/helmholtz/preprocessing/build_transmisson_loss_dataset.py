import pathlib

import numpy as np
import pandas as pd

from nos.data.helmholtz import HelmholtzDataset
from nos.data.helmholtz.domain_properties import read_from_json
from nos.data.helmholtz.eval import transmission_loss


def build_transmission_loss_dataset(in_dir: pathlib.Path, out_dir: pathlib.Path):
    data_files = in_dir.rglob("*.xdmf")

    df = pd.DataFrame(columns=["frequency", "transmission_loss", "grid_size", "radius", "inner_radius", "gap_width"])

    for data_file in data_files:
        u_id = data_file.name.split("_")[0]
        description_file = data_file.parent.joinpath(f"{u_id}_description.json")
        description = read_from_json(description_file)
        data_set = HelmholtzDataset.from_xdmf_file(data_file)

        tl = transmission_loss(data_set, 1.0)

        set_size = data_set.frequencies.size

        description_df = pd.DataFrame(
            {
                "frequency": data_set.frequencies,
                "transmission_loss": tl,
                "grid_size": description.crystal.grid_size * np.ones(set_size),
                "radius": description.crystal.radius * np.ones(set_size),
                "inner_radius": description.crystal.inner_radius * np.ones(set_size),
                "gap_width": description.crystal.gap_width * np.ones(set_size),
            }
        )

        df = pd.concat([df, description_df], ignore_index=True)

    out_dir.mkdir(exist_ok=True)

    file = out_dir.joinpath(f"{in_dir.name}_transmission_loss_dataset.csv")
    df.to_csv(file, index=False)
