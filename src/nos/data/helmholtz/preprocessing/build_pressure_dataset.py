import pathlib

import numpy as np
import pandas as pd

from nos.data.helmholtz import HelmholtzDataset
from nos.data.helmholtz.domain_properties import read_from_json


def build_pressure_dataset(in_dir: pathlib.Path, out_dir: pathlib.Path):
    data_files = in_dir.rglob("*.xdmf")

    df = pd.DataFrame(
        columns=["x", "y", "p_real", "p_imag", "frequency", "grid_size", "radius", "inner_radius", "gap_width"]
    )

    for data_file in data_files:
        u_id = data_file.name.split("_")[0]
        description_file = data_file.parent.joinpath(f"{u_id}_description.json")
        description = read_from_json(description_file)
        data_set = HelmholtzDataset.from_xdmf_file(data_file)

        relevant_idx = np.where(description.crystal_box.distance(data_set.x.T) < 1e-6)
        relevant_x = data_set.x[relevant_idx, :].squeeze()

        relevant_p = data_set.p[:, relevant_idx].squeeze()

        set_size = data_set.frequencies.size * relevant_x.shape[0]

        description_df = pd.DataFrame(
            {
                "x": np.outer(np.ones(data_set.frequencies.size), relevant_x[:, 0]).flatten(order="C"),
                "y": np.outer(np.ones(data_set.frequencies.size), relevant_x[:, 1]).flatten(order="C"),
                "p_real": np.real(relevant_p).flatten(order="C"),
                "p_imag": np.imag(relevant_p).flatten(order="C"),
                "frequency": np.outer(data_set.frequencies, np.ones(relevant_x.shape[0])).flatten(order="C"),
                "grid_size": description.crystal.grid_size * np.ones(set_size),
                "radius": description.crystal.radius * np.ones(set_size),
                "inner_radius": description.crystal.inner_radius * np.ones(set_size),
                "gap_width": description.crystal.gap_width * np.ones(set_size),
            }
        )

        df = pd.concat([df, description_df], ignore_index=True)

    out_dir.mkdir(exist_ok=True)

    file = out_dir.joinpath(f"{in_dir.name}_pressure_dataset.csv")
    df.to_csv(file, index=False)
