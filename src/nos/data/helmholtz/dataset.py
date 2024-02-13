import pathlib
import re
from dataclasses import dataclass, replace

import numpy as np

from nos.data.helmholtz.domain_properties import Description, read_from_json
from nos.data.utility import xdmf_to_numpy


@dataclass
class HelmholtzDataset:
    frequencies: np.array
    x: np.array
    p: np.array
    description: Description

    @staticmethod
    def from_xdmf_file(file_path: pathlib.Path):
        uid = file_path.name.split("_")[0]

        # load description
        json_file = file_path.parent.joinpath(f"{uid}_description.json")
        description = read_from_json(json_file)

        # load dataset
        data = xdmf_to_numpy(file_path)

        if data["Values"].ndim == 1:
            data["Values"] = data["Values"][np.newaxis, ...]

        return HelmholtzDataset(
            frequencies=data["Frequencies"],
            x=data["Geometry"],
            p=data["Values"],
            description=description,
        )

    @staticmethod
    def from_comsol_file(file_path: pathlib.Path, description: Description):
        data = np.genfromtxt(
            file_path,
            delimiter=",",
            skip_header=8,
            dtype=str,
        )
        complex_mapping = np.vectorize(lambda t: complex(t.replace("i", "j")))
        values = complex_mapping(data[1:, 2:]).T
        x = data[1:, :2].astype(float)

        # frequencies
        frequencies = data[0, 2:]
        # Searching the text for the pattern
        pattern = r"freq=(\d+\.?\d*)"
        frequencies = np.array([re.search(pattern, header).group(1) for header in frequencies], dtype=float)

        return HelmholtzDataset(
            frequencies=frequencies,
            x=x,
            p=values,
            description=replace(description),
        )
