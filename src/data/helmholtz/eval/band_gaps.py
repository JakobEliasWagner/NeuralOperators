import json
import pathlib
from typing import List, Tuple

import numpy as np

from src.data.utility import BoundingBox2D, xdmf_to_numpy


def get_dataset_information(xdmf_file: pathlib.Path) -> dict:
    json_path = xdmf_file.parent.joinpath(xdmf_file.name.split("_")[0] + "_description.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def get_sound_pressure_level(x, values, bbox: BoundingBox2D, p0: float = 1) -> np.ndarray:
    in_eval_area = bbox.inside(x)
    db_levels = 20 * np.log10(np.max(np.abs(values[:, in_eval_area]), axis=1) / p0)
    return db_levels


def get_band_gaps(data_dir: pathlib) -> Tuple[List[dict], List[np.ndarray], List[np.ndarray]]:
    data_sets = data_dir.glob("*.xdmf")
    band_gaps = []  # might have differing frequency sample sizes and shape
    frequencies = []
    descriptions = []
    for data_set in data_sets:
        description = get_dataset_information(data_set)
        descriptions.append(description)
        data = xdmf_to_numpy(data_set)
        frequencies.append(data["Frequencies"])

        # get band gaps
        r_bbox = BoundingBox2D(
            description["width"], 0.0, description["width"] + description["right_width"], description["height"]
        )
        band_gaps.append(get_sound_pressure_level(data["Geometry"], data["Values"], r_bbox))

    return descriptions, frequencies, band_gaps


if __name__ == "__main__":
    data_dir = pathlib.Path.cwd().joinpath("out", "20240113173310-5b2df42d-cd1c-48ed-a24c-125d885108a4")

    get_band_gaps(data_dir)
