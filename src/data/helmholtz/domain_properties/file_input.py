import configparser
import pathlib
from typing import List, Tuple

import numpy as np

from .description import CrystalDescription, CShapeDescription, CylinderDescription, Description

SAMPLING_STRATEGIES = {
    "linspace": np.linspace,
    "uniform": np.random.uniform,
}


def str_set_to_tuple(str_set: str) -> Tuple[float, float, int, str]:
    str_set = str_set.strip(" ()")
    str_set = str_set.split(",")
    return float(str_set[0]), float(str_set[1]), int(str_set[2]), str_set[3]


def sample(low: float, high: float, size: int, strategy_name: str) -> np.array:
    sampling_strategy = SAMPLING_STRATEGIES[strategy_name]
    return sampling_strategy(low, high, size)


def read_frequency(config: configparser.ConfigParser) -> np.array:
    freq_tuple = str_set_to_tuple(config["PHYSICS"]["samples"])
    samples = sample(*freq_tuple)
    if config["PHYSICS"]["sample_type"] == "frequency":
        return samples
    elif config["PHYSICS"]["sample_type"] == "wave-number":
        factor = float(config["PHYSICS"]["c"]) / (2.0 * np.pi)
        return factor / samples  # f = c / (2 pi k)
    else:
        raise TypeError(f'Unknown frequency encoding {config["PHYSICS"]["sample_type"]}!')


def read_crystal(config: configparser.ConfigParser) -> CrystalDescription:
    grid_size = float(config["CRYSTALS"][""])
    n_x = int(config["CRYSTALS"]["n_x"])
    n_y = int(config["CRYSTALS"]["n_y"])
    cut = bool(config["CRYSTALS"]["cut"])
    ref_index = float(config["CRYSTALS"]["ref_index"])

    return CrystalDescription("Crystal", grid_size, n_x, n_y, cut, ref_index)


def read_none_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    return []


def read_cylindrical_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    radii_tuple = str_set_to_tuple(config["CRYSTALS"]["CYLINDRICAL"]["radius"])
    radii = sample(*radii_tuple)

    c = read_crystal(config)

    return [CylinderDescription("Cylinder", c.grid_size, c.n_x, c.n_y, c.cut, c.ref_index, radius) for radius in radii]


def read_c_shaped_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    # generic crystal
    c = read_crystal(config)

    # c-shape
    outer_radii_tuple = str_set_to_tuple(config["CRYSTALS"]["C-SHAPED"]["outer_radius"])
    outer_radii = sample(*outer_radii_tuple)

    inner_radii_tuple = str_set_to_tuple(config["CRYSTALS"]["C-SHAPED"]["inner_radius"])
    inner_radii = sample(*inner_radii_tuple)

    gap_widths_tuple = str_set_to_tuple(config["CRYSTALS"]["C-SHAPED"]["gap_width"])
    gap_widths = sample(*gap_widths_tuple)

    # all permutations
    outer_radii, inner_radii, gap_widths = np.meshgrid(outer_radii, inner_radii, gap_widths)

    # prepare
    outer_radii = outer_radii.flatten()
    inner_radii = inner_radii.flatten()
    gap_widths = gap_widths.flatten()

    return [
        CShapeDescription("C-Shape", c.grid_size, c.n_x, c.n_y, c.cut, c.ref_index, outer, inner, gap)
        for outer, inner, gap in zip(outer_radii, inner_radii, gap_widths)
    ]


def read_config(file: pathlib.Path) -> List[Description]:
    config = configparser.ConfigParser()
    config.read(file)

    # crystal
    crystal_type = config["CRYSTALS"]["type"]
    if crystal_type == "None":
        read_func = read_none_crystal
    elif crystal_type == "Cylindrical":
        read_func = read_cylindrical_crystal
    elif crystal_type == "C-Shaped":
        read_func = read_c_shaped_crystal
    else:
        raise TypeError(f"Unknown crystal type {crystal_type}")
    crystal_descriptions = read_func(config)

    # domain
    frequencies = read_frequency(config)
    rho = float(config["PHYSICS"]["rho"])
    c = float(config["PHYSICS"]["c"])
    left_space = float(config["DOMAIN"]["left_space"])
    right_space = float(config["DOMAIN"]["right_space"])
    elements = float(config["DOMAIN"]["elements_per_wavelength"])

    # absorber
    absorber_depth = float(config["ABSORBER"]["depth"])
    round_trip = float(config["ABSORBER"]["round_trip"])
    directions = {
        "top": bool(config["ABSORBER"]["on_top"]),
        "left": bool(config["ABSORBER"]["on_left"]),
        "bottom": bool(config["ABSORBER"]["on_bottom"]),
        "right": bool(config["ABSORBER"]["on_right"]),
    }

    # indices
    domain_index_start = int(config["DOMAIN"]["cell_index_start"])
    absorber_index_start = int(config["ABSORBER"]["cell_index_start"])
    crystal_index_start = int(config["CRYSTALS"]["cell_index_start"])

    # assemble descriptions - currently only crystal descriptions need to be taken into account
    # as grid_size is constant the mesh generation will always generate the same mesh
    return [
        Description(
            frequencies=frequencies,
            rho=rho,
            c=c,
            left_space=left_space,
            right_space=right_space,
            depth=absorber_depth,
            round_trip=round_trip,
            directions=directions,
            crystal_description=description,
            domain_index_start=domain_index_start,
            absorber_index_start=absorber_index_start,
            crystal_index_start=crystal_index_start,
            elements=elements,
        )
        for description in crystal_descriptions
    ]
