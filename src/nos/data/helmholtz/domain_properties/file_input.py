import configparser
import pathlib
import re
from typing import List, Tuple

import numpy as np

from . import CShapeDescription, CylinderDescription, Description, NoneDescription
from .crystal_description import CrystalDescription

# available sampling strategies
SAMPLING_STRATEGIES = {
    "linspace": np.linspace,
    "uniform": np.random.uniform,
}


def str_set_to_tuple(str_set: str) -> Tuple[float, float, int, str]:
    """Converts str tuple to a tuple which can be used to sample a 1d domain.

    Args:
        str_set: tuple in str format with four elements

    Returns: Index 0 is the starting point, 1 is the end point, 2 is the number of
        samples, and 3 is the str name of the sampling strategy.

    """
    pattern = f'[{re.escape(" ()")}]'
    str_set = re.sub(pattern, "", str_set)
    str_set = str_set.split(",")
    return float(str_set[0]), float(str_set[1]), int(str_set[2]), str_set[3]


def sample(low: float, high: float, size: int, strategy_name: str) -> np.array:
    """Samples a 1d domain according to a specific sampling strategy.

    Args:
        low: lower limit
        high: upper limit
        size: number of samples within the interval
        strategy_name: str, name of the sampling strategy

    Returns: sampled 1d domain according to strategy.
    """
    sampling_strategy = SAMPLING_STRATEGIES[strategy_name]
    return sampling_strategy(low, high, size)


def read_frequency(config: configparser.ConfigParser) -> np.array:
    """Reads the frequency from a configparser

    The frequency can be inputted either as the frequency directly or using the wave number. Inputting it as the
    wavenumber requires additional processing.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: array, containing all frequencies that should be sampled.
    """
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
    """Read generic crystal properties from config.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: A description of a crystal.

    """
    grid_size = float(config["CRYSTAL"]["grid_size"])
    n_x = int(config["CRYSTAL"]["n_x"])
    n_y = int(config["CRYSTAL"]["n_y"])

    return CrystalDescription("Crystal", grid_size, n_x, n_y)


def read_none_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    """No crystal descriptions are required in an empty domain.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: empty list.
    """
    c = read_crystal(config)
    return [NoneDescription("None", c.grid_size, c.n_x, c.n_y)]


def read_cylindrical_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    """Reads cylindrical crystal properties from config.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: list, all possible crystal descriptions described by config.
    """
    radii_tuple = str_set_to_tuple(config["CRYSTAL-CYLINDRICAL"]["radius"])
    radii = sample(*radii_tuple)

    c = read_crystal(config)

    return [CylinderDescription("Cylinder", c.grid_size, c.n_x, c.n_y, radius) for radius in radii]


def read_c_shaped_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    """Reads C-shaped crystal properties from config.

    The C-shaped crystal is described by three individual properties. Each description is taken and arranged in a
    meshgrid to sample every possible permutation of parameters.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: list, all possible crystal descriptions described by config.
    """
    # generic crystal
    c = read_crystal(config)

    # c-shape
    outer_radii_tuple = str_set_to_tuple(config["CRYSTAL-C-SHAPED"]["outer_radius"])
    outer_radii = sample(*outer_radii_tuple)

    inner_radii_tuple = str_set_to_tuple(config["CRYSTAL-C-SHAPED"]["inner_radius"])
    inner_radii = sample(*inner_radii_tuple)

    gap_widths_tuple = str_set_to_tuple(config["CRYSTAL-C-SHAPED"]["gap_width"])
    gap_widths = sample(*gap_widths_tuple)

    # all permutations
    outer_radii, inner_radii, gap_widths = np.meshgrid(outer_radii, inner_radii, gap_widths)

    # prepare
    outer_radii = outer_radii.flatten()
    inner_radii = inner_radii.flatten()
    gap_widths = gap_widths.flatten()

    return [
        CShapeDescription("C-Shape", c.grid_size, c.n_x, c.n_y, outer, inner, gap)
        for outer, inner, gap in zip(outer_radii, inner_radii, gap_widths)
    ]


def read_config(file: pathlib.Path) -> List[Description]:
    """Reads a file and returns all possible descriptions of the domain that need to be solved.

    Args:
        file: ini-file, containing relevant information (template "input_file_template.ini").

    Returns: list of all possible domain descriptions.
    """
    config = configparser.ConfigParser()
    config.read(file)

    # crystal
    crystal_type = config["CRYSTAL"]["type"]
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
        "top": config["ABSORBER"].getboolean("on_top"),
        "left": False,  # excitation is applied over the boundary
        "bottom": config["ABSORBER"].getboolean("on_bottom"),
        "right": config["ABSORBER"].getboolean("on_right"),
    }

    # indices
    indices = {
        "excitation": int(config["INDICES"]["excitation"]),
        "left_side": int(config["INDICES"]["left_side"]),
        "crystal_domain": int(config["INDICES"]["crystal_domain"]),
        "crystals": int(config["INDICES"]["crystals"]),
        "right_side": int(config["INDICES"]["right_side"]),
        "absorber": int(config["INDICES"]["absorber"]),
    }

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
            indices=indices,
            elements=elements,
        )
        for description in crystal_descriptions
    ]
