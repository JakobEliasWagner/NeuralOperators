import configparser
import json
import pathlib
import re
from random import shuffle
from typing import List, Tuple

import numpy as np

from . import CShapeDescription, CylinderDescription, Description, NoneDescription
from .absorber_description import AbsorberDescription, AdiabaticAbsorberDescription
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
    """Reads the frequency from a configparser.

    The frequency can be inputted as the frequency.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: array, containing all frequencies that should be sampled.
    """
    freq_tuple = str_set_to_tuple(config["PHYSICS"]["frequencies"])
    samples = sample(*freq_tuple)
    return samples


def read_crystal(config: configparser.ConfigParser) -> CrystalDescription:
    """Read generic crystal properties from config.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns:
        A description of a crystal.
    """
    grid_size = float(config["CRYSTAL"]["grid_size"])
    n = int(config["CRYSTAL"]["n"])

    return CrystalDescription(grid_size, n)


def read_none_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    """No crystal descriptions are required in an empty domain.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: empty list.
    """
    c = read_crystal(config)
    return [NoneDescription(grid_size=c.grid_size, n=c.n)]


def read_cylindrical_crystal(config: configparser.ConfigParser) -> List[CrystalDescription]:
    """Reads cylindrical crystal properties from config.

    Args:
        config: configparser instance containing the description of the domain (conforming to template).

    Returns: list, all possible crystal descriptions described by config.
    """
    radii_tuple = str_set_to_tuple(config["CRYSTAL"]["radius"])
    radii = sample(*radii_tuple)

    c = read_crystal(config)

    return [CylinderDescription(c.grid_size, c.n, radius) for radius in radii]


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
    outer_radii_tuple = str_set_to_tuple(config["CRYSTAL"]["radius"])
    outer_radii = sample(*outer_radii_tuple)

    inner_radii_tuple = str_set_to_tuple(config["CRYSTAL"]["inner_radius"])
    inner_radii = sample(*inner_radii_tuple)

    gap_widths_tuple = str_set_to_tuple(config["CRYSTAL"]["gap_width"])
    gap_widths = sample(*gap_widths_tuple)

    # all permutations
    outer_radii, inner_radii, gap_widths = np.meshgrid(outer_radii, inner_radii, gap_widths)

    # prepare
    outer_radii = outer_radii.flatten()
    inner_radii = inner_radii.flatten()
    gap_widths = gap_widths.flatten()

    inner_radii = outer_radii * inner_radii
    gap_widths = inner_radii * gap_widths

    return [
        CShapeDescription(c.grid_size, c.n, outer, inner, gap)
        for outer, inner, gap in zip(outer_radii, inner_radii, gap_widths)
    ]


def read_crystal_config(config: configparser.ConfigParser) -> List[CrystalDescription]:
    """Defines a list of different crystal configurations.

    Args:
        config:

    Returns:
        list containing all crystal configurations.
    """
    crystal_type = config["CRYSTAL"]["type"]
    if crystal_type == "None":
        read_func = read_none_crystal
    elif crystal_type == "Cylindrical":
        read_func = read_cylindrical_crystal
    elif crystal_type == "C-Shaped":
        read_func = read_c_shaped_crystal
    else:
        raise TypeError(f"Unknown crystal type {crystal_type}")
    return read_func(config)


def read_adiabatic_absorber_config(
    config: configparser.ConfigParser, lambda_max: float
) -> AdiabaticAbsorberDescription:
    """Defines adiabatic absorber description from config.

    Args:
        lambda_max:
        config: ConfigParser object

    Returns:
        adiabatic absorber description
    """

    n = float(config["ABSORBER"]["n_lambda_depth"])
    rt = float(config["ABSORBER"]["round_trip"])
    deg = int(config["ABSORBER"]["degree"])
    return AdiabaticAbsorberDescription(depth=n * lambda_max, round_trip=rt, degree=deg)


def read_absorber_config(config: configparser.ConfigParser, lambda_max: float) -> AbsorberDescription:
    """Defines absorber description from config.

    Args:
        lambda_max:
        config: ConfigParser object

    Returns:
        a description of an absorber.
    """
    absorber_type = config["ABSORBER"]["type"]
    if absorber_type == "Adiabatic Absorber":
        read_func = read_adiabatic_absorber_config
    else:
        raise TypeError(f"Unknown absorber {absorber_type}")
    return read_func(config, lambda_max)


def read_config(file: pathlib.Path) -> List[Description]:
    """Reads a file and returns all possible descriptions of the domain that need to be solved.

    Args:
        file: ini-file, containing relevant information (template "input_file_template.ini").

    Returns:
        list of all possible domain descriptions.
    """
    config = configparser.ConfigParser()
    config.read(file)

    crystal_descriptions = read_crystal_config(config)

    # domain
    rho = float(config["PHYSICS"]["rho"])
    c = float(config["PHYSICS"]["c"])
    max_delta_lambda = float(config["DOMAIN"]["max_delta_lambda"])
    n_left = float(config["DOMAIN"]["n_left"])
    n_right = float(config["DOMAIN"]["n_right"])
    elements = float(config["DOMAIN"]["elements_per_wavelength"])

    descriptions = []
    for crystal_description in crystal_descriptions:
        frequencies = read_frequency(config)
        # find frequency spectra
        frequencies.sort()  # they might be sampled randomly
        split_frequencies = []

        arr_start = 0
        start_f = frequencies[0]
        for window_stop, end_f in enumerate(frequencies):
            if 1 - start_f / end_f > max_delta_lambda:
                split_frequencies.append(frequencies[arr_start:window_stop])
                # update window
                arr_start = window_stop
                start_f = end_f
        split_frequencies.append(frequencies[arr_start:])

        for frequency_spectrum in split_frequencies:
            absorber_description = read_absorber_config(config, c / min(frequency_spectrum))
            descriptions.append(
                Description(
                    frequencies=frequency_spectrum,
                    rho=rho,
                    c=c,
                    n_left=n_left,
                    n_right=n_right,
                    elements_per_lambda=elements,
                    absorber=absorber_description,
                    crystal=crystal_description,
                )
            )
    shuffle(descriptions)
    return descriptions


def read_from_json(json_file: pathlib.Path):
    with open(json_file, "r") as fh:
        json_str = fh.read()
    json_obj = json.loads(json_str)

    absorber = AdiabaticAbsorberDescription(
        json_obj["absorber"]["depth"], json_obj["absorber"]["round_trip"], json_obj["absorber"]["degree"]
    )

    gs = json_obj["crystal"]["grid_size"]
    n = json_obj["crystal"]["n"]
    if "inner_radius" in json_obj["crystal"]:
        crystal = CShapeDescription(
            gs, n, json_obj["crystal"]["radius"], json_obj["crystal"]["inner_radius"], json_obj["crystal"]["gap_width"]
        )
    elif "radius" in json_obj["crystal"]:
        crystal = CylinderDescription(gs, n, json_obj["crystal"]["radius"])
    else:
        crystal = NoneDescription(gs, n)

    return Description(
        frequencies=np.array(json_obj["frequencies"]),
        rho=json_obj["rho"],
        c=json_obj["c"],
        n_left=json_obj["n_left"],
        n_right=json_obj["n_right"],
        elements_per_lambda=json_obj["elements_per_lambda"],
        absorber=absorber,
        crystal=crystal,
    )
