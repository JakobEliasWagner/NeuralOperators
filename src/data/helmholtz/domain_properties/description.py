import dataclasses
from typing import Dict

import numpy as np


@dataclasses.dataclass
class CrystalDescription:
    """Holds information about a crystal."""

    type_name: str
    grid_size: float
    n_x: int
    n_y: int
    cut: bool
    ref_index: float


@dataclasses.dataclass
class Description:
    """Holds information of a domain on which only the frequency is varied.

    The reason for only varying the frequency in a single sample is that it may be easier to generate multiple samples
    on the same mesh. The frequency can but does not need to change the mesh of the domain. Compared to changing the
    parametrization of the crystals, this may save storage space.
    If the domain should be resampled for every simulation for every frequency, create one Description per frequency.

    """

    # physics
    frequencies: np.array
    rho: float
    c: float

    # domain
    left_space: float
    right_space: float
    elements: float
    domain_index_start: int  # for marking cell_indices

    # absorber
    depth: float
    round_trip: float
    directions: Dict[str, bool]
    absorber_index_start: int  # for marking cell_indices

    # crystals
    crystal_index_start: int  # for marking cell_indices
    crystal_description: CrystalDescription

    # derived properties
    height: float = dataclasses.field(init=False)  # height of the central stack
    width: float = dataclasses.field(init=False)  # width of the crystal domain
    absorber_depth: float = dataclasses.field(init=False)  # Description is used to create one mesh -> no array
    ks: np.array = dataclasses.field(init=False)
    wave_lengths: np.array = dataclasses.field(init=False)

    def __post_init__(self):
        self.height = self.crystal_description.n_y * self.crystal_description.grid_size
        self.width = self.crystal_description.n_x * self.crystal_description.grid_size
        self.wave_lengths = self.c / self.frequencies
        self.ks = 2 * np.pi * self.frequencies / self.c
        self.absorber_depth = max(self.wave_lengths) * self.absorber_depth


@dataclasses.dataclass
class CylinderDescription(CrystalDescription):
    """Holds information about a cylindrical crystal."""

    radius: float


@dataclasses.dataclass
class CShapeDescription(CrystalDescription):
    """Holds information about a C-shaped crystal."""

    outer_radius: float
    inner_radius: float
    gap_width: float


@dataclasses.dataclass
class NoneDescription(CrystalDescription):
    """A domain without crystals."""
