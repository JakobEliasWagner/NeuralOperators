import dataclasses
from typing import Dict

import numpy as np

from nos.utility import get_unique_id

from .crystal_description import CrystalDescription


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
    right_space: float  # in value of wavelengths
    elements: float
    # for marking cell_indices
    domain_index: int
    right_index: int
    excitation_index: int

    # absorber
    depth: float
    round_trip: float
    directions: Dict[str, bool]
    absorber_index: int  # for marking cell_indices

    # crystals
    crystal_index: int  # for marking cell_indices
    crystal_description: CrystalDescription

    # derived properties
    height: float = dataclasses.field(init=False)  # height of the central stack
    width: float = dataclasses.field(init=False)  # width of the crystal domain
    absorber_depth: float = dataclasses.field(init=False)  # Description is used to create one mesh -> no array
    ks: np.array = dataclasses.field(init=False)
    wave_lengths: np.array = dataclasses.field(init=False)
    unique_id: str = dataclasses.field(init=False)
    right_width: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.update_derived_properties()

    def update_derived_properties(self):
        self.height = self.crystal_description.n_y * self.crystal_description.grid_size
        self.width = self.crystal_description.n_x * self.crystal_description.grid_size
        self.wave_lengths = self.c / self.frequencies
        self.ks = 2 * np.pi * self.frequencies / self.c
        self.absorber_depth = max(self.wave_lengths) * self.depth
        self.right_width = max(self.right_space, 0) * max(self.wave_lengths)

        self.unique_id = get_unique_id()
