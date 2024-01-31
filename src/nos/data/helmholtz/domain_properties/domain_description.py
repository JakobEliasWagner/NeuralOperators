import dataclasses
import json
import pathlib
from typing import Dict

import numpy as np

from nos.utility import BoundingBox2D, UniqueId

from .absorber_description import AbsorberDescription
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
    n_left: float  # factor of grid_size
    n_right: float  # factor of grid_size
    elements_per_lambda: float

    # absorber
    absorber: AbsorberDescription

    # crystals
    crystal: CrystalDescription

    # indices
    indices: dict = dataclasses.field(
        default_factory=lambda: {
            "left_side": 10000,
            "crystal_domain": 10010,
            "right_side": 10020,
            "absorber": 20000,
        }
    )

    # derived properties
    height: float = dataclasses.field(init=False)
    ks: np.array = dataclasses.field(init=False)
    wave_lengths: np.array = dataclasses.field(init=False)
    unique_id: UniqueId = dataclasses.field(init=False)

    domain_box: BoundingBox2D = dataclasses.field(init=False)
    left_box: BoundingBox2D = dataclasses.field(init=False)
    crystal_box: BoundingBox2D = dataclasses.field(init=False)
    right_box: BoundingBox2D = dataclasses.field(init=False)
    absorber_boxes: Dict[str, BoundingBox2D] = dataclasses.field(init=False)

    def __post_init__(self):
        self.update_derived_properties()

    def update_derived_properties(self):
        self.wave_lengths = self.c / self.frequencies
        self.ks = 2 * np.pi * self.frequencies / self.c

        self.unique_id = UniqueId()

        # boxes
        self.height = self.crystal.grid_size
        offset = self.crystal.grid_size / 2.0
        # left box
        l_width = self.n_left * self.crystal.grid_size
        self.left_box = BoundingBox2D(-l_width - offset, 0 - offset, 0 - offset, self.height - offset)
        # domain
        d_width = self.crystal.n * self.crystal.grid_size
        self.crystal_box = BoundingBox2D(0 - offset, 0 - offset, d_width - offset, self.height - offset)
        # right box
        r_width = self.n_right * self.crystal.grid_size
        self.right_box = BoundingBox2D(d_width - offset, 0 - offset, d_width + r_width - offset, self.height - offset)
        # absorbers
        self.absorber_boxes = {
            "left": BoundingBox2D(
                -l_width - self.absorber.depth - offset, 0 - offset, -l_width - offset, self.height - offset
            ),
            "right": BoundingBox2D(
                d_width + r_width - offset,
                0 - offset,
                d_width + r_width + self.absorber.depth - offset,
                self.height - offset,
            ),
        }

        # overall bbox
        self.domain_box = BoundingBox2D(
            self.left_box.x_min, self.left_box.y_min, self.right_box.x_max, self.right_box.y_max
        )

    def serialize(self) -> dict:
        """Serializes this object to a dictionary.

        Returns:
            Dictionary containing the information of this class and its properties.
        """
        des_dict = dataclasses.asdict(self)
        des_dict["unique_id"] = str(self.unique_id)

        def transform_np_array(o):
            if isinstance(o, np.ndarray):
                return o.tolist()

            if isinstance(o, dict):
                for key, value in o.items():
                    o[key] = transform_np_array(value)

            return o

        des_dict = transform_np_array(des_dict)

        return des_dict

    def save_to_json(self, out_dir: pathlib.Path) -> None:
        """Saves this object to the provided dir.

        Args:
            out_dir: Path to the directory the json file is saved.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir.joinpath(f"{self.unique_id}_description.json")
        with open(file_path, "w") as file_handle:
            json.dump(self.serialize(), file_handle)
