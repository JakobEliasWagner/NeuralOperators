import dataclasses
from abc import ABC


@dataclasses.dataclass
class CrystalDescription(ABC):
    """Holds information about a crystal."""

    type_name: str
    grid_size: float
    n_x: int
    n_y: int


@dataclasses.dataclass
class CylinderDescription(CrystalDescription):
    """Holds information about a cylindrical crystal."""

    radius: float


@dataclasses.dataclass
class CShapeDescription(CrystalDescription):
    """Holds information about a C-shaped crystal."""

    radius: float
    inner_radius: float
    gap_width: float


@dataclasses.dataclass
class NoneDescription(CrystalDescription):
    """A domain without crystals."""
