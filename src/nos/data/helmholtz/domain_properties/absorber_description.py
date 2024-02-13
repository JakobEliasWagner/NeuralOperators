import dataclasses
from abc import ABC


@dataclasses.dataclass
class AbsorberDescription(ABC):
    """Holds information about absorbers.

    Absorbers are boundaries that truncate the domain in a finite space bigger than zero. Absorbers are placed commonly
    on the left and right hand side of the simulation domain.
    """

    lambda_depth: float  # depth expressed as multiples of a wavelength


@dataclasses.dataclass
class AdiabaticAbsorberDescription(AbsorberDescription):
    """Holds information about adiabatic absorbers."""

    round_trip: float = 1e-6  # factor to balance transition reflection
    degree: int = 2  # degree of the shape function governing
