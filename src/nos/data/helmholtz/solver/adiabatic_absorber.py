import numpy as np

from nos.data.helmholtz.domain_properties import Description
from nos.utility import BoundingBox2D

from .wave_number_function import WaveNumberFunction


class AdiabaticAbsorber(WaveNumberFunction):
    """Adiabatic layer class to truncate a domain with trivial reflections.

    Shown in Oskooi et al. "The failure of perfectly matched layers, and towards their redemption by adiabatic
    absorbers" Adiabatic layers asymptotically approach an adiabatic limit of zero reflections. This is done by
    gradually increasing absorption, which results in a truncated domain with only trivial reflections.
    This implementation assumes that the simulation domain without truncation is a rectangle.
    """

    def __init__(self, description: Description):
        self.bbox = BoundingBox2D(
            0.0, 0.0, description.left_width + description.domain_width + description.right_width, description.height
        )
        self.depth = description.absorber.lambda_depth
        self.round_trip = description.absorber.round_trip
        self.degree = description.absorber.degree
        self.sigma_0 = -(self.degree + 1) * np.log(self.round_trip) / (2.0 * self.depth)

    def eval(self, x: np.array) -> np.array:
        """Returns modification factor to the wave number caused by the adiabatic layer.

        Args:
            x: location vector

        Returns:
            zero in areas where the adiabatic is not active, and increasingly complex values inside absorber.
        """
        abs_distance = self.bbox.distance(x)
        rel_distance = abs_distance / self.depth

        return self.sigma_0 * 1j * rel_distance**self.degree
