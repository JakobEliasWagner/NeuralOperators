from abc import ABC, abstractmethod

import numpy as np

from src.data.helmholtz.domain_properties import Description


class WaveNumberModifier(ABC):
    @abstractmethod
    def eval(self, x):
        pass


class AdiabaticLayer(WaveNumberModifier):
    """Adiabatic layer to truncate a domain without reflections.

    Shown in Oskooi et al. "The failure of perfectly matched layers, and towards their redemption by adiabatic
    absorbers" Adiabatic layers asymptotically approach an adiabatic limit of zero reflections. This is done by
    gradually increasing absorption, which results in a truncated domain with only trivial reflections.
    This implementation assumes that the simulation domain without truncation is a rectangle.
    """

    def __init__(self, pd: Description, degree: int = 2):
        self.box_min = np.array([0.0, 0.0])
        self.box_max = np.array([pd.width + pd.right_width, pd.height])
        self.depth = pd.absorber_depth
        self.degree = degree
        rt = pd.round_trip
        self.sigma_0 = -(self.degree + 1) * np.log(rt) / (2.0 * self.depth)

    def eval(self, x: np.array) -> np.array:
        """Returns modification to the wave number caused by the adiabatic layer.

        Inside an absorbing layer, the wave number is modified with k = k0 + 1j * sigma, where sigma is a scaled shape
        function.

        Args:
            x:

        Returns:

        """
        # Clamp point coordinates to the range defined by the box
        clamped = np.maximum(self.box_min, np.minimum(x, self.box_max))

        # Compute the distance from the clamped point to the original point
        dist = np.linalg.norm(clamped - x)

        # Relative distance inside absorber
        dist = dist / self.depth

        return self.sigma_0 * np.sum(dist**self.degree)
