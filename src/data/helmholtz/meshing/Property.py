from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import numpy as np


class Property(ABC):
    """Defines a property on a domain

    Properties modify physical quantities of domains. These may be structures of physical parameters.

    """

    @abstractmethod
    def eval(self, x) -> np.array:
        """

        Args:
            x: tuple, position

        Returns: value by which property should be modified at location x

        """
        pass


class ConstantProperty(Property):
    """Constant.

    Returns always the same constant value.

    """

    def __init__(self, const: float):
        """

        Args:
            const: value of this property
        """
        self.value = const

    def eval(self, x) -> np.array:
        """

        Args:
            x: location vector

        Returns: the value of this property at the position x

        """
        return self.value * np.ones(x[0].shape)


class CylindricalCrystalProperty(Property):
    """Equally spaced crystals within a given rectangle

    Equally spaced cylindrical crystals within a rectangular domain.
    They are equally spaced to each other and to the boundaries of the rectangle.
    The cylinders manifest in a modification (by addition) of a property defined on the domain.
    Overlapping crystals are treated as a union of both (i.e., they do not amplify that region further).
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        n_x: int,
        n_y: int,
        radius: float,
        value: float,
    ):
        """

        Args:
            dx: size of rectangle in x direction
            dy: size of rectangle in y direction
            n_x: number of crystals in x direction
            n_y: number of crystals in y direction
            radius: radius of each crystal
            value: modification (by addition) of a given property within domain
        """
        stride = (dx / n_x, dy / n_y)  # start in rect (if it fits)
        self.centers = []
        for row in range(n_x):
            for col in range(n_y):
                self.centers.append((stride[0] * (row + 0.5), stride[1] * (col + 0.5)))
        self.squared_radius = radius**2
        self.value = value

    def eval(self, x) -> np.array:
        """

        Args:
            x: tuple/vector, position

        Returns: Value of the modified property at location x

        """
        in_circle = np.max(
            np.array([(x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2 <= self.squared_radius for c in self.centers]),
            axis=0,
        )
        return in_circle * self.value


class AdiabaticAbsorberProperty(Property):
    """Modifies wavenumber in property domain to truncate wave

    Adiabatic Absrobers are an extension to perfectly matched layers. They were introduced as PMLs sometimes fail to be
    reflectionless even in the limit of infinite resolution. These absorbers make the reflections negligible by
    gradually increasing the material absorption.
    Details can be found in Oskooi, A. F., Zhang, L., Avniel, Y. & Johnson, S. G. The failure of perfectly matched
    layers, and towards their redemption by adiabatic absorbers. Opt. Express 16, 11376–11392 (2008).
    """

    def __init__(
        self,
        degree: int,
        round_trip: float,
        value: float,
        direction_depth: dict[int, List[float]],
    ):
        self.direction_properties = defaultdict(lambda: [-1.0, -1.0])
        self.direction_properties.update(direction_depth)  # key: axis, value: [start, end]
        self.degree = degree
        self.round_trip = round_trip
        self.sigma_0 = {
            axis: -(self.degree + 1) * np.log(self.round_trip) / (2.0 * (pos[1] - pos[0])) * (-1) ** (pos[1] > pos[0])
            for axis, pos in self.direction_properties.items()
        }  # sign needs to be corrected
        self.value = value

    def eval(self, x) -> np.array:
        """

        Args:
            x: tuple/vector, location on which the property is evaluated

        Returns: Modified property value on location x

        """
        xi = {axis: (x[axis] - pos[0]) / (pos[1] - pos[0]) for axis, pos in self.direction_properties.items()}
        # bound xi as values above 1 and below 0 indicate that xi is not in absorption range
        for axis, value in xi.items():
            inside = (value > 0) * (value < 1)
            xi[axis] = value * inside
        sigma_x = []
        for axis, zeta in xi.items():
            sigma_x.append(self.sigma_0[axis] * (zeta**self.degree))
        sigma_x = np.sum(np.array(sigma_x), axis=0)
        return 2j * sigma_x * self.value - sigma_x**2
