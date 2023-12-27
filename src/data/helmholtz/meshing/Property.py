from abc import ABC, abstractmethod
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

    def __init__(self, dx: float, dy: float,
                 n_x: int, n_y: int, radius: float, value: float):
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
        self.squared_radius = radius ** 2
        self.value = value

    def eval(self, x) -> np.array:
        """

        Args:
            x: tuple/vector, position

        Returns: Value of the modified property at location x

        """
        in_circle = np.max(np.array([
            (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2 <= self.squared_radius
            for c in self.centers
        ]), axis=0)
        return in_circle * self.value
