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
