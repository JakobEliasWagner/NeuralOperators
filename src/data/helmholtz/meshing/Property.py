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
