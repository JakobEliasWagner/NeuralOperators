from __future__ import annotations
import numpy as np


class BoxDomain:
    """Maintains a subspace of the domain (which may be the entire domain)

    Top most domain will be placed at (0, 0) with its lower left corner. This class also manages its subdomains.
    The locations of the subdomains are placed in a relative context to this one.

    """

    def __init__(self):
        """

        Args:

        """
        pass

    def update_coordinates(self) -> None:
        """Sets coordinates within the contex of all domains for relative positioning to this one

        Returns:

        """
        pass

    def eval(self, property_name: str, x) -> np.array:
        """

        Args:
            property_name: physical property definition (as a domain can hold multiple different ones)
            x: tuple, position

        Returns:

        """
        pass
