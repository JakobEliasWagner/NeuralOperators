from __future__ import annotations
import numpy as np
from typing import Optional, List
from collections import defaultdict

from .util import Direction
from .Property import Property


class BoxDomain:
    """Maintains a subspace of the domain (which may be the entire domain)

    Top most domain will be placed at (0, 0) with its lower left corner. This class also manages its subdomains.
    The locations of the subdomains are placed in a relative context to this one.

    """

    def __init__(self, name: str = "BoxDomain",
                 dx: Optional[float] = None,
                 dy: Optional[float] = None,
                 properties: dict[str, Property] = None,
                 sub_domains: Optional[List[BoxDomain]] = None,
                 direction: Optional[Direction] = Direction.positive_x):
        """

        Args:
            name: to identify domain
            dx: extend in positive x direction
            dy: extend in positive y direction
            properties: dict, name of property with a property object, Properties defined on this domain
            sub_domains: List, of domains encompassed by this domain
            direction: Direction, domains are stacked as they are listed in sub_domains in this direction
        """
        if not any([properties, all([dx, dy])]):
            # if neither is provided, this would be an empty object
            raise ValueError("Domain is not well-defined.")
        self.name = name
        self.dimension = [dx, dy]
        if not all(self.dimension):
            self.dimension = [0.0, 0.0]
        if properties is None:
            properties = defaultdict(lambda: None)
        self.properties = properties  # holds information of physical parameters within the domain
        if sub_domains:
            self.sub_domains = [[s, [0., ] * 2] for s in
                                sub_domains]  # (absolute bbox coordinates in domain, sub_domain)
        else:
            self.sub_domains = []
        self.direction = (direction + 1) % 2
        self.orthogonal_direction = direction % 2

        # set properties
        self.update_coordinates()

        # bounding box
        self.bbox = [0., 0.]  # coordinate of upper right corner
        if self.sub_domains:
            self.bbox = self.sub_domains[-1][0].dimension.copy()  # orthogonal dimension
            self.bbox[self.direction] += self.sub_domains[-1][1][self.direction]  # sum of dx and position
        else:
            self.bbox = self.dimension

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
