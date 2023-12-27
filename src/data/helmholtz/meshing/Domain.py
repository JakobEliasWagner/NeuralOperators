from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from .Property import AdiabaticAbsorberProperty, Property
from .util import Direction


class BoxDomain:
    """Maintains a subspace of the domain (which may be the entire domain)

    Top most domain will be placed at (0, 0) with its lower left corner. This class also manages its subdomains.
    The locations of the subdomains are placed in a relative context to this one.

    """

    def __init__(
        self,
        name: str = "BoxDomain",
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        properties: dict[str, Property] = None,
        sub_domains: Optional[List[BoxDomain]] = None,
        direction: Optional[Direction] = Direction.positive_x,
    ):
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
            self.sub_domains = [
                [
                    s,
                    [
                        0.0,
                    ]
                    * 2,
                ]
                for s in sub_domains
            ]  # (absolute bbox coordinates in domain, sub_domain)
        else:
            self.sub_domains = []
        self.direction = (direction + 1) % 2
        self.orthogonal_direction = direction % 2

        # set properties
        self.update_coordinates()

        # bounding box
        self.bbox = [0.0, 0.0]  # coordinate of upper right corner
        if self.sub_domains:
            self.bbox = self.sub_domains[-1][0].dimension.copy()  # orthogonal dimension
            self.bbox[self.direction] += self.sub_domains[-1][1][self.direction]  # sum of dx and position
        else:
            self.bbox = self.dimension

    def update_coordinates(self) -> None:
        """Sets coordinates within the contex of all domains.

        Domains are stacked in the following order: 1) own domain 2) sub-domains in direction in the order of the list
        in self.sub_domains.

        Returns:

        """
        coordinates = [
            0.0,
        ] * 2

        # the dimension of this object is the first.
        coordinates[self.direction] += self.dimension[self.direction]

        # shift children accordingly
        if not self.sub_domains:
            return
        # check data alignment
        dimensions = set([self.dimension[self.orthogonal_direction]])
        for domain, _ in self.sub_domains:
            dimensions.add(domain.dimension[self.orthogonal_direction])
        dimensions.remove(0.0)  # domain without own "body"
        if len(dimensions) > 1:
            raise ValueError("Only rectangular domains allowed!")

        for i, (domain, _) in enumerate(self.sub_domains):
            # modify sub_domain information to match relative position from this domain
            self.sub_domains[i][1] = coordinates.copy()  # points at the lower left corner
            coordinates[self.direction] += domain.bbox[self.direction]

    def eval(self, property_name: str, x) -> np.array:
        """

        Args:
            property_name: physical property definition (as a domain can hold multiple different ones)
            x: tuple, position

        Returns:

        """
        output_value = np.zeros(x[0].shape, dtype=np.complex128)
        for sub_domain, pos in self.sub_domains:
            relative_x = [x[0] - pos[0], x[1] - pos[1]]
            # only values inside
            output_value += sub_domain.eval(property_name, relative_x)
        if self.properties[property_name]:
            inside = (0.0 <= x[0]) * (x[0] < self.bbox[0]) * (0.0 <= x[1]) * (x[1] < self.bbox[1])
            prop = self.properties[property_name]
            output_value += prop.eval(x) * inside
        return output_value


class AdiabaticLayer(BoxDomain):
    """Layer of an Adiabatic Absorber

    Adiabatic Absrobers are an extension to perfectly matched layers. They were introduced as PMLs sometimes fail to be
    reflectionless even in the limit of infinite resolution. These absorbers make the reflections negligible by
    gradually increasing the material absorption.
    Details can be found in Oskooi, A. F., Zhang, L., Avniel, Y. & Johnson, S. G. The failure of perfectly matched
    layers, and towards their redemption by adiabatic absorbers. Opt. Express 16, 11376–11392 (2008).
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        round_trip: float,
        direction_depth: dict[int, List[float, float]],
        properties: List[Tuple[str, float]],
    ):
        """

        Args:
            dx: extend in x direction
            dy: extend in y direction
            round_trip: float, governs the size of reflections caused by this layer
            direction_depth: dict, key: direction identifier, value: Starting and end point (key needs to be positive)
            properties: List of Tuples, defines which quantities/Properties should be modified by this domain
        """
        props = {prop[0]: AdiabaticAbsorberProperty(2, round_trip, prop[1], direction_depth) for prop in properties}
        super().__init__("Adiabaitc Absorber", dx, dy, props)
