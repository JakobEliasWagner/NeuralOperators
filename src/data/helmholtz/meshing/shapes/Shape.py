import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

import gmsh

from src.utility import run_once


class Shape(ABC):
    """Shape for gmsh meshing and cutting

    Attributes:

    """

    def __init__(self, *args, **kwargs):
        self.factory = gmsh.model.occ  # it is not encouraged to mix interfaces
        self.elements = defaultdict(lambda: set())  # holds elements associated with the top level shape (keys are dims)
        self.children = {}

        # generate shape
        gmsh.model.occ.synchronize()
        elements_before = gmsh.model.get_entities(-1)
        self.generate(*args, **kwargs)
        gmsh.model.occ.synchronize()
        elements_after = gmsh.model.get_entities(-1)
        new_elements = set(elements_after) - set(elements_before)

        # update elements
        for element in new_elements:
            self.elements[element[0]].add(element[1])

        # child elements need to be accessed via child
        for dim in self.elements:
            for child in self.children.values():
                self.elements[dim] -= child.elements[dim]

    @abstractmethod
    @run_once
    def generate(self, *args, **kwargs) -> None:
        """Initializes the entire shape, build all children and populates attributes of shape

        Args:
            *args:
            **kwargs:

        Returns:

        """

    def __call__(self, dim: int = None):
        """Get dimensions and tags identifying this shape

        Returns: List of tuples (dim, tag) pertaining to this shape (children excluded)

        """
        if not self.elements:
            # this shape holds no elements itself - collect recursively from children
            if not self.children:
                warnings.warn(f"Shape {self.__class__.__name__} holds not elements, but was called.")
                return []
            out = []
            for child in self.children.values():
                out += child()
            return out
        if not dim:
            # the highest dimensional element usually defines a shape
            dim = max(self.elements)
        return [(dim, element) for element in self.elements[dim]]

    def add_physical_group(self, dim: int, name: str) -> None:
        """Add physical group name to all elements of dim

        Args:
            dim: for elements of this dim
            name: physical group

        Returns:

        """
        gmsh.model.add_physical_group(dim, list(self.elements[dim]), name=name)
