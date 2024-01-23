from typing import List

from nos.data.helmholtz.domain_properties import Description

from .crystal_builder import CrystalBuilder, CShapedCrystalBuilder, CylindricalCrystalBuilder
from .gmsh_builder import GmshBuilder


class CrystalDomainBuilder(GmshBuilder):
    """Class for building simulation domains in which crystals are placed.

    The CrystalDomainBuilder class provides functionalities to create and manipulate
    the simulation domain in which crystals are placed for computational simulations.
    """

    def __init__(self, description: Description, crystal_builder: CrystalBuilder = None):
        """

        Args:
            description: domain description.
            crystal_builder: builder for actual crystals.
        """
        super().__init__(description)
        self.crystal_description = description.crystal
        if crystal_builder is None:
            crystal_builder = CrystalBuilder(description)
        self.crystal_builder = crystal_builder

    def build(self) -> List[int]:
        """Builds the domain according to description and crystal builder.

        Returns:
            indices to all elements of the newly created simulation domain.
        """
        rect = self.define_basic_shape()
        crystals = self.define_tools()
        if len(crystals) > 0:
            domain = self.cut(rect, crystals)
        else:
            domain = [rect]
        return domain

    def define_basic_shape(self) -> int:
        """Defines the shape of the encompassing domain.

        Returns:
            gmsh index to the rectangle.
        """
        domain = self.factory.addRectangle(
            self.description.left_width, 0.0, 0.0, self.description.domain_width, self.description.height
        )
        return domain

    def define_tools(self) -> List[int]:
        """Defines the tools which are used to cut the domain.

        Returns:
            indices to all elements that need to be cut from the domain.
        """
        return self.crystal_builder.build()

    def cut(self, obj: int, tools: List[int]) -> List[int]:
        """

        Args:
            obj: encompassing domain.
            tools: crystals which are cut from the encompassing domain.

        Returns:
            indices to all elements of the simulation domain.
        """
        tags, _ = self.factory.cut([(2, obj)], [(2, tool) for tool in tools])
        domain = [tag[1] for tag in tags]
        return domain


class CShapedCrystalDomainBuilder(CrystalDomainBuilder):
    """Class for building simulation domains in which C-shaped crystals are placed."""

    def __init__(self, description: Description):
        super().__init__(description, CShapedCrystalBuilder(description))


class CylindricalCrystalDomainBuilder(CrystalDomainBuilder):
    """Class for building simulation domains in which cylindrical crystals are placed."""

    def __init__(self, description: Description):
        super().__init__(description, CylindricalCrystalBuilder(description))
