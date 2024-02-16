from typing import List

from nos.data.helmholtz.domain_properties import Description

from .gmsh_builder import GmshBuilder


class CrystalBuilder(GmshBuilder):
    """Base class for crystal builders.

    Crystal buidlers are used to define all shapes of the crystals placed within a simulation domain. This base class
    does not place any crystals in the domain.
    """

    def __init__(self, description: Description):
        super().__init__(description)
        self.crystal_description = description.crystal
        self.box = self.description.crystal_box

    def build(self) -> List[int]:
        """Builds the crystals according to the domain description.

        Returns:
            indices to all newly created crystals.
        """
        objs = self.define_objects()
        tools = self.define_tools()
        if len(tools) > 0:
            crystals = self.cut(objs, tools)
        else:
            crystals = objs

        return crystals

    def define_objects(self) -> List[int]:
        """Defines the objects associated with constructing the crystal by cutting.

        Returns:
            indices to all objects.
        """
        return []

    def define_tools(self) -> List[int]:
        """Defines the tools associated with constructing the crystal by cutting.

        Returns:
            indices to all tools.
        """
        return []

    def cut(self, objs: List[int], tools: List[int]) -> List[int]:
        """Cuts the tools from the objects.

        Args:
            objs: basic shape of the crystals.
            tools: tools which are removed from the basic shape to create the crystal.

        Returns:
            indices to all crystals.
        """
        tags, _ = self.factory.cut([(2, obj) for obj in objs], [(2, tool) for tool in tools])
        domains = [tag[1] for tag in tags]
        return domains


class CylindricalCrystalBuilder(CrystalBuilder):
    def __init__(self, description: Description):
        super().__init__(description)

        # define center of crystals
        offset = self.crystal_description.grid_size / 2.0
        self.centers_x = [
            offset + col * self.crystal_description.grid_size + self.box.x_min
            for col in range(self.crystal_description.n)
        ]
        self.center_y = offset + self.box.y_min

    def define_objects(self) -> List[int]:
        """Defines disks associated with this type of crystal.

        Returns:
            indices to the surfaces of the disks.
        """
        crystals = []

        for center_x in self.centers_x:
            crystals.append(
                self.factory.addDisk(
                    center_x, self.center_y, 0.0, self.crystal_description.radius, self.crystal_description.radius
                )
            )

        return crystals


class CShapedCrystalBuilder(CylindricalCrystalBuilder):
    """C-shaped crystal builder.

    C-shaped sonic crystals use cylindrical shaped crystal as a precursor. The difference to a cylindrical shaped
    crystal is that a slot and a disk are cut from the disk.
    """

    def define_tools(self) -> List[int]:
        """Define slot and disc that needs to be cut from the disk.

        Returns:
            indices to both rectangles and discs used as tools in the cutting operation.
        """
        tools = []

        # inner radius and gap in terms of the outer radius
        inner_radius = self.crystal_description.inner_radius
        # gaps should be smaller than inner radius
        gap_height = self.crystal_description.gap_width
        tol = 1.1  # to prevent cutting artifacts
        gap_width = self.crystal_description.radius * tol

        for center_x in self.centers_x:
            inner_disk = self.factory.addDisk(center_x, self.center_y, 0, inner_radius, inner_radius)

            slot = self.factory.addRectangle(
                center_x - self.crystal_description.radius * tol,
                self.center_y - gap_height / 2,
                0,
                gap_width,
                gap_height,
            )

            tools.extend([inner_disk, slot])

        return tools
