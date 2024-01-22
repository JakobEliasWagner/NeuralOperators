import pathlib
import warnings
from collections import defaultdict
from typing import List, Tuple

import gmsh
import numpy as np

from nos.data.helmholtz.domain_properties import Description
from nos.utility import BoundingBox2D

from .crystal_domain_builder import CrystalDomainBuilder, CShapedCrystalDomainBuilder, CylindricalCrystalDomainBuilder
from .gmsh_builder import GmshBuilder


class MeshBuilder(GmshBuilder):
    def __init__(self, description: Description, out_file: pathlib.Path):
        super().__init__(description)
        self.out_file = out_file

        # set appropriate builders
        crystal_type = description.crystal_description.type_name
        if crystal_type == "Cylindrical":
            db = CylindricalCrystalDomainBuilder
        elif crystal_type == "C-Shaped":
            db = CShapedCrystalDomainBuilder
        elif crystal_type == "None":
            db = CrystalDomainBuilder
        else:
            warnings.warn(f"Unknown crystal type {crystal_type}. Defaulting to None!")
            db = CrystalDomainBuilder
        self.domain_builder = db(description)

    def build(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)  # set verbosity level (still prints warnings)
        gmsh.model.add(f"model_{self.description.unique_id}")

        self.build_basic_shapes()
        self.fragment_domain()
        groups = self.get_physical_groups()
        self.set_physical_groups(groups)
        self.set_mesh_properties()
        self.generate_mesh()

        self.save_mesh()
        gmsh.finalize()

    def set_mesh_properties(self):
        """Sets properties that the meshing algorithm needs."""
        gmsh.option.setNumber("Mesh.MeshSizeMax", min(self.description.wave_lengths) / self.description.elements)

    def generate_mesh(self):
        """Generates the mesh."""
        self.factory.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

    def build_basic_shapes(self) -> List[int]:
        """Builds all basic shapes.

        Basic shapes are rectangles within the domain. However, what is inside these rectangles may vary (i.e., cuts).
        Generates the crystal domain, a left and right space, and absorbers according to the description.

        Returns:
            indices to surfaces of generated shapes.
        """
        dd = self.description
        tags = []
        # left space, domain, right space
        if self.description.left_width > 0:
            tags.append(self.factory.addRectangle(0.0, 0.0, 0.0, self.description.left_width, self.description.height))
        tags.append(self.domain_builder.build())
        if self.description.right_width > 0:
            tags.append(
                self.factory.addRectangle(
                    self.description.left_width + self.description.width,
                    0.0,
                    0.0,
                    self.description.right_width,
                    self.description.height,
                )
            )
        # absorbers
        right_absorber_height = 0.0
        right_absorber_y = 0.0
        if dd.directions["top"]:
            tags.append(
                self.factory.add_rectangle(
                    0,
                    self.description.height,
                    0,
                    self.description.width + self.description.right_width,
                    self.description.absorber_depth,
                )
            )
            right_absorber_height += self.description.absorber_depth
        if self.description.directions["bottom"]:
            tags.append(
                self.factory.add_rectangle(
                    0,
                    -self.description.absorber_depth,
                    0,
                    self.description.width + self.description.right_width,
                    self.description.absorber_depth,
                )
            )
            right_absorber_height += self.description.absorber_depth
            right_absorber_y -= self.description.absorber_depth
        if self.description.directions["right"]:
            right_absorber_height += self.description.height
            tags.append(
                self.factory.add_rectangle(
                    self.description.width + self.description.right_width,
                    right_absorber_y,
                    0,
                    self.description.absorber_depth,
                    right_absorber_height,
                )
            )

        return tags

    def fragment_domain(self) -> List[int]:
        """Fragments all parts of the domain, to create one connected mesh.

        Returns:
            All surface indices in the domain.
        """
        all_surfaces = self.factory.get_entities(2)

        # fragment everything
        new_tags, _ = self.factory.fragment([all_surfaces[0]], all_surfaces[1:])

        return [tag[1] for tag in new_tags]

    def get_physical_groups(self) -> List[Tuple[int, List[int], int]]:
        """Takes the domain and identifies different physical groups.

        Returns:
            Tuples containing the dim, physical tag, and a list of surface tags within this group
        """
        # surface
        all_surfaces = self.factory.get_entities(2)
        surf_categories = defaultdict(list)
        domain_bbox = BoundingBox2D(
            0.0,
            0.0,
            self.description.left_width + self.description.width + self.description.right_width,
            self.description.height,
        )  # left space, domain, right space
        for _, surf in all_surfaces:
            com = np.array(self.factory.getCenterOfMass(2, surf)).reshape((1, 3))  # reshape to use bbox
            # find absorbers by calculating distance to inner box (including crystal domain and right spacer)
            is_inside = len(domain_bbox.inside(com)) > 0
            if not is_inside:
                surf_categories["absorber"].append(surf)
                continue
            # left spacer
            if com[0, 0] < self.description.left_width:
                surf_categories["left_side"].append(surf)
                continue
            # domain
            if com[0, 0] < self.description.left_width + self.description.width:
                surf_categories["crystal_domain"].append(surf)
                continue
            # right spacer
            surf_categories["right_side"].append(surf)
        categories = []
        for name, indices in surf_categories.items():
            categories.append((2, indices, self.description.indices[name]))

        # lines
        lines = self.factory.get_entities(1)
        excitation_boundary = None
        for _, line in lines:
            com = np.array(self.factory.getCenterOfMass(1, line))
            if np.allclose(com, [0, self.description.height / 2.0, 0]):
                excitation_boundary = line
                break

        categories.append((1, [excitation_boundary], self.description.indices["excitation"]))

        return categories

    def set_physical_groups(self, groups: List[Tuple[int, List[int], int]]) -> None:
        """Sets physical groups according to the given groups

        Args:
            groups: tuple containing (dim, surface_tags, physical group index)
        """
        self.factory.synchronize()
        for group in groups:
            gmsh.model.add_physical_group(*group)

    def save_mesh(self) -> None:
        """Saves raw mesh with physical groups to file."""
        self.out_file.parent.mkdir(exist_ok=True, parents=True)
        gmsh.write(str(self.out_file))  # gmsh does not accept pathlib path
