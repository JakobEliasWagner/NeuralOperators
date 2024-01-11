import dataclasses
import pathlib
from typing import List, Tuple

import dolfinx
import gmsh
import numpy as np
from mpi4py import MPI

from src.data.helmholtz.domain_properties import CShapeDescription, CylinderDescription, Description, NoneDescription

from .error import MeshConstructionError


class MeshFactory:
    f = gmsh.model.occ  # gmsh-factory

    @classmethod
    def define_cylindrical_crystals(cls, cd: CylinderDescription) -> List[int]:
        """Defines cylindrical sonic crystal lattice in gmsh.

        Args:
            cd: crystal description

        Returns: list surface tags of the individual cylinders.

        """
        crystals = []
        offset = cd.grid_size / 2.0

        for row in range(cd.n_y):
            for col in range(cd.n_x):
                center_x = offset + col * cd.grid_size
                center_y = offset + row * cd.grid_size
                crystals.append(cls.f.addDisk(center_x, center_y, 0.0, cd.radius, cd.radius))

        return crystals

    @classmethod
    def define_c_shaped_crystals(cls, cd: CShapeDescription) -> List[int]:
        """Defines C-shaped sonic crystal lattice in gmsh.

        Args:
            cd: crystal description

        Returns: list surface tags of the individual C-shapes.

        """
        offset = cd.grid_size / 2.0

        crystals = []
        for row in range(cd.n_y):
            for col in range(cd.n_x):
                center_x = offset + col * cd.grid_size
                center_y = offset + row * cd.grid_size

                # create basic shapes
                outer_disk = cls.f.addDisk(center_x, center_y, 0, cd.outer_radius, cd.outer_radius)
                inner_radius = cd.outer_radius * cd.inner_radius
                inner_disk = cls.f.addDisk(center_x, center_y, 0, inner_radius, inner_radius)
                gap_width = min([inner_radius, cd.gap_width * cd.outer_radius])  # should be smaller than inner radius
                tol = 1.1  # to prevent cutting artifacts
                slot = cls.f.addRectangle(
                    center_x - cd.outer_radius * tol, center_y - gap_width / 2, 0, cd.outer_radius * tol, gap_width
                )

                # create c-shape
                crystal, _ = cls.f.cut([(2, outer_disk)], [(2, inner_disk), (2, slot)])
                crystals.append(crystal[0][1])

        return crystals

    @classmethod
    def define_none_crystals(cls, cd: NoneDescription) -> List[int]:
        """Defines a domain without crystals

        Args:
            dd: domain description
            cd: crystal description

        Returns: Empty list

        """
        return []

    @classmethod
    def define_crystal_domain(cls, dd: Description) -> List[int]:
        """Defines the crystal domain.

        The crystal domain is made up of three rectangles. The actual crystal domain, and a right
        spacer. The crystals are either cut from the crystal domain or fragmented (this enables defining material
        properties on them). The right spacer can later be used to integrate and evaluate band gaps (this should be
        longer than half a wave-length for all frequencies). It Also sets the gmsh physical groups for fluids, crystals,
        excitation, and optionally of all crystals.

        Returns: list containing the surface indices of the crystal domain, right spacer, and optionally
        all indices to crystal surfaces.

        """
        # initialize basic shapes
        domain = cls.f.addRectangle(0, 0, 0, dd.width, dd.height)
        cls.f.addRectangle(dd.width, 0, 0, dd.right_width, dd.height)

        # initialize crystals
        if isinstance(dd.crystal_description, CylinderDescription):
            crystal_generator = cls.define_cylindrical_crystals
        elif isinstance(dd.crystal_description, CShapeDescription):
            crystal_generator = cls.define_c_shaped_crystals
        else:
            crystal_generator = cls.define_none_crystals
        crystals = crystal_generator(dd.crystal_description)

        # cut or fragment crystals from domain
        if dd.crystal_description.cut:
            crystal_domain, _ = cls.f.cut([(2, domain)], [(2, crystal) for crystal in crystals])
            if len(crystal_domain) != 1:
                # exactly one domain means that there were no overlapping crystals
                raise MeshConstructionError(
                    "The cutting operation from the domain was not successful. The generated "
                    "mesh does not yield a valid domain."
                )
            crystals = []  # surfaces no longer exist
            crystal_domain = [crystal_domain[0][1]]
        else:
            fragment_out, _ = cls.f.fragment([(2, domain)], [(2, crystal) for crystal in crystals])
            if len(fragment_out) != dd.crystal_description.n_x * dd.crystal_description.n_y + 1:
                raise MeshConstructionError(
                    "The fragmentation operation from the domain was not successful. "
                    "The generated mesh does not yield a valid domain."
                )
            crystal_domain = [f[1] for f in fragment_out]

        return crystal_domain

    @classmethod
    def define_absorbers(cls, dd: Description) -> List[int]:
        """The absorber domains are added onto the sides of the domain. Also, sets physical groups for them.

        Args:
            dd: Description of the domain.

        Returns: surface indices of the absorbers.

        """
        right_absorber_height = 0.0
        right_absorber_y = 0.0
        absorbers = []
        if dd.directions["top"]:
            absorbers.append(cls.f.addRectangle(0, dd.height, 0, dd.width + dd.right_width, dd.absorber_depth))
            right_absorber_height += dd.absorber_depth
        if dd.directions["bottom"]:
            absorbers.append(cls.f.addRectangle(0, -dd.absorber_depth, 0, dd.width + dd.right_width, dd.absorber_depth))
            right_absorber_height += dd.absorber_depth
            right_absorber_y -= dd.absorber_depth
        if dd.directions["right"]:
            right_absorber_height += dd.height
            absorbers.append(
                cls.f.addRectangle(
                    dd.width + dd.right_width, right_absorber_y, 0, dd.absorber_depth, right_absorber_height
                )
            )

        absorbers, _ = cls.f.fragment([(2, absorbers[0])], [(2, absorber) for absorber in absorbers[1:]])
        absorbers = [absorber[1] for absorber in absorbers]

        return absorbers

    @classmethod
    def set_mesh_properties(cls, dd: Description) -> None:
        """Sets properties that the meshing algorithm needs.

        Args:
            dd: Description of the domain

        Returns:

        """
        gmsh.option.setNumber("Mesh.MeshSizeMax", min(dd.wave_lengths) / dd.elements)

    @classmethod
    def save_mesh_to_file(cls, dd: Description, out_dir: pathlib.Path) -> None:
        """Saves raw mesh with physical groups to the output-dir.

        Args:
            dd: Description of the domain.
            out_dir: Directory to which the mesh file is saved.

        Returns:

        """
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = out_dir.joinpath(f"{dd.unique_id}_mesh.msh")
        gmsh.write(str(file_name))  # gmsh does not accept pathlib path

    @classmethod
    def get_mesh(
        cls, domain_description: Description, out_dir: pathlib.Path, frequency_idx: int = None
    ) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
        """Returns a dolfinx mesh for a specific domain description.

        Args:
            out_dir: Directory to which a copy in .msh-file format is saved.
            domain_description: Description of the domain.
            frequency_idx: index of the frequency in the frequency array if only a single frequency of the Description
                should be used.

        Returns: Tuple containing the mesh, cell and boundary tags.

        """
        if frequency_idx:
            domain_description = dataclasses.replace(domain_description)
            domain_description.frequencies = domain_description.frequencies[frequency_idx]
            domain_description.update_derived_properties()

        comm = MPI.COMM_WORLD.Get_rank()

        gmsh.initialize()
        gmsh.model.add("Sonic Crystal Domain")

        cls.define_basic_shapes(domain_description)
        cls.fragment_domain()
        cls.set_physical_groups(domain_description)
        cls.set_mesh_properties(domain_description)

        cls.f.synchronize()
        gmsh.model.mesh.generate(2)

        cls.save_mesh_to_file(domain_description, out_dir)

        return dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, comm)

    @classmethod
    def define_basic_shapes(cls, dd: Description) -> List[int]:
        """Defines the basic shapes (rectangles, crystals) of the domain. The objects are not fragmented/fused together.

        Args:
            dd: Description of the domain.

        Returns: list with all surface indices of the shapes.

        """
        crystal_domain = cls.define_crystal_domain(dd)
        absorbers = cls.define_absorbers(dd)
        return crystal_domain + absorbers

    @classmethod
    def fragment_domain(cls) -> List[int]:
        """Fragments all parts of the domain, to create one connected mesh.

        Returns: List with all surface indices in the domain.

        """
        all_surfaces = cls.f.get_entities(2)

        # fragment everything
        new_tags, _ = cls.f.fragment([all_surfaces[0]], all_surfaces[1:])

        return [tag[1] for tag in new_tags]

    @classmethod
    def get_surface_categories(cls, dd: Description) -> Tuple[List[int], List[int], List[int], List[int]]:
        """

        Args:
            dd: Description of the domain.

        Returns: surface indices of (crystals, crystal domain, right spacer, absorbers)

        """
        all_surfaces = cls.f.get_entities(2)

        crystals = []
        crystal_domain = []
        right_spacer = []
        absorbers = []
        for _, surf in all_surfaces:
            com = np.array(cls.f.getCenterOfMass(2, surf))
            # find absorbers by calculating distance to inner box (including crystal domain and right spacer)
            box_min = np.array([0.0, 0.0, 0.0])
            box_max = np.array([dd.width + dd.right_width, dd.height, 0.0])
            clamped = np.maximum(box_min, np.minimum(com, box_max))
            dist = np.linalg.norm(clamped - com, axis=0)
            if dist > 0:
                absorbers.append(surf)
                continue
            # find the right spacer
            if com[0] > dd.width:
                right_spacer.append(surf)
                continue

            # find crystals and crystal domain
            bbox = cls.f.get_bounding_box(2, surf)
            dims = np.array([x2 - x1 for x1, x2 in zip(bbox[:3], bbox[3:])])

            if all(dims < dd.crystal_description.grid_size):
                # crystals need to be smaller than the grid size to yield a valid domain
                crystals.append(surf)
                continue
            crystal_domain.append(surf)

        # ensure correct construction
        if len(crystal_domain) != 1:
            raise MeshConstructionError(
                "The crystal fluid domain could not be identified correctly."
                f"A total of {len(crystal_domain)} surfaces have been found!"
            )

        return crystals, crystal_domain, right_spacer, absorbers

    @classmethod
    def get_excitation_boundary(cls, dd: Description) -> List[int]:
        """Locates the excitation boundary

        Args:
            dd: Description of the domain.

        Returns: List with one element, which is the line index of the excitation boundary.

        """
        lines = cls.f.get_entities(1)
        excitation_boundary = []
        for _, line in lines:
            com = np.array(cls.f.getCenterOfMass(1, line))
            if np.allclose(com, [0, dd.height / 2.0, 0]):
                excitation_boundary.append(line)
                break
        else:
            raise MeshConstructionError("Excitation boundary could not be identified!")
        return excitation_boundary

    @classmethod
    def set_physical_groups(cls, dd: Description) -> None:
        """Identifies and sets the physical groups according to the description of the domain.

        Args:
            dd: Description of the domain.

        Returns:

        """
        crystals, crystal_domain, right_spacer, absorbers = cls.get_surface_categories(dd)
        excitation_boundary = cls.get_excitation_boundary(dd)

        # set physical groups
        cls.f.synchronize()

        gmsh.model.add_physical_group(2, crystal_domain, dd.domain_index)
        gmsh.model.add_physical_group(2, right_spacer, dd.right_index)
        gmsh.model.add_physical_group(2, absorbers, dd.absorber_index)
        if crystals:
            gmsh.model.add_physical_group(2, crystals, dd.crystal_index)

        gmsh.model.add_physical_group(1, excitation_boundary, dd.excitation_index)
