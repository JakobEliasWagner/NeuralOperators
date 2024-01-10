import pathlib
from typing import List

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
        right_spacer = cls.f.addRectangle(dd.width, 0, 0, dd.right_width, dd.height)

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
            crystal_domain = crystal_domain[0][1]
        else:
            fragment_out, _ = cls.f.fragment([(2, domain)], [(2, crystal) for crystal in crystals])
            if len(fragment_out) != dd.crystal_description.n_x * dd.crystal_description.n_y + 1:
                raise MeshConstructionError(
                    "The fragmentation operation from the domain was not successful. "
                    "The generated mesh does not yield a valid domain."
                )
            # identify crystal surfaces
            crystals = []
            crystal_domain = -1
            for element in fragment_out:
                bbox = cls.f.get_bounding_box(2, element[1])
                delta_x = bbox[3] - bbox[0]
                if delta_x <= dd.crystal_description.grid_size:
                    # crystals need to be smaller than grid
                    crystals.append(element)
                    continue
                crystal_domain = element[1]
            if crystal_domain == -1:
                raise MeshConstructionError("Crystal fluid domain could not be identified correctly!")

        # fragment all fluids (due to their construction, this does not change their tags)
        cls.f.fragment([(2, crystal_domain)], [(2, right_spacer)])

        # define physical properties
        gmsh.model.add_physical_group(2, [crystal_domain], dd.domain_index)
        gmsh.model.add_physical_group(2, [right_spacer], dd.right_index)
        # excitation boundary
        boundaries, _ = cls.f.get_curve_loops(crystal_domain)
        for boundary in boundaries:
            com = cls.f.get_center_of_mass(1, boundary)
            if np.allclose(com, [0, dd.height / 2, 0]):
                gmsh.model.add_physical_group(1, [boundary], dd.excitation_index)
                break
        else:
            raise MeshConstructionError("Excitation boundary could not be found!")
        # crystals
        if crystals:
            gmsh.model.add_physical_group(2, crystals, dd.crystal_index)

        return [crystal_domain, right_spacer] + crystals

    @classmethod
    def define_absorbers(cls) -> List[int]:
        """The absorber domains are added onto the sides of the domain. Also, sets physical groups for them.

        Returns:

        """
        pass

    @classmethod
    def set_mesh_properties(cls) -> None:
        """Sets properties that the meshing algorithm needs.

        Returns:

        """
        pass

    @classmethod
    def save_mesh_to_file(cls, dd: Description, out_dir: pathlib.Path) -> None:
        """Saves raw mesh with physical groups to the output-dir.

        Returns:

        """
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = out_dir.joinpath(f"{dd.unique_id}_mesh.msh")
        gmsh.write(str(file_name))  # gmsh does not accept pathlib path

    @classmethod
    def get_mesh(
        cls, domain_description: Description, out_dir: pathlib.Path, frequency_idx: int = -1
    ) -> dolfinx.mesh.Mesh:
        """Returns a dolfinx mesh for a specific domain description.

        Args:
            out_dir:
            domain_description:
            frequency_idx:

        Returns:

        """
        comm = MPI.COMM_WORLD.Get_rank()

        gmsh.initialize()
        gmsh.model.add("Sonic Crystal Domain")

        cls.define_crystal_domain(domain_description)
        cls.define_absorbers()
        cls.set_mesh_properties()

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

        cls.save_mesh_to_file(domain_description, out_dir)

        mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, comm)

        return mesh
