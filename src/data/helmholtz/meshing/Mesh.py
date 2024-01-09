import pathlib
from typing import List

import dolfinx
import gmsh
from mpi4py import MPI

from src.data.helmholtz.domain_properties import Description


class MeshFactory:
    f = gmsh.model.occ  # gmsh-factory

    @classmethod
    def define_crystal_domain(cls) -> List[int]:
        """Defines the crystal domain.

        The crystal domain is made up of three rectangles. The left spacer, the actual crystal domain, and a right
        spacer. The crystals are either cut from the crystal domain or fragmented (this enables defining material
        properties on them). The right spacer can later be used to integrate and evaluate band gaps (this should be
        longer than half a wave-length for all frequencies). It Also sets the gmsh physical groups for fluids, crystals,
        excitation, and optionally of all crystals.

        Returns: list containing the surface indices of the left spacer, crystal domain, right spacer, and optionally
        all indices to crystal surfaces.

        """
        pass

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
    def save_mesh_to_file(cls) -> None:
        """Saves raw mesh with physical groups to the output-dir.

        Returns:

        """
        pass

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

        cls.define_crystal_domain()
        cls.define_absorbers()
        cls.set_mesh_properties()

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

        cls.save_mesh_to_file()

        mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, comm)

        return mesh
