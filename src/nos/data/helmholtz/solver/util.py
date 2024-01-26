import pathlib
from typing import Tuple

import dolfinx
import gmsh
from mpi4py import MPI


def get_mesh(
    mesh_file: pathlib.Path, comm: MPI.Comm = MPI.COMM_SELF
) -> (Tuple)[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """Loads mesh from a msh file and returns it as a dolfinx mesh. Initializes mesh on rank 0 of the given
    communicator.

    Args:
        comm: communicator used for initialization.
        mesh_file: path to the mesh file.

    Returns:
        Tuple containing the dolfinx mesh, mesh-tags and boundary tags.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add(f"read form file {mesh_file.stem}")
    gmsh.merge(str(mesh_file))
    mesh, cell_tags, boundary_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm, 0)
    gmsh.finalize()
    return mesh, cell_tags, boundary_tags
