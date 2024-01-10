import pathlib

import dolfinx
from dolfinx.fem import Function, functionspace
from dolfinx.io import XDMFFile
from mpi4py import MPI


def write_function(mesh: dolfinx.mesh.Mesh, u: Function, out_dir: pathlib.Path, name: str) -> None:
    """

    Args:
        out_dir:
        mesh:
        u:
        name: name without file extension

    Returns:

    """
    # interpolate onto a mesh of order one
    # May introduce some problems
    # - loss of accuracy
    # - projection error
    # - smoothing of solution
    # - not efficient
    # higher order meshes can be written with the VTXWriter
    V_xdf = functionspace(mesh, ("Lagrange", 1))
    out_function = Function(V_xdf)
    out_function.interpolate(u)

    # Save solution in XDMF format (to be viewed in ParaView, for example)
    with XDMFFile(MPI.COMM_WORLD, out_dir / f"{name}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)
        file.write_function(out_function)
