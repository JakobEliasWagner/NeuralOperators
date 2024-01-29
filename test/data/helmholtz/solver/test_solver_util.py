import pathlib
import tempfile

import dolfinx
import gmsh
from mpi4py import MPI

from nos.data.helmholtz.solver import get_mesh


def test_get_mesh():
    # create mesh
    gmsh.initialize()
    gmsh.model.add("test_get_mesh")

    rect = gmsh.model.occ.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    circ = gmsh.model.occ.add_disk(0.0, 0.0, 0.0, 0.5, 0.5)

    objs, _ = gmsh.model.occ.fragment([(2, rect)], [(2, circ)])

    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.get_entities(2)
    surfs = [surf[1] for surf in surfs]
    lines = gmsh.model.occ.get_entities(1)
    lines = [line[1] for line in lines]

    for i, surf in enumerate(surfs):
        gmsh.model.add_physical_group(2, [surf], i)
    for i, line in enumerate(lines):
        gmsh.model.add_physical_group(1, [line], i)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    with tempfile.TemporaryDirectory() as tmp:
        # write file
        file_path = pathlib.Path(tmp).joinpath("test_mesh.msh")
        gmsh.write(str(file_path))
        gmsh.finalize()

        # load
        mesh, cell_tags, boundary_tags = get_mesh(file_path, MPI.COMM_WORLD)

    # test types
    assert isinstance(mesh, dolfinx.mesh.Mesh)
    assert isinstance(cell_tags, dolfinx.mesh.MeshTags)
    assert isinstance(boundary_tags, dolfinx.mesh.MeshTags)

    # test size
    assert len(lines) == len(set(boundary_tags.values))
    assert len(surfs) == len(set(cell_tags.values))
