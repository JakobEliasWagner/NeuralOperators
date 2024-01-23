import pathlib

import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from nos.data.helmholtz.domain_properties import Description
from nos.data.helmholtz.mesh import MeshBuilder

from .adiabatic_absorber import AdiabaticAbsorber
from .util import get_mesh


class HelmholtzSolver:
    def __init__(self, out_dir: pathlib.Path, element):
        self.out_dir = out_dir
        self.element = element

    def __call__(self, description: Description):
        # MESH
        mesh_path = self.out_dir.joinpath(f"{description.unique_id}_mesh.msh")
        mesh_builder = MeshBuilder(description, mesh_path)
        mesh_builder.build()
        mesh, ct, _ = get_mesh(mesh_path)

        # Define function space
        v = dolfinx.fem.FunctionSpace(mesh, self.element)
        v_plot = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

        # define domain parameters
        k = dolfinx.fem.Function(v)  # wave number
        trunc = AdiabaticAbsorber(description)
        p_i = dolfinx.fem.Function(v)  # incident wave
        d_i = ufl.Measure("dx", domain=mesh, subdomain_data=ct, subdomain_id=description.indices["left_side"])

        # Define variational problem
        p = ufl.TrialFunction(v)
        xi = ufl.TestFunction(v)

        p_sol = dolfinx.fem.Function(v)
        p_sol.name = "p"

        # Start writer
        filename = self.out_dir.joinpath(f"{description.unique_id}_solution.xdmf")
        out_file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5)
        out_file.write_mesh(mesh)

        for i, (k0, f) in enumerate(zip(description.ks, description.frequencies)):
            # frequency specific quantities
            k.interpolate(lambda x: k0 * (1 + trunc.eval(x)))
            p_i.interpolate(lambda x: k0**2 * np.exp(1j * k0 * x[0]))
            # assemble problem
            lhs = ufl.inner(ufl.grad(p), ufl.grad(xi)) * ufl.dx - k**2 * ufl.inner(p, xi) * ufl.dx
            rhs = k**2 * ufl.inner(p_i, xi) * d_i

            # compute solution
            problem = LinearProblem(lhs, rhs, u=p_sol, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem.solve()

            # write solution
            out_function = dolfinx.fem.Function(v_plot)
            out_function.interpolate(p_sol)
            out_file.write_function(out_function, f)

        # Close writer
        out_file.close()
