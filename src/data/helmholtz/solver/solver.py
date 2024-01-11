import pathlib

import dolfinx
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from src.data.helmholtz.domain_properties import Description
from src.data.helmholtz.meshing import MeshFactory

from .wave_number import AdiabaticLayer, Crystals


class HelmholtzSolver:
    def __init__(self, out_dir: pathlib.Path, element):
        self.out_dir = out_dir
        self.element = element

    def __call__(self, description: Description):
        mesh, cell_tags, facet_tags = MeshFactory.get_mesh(description, self.out_dir)

        # Define function space
        v = dolfinx.fem.FunctionSpace(mesh, self.element)
        v_plot = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
        v_dc = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))  # for discontinuous functions (e.g., crystals)

        # define domain parameters
        k = dolfinx.fem.Function(v)

        s = PETSc.ScalarType(1j * description.rho * description.c)

        trunc = AdiabaticLayer(description)
        crystals = Crystals(description)

        v_f = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1 + 1j)))
        d_excitation = ufl.Measure(
            "ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=description.excitation_index
        )

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
            # update k
            k.x.array[:] = k0 + trunc.eval(v, cell_tags).x.array
            k_crystal = k0 * crystals.eval(v_dc, cell_tags)

            # assemble problem
            lhs = ufl.inner(ufl.grad(p), ufl.grad(xi)) * ufl.dx - (k + k_crystal) ** 2 * ufl.inner(p, xi) * ufl.dx
            rhs = (k + k_crystal) * s * ufl.inner(v_f, xi) * d_excitation

            # compute solution
            problem = LinearProblem(lhs, rhs, u=p_sol, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem.solve()

            # write solution
            out_function = dolfinx.fem.Function(v_plot)
            out_function.interpolate(p_sol)
            out_file.write_function(out_function, f)

        # Close writer
        out_file.close()
