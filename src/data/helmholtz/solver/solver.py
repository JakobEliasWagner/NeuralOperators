import pathlib

import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from src.data.helmholtz.domain_properties import Description
from src.data.helmholtz.meshing import MeshFactory

from .wave_number import AdiabaticLayer


class HelmholtzSolver:
    def __init__(self, out_dir: pathlib.Path, element):
        self.out_dir = out_dir
        self.element = element

    def __call__(self, description: Description):
        mesh, cell_tags, facet_tags = MeshFactory.get_mesh(description, self.out_dir)

        # Define function space
        v = dolfinx.fem.FunctionSpace(mesh, self.element)
        v_plot = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))

        # define domain parameters
        k = dolfinx.fem.Function(v)

        s = PETSc.ScalarType(1j * description.rho * description.c)

        trunc = AdiabaticLayer(description)

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

        db_levels = np.empty((len(description.frequencies), 2))
        db_levels[:, 0] = description.frequencies

        for i, (k0, f) in enumerate(zip(description.ks, description.frequencies)):
            print(f"{(f - 4000) / 13000 * 100:.0f}%")
            # update k
            k.x.array[:] = k0 + trunc.eval(v, cell_tags).x.array

            # assemble problem
            lhs = ufl.inner(ufl.grad(p), ufl.grad(xi)) * ufl.dx - k**2 * ufl.inner(p, xi) * ufl.dx
            rhs = k * s * ufl.inner(v_f, xi) * d_excitation

            # compute solution
            problem = LinearProblem(lhs, rhs, u=p_sol, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem.solve()

            # write solution
            out_function = dolfinx.fem.Function(v_plot)
            out_function.interpolate(p_sol)
            out_file.write_function(out_function, f)

            # get sound pressure
            # TODO find better way of solving this
            print("rhs start")
            in_rhs = dolfinx.fem.Function(v)
            p_rhs = ufl.TrialFunction(v)

            in_rhs.interpolate(
                lambda x: (x[0] > description.width)
                * (x[0] < (description.width + description.right_width))
                * (x[1] > 0.0)
                * (x[1] < description.height)
            )
            a = ufl.inner(ufl.inner(in_rhs, p_sol), xi) * ufl.dx
            b = ufl.inner(p_rhs, xi) * ufl.dx
            problem = LinearProblem(a, b, u=p_rhs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem.solve()
            print("rhs stop")
            max_level = np.max(np.abs(p_rhs.x.array[:]))
            p0 = description.c**2 * description.rho
            db_level = 10 * np.log10(max_level / p0)
            db_levels[i, 1] = db_level

        # Close writer
        out_file.close()

        # write db levels
        np.savetxt(self.out_dir.joinpath("pressure_levels.csv"), db_levels, delimiter=",")
