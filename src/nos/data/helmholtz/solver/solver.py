import pathlib

import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import as_vector, grad, inner

from nos.data.helmholtz.domain_properties import Description
from nos.data.helmholtz.mesh import MeshBuilder

from .adiabatic_absorber import AdiabaticAbsorber
from .util import get_mesh


class HelmholtzSolver:
    def __init__(self, out_dir: pathlib.Path, element):
        self.out_dir = out_dir
        self.element = element

    def __call__(self, description: Description):
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        # MESH
        mesh_path = self.out_dir.joinpath(f"{description.unique_id}_mesh.msh")
        mesh_builder = MeshBuilder(description, mesh_path)
        mesh_builder.build()
        msh, ct, ft = get_mesh(mesh_path, comm=MPI.COMM_SELF)

        # Define function space
        v = dolfinx.fem.FunctionSpace(msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2))
        v_plot = dolfinx.fem.FunctionSpace(msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1))

        # define domain parameters
        p0 = 1.0
        ks = dolfinx.fem.Function(v)  # squared wave number with pml
        # alpha = dolfinx.fem.Function(v)
        sigma = AdiabaticAbsorber(description)

        p_i = dolfinx.fem.Function(v)  # incident wave
        dx = ufl.Measure("dx", msh, subdomain_data=ct)

        # Define variational problem
        p_s = ufl.TrialFunction(v)
        xi = ufl.TestFunction(v)

        p_sol = dolfinx.fem.Function(v)
        p_sol.name = "p"

        # Start writer
        filename = self.out_dir.joinpath(f"{description.unique_id}_solution.xdmf")
        writer = dolfinx.io.XDMFFile(MPI.COMM_SELF, filename, "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5)
        writer.write_mesh(msh)

        def grad_x(func):
            return inner(as_vector((1, 0)), grad(func))

        def grad_y(func):
            return inner(as_vector((0, 1)), grad(func))

        for i, (k0, f) in enumerate(zip(description.ks, description.frequencies)):
            # frequency specific quantities
            # alpha.interpolate(lambda x: 1 / (1 + 1j * sigma.eval(x)))
            # alpha.interpolate(lambda x: 1 / (1 + 1j * sigma.eval(x) / omega))
            ks.interpolate(lambda x: (k0 * (1 + 1j * sigma.eval(x) / k0)) ** 2)
            p_i.interpolate(lambda x: p0 * np.exp(1j * k0 * x[0]))

            # assemble problem
            lhs = ufl.inner(ufl.grad(p_s), ufl.grad(xi)) * dx - ks * ufl.inner(p_s, xi) * dx
            rhs = -ufl.inner(ufl.grad(p_i), ufl.grad(xi)) * dx + k0**2 * ufl.inner(p_i, xi) * dx

            """lhs = (
                    xi * alpha * grad_x(alpha) * grad_x(p_s) * dx  # 1
                    - (alpha ** 2) * grad_x(xi) * grad_x(p_s) * dx  # 2.1
                    - 2 * alpha * xi * grad_x(alpha) * grad_x(p_s) * dx  # 2.2
                    - grad_y(xi) * grad_y(p_s) * dx  # 3
                    + k0 ** 2 * xi * p_s * dx  # 4
            )
            rhs = (
                    - xi * alpha * grad_x(alpha) * grad_x(p_i) * dx  # 1
                    + (alpha ** 2) * grad_x(xi) * grad_x(p_i) * dx  # 2.1
                    + 2 * alpha * xi * grad_x(alpha) * grad_x(p_i) * dx  # 2.2
                    + grad_y(xi) * grad_y(p_i) * dx  # 3
                    - k0 ** 2 * xi * p_i * dx  # 4
            )"""

            # compute solution
            problem = LinearProblem(
                lhs,
                rhs,
                u=p_sol,
                petsc_options={"ksp_type": "preonly", "pc_type": "cholesky", "pc_factor_mat_solver_type": "mumps"},
            )
            problem.solve()

            p_sol.x.array[:] = p_sol.x.array[:] + p_i.x.array[:]

            # write solution
            out_function = dolfinx.fem.Function(v_plot)
            out_function.interpolate(p_sol)
            writer.write_function(out_function, f)

        # Close writer
        writer.close()
