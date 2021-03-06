from firedrake import *
from firedrake.utils import cached_property

import base


class CGProblem(base.Problem):

    name = "CG Helmholtz"

    @cached_property
    def function_space(self):
        return FunctionSpace(self.mesh, "CG", self.degree)

    @cached_property
    def u(self):
        return Function(self.function_space, name="Solution")

    @cached_property
    def forcing(self):
        V = FunctionSpace(self.mesh, "CG", self.degree + 3)
        f = Function(V, name="forcing")
        u = self.analytic_solution
        f.interpolate(-div(grad(u)) + u)
        return f

    @cached_property
    def a(self):
        V = self.function_space
        u = TrialFunction(V)
        v = TestFunction(V)
        return dot(grad(u), grad(v))*dx + inner(u, v)*dx

    @cached_property
    def L(self):
        V = self.function_space
        v = TestFunction(V)
        f = self.forcing
        return inner(f, v)*dx

    @cached_property
    def bcs(self):
        V = self.function_space
        if self.dim == 3 and self.quads:
            bcs = [DirichletBC(V, Constant(1), "on_boundary"),
                   DirichletBC(V, Constant(1), "top"),
                   DirichletBC(V, Constant(1), "bottom")]
        else:
            bcs = DirichletBC(V, Constant(1), "on_boundary")

        return bcs

    @cached_property
    def output(self):
        return self.u

    @cached_property
    def err(self):
        u_a = Function(self.function_space)
        u_a.interpolate(self.analytic_solution)
        return errornorm(self.u, u_a, norm_type="L2")

    @cached_property
    def true_err(self):
        return errornorm(self.analytic_solution, self.u, norm_type="L2")

    @cached_property
    def sol(self):
        u = Function(self.function_space, name="Analytic")
        u.interpolate(self.analytic_solution)
        return u
