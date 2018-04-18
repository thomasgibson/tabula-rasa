from firedrake import *
from firedrake.utils import cached_property

import base


class CGProblem(base.Problem):

    name = "CG Helmholtz"

    @cached_property
    def function_space(self):
        return FunctionSpace(self.mesh, "CG", self.degree)

    @cached_property
    def forcing(self):
        f = Function(self.function_space, name="forcing")
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
        return DirichletBC(self.function_space, Constant(1.), "on_boundary")

    @cached_property
    def output(self):
        return self.u

    @cached_property
    def err(self):
        u_a = Function(self.function_space)
        u_a.interpolate(self.analytic_solution)
        return errornorm(self.u, u_a)

    @cached_property
    def true_err(self):
        return errornorm(self.analytic_solution, self.u)

    @cached_property
    def sol(self):
        u = Function(self.function_space, name="Analytic")
        u.interpolate(self.analytic_solution)
        return u
