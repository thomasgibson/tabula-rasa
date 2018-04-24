from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_region

import base


class HDGProblem(base.Problem):

    name = "HDG Helmholtz"

    def __init__(self, N, degree, quadrilaterals, dimension):
        super(HDGProblem, self).__init__(N=N, degree=degree,
                                         quadrilaterals=quadrilaterals,
                                         dimension=dimension)
        self.Vpp = FunctionSpace(self.mesh, "DG", self.degree + 1)
        self.u_pp = Function(self.Vpp, name="Post-processed scalar")

    @cached_property
    def tau(self):
        # Stability parameter for the HDG method
        return Constant(1.0)

    @cached_property
    def u(self):
        return Function(self.function_space, name="Solution")

    @cached_property
    def function_space(self):
        if self.quads:
            dG = "DQ"
        else:
            dG = "DG"

        U = VectorFunctionSpace(self.mesh, dG, self.degree)
        V = FunctionSpace(self.mesh, dG, self.degree)
        T = FunctionSpace(self.mesh, "DGT", self.degree)
        return U * V * T

    @cached_property
    def forcing(self):
        V = FunctionSpace(self.mesh, "DG", self.degree + 3)
        f = Function(V, name="forcing")
        u = self.analytic_solution
        f.interpolate(-div(grad(u)) + u)
        return f

    @cached_property
    def a(self):
        W = self.function_space
        sigma, u, lambdar = TrialFunctions(W)
        tau, v, gamma = TestFunctions(W)
        n = FacetNormal(self.mesh)

        sigmahat = sigma + self.tau*(u - lambdar)*n

        a = (dot(sigma, tau)*dx - div(tau)*u*dx +
             lambdar('+')*jump(tau, n=n)*dS +
             lambdar*dot(tau, n)*ds -
             dot(grad(v), sigma)*dx +
             jump(sigmahat, n=n)*v('+')*dS +
             dot(sigmahat, n)*v*ds
             + v*u*dx
             + gamma('+')*jump(sigmahat, n=n)*dS)
        return a

    @cached_property
    def L(self):
        W = self.function_space
        _, v, _ = TestFunctions(W)
        f = self.forcing
        return inner(f, v)*dx

    @cached_property
    def analytic_flux(self):
        u = self.analytic_solution
        return -grad(u)

    @cached_property
    def bcs(self):
        # Trace variables enforce Dirichlet condition on scalar variable
        T = self.function_space.sub(2)

        if self.dim == 3 and self.quads:
            bcs = [DirichletBC(T, Constant(1), "on_boundary"),
                   DirichletBC(T, Constant(1), "top"),
                   DirichletBC(T, Constant(1), "bottom")]
        else:
            bcs = DirichletBC(T, Constant(1), "on_boundary")

        return bcs

    @cached_property
    def output(self):
        sigma, u, _ = self.u.split()
        return (sigma, u)

    @cached_property
    def err(self):
        u_a = Function(self.function_space[1])
        u_a.interpolate(self.analytic_solution)
        u_err = errornorm(self.u.split()[1], u_a, norm_type="L2")
        return u_err

    @cached_property
    def true_err(self):
        u_err = errornorm(self.analytic_solution, self.u.split()[1],
                          norm_type="L2")
        return u_err

    @cached_property
    def sol(self):
        sigma = Function(self.function_space[0], name="Analytic flux")
        u = Function(self.function_space[1], name="Analytic scalar")
        u.interpolate(self.analytic_solution)
        sigma.project(self.analytic_flux)
        return (sigma, u)

    @cached_property
    def post_processed_expr(self):

        V0 = FunctionSpace(self.mesh, "DG", 0)

        Wpp = self.Vpp * V0

        up, psi = TrialFunctions(Wpp)
        wp, phi = TestFunctions(Wpp)

        K = Tensor((inner(grad(up), grad(wp)) +
                    inner(psi, wp) +
                    inner(up, phi))*dx)

        q_h, u_h, _ = self.u.split()

        F = Tensor((-inner(q_h, grad(wp)) +
                    inner(u_h, phi))*dx)

        return K.inv * F

    def post_processed_sol(self):

        with timed_region("HDGPostprocessing"):
            assemble(self.post_processed_expr.block((0,)), tensor=self.u_pp)
            self.u_pp.dat._force_evaluation()

    @cached_property
    def pp_err(self):
        return errornorm(self.analytic_solution, self.u_pp, norm_type="L2")
