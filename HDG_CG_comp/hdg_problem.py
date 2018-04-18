from firedrake import *
from firedrake.utils import cached_property

import base


class HDGProblem(base.Problem):

    name = "HDG Helmholtz"

    @cached_property
    def tau(self):
        # Stability parameter for the HDG method
        return Constant(100.)

    @cached_property
    def function_space(self):
        U = VectorFunctionSpace(self.mesh, "DG", self.degree)
        V = FunctionSpace(self.mesh, "DG", self.degree)
        T = FunctionSpace(self.mesh, "DGT", self.degree)
        return U * V * T

    @cached_property
    def forcing(self):
        f = Function(self.function_space, name="forcing")
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

        def both(arg):
            return 2*avg(arg)

        a = (dot(sigma, tau)*dx - div(tau)*u*dx +
             both(sigmahat*dot(tau, n))*dS +
             sigmahat*dot(tau, n)*ds
             - dot(grad(v), sigma)*dx +
             both(dot(sigmahat, n)*v)*dS +
             dot(sigmahat, n)*v*ds
             + both(gamma*dot(sigmahat, n))*dS
             + gamma*lambdar*ds)
        return a

    @cached_property
    def L(self):
        W = self.function_space
        _, v, gamma = TestFunctions(W)
        f = self.forcing
        u0 = Constant(1.0)
        return inner(f, v)*dx + inner(u0, gamma)*ds

    @cached_property
    def analytic_flux(self):
        u = self.analytic_solution
        return -grad(u)

    @cached_property
    def bcs(self):
        # All bcs are weakly enforced
        return None

    @cached_property
    def output(self):
        sigma, u, _ = self.u.split()
        return (sigma, u)

    @cached_property
    def err(self):
        u_a = Function(self.function_space[1])
        u_a.interpolate(self.analytic_solution)
        sigma_a = FunctionSpace(self.function_space[0])
        sigma_a.project(self.analytic_flux)
        u_err = errornorm(self.u.split()[1], u_a)
        sigma_err = errornorm(self.u.split()[0], sigma_a)
        return (sigma_err, u_err)

    @cached_property
    def true_err(self):
        u_err = errornorm(self.analytic_solution, self.u.split()[1])
        sigma_err = errornorm(self.analytic_flux, self.u.split()[0])
        return (sigma_err, u_err)

    @cached_property
    def sol(self):
        sigma = Function(self.function_space[0], name="Analytic flux")
        u = Function(self.function_space[1], name="Analytic scalar")
        u.interpolate(self.analytic_solution)
        sigma.project(self.analytic_flux)
        return (sigma, u)
