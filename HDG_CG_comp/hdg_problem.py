from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_region

import base


class HDGProblem(base.Problem):

    name = "HDG Helmholtz"

    @cached_property
    def tau(self):
        # Stability parameter for the HDG method
        return Constant(1.)

    @cached_property
    def function_space(self):
        U = VectorFunctionSpace(self.mesh, "DG", self.degree)
        V = FunctionSpace(self.mesh, "DG", self.degree)
        T = FunctionSpace(self.mesh, "DGT", self.degree)
        return U * V * T

    @cached_property
    def forcing(self):
        f = Function(self.function_space[1], name="forcing")
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
             lambdar('+')*jump(tau, n=n)*dS +
             lambdar*dot(tau, n)*ds -
             dot(grad(v), sigma)*dx +
             jump(sigmahat, n=n)*v('+')*dS +
             dot(sigmahat, n)*v*ds
             + inner(u, v)*dx
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
        return DirichletBC(self.function_space[2], 1, "on_boundary")

    @cached_property
    def output(self):
        sigma, u, _ = self.u.split()
        return (sigma, u)

    @cached_property
    def err(self):
        u_a = Function(self.function_space[1])
        u_a.interpolate(self.analytic_solution)
        sigma_a = Function(self.function_space[0])
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

    def post_processed_sol(self):
        Vpp = FunctionSpace(self.mesh, "DG", self.degree + 1)
        V0 = FunctionSpace(self.mesh, "DG", 0)
        Wpp = Vpp * V0

        up, psi = TrialFunctions(Wpp)
        wp, phi = TestFunctions(Wpp)

        K = Tensor((inner(grad(up), grad(wp)) +
                    inner(psi, wp) +
                    inner(up, phi))*dx)

        q_h, u_h, _ = self.u.split()

        F = Tensor((-inner(q_h, grad(wp)) +
                    inner(u_h, phi))*dx)

        E = K.inv * F

        with timed_region("HDGPostprocessing"):
            u_pp = Function(Vpp, name="Post-processed scalar")
            assemble(E.block((0,)), tensor=u_pp)

        self.u_pp = u_pp

    @cached_property
    def pp_err(self):
        return errornorm(self.analytic_solution, self.u_pp)
