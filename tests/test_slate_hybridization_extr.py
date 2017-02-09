"""Tests hybridization of the nice-Helmholtz problem on extruded meshes
using Slate.
"""
from __future__ import absolute_import, print_function, division
import pytest

from firedrake import *


def test_slate_hybridization_extr():
    degree = 1
    base = UnitSquareMesh(1, 1, quadrilateral=False)
    mesh = ExtrudedMesh(base, 1)

    RT_elt = FiniteElement("RT", triangle, degree)
    DG = FiniteElement("DG", interval, degree - 1)
    DGh = FiniteElement("DG", triangle, degree - 1)
    CG = FiniteElement("CG", interval, degree)
    elem = EnrichedElement(HDiv(TensorProductElement(RT_elt, DG)),
                           HDiv(TensorProductElement(DGh, CG)))
    product_elt = BrokenElement(elem)
    V = FunctionSpace(mesh, product_elt)
    U = FunctionSpace(mesh, "DG", degree - 1)
    T = FunctionSpace(mesh, "HDiv Trace", (degree - 1, degree - 1))
    W = V * U
    n = FacetNormal(mesh)
    x, y, z = SpatialCoordinate(mesh)

    f = Function(U)
    f.interpolate((1+12*pi*pi)*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z))

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    gammar = TestFunction(T)

    mass_v = dot(sigma, tau) * dx
    mass_p = u * v * dx
    divgrad = div(sigma) * v * dx
    divgrad_adj = div(tau) * u * dx
    local_trace = (gammar('+') * dot(sigma, n) * dS_h +
                   gammar('+') * dot(sigma, n) * dS_v)
    L = f*v*dx

    bcs = DirichletBC(T, Constant(0.0), "on_boundary")

    A = Tensor(mass_v + mass_p + divgrad - divgrad_adj)
    K = Tensor(local_trace)
    Schur = -K * A.inv * K.T

    F = Tensor(L)
    RHS = - K * A.inv * F

    S = assemble(Schur, bcs=bcs)
    E = assemble(RHS)

    lambda_sol = Function(T)
    solve(S, lambda_sol, E, solver_parameters={'pc_type': 'lu',
                                               'ksp_type': 'cg'})

    sigma = TrialFunction(V)
    tau = TestFunction(V)
    u = TrialFunction(U)
    v = TestFunction(U)

    A_v = Tensor(dot(sigma, tau) * dx)
    A_p = Tensor(u * v * dx)
    B = Tensor(div(sigma) * v * dx)
    K = Tensor(dot(sigma, n) * gammar('+') * dS_h +
               dot(sigma, n) * gammar('+') * dS_v)
    F = Tensor(f * v * dx)

    # SLATE expression for pressure recovery:
    u_sol = (B * A_v.inv * B.T + A_p).inv * (F + B * A_v.inv * K.T * lambda_sol)
    u_h = assemble(u_sol)

    # SLATE expression for velocity recovery
    sigma_sol = A_v.inv * (B.T * u_h - K.T * lambda_sol)
    sigma_h = assemble(sigma_sol)

    exact = Function(U)
    exact.interpolate(cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z))

    # sigma = -grad(u)
    sig_error = sqrt(assemble(dot(sigma_h + grad(exact),
                                  sigma_h + grad(exact))*dx))
    u_error = sqrt(assemble((u_h - exact)*(u_h - exact)*dx))

    assert sig_error < 1e-11
    assert u_error < 1e-11


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
