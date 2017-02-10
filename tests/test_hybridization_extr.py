"""Tests hybridization of the nice-Helmholtz problem on extruded meshes"""
from __future__ import absolute_import, print_function, division
import pytest

from firedrake import *


def test_hybridization_extr():
    degree = 1
    base = UnitSquareMesh(4, 4, quadrilateral=False)
    mesh = ExtrudedMesh(base, 2)

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
    W = V * U * T
    n = FacetNormal(mesh)
    x, y, z = SpatialCoordinate(mesh)

    f = Function(U)
    f.interpolate((1+12*pi*pi)*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z))

    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    a_dx = (dot(tau, sigma) - div(tau)*u + v*u + v*div(sigma))*dx
    a_dS = ((jump(tau, n=n)*lambdar('+') + gammar('+')*jump(sigma, n=n))*dS_h +
            (jump(tau, n=n)*lambdar('+') + gammar('+')*jump(sigma, n=n))*dS_v)
    a = a_dx + a_dS
    L = f*v*dx

    bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

    w = Function(W)
    solve(a == L, w, solver_parameters={'ksp_rtol': 1e-14,
                                        'ksp_max_it': 30000},
          bcs=bcs)
    Hsigma, Hu, Hlambdar = w.split()

    exact = Function(U)
    exact.interpolate(cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z))

    # sigma = -grad(u)
    sig_error = sqrt(assemble(dot(Hsigma + grad(exact),
                                  Hsigma + grad(exact))*dx))
    u_error = sqrt(assemble((Hu - exact)*(Hu - exact)*dx))

    assert sig_error < 1e-11
    assert u_error < 1e-11


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
