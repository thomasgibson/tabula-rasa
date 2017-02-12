from __future__ import absolute_import, print_function, division

from firedrake import *


def test_slate_hybridization_extr(degree, resolution, layers):
    base = UnitSquareMesh(2 ** resolution, 2 ** resolution,
                          quadrilateral=False)
    mesh = ExtrudedMesh(base, layers=layers, layer_height=0.025)

    RT_elt = FiniteElement("RT", triangle, degree + 1)
    DG = FiniteElement("DG", interval, degree)
    DGh = FiniteElement("DG", triangle, degree)
    CG = FiniteElement("CG", interval, degree + 1)
    elem = EnrichedElement(HDiv(TensorProductElement(RT_elt, DG)),
                           HDiv(TensorProductElement(DGh, CG)))
    V = FunctionSpace(mesh, elem)
    U = FunctionSpace(mesh, "DG", degree)
    W = V * U
    x, y, z = SpatialCoordinate(mesh)

    f = Function(U)
    expr = (1+12*pi*pi)*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z)
    f.interpolate(expr)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    mass_v = dot(sigma, tau) * dx
    mass_p = u * v * dx
    divgrad = div(sigma) * v * dx
    divgrad_adj = div(tau) * u * dx
    L = f*v*dx
    a = mass_v + divgrad - divgrad_adj + mass_p

    w = Function(W)
    solve(a == L, w, solver_parameters={'ksp_rtol': 1e-13})

    sigma_h, u_h = w.split()

    File("ref3D-hybrid.pvd").write(sigma_h, u_h)

test_slate_hybridization_extr(degree=0, resolution=4, layers=1)
