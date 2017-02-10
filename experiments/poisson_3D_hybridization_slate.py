from __future__ import absolute_import, print_function, division

from firedrake import *


def test_3d_poisson_hybridization(degree, resolution, layers):
    base = UnitSquareMesh(2 ** resolution, 2 ** resolution,
                          quadrilateral=False)
    mesh = ExtrudedMesh(base, layers=layers, layer_height=0.125)

    RT_elt = FiniteElement("RT", triangle, degree + 1)
    DG = FiniteElement("DG", interval, degree)
    DGh = FiniteElement("DG", triangle, degree)
    CG = FiniteElement("CG", interval, degree + 1)
    elem = EnrichedElement(HDiv(TensorProductElement(RT_elt, DG)),
                           HDiv(TensorProductElement(DGh, CG)))
    product_elt = BrokenElement(elem)
    V = FunctionSpace(mesh, product_elt)
    U = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", (degree, degree))
    W = V * U
    n = FacetNormal(mesh)
    x, y, z = SpatialCoordinate(mesh)

    f = Function(U)
    expr = 120*pi*sin(10*(x*x + y*y + z*z))
    f.interpolate(expr)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    gammar = TestFunction(T)

    mass_v = dot(sigma, tau) * dx
    divgrad = div(sigma) * v * dx
    divgrad_adj = div(tau) * u * dx
    local_trace = (gammar('+') * dot(sigma, n) * dS_h +
                   gammar('+') * dot(sigma, n) * dS_v)
    L = f*v*dx

    bcs = [DirichletBC(T, Constant(0.0), "on_boundary"),
           DirichletBC(T, Constant(0.0), "top"),
           DirichletBC(T, Constant(0.0), "bottom")]

    A = Tensor(mass_v + divgrad - divgrad_adj)
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
    B = Tensor(div(sigma) * v * dx)
    K = Tensor(dot(sigma, n) * gammar('+') * dS_h +
               dot(sigma, n) * gammar('+') * dS_v)
    F = Tensor(f * v * dx)

    # SLATE expression for pressure recovery:
    u_sol = (B * A_v.inv * B.T).inv * (F + B * A_v.inv * K.T * lambda_sol)
    u_h = assemble(u_sol)

    # SLATE expression for velocity recovery
    sigma_sol = A_v.inv * (B.T * u_h - K.T * lambda_sol)
    sigma_h = assemble(sigma_sol)

    new_sigma_h = project(sigma_h, FunctionSpace(mesh, elem))

    File("Poisson-3D-hybrid.pvd").write(new_sigma_h, u_h)

test_3d_poisson_hybridization(degree=0, resolution=10, layers=1)
