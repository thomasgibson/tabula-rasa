from __future__ import absolute_import, print_function, division

from firedrake import *


def poisson_sphere_hybridization(degree, refinement):

    mesh = UnitIcosahedralSphereMesh(refinement_level=refinement)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    RT_elt = FiniteElement("RT", triangle, degree + 1)
    V = FunctionSpace(mesh, BrokenElement(RT_elt))
    U = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)
    W = V * U
    n = FacetNormal(mesh)

    f = Function(U)
    expr = Expression("x[0]*x[1]*x[2]")
    f.interpolate(expr)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    gammar = TestFunction(T)

    mass_v = dot(sigma, tau) * dx
    divgrad = div(sigma) * v * dx
    divgrad_adj = div(tau) * u * dx
    local_trace = gammar('+') * dot(sigma, n) * dS
    L = f*v*dx

    bcs = DirichletBC(T, Constant(0.0), "on_boundary")

    A = Tensor(mass_v + divgrad - divgrad_adj)
    K = Tensor(local_trace)
    Schur = -K * A.inv * K.T

    F = Tensor(L)
    RHS = - K * A.inv * F

    S = assemble(Schur, bcs=bcs)
    E = assemble(RHS)

    lambda_sol = Function(T)
    nullsp = VectorSpaceBasis(constant=True)
    solve(S, lambda_sol, E, nullspace=nullsp,
          solver_parameters={'pc_type': 'lu',
                             'ksp_type': 'cg'})

    sigma = TrialFunction(V)
    tau = TestFunction(V)
    u = TrialFunction(U)
    v = TestFunction(U)

    A_v = Tensor(dot(sigma, tau) * dx)
    B = Tensor(div(sigma) * v * dx)
    K = Tensor(dot(sigma, n) * gammar('+') * dS)
    F = Tensor(f * v * dx)

    # SLATE expression for pressure recovery:
    u_sol = (B * A_v.inv * B.T).inv * (F + B * A_v.inv * K.T * lambda_sol)
    u_h = assemble(u_sol)

    # SLATE expression for velocity recovery
    sigma_sol = A_v.inv * (B.T * u_h - K.T * lambda_sol)
    sigma_h = assemble(sigma_sol)

    new_sigma_h = project(sigma_h, FunctionSpace(mesh, RT_elt))

    File("SpherePoisson-hybrid.pvd").write(new_sigma_h, u_h)

poisson_sphere_hybridization(degree=0, refinement=4)
