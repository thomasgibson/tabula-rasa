"""Solve a mixed Helmholtz problem

sigma + grad(u) = 0,
u + div(sigma) = f,

using hybridisation with SLATE performing the forward elimination and
backwards reconstructions. The corresponding finite element variational
problem is:

dot(sigma, tau)*dx - u*div(tau)*dx + lambdar*dot(tau, n)*dS = 0
div(sigma)*v*dx + u*v*dx = f*v*dx
gammar*dot(sigma, n)*dS = 0

for all tau, v, and gammar.

This is solved using broken Raviart-Thomas elements of degree k for
(sigma, tau), discontinuous Galerkin elements of degree k - 1
for (u, v), and HDiv-Trace elements of degree k - 1 for (lambdar, gammar).

The forcing function is chosen as:

(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2),

which reproduces the known analytical solution:

sin(x[0]*pi*2)*sin(x[1]*pi*2)
"""

from __future__ import absolute_import, print_function, division

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


def l2_error(a, b):
    return sqrt(assemble(inner(a - b, a - b)*dx))


def test_slate_hybridization(degree, resolution):
    # Create a mesh
    mesh = UnitSquareMesh(2 ** resolution, 2 ** resolution)
    RT_elt = FiniteElement("RT", triangle, degree + 1)
    broken_RT = BrokenElement(RT_elt)
    RT_d = FunctionSpace(mesh, broken_RT)
    DG = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)
    W = RT_d * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    gammar = TestFunction(T)
    n = FacetNormal(mesh)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define the variational forms
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx
    local_trace = gammar('+') * dot(sigma, n) * dS

    bcs = [DirichletBC(T, Constant(0.0), "on_boundary")]

    A = Tensor(a)
    K = Tensor(local_trace)
    Schur = K * A.inv * K.T

    F = Tensor(L)
    RHS = K * A.inv * F

    S = assemble(Schur, bcs=bcs)
    E = assemble(RHS)

    lambda_sol = Function(T)
    solve(S, lambda_sol, E, solver_parameters={'pc_type': 'lu',
                                               'ksp_type': 'cg'})

    sigma = TrialFunction(RT_d)
    tau = TestFunction(RT_d)
    u = TrialFunction(DG)
    v = TestFunction(DG)

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

    # Post processing
    DG_k1 = FunctionSpace(mesh, "DG", degree + 1)
    ustar = Function(DG_k1)
    u = TrialFunction(DG_k1)
    v = TestFunction(DG)
    gammar = TestFunction(T)
    if degree < 2:
        r = u*gammar('+')*dS + u*gammar*ds
        Lr = lambda_sol*gammar('+')*dS + lambda_sol*gammar*ds
        R = Tensor(r)
        LR = Tensor(Lr)
        assemble(R.inv * LR, tensor=ustar)
    else:
        raise ValueError

    exact_u = sin(x*pi*2)*sin(y*pi*2)
    err = l2_error(exact_u, ustar)
    # File("helmholtz_2D-hybrid.pvd").write(new_sigma_h, ustar)
    return err


errs = []
ref_levels = range(3, 8)
d = 0
for i in ref_levels:
    err = test_slate_hybridization(degree=d, resolution=i)
    errs.append(err)

errRT_u = np.asarray(errs)

fig = plt.figure()
ax = fig.add_subplot(111)

res = [2 ** r for r in ref_levels]
dh = np.asarray(res)
k = d + 2
dh_arry = dh ** k
dh_arry = 0.001 * dh_arry

orange = '#FF6600'
lw = '5'
ms = 15

if k == 1:
    dhlabel = '$\propto \Delta x$'
else:
    dhlabel = '$\propto \Delta x^%d$' % k

ax.loglog(res, errRT_u, color='r', marker='o',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$DG_1$ $p_h$')
ax.loglog(res, dh_arry[::-1], color='k', linestyle=':',
          linewidth=lw, label=dhlabel)
ax.grid(True)
plt.title("Resolution test for lowest order H-RT method")
plt.xlabel("Mesh resolution in all spatial directions $2^r$")
plt.ylabel("$L^2$-error against projected exact solution")
plt.gca().invert_xaxis()
plt.legend(loc=2)
font = {'family': 'normal',
        'weight': 'bold',
        'size': 28}
plt.rc('font', **font)
plt.show()
