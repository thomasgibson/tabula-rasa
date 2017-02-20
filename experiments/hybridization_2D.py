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


def test_slate_hybridization(degree, resolution, quads=False):
    # Create a mesh
    mesh = UnitSquareMesh(2 ** resolution, 2 ** resolution,
                          quadrilateral=quads)

    # Define relevant function spaces
    if quads:
        eRT = FiniteElement("RTCF", quadrilateral, degree + 1)
    else:
        eRT = FiniteElement("RT", triangle, degree + 1)

    RT = FunctionSpace(mesh, eRT)
    DG = FunctionSpace(mesh, "DG", degree)

    W = RT * DG

    # Define the trial and test functions
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    expr = sin(10*(x*x + y*y))/10
    # expr = (1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2)
    f.interpolate(expr)

    # Define finite element variational forms
    Mass_v = dot(sigma, tau) * dx
    Mass_p = u * v * dx
    Div = div(sigma) * v * dx
    Div_adj = div(tau) * u * dx
    a = Mass_v - Div_adj + Div + Mass_p
    L = f * v * dx

    solver_parameters = {'mat_type': 'matfree',
                         'pc_type': 'python',
                         'pc_python_type': 'firedrake.HybridizationPC',
                         'trace_ksp_rtol': 1e-8,
                         'trace_pc_type': 'lu',
                         'trace_ksp_type': 'preonly',
                         'ksp_monitor': True,
                         'trace_ksp_monitor': True}
    w = Function(W)
    solve(a == L, w, solver_parameters=solver_parameters)
    sigma_h, u_h = w.split()

    File("hybrid-2d.pvd").write(sigma_h, u_h)

test_slate_hybridization(degree=0, resolution=8, quads=True)
