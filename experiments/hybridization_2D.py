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
    mesh = UnitSquareMesh(2 ** resolution, 2 ** resolution)
    RT = FunctionSpace(mesh, "RT", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define the variational forms
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx - 42 * dot(tau, n)*ds

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    solve(a == L, w,
          solver_parameters={'mat_type': 'matfree',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization_fieldsplit_schur_fact_type': 'lower',
                             'hybridization_ksp_rtol': 1e-8,
                             'hybridization_pc_type': 'lu',
                             'hybridization_ksp_type': 'preonly',
                             'hybridization_projector_tolerance': 1e-14})
    sigma_h, u_h = w.split()

    File("hybrid-2d.pvd").write(sigma_h, u_h)

test_slate_hybridization(degree=1, resolution=5)
