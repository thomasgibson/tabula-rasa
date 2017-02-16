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

# Create a mesh
mesh = UnitSquareMesh(120, 120)

# Define relevant function spaces
degree = 2
RT = FiniteElement("RT", triangle, degree)
BRT = FunctionSpace(mesh, RT)
DG = FunctionSpace(mesh, "DG", degree - 1)

W = BRT * DG

# Define the trial and test functions
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

# Define the source function
f = Function(DG)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

# Define finite element variational forms
a = (dot(sigma, tau) + u * v + div(sigma) * v - div(tau) * u) * dx
L = f * v * dx
w = Function(W)
parameters = {"mat_type": "matfree",
              "pc_type": "python",
              "pc_python_type": "firedrake.HybridizationPC",
              "trace_pc_type": "lu",
              "trace_ksp_type": "preonly",
              "trace_ksp_monitor_true_residual": True,
              "ksp_monitor_true_residual": True}
solve(a == L, w, solver_parameters=parameters)

u, p = w.split()
u.rename("Velocity")
p.rename("Pressure")

File("pchybrid.pvd").write(u, p)
