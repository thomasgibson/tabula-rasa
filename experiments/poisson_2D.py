from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitSquareMesh(2 ** 9, 2 ** 9, quadrilateral=False)

degree = 0
V = FunctionSpace(mesh, "BDM", degree + 1)
U = FunctionSpace(mesh, "DG", degree)
W = V * U

f = Function(U)
x, y = SpatialCoordinate(mesh)
expr = -2.0 * (x - 1) * x - 2.0 * (y - 1) * y
f.interpolate(expr)

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

a = (dot(sigma, tau) - div(tau) * u + v * div(sigma)) * dx
L = f * v * dx
w = Function(W)

solver_parameters = {'mat_type': 'matfree',
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.HybridizationPC',
                     'trace_ksp_rtol': 1e-10,
                     'trace_pc_type': 'lu',
                     'trace_ksp_type': 'preonly',
                     'ksp_monitor': True,
                     'trace_ksp_monitor': True}
solve(a == L, w, solver_parameters=solver_parameters)
sigma_h, u_h = w.split()

File("Poisson2D.pvd").write(simga_h, u_h)
