from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitCubedSphereMesh(refinement_level=6)
mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

degree = 0
RT_elt = FiniteElement("RTCF", quadrilateral, degree + 1)
V = FunctionSpace(mesh, RT_elt)
U = FunctionSpace(mesh, "DG", degree)
W = V * U

f = Function(U)
expr = Expression("x[0]*x[1]*x[2]")
f.interpolate(expr)

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

a = (dot(sigma, tau) - div(tau) * u + v * div(sigma) + u * v) * dx
L = f * v * dx
w = Function(W)

solver_parameters = {'mat_type': 'matfree',
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.HybridizationPC',
                     'trace_ksp_rtol': 1e-13,
                     'trace_pc_type': 'lu',
                     'trace_ksp_type': 'preonly',
                     'trace_ksp_monitor_true_residual': True}

solve(a == L, w, solver_parameters=solver_parameters)
sigma_h, u_h = w.split()

File("SphereHelmholtz.pvd").write(sigma_h, u_h)
