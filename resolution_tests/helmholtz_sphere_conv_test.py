from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

from firedrake import *


def RT0DGO0(res):
    mesh = UnitIcosahedralSphereMesh(refinement_level=res)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

    degree = 0
    V = FunctionSpace(mesh, "RT", degree + 1)
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
    _, u_h = w.split()
    true_u = Function(U).interpolate(Expression("x[0]*x[1]*x[2]/13.0"))

    error = errornorm(true_u, u_h, degree_rise=0)
    return error


def BDM1DGO0(res):
    mesh = UnitIcosahedralSphereMesh(refinement_level=res)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

    degree = 0
    V = FunctionSpace(mesh, "BDM", degree + 1)
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
    _, u_h = w.split()
    true_u = Function(U).interpolate(Expression("x[0]*x[1]*x[2]/13.0"))

    error = errornorm(true_u, u_h, degree_rise=0)
    return error

ref_levels = range(2, 7)
l2errorsRT = np.asarray([RT0DGO0(r) for r in ref_levels])
l2errorsBDM = np.asarray([BDM1DGO0(r) for r in ref_levels])

print(np.log2(l2errorsRT[:-1]/l2errorsRT[1:]))
print(np.log2(l2errorsBDM[:-1]/l2errorsBDM[1:]))

plt.semilogy(ref_levels, l2errorsRT)
plt.semilogy(ref_levels, l2errorsBDM)
plt.show()
