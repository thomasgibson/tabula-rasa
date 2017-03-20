from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

from firedrake import *


def RT0DGO0(res, degree):
    mesh = UnitSquareMesh(2 ** res, 2 ** res, quadrilateral=False)

    V = FunctionSpace(mesh, "RT", degree + 1)
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
    _, u_h = w.split()
    true_u = Function(U).interpolate(x*(1 - x)*y*(1 - y))

    error = errornorm(true_u, u_h, degree_rise=0)
    return error


def BDM1DGO0(res, degree):
    mesh = UnitSquareMesh(2 ** res, 2 ** res, quadrilateral=False)

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
    _, u_h = w.split()
    true_u = Function(U).interpolate(x*(1 - x)*y*(1 - y))

    error = errornorm(true_u, u_h, degree_rise=0)
    return error
    return error

ref_levels = range(1, 8)
degree = 0
l2errorsRT = np.asarray([RT0DGO0(r, degree) for r in ref_levels])
l2errorsBDM = np.asarray([BDM1DGO0(r, degree) for r in ref_levels])
conv_rate_RT = np.log2(l2errorsRT[:-1]/l2errorsRT[1:])[-1]
conv_rate_BDM = np.log2(l2errorsBDM[:-1]/l2errorsBDM[1:])[-1]
text_RT = "Rate for RT0-DG0: %f" % conv_rate_RT
text_BDM = "Rate for BDM1-DG0: %f" % conv_rate_BDM

print(np.log2(l2errorsRT[:-1]/l2errorsRT[1:]))
print(np.log2(l2errorsBDM[:-1]/l2errorsBDM[1:]))

fig = plt.figure()
ax = fig.add_subplot(111)

res = [2 ** r for r in ref_levels]

ax.loglog(res, l2errorsRT, color="r", marker="o",
          linestyle="-", linewidth="2",
          label="RT0-DG0")
ax.loglog(res, l2errorsBDM, color="b", marker="^",
          linestyle="-", linewidth="2",
          label="BDM1-DG0")
ax.grid(True)
ax.annotate(text_RT + '\n' + text_BDM, xy=(res[-1], 1e-2))
plt.title("Resolution Test for RT0-DG0 and BDM1-DG0")
plt.xlabel("Mesh resolution $2^r$")
plt.ylabel("$L_2$ error in pressure approximation")
plt.xlim(1, 2**(ref_levels[-1]+1))
plt.ylim(1e-8, 1e-1)
plt.gca().invert_xaxis()
plt.legend(loc=1)
plt.show()
