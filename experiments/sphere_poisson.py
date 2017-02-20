from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitIcosahedralSphereMesh(refinement_level=4)
mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

degree = 0
RT_elt = FiniteElement("RT", triangle, degree + 1)
V = FunctionSpace(mesh, RT_elt)
U = FunctionSpace(mesh, "DG", degree)
W = V * U

f = Function(U)
expr = Expression("x[0]*x[1]*x[2]")
f.interpolate(expr)

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

a = (dot(sigma, tau) - div(tau) * u + v * div(sigma)) * dx
L = f * v * dx
w = Function(W)

# nullspace = VectorSpaceBasis(constant=True)
# solver_parameters = {'mat_type': 'matfree',
#                      'pc_type': 'python',
#                      'pc_python_type': 'firedrake.HybridizationPC',
#                      'trace_ksp_rtol': 1e-13,
#                      'trace_pc_type': 'lu',
#                      'trace_ksp_type': 'preonly',
#                      'ksp_monitor': True}

nullspace = MixedVectorSpaceBasis(W, [W[1], VectorSpaceBasis(constant=True)])
solver_parameters = {'pc_type': 'fieldsplit',
                     'pc_fieldsplit_type': 'schur',
                     'fieldsplit_0_pc_type': 'bjacobi',
                     'fieldsplit_0_sub_pc_type': 'ilu',
                     'fieldsplit_1_pc_type': 'none',
                     'pc_fieldsplit_schur_fact_type': 'FULL',
                     'fieldsplit_0_ksp_max_it': 100}
solve(a == L, w, nullspace=nullspace, solver_parameters=solver_parameters)
sigma_h, u_h = w.split()

File("SpherePoisson-hybrid.pvd").write(sigma_h, u_h)
