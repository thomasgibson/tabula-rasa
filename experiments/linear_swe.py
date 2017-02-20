from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitIcosahedralSphereMesh(refinement_level=4)
mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

degree = 0
elem = FiniteElement("RT", triangle, degree + 1)
V = FunctionSpace(mesh, elem)
DG = FunctionSpace(mesh, "DG", degree)

W = V * DG

f = Constant(1.0)

# Initial conditions for fields
nu_0 = Function(DG)
nu_0.interpolate(Expression("sin(4*pi*x[0])*sin(2*pi*x[1])"))
u_n = Function(V)
nu_n = Function(DG).assign(nu_0)

T = 0.5
t = 0
dt = 0.0025

output = File("output.pvd")
output.write(u_n, nu_n)

while t < T:
    u, nu = TrialFunctions(W)
    w, alpha = TestFunctions(W)

    a = (dot(w, u) - 0.5*dt*nu*div(w))*dx \
        + alpha*nu*dx + 0.5*dt*alpha*div(u)*dx

    L = 0.5*dt*nu_n*div(w)*dx - 0.5*dt*div(u_n)*alpha*dx

    s = Function(W)
    solve(a == L, s, solver_parameters={'mat_type': 'matfree',
                                        'pc_type': 'python',
                                        'pc_python_type': 'firedrake.HybridizationPC',
                                        'trace_ksp_rtol': 1e-8,
                                        'trace_pc_type': 'lu',
                                        'trace_ksp_type': 'preonly'})

    new_u, new_nu = s.split()
    nu_n.assign(new_nu)
    u_n.assign(new_u)
    t += dt
    print(t)
    output.write(u_n, nu_n, time=t)
