from __future__ import absolute_import, print_function, division
from firedrake import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "RT", 1)
U = FunctionSpace(mesh, "DG", 0)
T = FunctionSpace(mesh, "HDiv Trace", 0)

W = V * U

w, phi = TestFunctions(W)
u, rho = TrialFunctions(W)
l0 = TrialFunction(T)
dl = TestFunction(T)

n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
rhobar = Function(U).interpolate(x[0] ** 2 + x[1] ** 2)
rhobar_tr = Function(T)
rbareqn = (l0('+') - avg(rhobar))*dl('+')*dS + (l0 - rhobar)*dl*ds
rhobar_prob = LinearVariationalProblem(lhs(rbareqn), rhs(rbareqn), rhobar_tr)
rhobar_solver = LinearVariationalSolver(rhobar_prob,
                                        solver_parameters={'ksp_type': 'preonly',
                                                           'pc_type': 'bjacobi',
                                                           'pc_sub_type': 'lu'})
rhobar_solver.solve()

Aeqn = (inner(w, u)*dx
        - div(w)*rho*dx
        + (phi*rho - inner(grad(phi), u)*rhobar)*dx
        + rhobar_tr*inner(phi*u, n)*dS)

Aop = Tensor(lhs(Aeqn))
Arhs = rhs(Aeqn)

#  (A K)(U) = (U_r)
#  (L 0)(l)   (0  )

dl = dl('+')
l0 = l0('+')

K = Tensor(inner(w, n)*l0*dS)
L = Tensor(dl*inner(u, n)*dS)

assemble(L * Aop.inv * K).force_evaluation()
