from __future__ import absolute_import, print_function, division

from firedrake import *

qflag = False
degree = 1

mesh = UnitSquareMesh(2, 2, quadrilateral=qflag)
n = FacetNormal(mesh)

if qflag:
    RT = FiniteElement("RTCF", quadrilateral, degree)
    DG = FiniteElement("DQ", quadrilateral, degree - 1)
    Te = FiniteElement("HDiv Trace", quadrilateral, degree - 1)

else:
    RT = FiniteElement("RT", triangle, degree)
    DG = FiniteElement("DG", triangle, degree - 1)
    Te = FiniteElement("HDiv Trace", triangle, degree - 1)

V = FunctionSpace(mesh, BrokenElement(RT))
U = FunctionSpace(mesh, DG)
T = FunctionSpace(mesh, Te)

W = V * U

u, p = TrialFunctions(W)
v, q = TestFunctions(W)
gammar = TestFunction(T)

t = Function(T).assign(3.14)

a_dx = (dot(u, v) + div(v)*p + q*div(u) + p*q)*dx
a_dS = t*dot(u*q, n)*dS
a = a_dx + a_dS

tr = gammar('+')*dot(u, n)*dS

A = Tensor(a)
K = Tensor(tr)

S = assemble(K * A.inv * K.T)
S.force_evaluation()
