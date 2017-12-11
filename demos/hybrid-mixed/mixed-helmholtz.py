"""This demonstration generates the relevant code from the Slate
expressions. Note that this is for code only; it's not solving any
particular PDE with given data.

See the main hybrid-mixed folder for an actual solution to a
mixed system using hybridization and static condensation.
"""
from firedrake import *

mesh = UnitSquareMesh(8, 8, quadrilateral=False)
RT = FiniteElement("RT", triangle, 1)
DG = FiniteElement("DG", triangle, 0)
T = FiniteElement("HDiv Trace", triangle, 0)
U = FunctionSpace(mesh, RT)
V = FunctionSpace(mesh, DG)
M = FunctionSpace(mesh, T)
W = U * V

n = FacetNormal(mesh)
p0 = Function(V)
g = Function(V)
f = Function(V)
mu = Function(V)
c = Function(V)
u, p = TrialFunctions(W)
w, phi = TestFunctions(W)
gamma = TestFunction(M)

A = Tensor(dot(w, mu*u)*dx - div(w)*p*dx
           + phi*div(u)*dx + phi*c*p*dx)
C = Tensor(gamma*dot(u, n)*dS + gamma*dot(u, n)*ds(2))
F = Tensor(-dot(w, n)*p0*ds(1) + phi*f*dx)
G = Tensor(gamma*g*ds(2))

S = C * A.inv * C.T
E = C * A.inv * F - G
Smat = assemble(S).force_evaluation()
Evec = assemble(E).dat.data
