from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitSquareMesh(2, 2, quadrilateral=False)
n = FacetNormal(mesh)

degree = 1
V = FunctionSpace(mesh, "RT", degree)
U = FunctionSpace(mesh, "DG", degree - 1)
W = V * U

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = (dot(u, v) + div(v)*p + q*div(u))*dx

x = SpatialCoordinate(mesh)
f = Function(U).assign(0)

L = -f*q*dx + 42*dot(v, n)*ds(4)

bcs = [DirichletBC(W.sub(0), Expression(("0", "0")), (1, 2))]
L = assemble(L)
y = Function(W)
for bc in bcs:
    bc.apply(y)

rhs = assemble(assemble(action(a, y)) - L)

for bc in bcs:
    bc.apply(rhs)
