from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitSquareMesh(2, 2)

V = FunctionSpace(mesh, "RT", 1)
f1 = Function(V)
f2 = Function(V)

# Apply a strong condition at the top of the square domain
bc1 = DirichletBC(V, Constant((0.0, 10.0)), 4)

bc1.apply(f1)

bc2 = DirichletBC(V, Expression(("0.0", "10.0")), 4)
bc2.apply(f2)

print(f1.dat.data)
print(f2.dat.data)
