from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitSquareMesh(2, 2)

RT = FiniteElement("RT", triangle, 2)
BRT = BrokenElement(RT)

U = FunctionSpace(mesh, RT)
Ud = FunctionSpace(mesh, BRT)

x = SpatialCoordinate(mesh)
fct = as_vector([x[0] ** 2, x[1] ** 2])
u0 = Function(U).project(fct)
u = Function(Ud).project(u0)

ans = Function(U)

count_code = """
for (int i; i<count.dofs; ++i) {
    count[i][0] += 1.0;
}
"""

kernel_code = """
for (int i; i<ubar.dofs; ++i) {
    ubar[i][0] += u[i][0]/one[i][0];
}
"""

One = Function(U)

par_loop(count_code, dx, {"count":(One, INC)})
par_loop(kernel_code, dx, {"ubar":(ans, INC), "u":(u, READ), "one":(One, READ)})
