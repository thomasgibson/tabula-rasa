from __future__ import absolute_import, print_function, division

from firedrake import *

r = 32
mesh = UnitSquareMesh(r, r)
x = SpatialCoordinate(mesh)
V = TensorFunctionSpace(mesh, "CG", 1)
V_ho = TensorFunctionSpace(mesh, "CG", 5)

# bc1 = Expression([["0.5", "0.5"], ["0.5", "0.5"]])
bc2 = Expression([["-0.5", "-0.5"], ["-0.5", "-0.5"]])

bc1 = Constant(((0.5, 0.5), (0.5, 0.5)))

bcs = [DirichletBC(V_ho, bc1, (1, 3)),
       DirichletBC(V_ho, bc2, (2, 4))]

expr = as_tensor([[cos(x[0]*pi*2)*sin(x[1]*pi*2),
                   cos(x[0]*pi*2)*sin(x[1]*pi*2)],
                  [cos(x[0]*pi*2)*sin(x[1]*pi*2),
                   cos(x[0]*pi*2)*sin(x[1]*pi*2)]])

v = Function(V).project(expr)
vhbc = Function(V_ho, name="With manual solve")
p = TrialFunction(V_ho)
q = TestFunction(V_ho)
a = inner(p, q)*dx
f = Function(V_ho).interpolate(expr)
L = inner(f, q)*dx
solve(a == L, vhbc, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                "pc_type": "lu"})
vp = Function(V_ho, name="With manual bcs applied").interpolate(expr)
for bc in bcs:
    bc.apply(vp)

vpbc = Function(V_ho, name="With projector")
proj = Projector(v, vpbc, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "lu"}).project()
print(errornorm(vpbc, vhbc))
print(errornorm(vp, vhbc))
