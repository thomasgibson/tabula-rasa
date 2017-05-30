from __future__ import absolute_import, print_function, division

from firedrake import *

r = 4
base_mesh = UnitSquareMesh(r, r)
mesh = ExtrudedMesh(base_mesh, layers=r, layer_height=1.0/r)

V = FunctionSpace(mesh, "CG", 1)

v = Function(V)
w = Function(V)

ids = tuple(map(int, mesh.topology.exterior_facets.unique_markers))

print(ids)

bcs = [DirichletBC(V, Constant(1.2), (1, 2, 3, 4)),
       DirichletBC(V, Constant(1.2), "top"),
       DirichletBC(V, Constant(1.2), "bottom")]

bcs2 = DirichletBC(V, Constant(1.2), "on_boundary")

for bc in bcs:
    bc.apply(v)

bcs2.apply(w)

print(errornorm(v, w))
print(v.dat.data)
print(w.dat.data)
