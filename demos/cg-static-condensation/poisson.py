"""A simple demonstration showing the use of Slate for performing static
condensation on a Poisson problem with strong conditions on the x=0, x=1
sides of a unit square.

Sections of the generated code from this example is displayed in the
main manuscript.
"""
from firedrake import *
import numpy as np

mesh = UnitSquareMesh(8, 8, quadrilateral=False)
degree = 8
element = FiniteElement("Lagrange", triangle, degree)
V = FunctionSpace(mesh, element)
Vo = FunctionSpace(mesh, element["interior"])
Vf = FunctionSpace(mesh, element["facet"])
V = Vo * Vf

vo, vf = TestFunctions(V)
uo, uf = TrialFunctions(V)

v = vo + vf
u = uo + uf

a = dot(grad(u), grad(v))*dx
L = Constant(0.0)*v*dx

bcs = [DirichletBC(Vf, 0, 3),
       DirichletBC(Vf, 42, 4)]

A = Tensor(a)
b = Tensor(L)

S = A[1, 1] - A[1, 0] * A[0, 0].inv * A[0, 1]
E = b[1] - A[1, 0] * A[0, 0].inv * b[0]

Smat = assemble(S, bcs=bcs)
E = assemble(E)

xf = Function(Vf)
solve(Smat, xf, E, solver_parameters={"ksp_type": "preonly",
                                      "pc_type": "lu"})

Xf = AssembledVector(xf)
xo = assemble(A[0, 0].inv * (b[0] - A[0, 1] * Xf))

V = FunctionSpace(mesh, "CG", degree)
u_h = Function(V, name="Computed solution")
sol = Function(V, name="Analytic").interpolate(Expression("42*x[1]"))

# Custom kernel to join the two data sets together.
# NOTE: This is done automatically in the PC version of this method.
dim = V.finat_element._element.ref_el.get_dimension()
offset = V.finat_element.entity_dofs()[dim][0][0]
args = (Vo.finat_element.space_dimension(), np.prod(Vo.shape),
        offset,
        Vf.finat_element.space_dimension(), np.prod(Vf.shape))

join = """
for (int i=0; i<%d; ++i){
for (int j=0; j<%d; ++j){
x[i + %d][j] = x_int[i][j];
}
}

for (int i=0; i<%d; ++i){
for (int j=0; j<%d; ++j){
x[i][j] = x_facet[i][j];
}
}""" % args

par_loop(join, dx, {"x_int": (xo, READ),
                    "x_facet": (xf, READ),
                    "x": (u_h, WRITE)})
File("ex-poisson.pvd").write(u_h, sol)
