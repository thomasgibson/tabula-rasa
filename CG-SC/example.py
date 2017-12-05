from firedrake import *

# Test assembly of Schur-complement using indexed tensors.
# This generates the code in the examples section featuring
# the Slate language.
mesh = UnitSquareMesh(8, 8)
element = FiniteElement("Lagrange", triangle, 3)
Vo = FunctionSpace(mesh, element["interior"])
Vf = FunctionSpace(mesh, element["facet"])
W = Vo * Vf
vo, vf = TestFunctions(W)
uo, uf = TrialFunctions(W)
u = uo + uf
v = vo + vf
a = (dot(grad(v), grad(u)) + inner(v, u))*dx
A = Tensor(a)
S = A[1, 1] - A[1, 0] * A[0, 0].inv * A[0, 1]
mat = assemble(S)
mat.force_evaluation()
