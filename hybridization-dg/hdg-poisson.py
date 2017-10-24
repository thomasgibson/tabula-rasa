from firedrake import *

degree = 1
res = 2
mesh = UnitSquareMesh(2**res, 2**res)
x = SpatialCoordinate(mesh)
U = VectorFunctionSpace(mesh, "DG", degree)
V = FunctionSpace(mesh, "DG", degree)
M = FunctionSpace(mesh, "HDiv Trace", degree)

# Stability parameter
tau = 1.0

W = U * V
q, u = TrialFunctions(W)
w, v = TestFunctions(W)
lambdar = TrialFunction(M)
gammar = TestFunction(M)
bcs = DirichletBC(M, Constant(0.0), "on_boundary")

f = Function(V).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

# The H-DG system for the Poisson equation is a 3x3 system of the form:
#
# |A Bt Ct| |q|         |0|
# |B D  Gt| |u|       = |F|
# |-C -G H| |lambdar|   |0| <- Strong BCs would be imposed here

# Here we set up the block:
# |A Bt|
# |B D |
# and RHS: |F|
n = FacetNormal(mesh)
Atilde = Tensor(inner(w, q)*dx - div(w)*u*dx +
                inner(grad(v), q)*dx + inner(v, tau*u)*dS +
                inner(v, dot(q, n))*dS)
F = Tensor(-inner(v, f)*dx)

# Now we create the subblocks:
# |Ct|
# |Gt|
# and
# |-C -G|
CGt = Tensor(inner(w, lambdar*n)*dS +
             inner(v, -tau*lambdar)*dS)
CG = Tensor(-inner(gammar, dot(q, n) + tau*u)*dS)

# Finally, we define the trace mass term: H
H = Tensor(inner(gammar, tau*lambdar)*dS)

# We now statically condense to arrive at a SPD system for lambdar
K = H + CG * Atilde.inv * CGt
R = CG * Atilde.inv * F

# Solve for the multipliers
lambda_h = Function(M)
Kmat = assemble(K, bcs=bcs)
Rvec = assemble(R)
solve(Kmat, lambda_h, Rvec)
