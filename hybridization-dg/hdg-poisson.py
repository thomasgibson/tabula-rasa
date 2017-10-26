"""This module solves the Dirichlet Poisson problem:


-div(grad(u)) = f in [0, 1]^2

u = 0 on the boundary (all sides of the domain),


in a mixed formulation:


sigma = -grad(u),

div(sigma) = f

u = 0 on the boundary,


using a hybridized discontinuous Galerkin (HDG) method.
The method is implemented using the Slate DSL for
constructing the Schur complement system and local recovery.
The weak formulation for the HDG method reads as follows:

find sigma, u, lambda in (DG_k)^2 x DG_k, DG_Trace_k such that


dot(tau, sigma)*dx - div(tau)*u*dx + dot(tau, lambda*n)*(dS + ds) = 0,

dot(grad(v), sigma)*dx + v*(dot(sigma, n) + T*(u - lambda))*(dS + ds) = v*f*dx,

-gamma*(dot(sigma, n) + T*(u - lambda))*(dS + ds) = 0 (no Neumman BCs)


for all tau, v, gamma in (DG_k)^2 x DG_k, DG_Trace_k, where T is a non-negative
stability parameter.
"""
from firedrake import *

degree = 2
res = 1
mesh = UnitSquareMesh(2**res, 2**res)
x = SpatialCoordinate(mesh)
U = VectorFunctionSpace(mesh, "DG", degree)
V = FunctionSpace(mesh, "DG", degree)
T = FunctionSpace(mesh, "HDiv Trace", degree)

# Stability parameter
tau = 1.0

W = U * V
q, u = TrialFunctions(W)
w, v = TestFunctions(W)
lambdar = TrialFunction(T)
gammar = TestFunction(T)
bcs = DirichletBC(T, Constant(0.0), "on_boundary")

f = Function(V).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

# The H-DG system for the Poisson equation is a 3x3 system of the form:
#
# |A Bt Ct| |q|         |0|
# |B D  Gt| |u|       = |F|
# |-C -G H| |lambdar|   |0| <- Strong BCs would be imposed here

# Here we set up the block:
# |A Bt|
# |B D |
# as M, and RHS:
# |0|
# |F|
# as Fbar
n = FacetNormal(mesh)
M = Tensor(dot(w, q)*dx - inner(div(w), u)*dx +
           inner(v, div(q))*dx + inner(v, dot(q, n))*dS +
           inner(v, tau*u)*dS)
Fbar = Tensor(-inner(v, f)*dx)

# Now we create the subblocks:
# |Ct|
# |Gt|
# as N and
# |-C -G|
# as P
N = Tensor(inner(w, lambdar*n)*dS -
           inner(v, tau*lambdar)*dS)
P = Tensor(-inner(gammar, dot(q, n))*dS -
           inner(gammar, tau*u)*dS)

# Finally, we define the trace mass term: H
H = Tensor(inner(gammar, tau*lambdar)*dS)

# We now statically condense to arrive at a SPD system for lambdar
K = H - P * M.inv * N
R = -P * M.inv * Fbar

# Solve for the multipliers
lambda_h = Function(T)
Kmat = assemble(K, bcs=bcs)
Rvec = assemble(R)

# import ipdb; ipdb.set_trace()
solve(Kmat, lambda_h, Rvec, solver_parameters={"ksp_type": "cg"})

# Now we reconstruct q and u
# Can't write into mixed-dats, so we need to split EVERYTHING!
from firedrake.formmanipulation import split_form

split_mixed_op = dict(split_form(Atilde.form))
split_trace_op = dict(split_form(CG.form))
split_rhs = dict(split_form(F.form))

# Local tensors
A = Tensor(split_mixed_op[(0, 0)])
B = Tensor(split_mixed_op[(0, 1)])
C = Tensor(split_mixed_op[(1, 0)])
D = Tensor(split_mixed_op[(1, 1)])
K_0 = Tensor(split_trace_op[(0, 0)])
K_1 = Tensor(split_trace_op[(0, 1)])
G = Tensor(split_rhs[(0,)])
F = Tensor(split_rhs[(1,)])

# Functions for solutions
q_h = Function(U, name="Flux")
u_h = Function(V, name="Scalar")

# Assemble u_h
M = D - C * A.inv * B
R = K_1.T - C * A.inv * K_0.T
L_h = AssembledVector(lambda_h)
u_rec = M.inv * (F + R * L_h)
assemble(u_rec, tensor=u_h)

U_h = AssembledVector(u_h)
q_rec = A.inv * (B * U_h - K_0.T * L_h)
assemble(q_rec, tensor=q_h)

analytic = Function(V, name="Analytical Scalar")
analytic.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))
File("hdg_poisson.pvd").write(q_h, u_h, analytic)
