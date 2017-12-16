from firedrake import *

# Define problem context
mesh = UnitSquareMesh(8, 8)
DG = FiniteElement("Discontinuous Lagrange", triangle, 1)
T = FiniteElement("HDiv Trace", triangle, 1)
U = VectorFunctionSpace(mesh, DG)
V = FunctionSpace(mesh, DG)
M = FunctionSpace(mesh, T)
W = U * V * M

n = FacetNormal(mesh)
p0 = Function(V)
g = Function(V)
f = Function(V)
mu = Function(V)
c = Function(V)
u, p, lambdar = TrialFunctions(W)
w, phi, gammar = TestFunctions(W)

# LDG-H fluxes
# In this example, tau = 1. But we could also use
# some function of the facet area (FacetArea(mesh)).
tau = Constant(1)
phat = lambdar
uhat = u + tau*(p - phat)*n

# Entire elemental 3-field matrix system: Rx = Z, x = {U, P, Lambda},
#
# R = [A B K]
#     [C D L]
#     [M N Q]
#
# Z = [F]
#     [G]
#     [H]
R = Tensor(dot(w, mu*u)*dx - div(w)*p*dx - dot(grad(phi), u)*dx + phi*c*p*dx +
           lambdar*jump(w, n=n)*dS + lambdar*dot(w, n)*ds +
           phi*jump(uhat, n=n)*dS + phi*dot(uhat, n)*ds +
           gammar*jump(uhat, n=n)*dS + gammar*dot(uhat, n)*ds(2) +
           gammar*lambdar*ds(1))
Z = Tensor(phi*f*dx + gammar*g*ds(2) + gammar*p0*ds(1))

S = R.block((2, 2)) - \
    R.block((2, (0, 1))) * R.block(((0, 1), (0, 1))).inv * R.block(((0, 1), 2))
E = Z.block((2,)) - \
    R.block((2, (0, 1))) * R.block(((0, 1), (0, 1))).inv * Z.block(((0, 1),))

# Force assembly to generate code
assemble(S).force_evaluation()
assemble(E).dat.data

u_h = Function(U)
p_h = Function(V)
lambda_h = Function(M)

# Individual blocks of R
A = R.block((0, 0))
B = R.block((0, 1))
C = R.block((1, 0))
D = R.block((1, 1))
K = R.block((0, 2))
L = R.block((1, 2))

# Individual blocks of Z
F = Z.block((0,))
G = Z.block((1,))

# Algebraic expressions in reconstructions
Sd = D - C * A.inv * B
Sl = L - C * A.inv * K

# Coefficient vector of lambda_h
Lambda = AssembledVector(lambda_h)

# Local solve for p_h
assemble(Sd.inv * (G - C * A.inv * F - Sl * Lambda), p_h)

# Force code generation for p_h
p_h.dat.data

# Coefficient vector for p_h
P = AssembledVector(p_h)

# local solve for u_h
assemble(A.inv * (F - B * P - K * Lambda), u_h)

# Force code generation for u_h
u_h.dat.data
