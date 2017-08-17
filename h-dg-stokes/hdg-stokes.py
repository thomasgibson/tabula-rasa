from firedrake import *

# Problem domain
mesh = UnitSquareMesh(8, 8)

# Facet normal and cellsize
n = FacetNormal(mesh)
he = CellSize(mesh)

# Polyomial order
k = 2

# Relevant constants
nu = Constant(1.0)
alpha = Constant(6.0*k*k)
beta = Constant(1e-4)

# Analytic expressions
Wu = VectorFunctionSpace(mesh, "CG", k+1)
Qp = FunctionSpace(mesh, "CG", k+1)
u_expr = Expression(("x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                      - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])",
                     "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                      - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"))
p_expr = Expression("x[0]*(1.0 - x[0])")
u_exact = Function(Wu, name="Exact velocity").project(u_expr)
p_exact = Function(Qp, name="Exact pressure").interpolate(p_expr)

# Function spaces (velocity and pressure)
W = VectorFunctionSpace(mesh, "DG", k)
Q = FunctionSpace(mesh, "DG", k)
WQ = W * Q

# Space of continuous Lagrange multipliers (vector and scalar)
CG_trace = FiniteElement("CG", mesh.ufl_cell(), k)['facet']
Wbar = VectorFunctionSpace(mesh, CG_trace)
Qbar = FunctionSpace(mesh, CG_trace)

# Source term (computed from exact expressions)
f = div(p_exact * Identity(2) - 2 * nu * sym(grad(u_exact)))

# Test and trial functions
u, p = TrialFunctions(WQ)
v, q = TestFunctions(WQ)

# Numerical fluxes
ubar = TrialFunction(Wbar)
vbar = TestFunction(Wbar)
pbar = TrialFunction(Qbar)
qbar = TestFunction(Qbar)

# Boundary conditions
bc1 = DirichletBC(Wbar, Constant((0.0, 0.0)), "on_boundary")
bc2 = DirichletBC(Qbar, Constant(0.0), "on_boundary")
# bcs = [bc1, bc2]

# Relevant tensor expressions for Stokes formulation
pI = p*Identity(2)
pbI = pbar*Identity(2)

# Slate tensors for the momentum equations
A = Tensor(inner(2*nu*sym(grad(u)), grad(v))*dx
           + dot(-2*nu*sym(grad(u))*n + (2*nu*alpha/he)*u, v)*(dS + ds)
           + dot(-2*nu*u, sym(grad(v))*n)*(dS + ds))
B = Tensor(-dot(p, div(v))*dx)
C = Tensor(-alpha/he*2*nu*inner(ubar, v)*(dS + ds)
           + 2*nu*inner(ubar, sym(grad(v))*n)*(dS + ds))
D = Tensor(dot(pbI*n, v)*(dS + ds))

# Right-hand side tensor
Q = Tensor(inner(f, v)*dx)

# Remaining momentum surface terms
K = Tensor(alpha/he * 2*nu*dot(ubar, vbar)*(dS + ds))
L = Tensor(-dot(pbar*n, vbar)*(dS + ds))

# Stabilization tensors
F = Tensor(-beta*he/(nu + 1) * dot(p, q)*(dS + ds))
H = Tensor(beta*he/(nu + 1) * dot(pbar, q)*(dS + ds))
P = Tensor(-beta*he/(nu + 1) * pbar * qbar * (dS + ds))
S = Tensor(dot(Constant((0.0, 0.0)), vbar)*dx)

# Static condensation using the local element Slate tensors:
# Intermediate tensors to simplify notation
M = B.T * A.inv * B - F
N = A.inv * B * M.inv * B.T * A.inv
T = C.T * (N - A.inv) * D - C.T * A.inv * B * M.inv * H + L
W = D.T * A.inv * B * M.inv * H
U = C.T * (N - A.inv) * C + K
V = S + C.T * (N - A.inv) * Q
X = (D.T * (N - A.inv) - H.T * M.inv * B.T * A.inv) * Q
Y = D.T * (N - A.inv) * D - (W + W.T) + H.T * M.inv * H + P

# pbar system:
Mpbar = Y - T.T * U.inv * T
SS = X - T.T * U.inv * V
MATpbar = assemble(Mpbar, bcs=bc2)
MATpbar.force_evaluation()
RHSpbar = assemble(SS)
xpbar = Function(Qbar, name="Pressure flux")

pbar_params = {"ksp_type": "preonly",
               "pc_type": "lu"}
solve(MATpbar, xpbar, RHSpbar, solver_parameters=pbar_params)


File("hdg_stokes.pvd").write(u_exact, p_exact)
