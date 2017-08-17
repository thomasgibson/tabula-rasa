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

# Function spaces
W = VectorFunctionSpace(mesh, "DG", k)
Q = FunctionSpace(mesh, "DG", k)
Wbar = VectorFunctionSpace(mesh, "Discontinuous Lagrange Trace", k)
Qbar = FunctionSpace(mesh, "Discontinuous Lagrange Trace", k)

# Source term (computed from exact expressions)
f = div(p_exact * Identity(2) - 2 * nu * sym(grad(u_exact)))

# Test and trial functions
u = TrialFunction(W)
v = TestFunction(W)
p = TrialFunction(Q)
q = TestFunction(Q)
ubar = TrialFunction(Wbar)
vbar = TestFunction(Wbar)
pbar = TrialFunction(Qbar)
qbar = TestFunction(Qbar)

# Relevant tensor expressions for Stokes formulation
pI = p*Identity(2)
pbI = pbar*Identity(2)

# Slate tensors
A = Tensor(inner(2*nu*sym(grad(u)), grad(v))*dx
           + dot(-2*nu*sym(grad(u))*n + (2*nu*alpha/he)*u, v)*(dS + ds)
           + dot(-2*nu*u, sym(grad(v))*n)*(dS + ds))
B = Tensor(-dot(p, div(v))*dx)
C = Tensor(-alpha/he*2*nu*inner(ubar, v)*(dS + ds)
           + 2*nu*inner(ubar, sym(grad(v))*n)*(dS + ds))
D = Tensor(dot(pbI*n, v)*(dS + ds))
Q = Tensor(inner(f, v)*dx)

K = Tensor(alpha/he * 2*nu*dot(ubar, vbar)*(dS + ds))
L = Tensor(-dot(pbar*n, vbar)*(dS + ds))

F = Tensor(-beta*he/(nu + 1) * dot(p, q)*(dS + ds))
H = Tensor(beta*he/(nu + 1) * dot(pbar, q)*(dS + ds))
P = Tensor(-beta*he/(nu + 1) * pbar * qbar * (dS + ds))
S = Tensor(dot(Constant((0.0, 0.0)), vbar)*dx)

File("hdg_stokes.pvd").write(u_exact, p_exact)
