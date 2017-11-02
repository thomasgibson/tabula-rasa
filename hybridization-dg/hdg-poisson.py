from firedrake import *

degree = 2
res = 3
mesh = UnitSquareMesh(2**res, 2**res)
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)
U = VectorFunctionSpace(mesh, "DG", degree)
V = FunctionSpace(mesh, "DG", degree)
T = FunctionSpace(mesh, "HDiv Trace", degree)

W = U * V * T
q, u, uhat = TrialFunctions(W)
v, w, mu = TestFunctions(W)

# Lagrange multipliers enforce the Dirichlet condition
bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

f = Function(V).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

# $qhat\cdot n$
tau = Constant(10)/CellVolume(mesh)
qhat = q + tau*(u - uhat)*n


def both(a):
    return 2*avg(a)


a = (
    (dot(v, q) - div(v)*u)*dx
    + both(uhat*inner(v, n))*dS
    # + uhat*inner(v, n)*ds
    + dot(grad(w), q)*dx
    - both(inner(qhat, n)*w)*dS
    - inner(qhat, n)*w*ds
    + both(mu*inner(qhat, n))*dS
    # + mu*inner(qhat, n)*ds
)

L = -w*f*dx
w = Function(W, name="solutions")
params = {'mat_type': 'matfree',
          'ksp_type': 'gmres',
          'pc_type': 'python',
          'ksp_monitor': True,
          'pc_python_type': 'firedrake.HybridStaticCondensationPC',
          'hybrid_sc': {'ksp_type': 'preonly',
                        'pc_type': 'lu'}}
solve(a == L, w, bcs=bcs, solver_parameters=params)
q_h, u_h, uhat_h = w.split()
analytic = Function(V, name="Analytical Scalar")
analytic.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))
File("hdg-test.pvd").write(q_h, u_h, analytic)
