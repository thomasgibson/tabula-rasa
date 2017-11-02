from firedrake import *

n = 4
mesh = UnitSquareMesh(n, n)
n = FacetNormal(mesh)
RTd = FunctionSpace(mesh, BrokenElement(FiniteElement("RT", triangle, 1)))
DG = FunctionSpace(mesh, "DG", 0)
T = FunctionSpace(mesh, "HDiv Trace", 0)

Wd = RTd * DG * T

sigma, u, lambdar = TrialFunctions(Wd)
tau, v, gammar = TestFunctions(Wd)

bcs = DirichletBC(Wd.sub(2), Constant(0.0), "on_boundary")

adx = (dot(sigma, tau) - div(tau)*u + div(sigma)*v + u*v)*dx
adS = (jump(sigma, n=n)*gammar('+') + jump(tau, n=n)*lambdar('+'))*dS
a = adx + adS

f = Function(DG)
f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
L = v*f*dx

w = Function(Wd)
params = {'mat_type': 'matfree',
          'ksp_type': 'gmres',
          'pc_type': 'python',
          'ksp_monitor': True,
          'pc_python_type': 'firedrake.HybridStaticCondensationPC',
          'hybrid_sc': {'ksp_type': 'preonly',
                        'pc_type': 'lu'}}
solve(a == L, w, bcs=bcs, solver_parameters=params)
sigma_h, u_h, lambdar_h = w.split()
File("hybrid-mixed-test.pvd").write(sigma_h, u_h)
