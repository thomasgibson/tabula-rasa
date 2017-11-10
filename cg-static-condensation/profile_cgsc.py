from firedrake import *
import sys

parameters["pyop2_options"]["lazy_evaluation"] = False


def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# NOTE: ksp_monitor is on to monitor convergence of the
# preconditioned (AMG) Krylov method
if '--scpc' in sys.argv:
    parameters = {'mat_type': 'matfree',
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.StaticCondensationPC',
                  'static_condensation': {'ksp_type': 'cg',
                                          'pc_type': 'hypre',
                                          'pc_hypre_type': 'boomeramg',
                                          'pc_hypre_boomeramg_P_max': 4,
                                          'ksp_monitor': True,
                                          'ksp_rtol': 1e-10}}
else:
    parameters = {'ksp_type': 'cg',
                  'pc_type': 'hypre',
                  'pc_hypre_type': 'boomeramg',
                  'pc_hypre_boomeramg_P_max': 4,
                  'ksp_monitor': True,
                  'ksp_rtol': 1e-10}

# Set up unit cube mesh with h = (1/2**r) in all spatial
# directions
if is_intstring(sys.argv[1]):
    r = int(sys.argv[1])
else:
    r = 3

print("Resolution parameter is: %d" % r)

mesh = UnitCubeMesh(2 ** r, 2 ** r, 2 ** r)
x = SpatialCoordinate(mesh)

# Set up H1 function space and test/trial functions
d = 4
V = FunctionSpace(mesh, "CG", degree=d)
u = TrialFunction(V)
v = TestFunction(V)

f = Function(FunctionSpace(mesh, "CG", degree=d+1))
f.interpolate((1 + 108*pi*pi)*cos(6*pi*x[0])*cos(6*pi*x[1])*cos(6*pi*x[2]))

# Set a(u, v) = L(v)
# NOTE: This problem has homogeneous Neumman conditions
# applied weakly on all sides of the cube
a = inner(grad(u), grad(v))*dx + u*v*dx
L = f*v*dx

Uh = Function(V, name="Approximate Solution")
solve(a == L, Uh, solver_parameters=parameters)

# Compare with exact solution
V_a = FunctionSpace(mesh, "CG", d + 2)
exact = Function(V_a, name="Exact Solution")
exact.interpolate(cos(6*pi*x[0])*cos(6*pi*x[1])*cos(6*pi*x[2]))
error = errornorm(Uh, exact)
print("Error between computed solution and exact: %0.8f" % error)

# Write output file
File("SCPC-3DHelmholtz-r%d.pvd" % r).write(Uh, exact)
