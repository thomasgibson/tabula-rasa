from firedrake import *


def run_helmholtz_3D_example(d, r, alpha=None, beta=None):
    """Solves the Helmholtz equation with homogenous Neumman
    conditions imposed on the domain boundary.
    """
    # Define computational domain
    mesh = UnitCubeMesh(2 ** r, 2 ** r, 2 ** r)
    x = SpatialCoordinate(mesh)

    # H1 function space
    V = FunctionSpace(mesh, "CG", degree=d)
    v = TestFunction(V)
    u = TrialFunction(V)

    # Forcing function
    f = Function(V)
    f.interpolate((1 + 12*pi*pi)*cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2]))

    # Coefficients
    if alpha is None:
        alpha = 1
    if beta is None:
        beta = 1

    # Bilinear and linear forms
    a = inner(grad(v), alpha*grad(u))*dx + v*beta*u*dx
    L = v*f*dx

    # Solve with static condensation
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.StaticCondensationPC',
              'static_condensation': {'ksp_type': 'cg',
                                      'pc_type': 'ilu',
                                      'ksp_rtol': 1e-8}}
    u_h_sc = Function(V, name="approx-sc")
    solve(a == L, u_h_sc, solver_parameters=params)

    # Solve again without static condensation
    params = {'ksp_type': 'cg',
              'pc_type': 'ilu',
              'ksp_rtol': 1e-8}
    u_h = Function(V, name="approx-no-sc")
    solve(a == L, u_h, solver_parameters=params)

    # Compare with exact solution
    exact = Function(V, name="exact")
    exact.interpolate(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2]))
    error_sc_exact = errornorm(u_h_sc, exact)

    # Compare two solutions (should close)
    error_solutions = errornorm(u_h_sc, u_h)

    return u_h_sc, u_h, exact, error_sc_exact, error_solutions

# Run problem
degree = 4
resolution = 3
u_h_sc, u_h, u, err, err_comp = run_helmholtz_3D_example(d=degree,
                                                         r=resolution)

print("Error between static condensation solution and exact: %.6f"
      % err)

File("sc_3d_helmholtz.pvd").write(u_h_sc, u_h, u)

print("Error between solutions computed with "
      "and without static condensation: %.6f"
      % err_comp)
