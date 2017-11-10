from firedrake import *


def run_helmholtz_3D_example(d, r, write=False):
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
    f.interpolate((1 + 108*pi*pi)*cos(6*pi*x[0])*cos(6*pi*x[1])*cos(6*pi*x[2]))

    # Bilinear and linear forms
    a = inner(grad(v), grad(u))*dx + v*u*dx
    L = v*f*dx

    # Solve with static condensation
    params = {'mat_type': 'matfree',
              'ksp_type': 'gmres',
              'ksp_monitor': True,
              'pc_type': 'python',
              'pc_python_type': 'firedrake.StaticCondensationPC',
              'static_condensation': {'ksp_type': 'cg',
                                      'pc_type': 'hypre',
                                      'pc_hypre_type': 'boomeramg',
                                      'pc_hypre_boomeramg_P_max': 4,
                                      'ksp_monitor': True,
                                      'ksp_rtol': 1e-8}}
    u_h_sc = Function(V, name="approx-sc")
    solve(a == L, u_h_sc, solver_parameters=params)

    # Solve again without static condensation
    params = {'ksp_type': 'cg',
              'pc_type': 'hypre',
              'pc_hypre_type': 'boomeramg',
              'pc_hypre_boomeramg_P_max': 4,
              'ksp_monitor': True,
              'ksp_rtol': 1e-8}
    u_h = Function(V, name="approx-no-sc")
    solve(a == L, u_h, solver_parameters=params)

    # Compare with exact solution
    V_a = FunctionSpace(mesh, "CG", d + 4)
    exact = Function(V_a, name="exact")
    exact.interpolate(cos(6*pi*x[0])*cos(6*pi*x[1])*cos(6*pi*x[2]))
    error_sc_exact = errornorm(u_h_sc, exact)

    # Compare two solutions (should close)
    error_solutions = errornorm(u_h_sc, u_h)

    if write:
        File("3DSC.pvd").write(u_h_sc, exact)

    return u_h_sc, u_h, exact, error_sc_exact, error_solutions

# Run problem
degree = 4
resolution = 5
u_h_sc, u_h, u, err, err_comp = run_helmholtz_3D_example(d=degree,
                                                         r=resolution,
                                                         write=True)

print("Error between static condensation solution and exact: %.6f"
      % err)

File("sc_3d_helmholtz.pvd").write(u_h_sc, u_h, u)

print("Error between solutions computed with "
      "and without static condensation: %.6f"
      % err_comp)
