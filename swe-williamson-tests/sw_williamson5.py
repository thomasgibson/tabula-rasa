from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace
import sys


parameters["pyop2_options"]["lazy_evaluation"] = False


def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Use a specific dumpfreq
if is_intstring(sys.argv[1]):
    dumpfreq = int(sys.argv[1])
else:
    dumpfreq = 100

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
elif '--run-test-lvl3' in sys.argv:
    ref_dt = {3: 900.}
    tmax = 50*day
elif '--run-test-lvl4' in sys.argv:
    ref_dt = {4: 450.}
    tmax = 50*day
elif '--run-test-lvl5' in sys.argv:
    ref_dt = {5: 225.}
    tmax = 50*day
elif '--run-test-lvl6' in sys.argv:
    ref_dt = {6: 112.5}
    tmax = 50*day
elif '--run-test-lvl7' in sys.argv:
    ref_dt = {7: 56.25}
    tmax = 50*day
elif '--profile-mode' in sys.argv:
    ref_dt = {7: 56.25}
    tmax = 1125.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 900., 4: 450., 5: 225., 6: 112.5}
    tmax = 50*day

# Turn hybridization on/off
if '--hybrid-mixed-method' in sys.argv:
    params = {'ksp_type': 'preonly',
              'mat_type': 'matfree',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'gamg',
                                'ksp_monitor': True,
                                'mg_levels_ksp_type': 'chebyshev',
                                'mg_levels_ksp_max_it': 2,
                                'mg_levels_pc_type': 'bjacobi',
                                'mg_levels_sub_pc_type': 'ilu',
                                # Construct broken residual
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu',
                                                  'ksp_rtol': 1e-8,
                                                  'ksp_monitor': True},
                                # Reconstruct HDiv vector field
                                # via local averaging
                                # (Alternatively, one could also use
                                # a Galerkin project onto the HDiv space)
                                # 'hdiv_projection':{'ksp_type': 'cg',
                                #                    'pc_type': 'bjacobi',
                                #                    'sub_pc_type': 'ilu',
                                #                    'ksp_rtol': 1e-8,
                                #                    'ksp_monitor': True}}}
                                'use_reconstructor': True}}
else:
    params = {'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'schur',
              'ksp_type': 'gmres',
              'ksp_monitor': True,
              'ksp_max_it': 100,
              'ksp_gmres_restart': 50,
              'pc_fieldsplit_schur_fact_type': 'FULL',
              'pc_fieldsplit_schur_precondition': 'selfp',
              'fieldsplit_0_ksp_type': 'preonly',
              'fieldsplit_0_pc_type': 'bjacobi',
              'fieldsplit_0_sub_pc_type': 'ilu',
              'fieldsplit_1_ksp_type': 'cg',
              'fieldsplit_1_pc_type': 'gamg',
              'fieldsplit_1_ksp_monitor': True,
              'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
              'fieldsplit_1_mg_levels_ksp_max_it': 2,
              'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
              'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

# Setup shallow water parameters
R = 6371220.
H = 5960.

# Setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.items():

    dirname = "sw_W5_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname,
                              dumplist_latlon=['D'],
                              dumpfreq=100)
    diagnostic_fields = [Sum('D', 'topography')]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # Interpolate initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    x = SpatialCoordinate(mesh)
    # Maximum amplitude of the zonal wind (m/s)
    u_max = 20.
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    theta, lamda = latlon_coords(mesh)
    Omega = parameters.Omega
    g = parameters.g
    Rsq = R**2
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = Min(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    # Coriolis frequency (1/s)
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)
    b = state.fields("topography", D0.function_space())
    b.interpolate(bexpr)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    # Advection for velocity and depth fields
    ueqn = AdvectionEquation(state,
                             u0.function_space(),
                             vector_manifold=True)
    Deqn = AdvectionEquation(state,
                             D0.function_space(),
                             equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

    linear_solver = ShallowWaterSolver(state,
                                       solver_parameters=params,
                                       overwrite_solver_parameters=True)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    # Build time stepper using Crank-Nicolson
    stepper = CrankNicolson(state,
                            advected_fields,
                            linear_solver,
                            sw_forcing)

    stepper.run(t=0, tmax=tmax)
