from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace, COMM_WORLD, parameters
from firedrake.petsc import PETSc
from crank_nicolson import CrankNicolsonStepper
from linear_solver import LinearizedShallowWaterSolver
from argparse import ArgumentParser

import pandas as pd


parameters["pyop2_options"]["lazy_evaluation"] = False


PETSc.Log.begin()
parser = ArgumentParser(description="""Run Williamson test case 5""",
                        add_help=False)

parser.add_argument("--hybridization",
                    default=False,
                    type=bool,
                    action="store",
                    help="Select 'True' or 'False'. Default is 'False'.")

parser.add_argument("--testing",
                    default=False,
                    type=bool,
                    action="store",
                    help=("Select 'True' or 'False' to enable a test run. "
                          "Default is False."))

parser.add_argument("--dumpfreq",
                    default=0,
                    action="store",
                    help="Dump frequency of output.")

parser.add_argument("--refinements",
                    action="store",
                    default=3,
                    type=int,
                    choices=[3, 4, 5, 6, 7],
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--profile",
                    action="store",
                    default=False,
                    type=bool,
                    help="Turn profiling on? This forces 7 mesh refinements.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.hybridization is not None:
    if args.hybridization not in [True, False]:
        raise ValueError("Unrecognized argument '%s', use a boolean"
                         % args.hybridization)
    hybridize = args.hybridization

else:
    hybridize = False


day = 24.*60.*60.
if args.testing:
    ref_dt = {3: 3000.}
    tmax = 3000.
elif args.profiling:
    ref_dt = {7: 56.25}
    tmax = 1125.
elif args.refinements == 3:
    ref_dt = {3: 900.}
    tmax = 50*day
elif args.refinements == 4:
    ref_dt = {4: 450.}
    tmax = 50*day
elif args.refinements == 5:
    ref_dt = {5: 225.}
    tmax = 50*day
elif args.refinements == 6:
    ref_dt = {6: 112.5}
    tmax = 50*day
elif args.refinements == 7:
    ref_dt = {7: 56.25}
    tmax = 50*day
else:
    raise ValueError("What?")

# Setup shallow water parameters
R = 6371220.
H = 5960.

# Setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.items():

    if hybridized:
        dirname = "results/hybrid_sw_W5_ref%s_dt%s" % (ref_level, dt)
    else:
        dirname = "results/sw_W5_ref%s_dt%s" % (ref_level, dt)

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

    linear_solver = LinearizedShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    # Build time stepper using Crank-Nicolson
    stepper = CrankNicolsonStepper(state,
                                   advected_fields,
                                   linear_solver,
                                   sw_forcing)

    stepper.run(t=0, tmax=tmax)

    if COMM_WORLD.rank == 0:
        if hybridized:
            if args.profile:
                results = "hybrid_profiling_sw_W5_ref%s_dt%s" % (ref_level, dt)
            else:
                results = "hybrid_sw_W5_ref%s_dt%s" % (ref_level, dt)
        else:
            if args.profile:
                results = "profiling_sw_W5_ref%s_dt%s" % (ref_level, dt)
            else:
                results = "sw_W5_ref%s_dt%s" % (ref_level, dt)

        data = {"Timestep": stepper.t_array,
                "SolveTime": stepper.solve_time_array,
                "OuterKSPIterations": stepper.ksp_iter_array,
                "InnerKSPIterations": stepper.inner_ksp_iter_array}

        df = pd.DataFrame(data)
        df.to_csv(results, index=False, mode="w", header=True)
