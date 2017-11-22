from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace, COMM_WORLD, parameters
from firedrake.petsc import PETSc
from crank_nicolson import CrankNicolsonStepper
from linear_solver import LinearizedShallowWaterSolver
from argparse import ArgumentParser
from mpi4py import MPI

import pandas as pd
import sys


PETSc.Log.begin()
parser = ArgumentParser(description="""Run Williamson test case 5""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--run-full-sim",
                    action="store_true",
                    help="Run full 50 day simulation.")

parser.add_argument("--verification",
                    action="store_true",
                    help=("Turn verification mode on? "
                          "This enables GMRES residual monitors"))

parser.add_argument("--testing",
                    action="store_true",
                    help=("Select 'True' or 'False' to enable a test run. "
                          "Default is False."))

parser.add_argument("--dumpfreq",
                    default=100,
                    type=int,
                    action="store",
                    help="Dump frequency of output.")

parser.add_argument("--refinements",
                    action="store",
                    default=3,
                    type=int,
                    choices=[3, 4, 5, 6, 7],
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn profiling on for 20 timesteps.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.profile:
    # Ensures accurate timing of parallel loops
    parameters["pyop2_options"]["lazy_evaluation"] = False

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)

if args.refinements == 4:
    ref_dt = {4: 450.}
elif args.refinements == 5:
    ref_dt = {5: 225.}
elif args.refinements == 6:
    ref_dt = {6: 112.5}
elif args.refinements == 7:
    ref_dt = {7: 56.25}
else:
    ref_dt = {3: 900.}

# Run full 50 simulation if requested
if args.run_full_sim:
    day = 24.*60.*60.
    tmax = 50*day
else:
    # If testing, adjust tmax
    dt, = ref_dt.values()
    if args.testing:
        # 5 time steps
        tmax = dt*5
    else:
        tmax = dt*20

# Setup shallow water parameters
R = 6371220.
H = 5960.

# Setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)
hybridize = args.hybridization

for ref_level, dt in ref_dt.items():

    if hybridize:
        dirname = "hybrid_sw_W5_ref%s_dt%s" % (ref_level, dt)
    else:
        dirname = "sw_W5_ref%s_dt%s" % (ref_level, dt)

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname,
                              dumpfreq=args.dumpfreq)
    diagnostic_fields = [Sum('D', 'topography'), PotentialVorticity()]

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

    verify = args.verification
    linear_solver = LinearizedShallowWaterSolver(state,
                                                 hybridization=hybridize,
                                                 verification=verify,
                                                 profiling=args.profile)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    # Build time stepper using Crank-Nicolson
    stepper = CrankNicolsonStepper(state,
                                   advected_fields,
                                   linear_solver,
                                   sw_forcing)

    stepper.run(t=0, tmax=tmax)

    if COMM_WORLD.rank == 0:
        if args.profile:
            if hybridize is True:
                results = "hybrid_profiling_sw_W5_ref%s_dt%s.csv" % (ref_level,
                                                                     dt)
            else:
                results = "profiling_sw_W5_ref%s_dt%s.csv" % (ref_level, dt)

            solver = stepper.linear_solver
            comm = solver.uD_solver._problem.u.comm
            ksp_solve = PETSc.Log.Event("KSPSolve").getPerfInfo()
            pc_setup = PETSc.Log.Event("PCSetUp").getPerfInfo()
            pc_apply = PETSc.Log.Event("PCApply").getPerfInfo()
            outer_its = stepper.ksp_iter_array
            inner_its = stepper.inner_ksp_iter_array

            # Round down to nearest int
            avg_outer_its = int(sum(outer_its)/len(outer_its))
            avg_inner_its = int(sum(inner_its)/len(inner_its))

            # Average times from log
            ksp_time = comm.allreduce(ksp_solve["time"], op=MPI.SUM)/comm.size
            setup_time = comm.allreduce(pc_setup["time"], op=MPI.SUM)/comm.size
            apply_time = comm.allreduce(pc_apply["time"], op=MPI.SUM)/comm.size

            data = {"SolveTime": ksp_time,
                    "SetUpTime": setup_time,
                    "ApplyTime": apply_time,
                    "AvgOuterIters": avg_outer_its,
                    "AvgInnerIters": avg_inner_its}

            df = pd.DataFrame(data, index=[0])
            df.to_csv(results, index=False, mode="w", header=True)
