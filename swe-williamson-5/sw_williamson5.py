from gusto import *
from gusto.state import get_latlon_mesh
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace, COMM_WORLD, parameters, \
    File, Function, functionspaceimpl
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

parser.add_argument("--compare",
                    action="store_true",
                    help=("Run and compare the hybridized "
                          "and unhybridized solutions."))

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
    ref_lvl = 4
    dt = 450.
elif args.refinements == 5:
    ref_lvl = 5
    dt = 225.
elif args.refinements == 6:
    ref_lvl = 6
    dt = 112.5
elif args.refinements == 7:
    ref_lvl = 3
    dt = 56.25
else:
    ref_lvl = 3
    dt = 900.

# Run full simulation up to 15 days if requested
if args.run_full_sim:
    day = 24.*60.*60.
    tmax = 15*day
else:
    # If testing, adjust tmax
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

if args.compare:
    test_case = {(ref_lvl, dt): [False, True]}
else:
    test_case = {(ref_lvl, dt): [args.hybridization]}

for ref_dt, params in test_case.items():
    ref_level, dt = ref_dt
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level,
                                 degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    solutions = []
    for hybrid_flag in params:
        timestepping = TimesteppingParameters(dt=dt)
        diagnostic_fields = [Sum('D', 'topography'), PotentialVorticity()]

        if hybrid_flag:
            dirname = "hybrid_sw_W5_ref%s_dt%s" % (ref_level, dt)
        else:
            dirname = "sw_W5_ref%s_dt%s" % (ref_level, dt)

        output = OutputParameters(dirname=dirname,
                                  dumplist_latlon=['D',
                                                   'u',
                                                   'PotentialVorticity'],
                                  dumpfreq=args.dumpfreq)
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
                                                     hybridization=hybrid_flag,
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
        solutions.append((state.fields.u, state.fields.D))

        if args.profile:
            solver = stepper.linear_solver
            ksp_solve = PETSc.Log.Event("KSPSolve").getPerfInfo()
            pc_setup = PETSc.Log.Event("PCSetUp").getPerfInfo()
            pc_apply = PETSc.Log.Event("PCApply").getPerfInfo()
            outer_its = stepper.ksp_iter_array
            inner_its = stepper.inner_ksp_iter_array

            # Round down to nearest int
            avg_outer_its = int(sum(outer_its)/len(outer_its))
            avg_inner_its = int(sum(inner_its)/len(inner_its))

            # Average times from log
            comm = mesh.comm
            ksp_time = comm.allreduce(ksp_solve["time"], op=MPI.SUM)
            setup_time = comm.allreduce(pc_setup["time"], op=MPI.SUM)
            apply_time = comm.allreduce(pc_apply["time"], op=MPI.SUM)
            # ksp_time = sum(stepper.ksp_times)
            # setup_time = sum(stepper.setup_times)
            # apply_time = sum(stepper.apply_times)

            if COMM_WORLD.rank == 0:
                if hybrid_flag:
                    results = "hybrid_profiling_sw_W5_ref%s.csv" % ref_level
                else:
                    results = "profiling_sw_W5_ref%s.csv" % ref_level

                data = {"SolveTime": ksp_time,
                        "SetUpTime": setup_time,
                        "ApplyTime": apply_time,
                        "AvgOuterIters": avg_outer_its,
                        "AvgInnerIters": avg_inner_its}

                df = pd.DataFrame(data, index=[0])
                df.to_csv(results, index=False, mode="w", header=True)

    if args.compare:
        uD1, uD2 = solutions
        u1, d1 = uD1
        u2, d2 = uD2
        V = u1.function_space()
        U = d1.function_space()
        erru = Function(V, name="Velocity difference").assign(u1 - u2)
        errD = Function(U, name="Depth difference").assign(d1 - d2)

        mesh_ll = get_latlon_mesh(mesh)
        erru_ll = Function(functionspaceimpl.WithGeometry(V, mesh_ll),
                           val=erru.topological, name=erru.name()+'_ll')
        errd_ll = Function(functionspaceimpl.WithGeometry(U, mesh_ll),
                           val=errD.topological, name=errD.name()+'_ll')
        File("results/comp_sw_W5_ref%s/output_tmax%s.pvd"
             % (ref_level, tmax)).write(erru_ll, errd_ll)
