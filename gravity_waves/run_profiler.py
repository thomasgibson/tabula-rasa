from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, parameters
from argparse import ArgumentParser
from pyop2.profiling import timed_stage
from mpi4py import MPI
import pandas as pd
import sys

from profile_problem import ProfileGravityWaveSolver as Solver


parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Profile the gravity wave solver.""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--X",
                    action="store",
                    default=1.0,
                    type=float,
                    help="Factor to scale the Earth's radius")

parser.add_argument("--H",
                    action="store",
                    default=1.0e4,
                    type=float,
                    help="Atmospheric lid")

parser.add_argument("--cfl",
                    action="store",
                    default=1.0,
                    type=float,
                    help="Horizontal CFL number")

parser.add_argument("--rtol",
                    action="store",
                    default=1.0e-5,
                    type=float,
                    help="Solver rtolerance for the u-p system.")

parser.add_argument("--model_degree",
                    action="store",
                    type=int,
                    default=1,
                    help="Degree of the finite element model.")

parser.add_argument("--method",
                    action="store",
                    default="RTCF",
                    choices=["RT", "RTCF", "BDFM"],
                    help="Mixed method type.")

parser.add_argument("--refinements",
                    action="store",
                    default=4,
                    type=int,
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--nlayers",
                    action="store",
                    default=20,
                    type=int,
                    help="Number of vertical layers.")

parser.add_argument("--monitor",
                    action="store_true",
                    help="Turn on KSP monitors")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")


args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


PETSc.Log.begin()

solver = Solver(refinement_level=args.refinements,
                nlayers=args.nlayers,
                model_degree=args.model_degree,
                method=args.method,
                X=args.X,
                H=args.H,
                rtol=args.rtol,
                hybridization=args.hybridization,
                cfl=args.cfl,
                monitor=args.monitor)

PETSc.Sys.Print("""Warming up solver with parameters:\n
Planet radius (m): %s,\n
Atmospheric lid (m) %s,\n
Mixed method: %s,\n
Model degree: %s,\n
Hybridization: %s,\n
Horizontal CFL: %s,\n
Dt (s): %s,\n
Dx (km): %s,\n
Dz (m): %s,\n
Solver rtol: %s,\n
KSP monitor: %s.
""" % (solver._R,
       solver.H,
       solver.method,
       solver.model_degree,
       solver.hybridization,
       solver.courant,
       solver.Dt,
       solver.dx_max / 1000,
       solver.dz,
       solver.rtol,
       solver.monitor))

with timed_stage("Warm up"):
    solver.warmup()

PETSc.Sys.Print("""Warm up complete. Profiling linear solver.""")

solver.run_profile()
