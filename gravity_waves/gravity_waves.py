"""
This script runs a linear compressible Boussinesq system describing
a simplified atmospheric model on an Earth-sized sphere mesh.
This model problem is designed from the Skamarock and Klemp
gravity wave test case. This is also the model for the DCMIP
test case 3-1.

We mimic the UK Met Office's approach by point-wise eliminating
the buoyancy variable and solve a coupled mixed system for the
velocity and pressure. Once that system is solve, the buoyancy
variable is reconstructed using the previously computed fields.
"""

from firedrake.petsc import PETSc
from firedrake import parameters
from argparse import ArgumentParser
import sys

import problem as module


parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Run the gravity wave test""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--X",
                    action="store",
                    default=125.0,
                    type=float,
                    help="Factor to scale the Earth's radius")

parser.add_argument("--H",
                    action="store",
                    default=1.0e4,
                    type=float,
                    help="Atmospheric lid")

parser.add_argument("--dt",
                    action="store",
                    type=float,
                    default=100.0,
                    help="The time-step size.")

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

parser.add_argument("--use_dt_from_cfl",
                    action="store_true",
                    help="Overwrite Dt and deduce a value of Dt using CFL")

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

parser.add_argument("--tmax",
                    action="store",
                    default=3600.0,
                    type=float,
                    help="Max time.")

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

parser.add_argument("--write",
                    action="store_true",
                    help="Write output.")

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


def run_gravity_waves(problem_cls, Dt, cfl, refinements, nlayers, method,
                      model_degree, hybridization, write=False):

    # Max height (m)
    thickness = args.H

    problem = problem_cls(refinement_level=refinements,
                          nlayers=nlayers,
                          Dt=Dt,
                          method=method,
                          X=args.X,
                          thickness=thickness,
                          model_degree=model_degree,
                          rtol=args.rtol,
                          hybridization=hybridization,
                          cfl=cfl,
                          monitor=args.monitor,
                          use_dt_from_cfl=args.use_dt_from_cfl)

    Dt = problem.Dt
    cfl = problem.courant
    dx_max = problem.dx_max
    dz = problem.dz

    tmax = args.tmax

    PETSc.Sys.Print("""
Dt = %s,\n
Horizontal Courant number (approximate): %s,\n
Dx (max): %s km,
Dz: %s m,\n
tmax: %s s
""" % (Dt, cfl, dx_max/1000, dz, tmax))

    dumpfreq = 100 / Dt

    PETSc.Sys.Print("Warm up with one-step.\n")
    problem.warmup()

    # Run the problem
    problem.run_simulation(tmax, write=write, dumpfreq=dumpfreq)

    PETSc.Sys.Print("Simulation complete.\n")


GWProblem = module.GravityWaveProblem
method = args.method
model_degree = args.model_degree
refinements = args.refinements
nlayers = args.nlayers
hybridization = args.hybridization
Dt = args.dt
cfl = args.cfl

run_gravity_waves(problem_cls=GWProblem,
                  Dt=Dt,
                  cfl=cfl,
                  refinements=refinements,
                  nlayers=nlayers,
                  method=method,
                  model_degree=model_degree,
                  hybridization=hybridization,
                  write=args.write)
