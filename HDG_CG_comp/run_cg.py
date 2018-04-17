from argparse import ArgumentParser
from collections import defaultdict
from firedrake import COMM_WORLD, parameters
from firedrake.petsc import PETSc
from mpi4py import MPI

import os
import pandas as pd
import cg_problem as module

parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Profile CG solver.""",
                        add_help=False)

parser.add_argument("--results_file", action="store",
                    default="CG-timings.csv",
                    help="Where to put the results.")

parser.add_argument("--degree", action="store", default=1,
                    type=int, help="Degree of approximation.")

parser.add_argument("--help", action="store_true", help="Show help.")

args, _ = parser.parse_known_args()

if args.help:
    import sys
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(0)


problem_cls = module.CGProblem
results = os.path.abspath(args.results_file)

warm = defaultdict(bool)

PETSc.Log.begin()


def run_solver(problem_cls, degree, size, rtol):

    params = {"ksp_type": "cg",
              "ksp_monitor_true_residual": True,
              "ksp_rtol": rtol,
              "pc_type": "hypre",
              "pc_hypre_type": "boomeramg",
              "pc_hypre_boomeramg_no_CF": True,
              "pc_hypre_boomeramg_coarsen_type": "HMIS",
              "pc_hypre_boomeramg_interp_type": "ext+i",
              "pc_hypre_boomeramg_P_max": 4,
              "pc_hypre_boomeramg_agg_nl": 1}

    problem = problem_cls(degree=degree, N=size)
    name = getattr(problem, "name")
    solver = problem.solver(parameters=params)

    PETSc.Sys.Print("""
\nSolving problem: %s.\n
Approximation degree: %s\n
Problem size: %s ^ 3\n
""" % (name, problem.degree, problem.N))

    if not warm[(name, degree)]:
        PETSc.Sys.Print("Warmup solve\n")
        problem.u.assign(0)

        with PETSc.Log.Stage("Warmup..."):
            solver.solve()

        warm[(name, degree)] = True

    solver = problem.solver(parameters=params)
    problem.u.assign(0)

    PETSc.Sys.Print("Timed solve...")
    solver.snes.setConvergenceHistory()
    solver.snes.ksp.setConvergenceHistory()
    with PETSc.Log.Stage("%s(degree=%s, size=%s) Warm solve\n" %
                         (name, degree, size)):
        solver.solve()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        ksp_time = problem.comm.allreduce(ksp["time"], op=MPI.SUM) / problem.comm.size
        pcsetup_time = problem.comm.allreduce(pcsetup["time"], op=MPI.SUM) / problem.comm.size
        pcapply_time = problem.comm.allreduce(pcapply["time"], op=MPI.SUM) / problem.comm.size
        num_cells = problem.comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)
        err = problem.err

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            data = {"KSPSolve": ksp_time,
                    "PCSetUp": pcsetup_time,
                    "PCApply": pcapply_time,
                    "num_processes": problem.comm.size,
                    "mesh_size": problem.N,
                    "num_cells": num_cells,
                    "degree": problem.degree,
                    "parameter_name": name,
                    "dofs": problem.u.dof_dset.layout_vec.getSize(),
                    "name": problem.name,
                    "disc_error": err}

            df = pd.DataFrame(data, index=[0])
            df.to_csv(results, index=False, mode="w", header=True)

    PETSc.Sys.Print("Solving %s(degree=%s, size=%s) ... finished." %
                    (name, problem.degree, problem.N))


degree = args.degree
run_solver(problem_cls, degree, size=32, rtol=1e-8)
