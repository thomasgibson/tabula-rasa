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
                    default="CG-timings",
                    help="Where to put the results.")

parser.add_argument("--degree", action="store", default=1,
                    type=int, help="Degree of approximation.")

parser.add_argument("--size", action="store", default=10,
                    type=int, help="Number of cells in each direction")

parser.add_argument("--rtol", action="store", default=1.0e-8,
                    type=float, help="Relative tolerance of solver.")

parser.add_argument("--dim", action="store", default=2,
                    type=int, choices=[2, 3], help="Problem dimension.")

parser.add_argument("--quads", action="store_true",
                    help="Use quadrilateral elements")

parser.add_argument("--write_output", action="store_true",
                    help="Plot analytic and computed solution.")

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


def run_solver(problem_cls, degree, size, rtol, quads, dim):

    params = {'ksp_type': 'cg',
              'ksp_rtol': rtol,
              'ksp_monitor_true_residual': True,
              'pc_type': 'bjacobi',
              'sub_pc_type': 'ilu'}

    # params = {'ksp_type': 'cg',
    #           'pc_type': 'gamg',
    #           'ksp_rtol': rtol,
    #           'ksp_monitor_true_residual': True,
    #           'mg_levels': {'ksp_type': 'chebyshev',
    #                         'ksp_max_it': 1,
    #                         'pc_type': 'bjacobi',
    #                         'sub_pc_type': 'ilu'}}

    problem = problem_cls(degree=degree, N=size,
                          quadrilaterals=quads, dimension=dim)
    name = getattr(problem, "name")
    solver = problem.solver(parameters=params)

    PETSc.Sys.Print("""
\nSolving problem: %s.\n
Approximation degree: %s\n
Problem size: %s ^ %s\n
Quads: %s\n
""" % (name, problem.degree, problem.N, problem.dim, problem.quads))

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
    with PETSc.Log.Stage("%s(degree=%s, size=%s, dimension=%s) Warm solve\n" %
                         (name, degree, size, dim)):
        solver.solve()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        ksp_time = problem.comm.allreduce(ksp["time"], op=MPI.SUM) / problem.comm.size
        pcsetup_time = problem.comm.allreduce(pcsetup["time"], op=MPI.SUM) / problem.comm.size
        pcapply_time = problem.comm.allreduce(pcapply["time"], op=MPI.SUM) / problem.comm.size
        num_cells = problem.comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)
        err = problem.err
        true_err = problem.true_err

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
                    "problem_name": name,
                    "dofs": problem.u.dof_dset.layout_vec.getSize(),
                    "name": problem.name,
                    "disc_error": err,
                    "true_err": true_err}

            df = pd.DataFrame(data, index=[0])
            if problem.quads:
                result_file = results + "_quads.csv"
            else:
                result_file = results + ".csv"

            df.to_csv(result_file, index=False, mode="w", header=True)

    PETSc.Sys.Print("Solving %s(degree=%s, size=%s, dimension=%s) finished." %
                    (name, problem.degree, problem.N, problem.dim))

    if args.write_output:
        from firedrake import File
        File("cg_output.pvd").write(problem.u, problem.sol)


degree = args.degree
size = args.size
rtol = args.rtol
dim = args.dim
quads = args.quads
run_solver(problem_cls, degree, size, rtol, quads, dim)
