from argparse import ArgumentParser
from collections import defaultdict
from firedrake import COMM_WORLD, parameters
from firedrake.petsc import PETSc
from mpi4py import MPI

import os
import pandas as pd
import hdg_problem as module

parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Profile HDG solver.""",
                        add_help=False)

parser.add_argument("--results_file", action="store",
                    default="results/MG_TEST_HDG_data",
                    help="Where to put the results.")

parser.add_argument("--dim", action="store", default=3,
                    type=int, choices=[2, 3], help="Problem dimension.")

parser.add_argument("--quads", action="store_true",
                    help="Use quadrilateral elements")

parser.add_argument("--help", action="store_true", help="Show help.")

args, _ = parser.parse_known_args()

if args.help:
    import sys
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(0)

results = os.path.abspath(args.results_file)

warm = defaultdict(bool)


PETSc.Log.begin()


def run_solver(problem_cls, degree, size, rtol, quads, dim, cold=False):

    pcg_params = {"ksp_type": "cg",
                  "ksp_rtol": rtol,
                  "ksp_monitor_true_residual": None,
                  "pc_type": "hypre",
                  "pc_hypre_type": "boomeramg",
                  "pc_hypre_boomeramg_strong_threshold": 0.75,
                  "pc_hypre_boomeramg_agg_nl": 2}

    params = {'mat_type': 'matfree',
              'pmat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type':  'firedrake.SCPC',
              'pc_sc_eliminate_fields': '0, 1',
              'condensed_field': pcg_params}

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


    with PETSc.Log.Stage("MG Test N=%s, k=%s" % (problem.N, problem.degree)):
        solver.solve()

        num_cells = comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            q_h, u_h, lambdar_h = problem.u.split()

            ksp = solver.snes.ksp.getPC().getPythonContext().condensed_ksp
            data = {"num_processes": problem.comm.size,
                    "mesh_size": problem.N,
                    "num_cells": num_cells,
                    "degree": problem.degree,
                    "scalar_dofs": u_h.dof_dset.layout_vec.getSize(),
                    "flux_dofs": q_h.dof_dset.layout_vec.getSize(),
                    "trace_dofs": lambdar_h.dof_dset.layout_vec.getSize(),
                    "name": problem.name,
                    "ksp_iters": ksp.getIterationNumber()}

            df = pd.DataFrame(data, index=[0])
            if problem.quads:
                result_file = results + "_N%d_deg%d_quads.csv" % (problem.N,
                                                                  problem.degree)
            else:
                result_file = results + "_N%d_deg%d.csv" % (problem.N,
                                                            problem.degree)

            df.to_csv(result_file, index=False, mode="w", header=True)

    PETSc.Sys.Print("Solving %s(deg=%s, N=%s, dim=%s) finished.\n" %
                    (name, problem.degree, problem.N, problem.dim))


dim = args.dim
if dim == 3:
    # (degree, size, rtol) NOTE: rtol is chosen such that the
    # iterative solver reaches the minimal algebraic error
    # so that we avoid "oversolving"
    hdg_params = [(1, 4, 1.0e-4),
                  (1, 8, 1.0e-5),
                  (1, 16, 1.0e-6),
                  (1, 32, 1.0e-7),
                  (1, 64, 1.0e-8),
                  (1, 128, 1.0e-9),
                  # Degree 2 set
                  (2, 4, 1.0e-6),
                  (2, 8, 1.0e-7),
                  (2, 16, 1.0e-8),
                  (2, 32, 1.0e-9),
                  (2, 64, 1.0e-10),
                  # Degree 3 set
                  (3, 4, 1.0e-8),
                  (3, 8, 1.0e-9),
                  (3, 16, 1.0e-10),
                  (3, 32, 1.0e-11),
                  (3, 64, 1.0e-12)]

    cold_params = [(1, 4, 1.0e-4),
                   (2, 4, 1.0e-6),
                   (3, 4, 1.0e-8)]
else:
    # If reviewers want a 2D test, we can give them one.
    raise NotImplementedError("Dim %s not set up yet." % dim)


for hdg_param in hdg_params:

    degree, size, rtol = hdg_param
    run_solver(problem_cls=problem_cls, degree=degree,
               size=size, rtol=1e-8, quads=quads, dim=dim,
               cold=False)
