from argparse import ArgumentParser
from collections import defaultdict
from firedrake import COMM_WORLD, parameters
from firedrake.petsc import PETSc
from mpi4py import MPI

from pyop2.profiling import timed_region

import os
import pandas as pd
import hdg_problem as module

parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Profile HDG solver.""",
                        add_help=False)

parser.add_argument("--results_file", action="store",
                    default="HDG-timings",
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


problem_cls = module.HDGProblem
results = os.path.abspath(args.results_file)

warm = defaultdict(bool)

PETSc.Log.begin()


def run_solver(problem_cls, degree, size, rtol, quads, dim):

    pcg_params = {'ksp_type': 'cg',
                  'ksp_rtol': rtol,
                  'ksp_monitor_true_residual': True,
                  'pc_type': 'bjacobi',
                  'sub_pc_type': 'ilu'}

    # pcg_params = {'ksp_type': 'cg',
    #               'pc_type': 'gamg',
    #               'ksp_rtol': rtol,
    #               'ksp_monitor_true_residual': True,
    #               'mg_levels': {'ksp_type': 'chebyshev',
    #                             'ksp_max_it': 1,
    #                             'pc_type': 'bjacobi',
    #                             'sub_pc_type': 'ilu'}}

    params = {'mat_type': 'matfree',
              'pmat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'scpc.HybridSCPC',
              'hybrid_sc': pcg_params}

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
            problem.post_processed_sol()

        warm[(name, degree)] = True

    solver = problem.solver(parameters=params)
    problem.u.assign(0)
    problem.u_pp.assign(0)

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

        # HDG-specific timings
        HDGinit = PETSc.Log.Event("HybridSCInit").getPerfInfo()
        HDGrhs = PETSc.Log.Event("HybridSCRHS").getPerfInfo()
        HDGrecon = PETSc.Log.Event("HybridSCReconstruct").getPerfInfo()
        HDGSolve = PETSc.Log.Event("HybridSCSolve").getPerfInfo()
        hdginit_time = problem.comm.allreduce(HDGinit["time"], op=MPI.SUM) / problem.comm.size
        hdgrhs_time = problem.comm.allreduce(HDGrhs["time"], op=MPI.SUM) / problem.comm.size
        hdgrecon_time = problem.comm.allreduce(HDGrecon["time"], op=MPI.SUM) / problem.comm.size
        hdgsolve_time = problem.comm.allreduce(HDGSolve["time"], op=MPI.SUM) / problem.comm.size

        # Should total to KSPSolve time (approximately)
        hdg_total_solve = hdginit_time + hdgrhs_time + hdgsolve_time + hdgrecon_time

        problem.post_processed_sol()
        HDGPP = PETSc.Log.Event("HDGPostprocessing").getPerfInfo()
        pp_time = problem.comm.allreduce(HDGPP["time"], op=MPI.SUM) / problem.comm.size

        # Total HDG time (with pp)
        hdg_total_time = hdg_total_solve + pp_time

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            _, u_h, lambdar_h = problem.u.split()

            data = {"KSPSolve": ksp_time,
                    "PCSetUp": pcsetup_time,
                    "PCApply": pcapply_time,
                    "num_processes": problem.comm.size,
                    "mesh_size": problem.N,
                    "num_cells": num_cells,
                    "degree": problem.degree,
                    "problem_name": name,
                    "u_dofs": u_h.dof_dset.layout_vec.getSize(),
                    "trace_dofs": lambdar_h.dof_dset.layout_vec.getSize(),
                    "name": problem.name,
                    "disc_error_u": err,
                    "true_err_u": true_err,
                    "HDGInit": hdginit_time,
                    "HDGRhs": hdgrhs_time,
                    "HDGRecover": hdgrecon_time,
                    "HDGSolve": hdgsolve_time,
                    "HDGTotalSolve": hdg_total_solve,
                    "HDGTotal": hdg_total_time,
                    "HDGPPTime": pp_time,
                    "ErrorPP": problem.pp_err}

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
        sigma_h, u_h, _ = problem.u.split()
        u_pp = problem.u_pp
        File("hdg_output.pvd").write(sigma_h, u_h,
                                     problem.sol[0], problem.sol[1], u_pp)


degree = args.degree
size = args.size
rtol = args.rtol
dim = args.dim
quads = args.quads
run_solver(problem_cls, degree, size, rtol, quads, dim)
