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
                    default="HDG_data",
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


problem_cls = module.HDGProblem
results = os.path.abspath(args.results_file)

warm = defaultdict(bool)

PETSc.Log.begin()


def run_solver(problem_cls, degree, size, rtol, quads, dim):

    pcg_params = {"ksp_type": "cg",
                  "ksp_rtol": rtol,
                  "pc_type": "hypre",
                  "pc_hypre_type": "boomeramg",
                  "pc_hypre_boomeramg_strong_threshold": 0.75,
                  "pc_hypre_boomeramg_agg_nl": 2}

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

    if not warm[(name, degree, size)]:
        PETSc.Sys.Print("Warmup solve\n")
        problem.u.assign(0)

        with PETSc.Log.Stage("Warmup..."):
            solver.solve()
            problem.post_processed_sol()

        warm[(name, degree, size)] = True

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
        mat_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        ksp_time = problem.comm.allreduce(ksp["time"], op=MPI.SUM) / problem.comm.size
        pcsetup_time = problem.comm.allreduce(pcsetup["time"], op=MPI.SUM) / problem.comm.size
        pcapply_time = problem.comm.allreduce(pcapply["time"], op=MPI.SUM) / problem.comm.size
        assembly_time = problem.comm.allreduce(mat_eval["time"], op=MPI.SUM) / problem.comm.size
        num_cells = problem.comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)
        err = problem.err
        true_err = problem.true_err

        # HDG-specific timings
        HDGinit = PETSc.Log.Event("HybridSCInit").getPerfInfo()
        HDGUpdate = PETSc.Log.Event("HybridSCUpdate").getPerfInfo()
        HDGrhs = PETSc.Log.Event("HybridSCRHS").getPerfInfo()
        HDGrecon = PETSc.Log.Event("HybridSCReconstruct").getPerfInfo()
        HDGSolve = PETSc.Log.Event("HybridSCSolve").getPerfInfo()
        hdginit_time = problem.comm.allreduce(HDGinit["time"], op=MPI.SUM) / problem.comm.size
        hdgrhs_time = problem.comm.allreduce(HDGrhs["time"], op=MPI.SUM) / problem.comm.size
        hdgrecon_time = problem.comm.allreduce(HDGrecon["time"], op=MPI.SUM) / problem.comm.size
        hdgsolve_time = problem.comm.allreduce(HDGSolve["time"], op=MPI.SUM) / problem.comm.size
        hdgupdate_time = problem.comm.allreduce(HDGUpdate["time"], op=MPI.SUM) / problem.comm.size
        # Should total to KSPSolve time (approximately)
        hdg_total_solve = hdginit_time + hdgrhs_time + hdgsolve_time + hdgrecon_time + hdgupdate_time

        problem.post_processed_sol()
        HDGPP = PETSc.Log.Event("HDGPostprocessing").getPerfInfo()
        pp_time = problem.comm.allreduce(HDGPP["time"], op=MPI.SUM) / problem.comm.size

        # Total HDG time (with pp)
        hdg_total_time = hdg_total_solve + pp_time

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            _, u_h, lambdar_h = problem.u.split()

            ksp = solver.snes.ksp.getPC().getPythonContext().trace_ksp
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
                    "ErrorPP": problem.pp_err,
                    "ksp_iters": ksp.getIterationNumber(),
                    "jac_eval": assembly_time,
                    "HDGUpdate": hdgupdate_time}

            df = pd.DataFrame(data, index=[0])
            if problem.quads:
                result_file = results + "_N%d_deg%d_quads.csv" % (problem.N, problem.degree)
            else:
                result_file = results + "_N%d_deg%d.csv" % (problem.N, problem.degree)

            df.to_csv(result_file, index=False, mode="w", header=True)

    PETSc.Sys.Print("Solving %s(degree=%s, size=%s, dimension=%s) finished.\n" %
                    (name, problem.degree, problem.N, problem.dim))

    PETSc.Sys.Print("L2 error: %s\n" % true_err)
    PETSc.Sys.Print("L2 error (post-processed): %s\n" % problem.pp_err)
    PETSc.Sys.Print("Algebraic error: %s\n" % err)
    PETSc.Sys.Print("Relative tolerance: %s\n" % rtol)


dim = args.dim
if dim == 3:
    # (degree, size, rtol)
    hdg_params = [(1, 4, 1.0e-3),
                  (1, 8, 1.0e-4),
                  (1, 16, 1.0e-5),
                  (1, 32, 1.0e-6),
                  (1, 64, 1.0e-7),
                  # Degree 2 set
                  (2, 4, 1.0e-5),
                  (2, 8, 1.0e-6),
                  (2, 16, 1.0e-7),
                  (2, 32, 1.0e-8),
                  (2, 64, 1.0e-9),
                  # Degree 3 set
                  (3, 4, 1.0e-7),
                  (3, 8, 1.0e-8),
                  (3, 16, 1.0e-9),
                  (3, 32, 1.0e-10),
                  (3, 64, 1.0e-11),
                  # Degree 4 set
                  (4, 4, 1.0e-9),
                  (4, 8, 1.0e-10),
                  (4, 16, 1.0e-11),
                  (4, 32, 1.0e-12),
                  (4, 64, 1.0e-13)]
else:
    raise NotImplementedError("Dim %s not set up yet." % dim)

quads = args.quads
for hdg_param in hdg_params:

    degree, size, rtol = hdg_param
    run_solver(problem_cls=problem_cls, degree=degree,
               size=size, rtol=rtol, quads=quads, dim=dim)
