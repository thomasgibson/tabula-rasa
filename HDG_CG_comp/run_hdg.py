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

results = os.path.abspath(args.results_file)

warm = defaultdict(bool)


PETSc.Log.begin()


def run_solver(problem_cls, degree, size, rtol, quads, dim, cold=False):

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
              'pc_python_type':  'firedrake.SCPC',
              'pc_sc_eliminate_fields': '0, 1',
              'condensed_field': pcg_params}

    problem = problem_cls(degree=degree, N=size,
                          quadrilaterals=quads, dimension=dim)
    name = getattr(problem, "name")
    solver = problem.solver(parameters=params)

    if cold:
        PETSc.Sys.Print("""
Running cold solve on coarse mesh for degree %d.\n
""" % degree)
        solver.solve()
        problem.post_processed_sol()
        return

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
    warm_stage = "%s(deg=%s, N=%s, dim=%s) Warm solve\n" % (name,
                                                            degree,
                                                            size,
                                                            dim)
    with PETSc.Log.Stage(warm_stage):
        solver.solve()

        snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

        comm = problem.comm
        snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
        ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
        pcsetup_time = comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size
        pcapply_time = comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size
        jac_time = comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size
        res_time = comm.allreduce(residual["time"], op=MPI.SUM) / comm.size

        num_cells = comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)
        err = problem.err
        true_err = problem.true_err

        # HDG-specific timings
        HDGinit = PETSc.Log.Event("SCPCInit").getPerfInfo()
        HDGUpdate = PETSc.Log.Event("SCPCUpdate").getPerfInfo()
        HDGrhs = PETSc.Log.Event("SCForwardElim").getPerfInfo()
        HDGrecon = PETSc.Log.Event("SCBackSub").getPerfInfo()
        HDGSolve = PETSc.Log.Event("SCSolve").getPerfInfo()

        hdginit_time = comm.allreduce(HDGinit["time"], op=MPI.SUM) / comm.size
        hdgrhs_time = comm.allreduce(HDGrhs["time"], op=MPI.SUM) / comm.size
        hdgrecon_time = comm.allreduce(HDGrecon["time"], op=MPI.SUM) / comm.size
        hdgsolve_time = comm.allreduce(HDGSolve["time"], op=MPI.SUM) / comm.size
        hdgupdate_time = comm.allreduce(HDGUpdate["time"], op=MPI.SUM) / comm.size

        # Should total to KSPSolve time (approximately)
        hdg_total_solve = (hdginit_time + hdgrhs_time + hdgsolve_time
                           + hdgrecon_time + hdgupdate_time)

        problem.post_processed_sol()

        HDGPP = PETSc.Log.Event("HDGPostprocessing").getPerfInfo()
        pp_time = comm.allreduce(HDGPP["time"], op=MPI.SUM) / comm.size

        # Total HDG time (with pp)
        hdg_total_time = hdg_total_solve + pp_time

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            q_h, u_h, lambdar_h = problem.u.split()

            ksp = solver.snes.ksp.getPC().getPythonContext().condensed_ksp
            data = {"SNESSolve": snes_time,
                    "KSPSolve": ksp_time,
                    "PCSetUp": pcsetup_time,
                    "PCApply": pcapply_time,
                    "SNESJacobianEval": jac_time,
                    "SNESFunctionEval": res_time,
                    "num_processes": problem.comm.size,
                    "mesh_size": problem.N,
                    "num_cells": num_cells,
                    "degree": problem.degree,
                    "scalar_dofs": u_h.dof_dset.layout_vec.getSize(),
                    "flux_dofs": q_h.dof_dset.layout_vec.getSize(),
                    "trace_dofs": lambdar_h.dof_dset.layout_vec.getSize(),
                    "name": problem.name,
                    "disc_error_u": err,
                    "true_err_u": true_err,
                    "HDGInit": hdginit_time,
                    "HDGUpdate": hdgupdate_time,
                    "HDGRhs": hdgrhs_time,
                    "HDGRecover": hdgrecon_time,
                    "HDGTraceSolve": hdgsolve_time,
                    "HDGPPTime": pp_time,
                    "HDGTotal": hdg_total_time,
                    "ErrorPP": problem.pp_err,
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

    PETSc.Sys.Print("L2 error: %s\n" % true_err)
    PETSc.Sys.Print("L2 error (post-processed): %s\n" % problem.pp_err)
    PETSc.Sys.Print("Algebraic error: %s\n" % err)


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

problem_cls = module.HDGProblem
quads = args.quads
for cold_param in cold_params:
    degree, size, rtol = cold_param
    run_solver(problem_cls=problem_cls, degree=degree,
               size=size, rtol=rtol, quads=quads, dim=dim,
               cold=True)

# Now we profile once the code has been generated
for hdg_param in hdg_params:

    degree, size, rtol = hdg_param
    run_solver(problem_cls=problem_cls, degree=degree,
               size=size, rtol=rtol, quads=quads, dim=dim,
               cold=False)
