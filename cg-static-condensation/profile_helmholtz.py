"""This module is an adapted profiling set up authored by
Lawrence Mitchell, a co-author of the paper where these results
will be published.
"""
from helmholtz import HelmholtzProblem

from argparse import ArgumentParser
from collections import defaultdict
from firedrake import COMM_WORLD, parameters
from firedrake.petsc import PETSc
from mpi4py import MPI

import os
import _pickle as cPickle
import pandas


parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()


problem = HelmholtzProblem()
parser = ArgumentParser(description="""Profile 3D Helmholtz solve""",
                        add_help=False)


parser.add_argument("--results-file",
                    action="store",
                    default="helmholtz-timings.csv",
                    help="Where to put the results")

parser.add_argument("--overwrite",
                    action="store_true",
                    default=False,
                    help="Overwrite existing output? Default is to append.")

parser.add_argument("--parameters",
                    default=None,
                    action="store",
                    help="Select parameter set 'scpc_hypre' or 'hypre'.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")


args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)


if args.parameters is not None:
    if args.parameters not in problem.parameter_names:
        raise ValueError("Unrecognized parameter '%s', not in %s",
                         args.parameters,
                         problem.parameter_names)
    parameter_names = [args.parameters]

else:
    parameter_names = problem.parameter_names


results = os.path.abspath(args.results_file)


warm = defaultdict(bool)


def run_helmholtz_solve(problem, degree, mesh_size):
    problem.re_initialize(degree, mesh_size)

    first = True
    for param_name in parameter_names:
        parameters = getattr(problem, param_name)
        solver = problem.solver(parameters=parameters)
        PETSc.Sys.Print(
            "\nSolving 3D %s problem of degree %s, mesh size %s, "
            "and parameter set %s\n" % (problem.name,
                                        problem.degree,
                                        problem.mesh_size,
                                        param_name)
        )
        if not warm[(param_name, degree)]:
            PETSc.Sys.Print("\nWarm up run...\n")
            problem.u.assign(0.0)

            with PETSc.Log.Stage("Warmup"):
                try:
                    solver.solve()
                except:
                    PETSc.Sys.Print(
                        "\nError: Unable to solve %s problem "
                        "with degree %s, mesh size %s, and "
                        "parameter set %s" % (problem.name,
                                              problem.degree,
                                              problem.mesh_size,
                                              param_name)
                    )
                    PETSc.Sys.Print("************************************")
                    import traceback
                    PETSc.Sys.Print(*traceback.format_stack())
                    PETSc.Sys.Print("************************************")
                    continue

            warm[(param_name, degree)] = True

        problem.u.assign(0.0)

        PETSc.Sys.Print("Timing the solve.\n")
        solver.snes.setConvergenceHistory()
        solver.snes.ksp.setConvergenceHistory()
        with PETSc.Log.Stage("%s(%d, %d) Warm solve %s" % (problem.name,
                                                           degree,
                                                           mesh_size,
                                                           param_name)):
            try:
                solver.solve()

                # Collect events
                size = problem.comm.size
                snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
                ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
                pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
                pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
                jac = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
                residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
                scrhs = PETSc.Log.Event("SCRHS").getPerfInfo()
                scsolve = PETSc.Log.Event("SCSolve").getPerfInfo()
                screcover = PETSc.Log.Event("SCRecover").getPerfInfo()

                # Collect times
                snes_time = problem.comm.allreduce(snes["time"],
                                                   op=MPI.SUM)/size
                jac_time = problem.comm.allreduce(jac["time"],
                                                  op=MPI.SUM)/problem.comm.size
                residual_time = problem.comm.allreduce(residual["time"],
                                                       op=MPI.SUM)/size
                ksp_time = problem.comm.allreduce(ksp["time"],
                                                  op=MPI.SUM)/size
                pcsetup_time = problem.comm.allreduce(pcsetup["time"],
                                                      op=MPI.SUM)/size
                pcapply_time = problem.comm.allreduce(pcapply["time"],
                                                      op=MPI.SUM)/size
                scrhs_time = problem.comm.allreduce(scrhs["time"],
                                                    op=MPI.SUM)/size
                scsolve_time = problem.comm.allreduce(scsolve["time"],
                                                      op=MPI.SUM)/size
                screcover_time = problem.comm.allreduce(screcover["time"],
                                                        op=MPI.SUM)/size

                newton_its = solver.snes.getIterationNumber()
                ksp_its = solver.snes.getLinearSolveIterations()

                cell_set_size = problem.mesh.cell_set.size
                num_cells = problem.comm.allreduce(cell_set_size,
                                                   op=MPI.SUM)

                if COMM_WORLD.rank == 0:
                    if not os.path.exists(os.path.dirname(results)):
                        os.makedirs(os.path.dirname(results))

                    if args.overwrite:
                        if first:
                            mode = "w"
                            header = True
                        else:
                            mode = "a"
                            header = False
                        first = False
                    else:
                        mode = "a"
                        header = not os.path.exists(results)

                    conv_history = solver.snes.getConvergenceHistory()
                    snes_history, linear_its = conv_history
                    ksp_history = solver.snes.ksp.getConvergenceHistory()
                    data = {"snes_its": newton_its,
                            "ksp_its": ksp_its,
                            "snes_history": cPickle.dumps(snes_history),
                            "linear_its": cPickle.dumps(linear_its),
                            "ksp_history": cPickle.dumps(ksp_history),
                            "SNESSolve": snes_time,
                            "KSPSolve": ksp_time,
                            "PCSetUp": pcsetup_time,
                            "PCApply": pcapply_time,
                            "JacobianEval": jac_time,
                            "FunctionEval": residual_time,
                            "num_processes": problem.comm.size,
                            "mesh_size": problem.mesh_size,
                            "num_cells": num_cells,
                            "degree": problem.degree,
                            "solver_parameters": cPickle.dumps(solver.parameters),
                            "parameter_name": param_name,
                            "dofs": problem.u.dof_dset.layout_vec.getSize(),
                            "name": problem.name,
                            "SCPC_rhs": scrhs_time,
                            "SCPC_solve": scsolve_time,
                            "SCPC_recover": screcover_time}

                    if param_name == "scpc_hypre":
                        SCPC = solver.snes.ksp.getPC()
                        scpc_cxt = SCPC.getPythonContext()
                        sc_ksp = scpc_cxt.sc_ksp
                        sc_ksp_its = sc_ksp.getIterationNumber()
                        data.update({"SCPC_ksp_its": sc_ksp_its})

                    df = pandas.DataFrame(data, index=[0])
                    df.to_csv(results, index=False, mode=mode, header=header)
            except:
                PETSc.Sys.Print(
                    "\nError: Unable to solve %s problem "
                    "with degree %s, mesh size %s, and "
                    "parameter set %s" % (problem.name,
                                          problem.degree,
                                          problem.mesh_size,
                                          param_name)
                )
                PETSc.Sys.Print("************************************")
                import traceback
                PETSc.Sys.Print(*traceback.format_stack())
                PETSc.Sys.Print("************************************")
                continue
        PETSc.Sys.Print(
            "\nSolver complete for the 3D %s problem of degree %s, "
            "mesh size %s, and parameter set %s\n" % (problem.name,
                                                      problem.degree,
                                                      problem.mesh_size,
                                                      param_name)
        )


for size in [16]:
    for degree in range(4, 5):
        run_helmholtz_solve(problem, degree, size)
