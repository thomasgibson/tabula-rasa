"""
This script runs a nonlinear shallow water system describing
a simplified atmospheric model on an Earth-sized sphere mesh.
This model problem is designed from the Williamson test case
suite (1992), specifically the mountain test case (case 5).

A simple DG-advection scheme is used for the advection of the
depth-field, and an upwinded-DG method is used for velocity.
The nonlinear system is solved using a Picard method for computing
solution updates (currently set to 4 iterations). The implicit
midpoint rule is employed for time-integration.

The resulting implicit linear system for the updates in each
Picard iteration is solved using either a precondtioned Schur
complement method with GMRES, or a single application of a
Firedrake precondtioner using a mixed-hybrid method. This purpose
of this script is to compare both approaches via profiling and
computing the reductions in the problem residual for the implicit
linear system.
"""

from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, parameters
from argparse import ArgumentParser
from mpi4py import MPI
import pandas as pd
import sys

import solver as module


parameters["pyop2_options"]["lazy_evaluation"] = False


ref_to_dt = {3: 900.0,
             4: 450.0,
             5: 225.0,
             6: 112.5,
             7: 56.25}


PETSc.Log.begin()
parser = ArgumentParser(description="""Run Williamson test case 5""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--verbose",
                    action="store_true",
                    help="Turn on energy and output print statements.")

parser.add_argument("--model_degree",
                    action="store",
                    type=int,
                    default=2,
                    help="Degree of the finite element model.")

parser.add_argument("--method",
                    action="store",
                    default="BDM",
                    choices=["RT", "RTCF", "BDM"],
                    help="Mixed method type for the SWE.")

parser.add_argument("--dumpfreq",
                    default=10,
                    type=int,
                    action="store",
                    help="Dump frequency of output.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Start profile of all methods for 100 time-steps.")

parser.add_argument("--refinements",
                    action="store",
                    default=3,
                    type=int,
                    choices=[3, 4, 5, 6, 7],
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--write",
                    action="store_true",
                    help="Write output.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


def run_williamson5(problem, write=False):

    if args.profile:
        tmax = 10*problem.Dt
        PETSc.Sys.Print("Taking 10 time-steps\n")
    else:
        day = 24.*60.*60.
        tmax = 15*day
        PETSc.Sys.Print("Running 15 day simulation\n")

    problem.warmup()

    problem.run_simulation(tmax, write=write, dumpfreq=args.dumpfreq)

    if COMM_WORLD.rank == 0:
        PETSc.Log.Stage("Linear solve").push()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        ksp_time = ksp["time"]
        pc_setup_time = pcsetup["time"]
        pc_apply_time = pcapply["time"]
        ref = refinement_level
        if hybridization:
            results_data = "hybrid_%s_data_W5_ref%s.csv" % (method, ref)
            results_timings = "hybrid_%s_profile_W5_ref%s.csv" % (method, ref)
        else:
            results_data = "gmres_%s_data_W5_ref%s.csv" % (method, ref)
            results_timings = "gmres_%s_profile_W5_ref%s.csv" % (method, ref)

        data = {"OuterIters": problem.ksp_outer_its,
                "InnerIters": problem.ksp_inner_its,
                "PicardIters": problem.picard_seq,
                "SimTime": problem.sim_time,
                "ResidualReductions": problem.reductions}

        dofs = problem.DU.dof_dset.layout_vec.getSize()

        # Time spent computing diagnostic information
        diagnostic_time = (problem.time_assembling_residuals +
                           problem.time_writing_output +
                           problem.time_getting_ksp_info)

        time_data = {"PETScLogKSPSolve": ksp_time,
                     "PETSCLogPCApply": pc_apply_time,
                     "PETSCLogPCSetup": pc_setup_time,
                     "num_processes": problem.comm.size,
                     "method": problem.method,
                     "model_degree": problem.model_degree,
                     "refinement_level": problem.refinement_level,
                     "total_dofs": dofs,
                     # Time spent in just the linear solver bit.
                     "LinearSolve": problem.LinearSolve_time,
                     # Time spent setting up the stabilized residual RHS
                     # for the implicit linear system.
                     "DUResidual": problem.DUResidual_time,
                     # Total time to run the problem, including time
                     # spent computing residuals.
                     "TotalRunTime": problem.elapsed_time,
                     # To accurately record the time spend in the actual
                     # simulation, we remove the time spent computing
                     # residuals, writing output (if written), and
                     # time gathering KSP information.
                     "SimTime": problem.elapsed_time - diagnostic_time,
                     # We still record these here for reference.
                     "ComputingResiduals": problem.time_assembling_residuals,
                     "TimeWritingOutput": problem.time_writing_output,
                     "TimeGatheringKSPInfo": problem.time_getting_ksp_info}

        if hybridization:
            RHS = PETSc.Log.Event("HybridRHS").getPerfInfo()
            trace = PETSc.Log.Event("HybridSolve").getPerfInfo()
            recover = PETSc.Log.Event("HybridRecover").getPerfInfo()
            recon = PETSc.Log.Event("HybridRecon").getPerfInfo()
            reconstruction = recover["time"] + recon["time"]
            hybridupdate = PETSc.Log.Event("HybridUpdate").getPerfInfo()
            update_time = hybridupdate["time"]
            trace_solve = trace["time"]
            rhstime = RHS["time"]
            other = ksp_time - (update_time + trace_solve +
                                reconstruction + rhstime)
            updates = {"HybridTraceSolve": trace_solve,
                       "HybridRHS": rhstime,
                       "HybridReconstruction": reconstruction,
                       "HybridUpdate": update_time,
                       "HybridOther": other}

        else:
            KSPSchur = PETSc.Log.Event("KSPSolve_FS_Schu").getPerfInfo()
            schur_time = KSPSchur["time"]
            KSPF0 = PETSc.Log.Event("KSPSolve_FS_0").getPerfInfo()
            KSPLow = PETSc.Log.Event("KSPSolve_FS_Low").getPerfInfo()
            f0_time = KSPF0["time"]
            other = ksp_time - (schur_time + f0_time + KSPLow["time"])
            updates = {"KSPSchur": schur_time,
                       "KSPF0": f0_time,
                       "KSPother": other}

        time_data.update(updates)

        df_data = pd.DataFrame(data)
        df_data.to_csv(results_data, index=False,
                       mode="w", header=True)

        df_time = pd.DataFrame(time_data, index=[0])
        df_time.to_csv(results_timings, index=False,
                       mode="w", header=True)


if args.profile:
    param = (args.method, args.model_degree)
    method, model_degree = param
    ref_level = args.refinements
    Dt = ref_to_dt[ref_level]
    R = 6371220.0
    H = 5960.0

    W5Problem = module.W5Problem(refinement_level=ref_level,
                                 R=R,
                                 H=H,
                                 Dt=Dt,
                                 method=method,
                                 hybridization=args.hybridization,
                                 model_degree=model_degree)
    run_williamson5(W5Problem,
                    write=args.write)

else:
    Dt = ref_to_dt[args.refinements]
    R = 6371220.0
    H = 5960.0

    W5Problem = module.W5Problem(refinement_level=args.refinements,
                                 R=R,
                                 H=H,
                                 Dt=Dt,
                                 method=args.method,
                                 hybridization=args.hybridization,
                                 model_degree=args.model_degree)
    run_williamson5(W5Problem,
                    write=args.write)
