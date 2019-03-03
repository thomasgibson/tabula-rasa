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
from pyop2.profiling import timed_stage
from mpi4py import MPI
import pandas as pd
import sys
import os

import solver as module


parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Run Williamson test case 5""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--dt",
                    action="store",
                    type=float,
                    default=1000.0,
                    help="The time-step size.")

parser.add_argument("--mesh_degree",
                    action="store",
                    type=int,
                    default=1,
                    help="Degree of the mesh")

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

parser.add_argument("--profile",
                    action="store_true",
                    help="Start profiler.")

parser.add_argument("--nsteps",
                    action="store",
                    default=20,
                    type=int,
                    help="Number of steps to profile.")

parser.add_argument("--refinements",
                    action="store",
                    default=4,
                    type=int,
                    choices=[3, 4, 5, 6, 7, 8],
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--write",
                    action="store_true",
                    help="Write output.")

parser.add_argument("--monitor",
                    action="store_true",
                    help="Turn on KSP monitors for debugging")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


PETSc.Log.begin()


def run_williamson5(problem_cls, Dt, refinements, method,
                    model_degree, mesh_degree, nsteps,
                    hybridization, write=False, cold=False):

    # Radius of the Earth (m)
    R = 6371220.0

    # Max depth height (m)
    H = 5960.0

    if cold:
        PETSc.Sys.Print("""
Running cold initialization for the problem set:\n
method: %s,\n
model degree: %s,\n
hybridization: %s,\n
""" % (method, model_degree, bool(hybridization)))
        problem = problem_cls(refinement_level=refinements,
                              R=R,
                              H=H,
                              Dt=Dt,
                              method=method,
                              hybridization=hybridization,
                              model_degree=model_degree,
                              mesh_degree=mesh_degree,
                              monitor=args.monitor)
        problem.warmup()
        return

    problem = problem_cls(refinement_level=refinements,
                          R=R,
                          H=H,
                          Dt=Dt,
                          method=method,
                          hybridization=hybridization,
                          model_degree=model_degree,
                          mesh_degree=mesh_degree,
                          monitor=args.monitor)

    cfl = problem.courant
    dx_max = problem.dx_max

    PETSc.Sys.Print("""
Dt = %s,\n
Courant number (approximate): %s,\n
Dx (max): %s km,\n
nsteps: %s.
""" % (Dt, cfl, dx_max/1000, nsteps))

    comm = problem.comm
    day = 24.*60.*60.

    if args.profile:
        tmax = nsteps*Dt
    else:
        tmax = 15*day
        PETSc.Sys.Print("Running 15 day simulation\n")

    # If writing simulation output, write out fields in 5-day intervals
    dumpfreq = 5*day / Dt

    PETSc.Sys.Print("Warm up with one-step.\n")
    with timed_stage("Warm up"):
        problem.warmup()
        PETSc.Log.Stage("Warm up: Linear solve").push()
        prepcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pre_res_eval = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        pre_jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()

        pre_res_eval_time = comm.allreduce(pre_res_eval["time"],
                                           op=MPI.SUM) / comm.size
        pre_jac_eval_time = comm.allreduce(pre_jac_eval["time"],
                                           op=MPI.SUM) / comm.size
        pre_setup_time = comm.allreduce(prepcsetup["time"],
                                        op=MPI.SUM) / comm.size

        if problem.hybridization:
            prehybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()
            prehybridinit_time = comm.allreduce(prehybridinit["time"],
                                                op=MPI.SUM) / comm.size

        PETSc.Log.Stage("Warm up: Linear solve").pop()

    PETSc.Sys.Print("Warm up done. Profiling run for %d steps.\n" % nsteps)
    problem.initialize()
    problem.run_simulation(tmax, write=write, dumpfreq=dumpfreq)
    PETSc.Sys.Print("Simulation complete.\n")

    PETSc.Log.Stage("Linear solve").push()

    snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
    ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
    pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
    pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
    jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
    residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

    snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
    ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
    pc_setup_time = comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size
    pc_apply_time = comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size
    jac_eval_time = comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size
    res_eval_time = comm.allreduce(residual["time"], op=MPI.SUM) / comm.size

    ref = problem.refinement_level
    num_cells = comm.allreduce(problem.num_cells, op=MPI.SUM)

    if problem.hybridization:
        results_data = "results/hybrid_%s_data_W5_ref%d_Dt%s_NS%d.csv" % (
            problem.method,
            ref,
            Dt,
            nsteps
        )
        results_timings = "results/hybrid_%s_profile_W5_ref%d_Dt%s_NS%d.csv" % (
            problem.method,
            ref,
            Dt,
            nsteps
        )

        RHS = PETSc.Log.Event("HybridRHS").getPerfInfo()
        trace = PETSc.Log.Event("SCSolve").getPerfInfo()
        proj = PETSc.Log.Event("HybridProject").getPerfInfo()
        full_recon = PETSc.Log.Event("SCBackSub").getPerfInfo()
        hybridbreak = PETSc.Log.Event("HybridBreak").getPerfInfo()
        hybridupdate = PETSc.Log.Event("HybridUpdate").getPerfInfo()
        hybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()

        # Time to reconstruct (backsub) and project
        full_recon_time = comm.allreduce(full_recon["time"],
                                         op=MPI.SUM) / comm.size
        # Project only
        projection = comm.allreduce(proj["time"], op=MPI.SUM) / comm.size
        # Backsub only = Total Recon time - projection time
        recon_time = full_recon_time - projection

        transfer = comm.allreduce(hybridbreak["time"], op=MPI.SUM) / comm.size
        update_time = comm.allreduce(hybridupdate["time"],
                                     op=MPI.SUM) / comm.size
        trace_solve = comm.allreduce(trace["time"], op=MPI.SUM) / comm.size
        rhstime = comm.allreduce(RHS["time"], op=MPI.SUM) / comm.size
        inittime = comm.allreduce(hybridinit["time"], op=MPI.SUM) / comm.size
        other = ksp_time - (trace_solve + transfer
                            + projection + recon_time + rhstime)
        full_solve = (transfer + trace_solve + rhstime
                      + recon_time + projection + update_time)
    else:
        results_data = "results/gmres_%s_data_W5_ref%d_Dt%s_NS%d.csv" % (
            problem.method,
            ref,
            Dt,
            nsteps
        )
        results_timings = "results/gmres_%s_profile_W5_ref%d_Dt%s_NS%d.csv" % (
            problem.method,
            ref,
            Dt,
            nsteps
        )

        KSPSchur = PETSc.Log.Event("KSPSolve_FS_Schu").getPerfInfo()
        KSPF0 = PETSc.Log.Event("KSPSolve_FS_0").getPerfInfo()
        KSPLow = PETSc.Log.Event("KSPSolve_FS_Low").getPerfInfo()

        schur_time = comm.allreduce(KSPSchur["time"], op=MPI.SUM) / comm.size
        f0_time = comm.allreduce(KSPF0["time"], op=MPI.SUM) / comm.size
        ksplow_time = comm.allreduce(KSPLow["time"], op=MPI.SUM) / comm.size
        other = ksp_time - (schur_time + f0_time + ksplow_time)

    PETSc.Log.Stage("Linear solve").pop()
    if COMM_WORLD.rank == 0:

        if not os.path.exists(os.path.dirname('results/')):
            os.makedirs(os.path.dirname('results/'))

        data = {"OuterIters": problem.ksp_outer_its,
                "InnerIters": problem.ksp_inner_its,
                "PicardIters": problem.picard_seq,
                "SimTime": problem.sim_time,
                "ResidualReductions": problem.reductions}

        dofs = problem.DU.dof_dset.layout_vec.getSize()

        time_data = {"PETSCLogKSPSolve": ksp_time,
                     "PETSCLogPCApply": pc_apply_time,
                     "PETSCLogPCSetup": pc_setup_time,
                     "PETSCLogPreSetup": pre_setup_time,
                     "PETSCLogPreSNESJacobianEval": pre_jac_eval_time,
                     "PETSCLogPreSNESFunctionEval": pre_res_eval_time,
                     "SNESSolve": snes_time,
                     "SNESFunctionEval": res_eval_time,
                     "SNESJacobianEval": jac_eval_time,
                     "num_processes": problem.comm.size,
                     "method": problem.method,
                     "model_degree": problem.model_degree,
                     "refinement_level": problem.refinement_level,
                     "total_dofs": dofs,
                     "num_cells": num_cells,
                     "Dt": Dt,
                     "CFL": cfl,
                     "nsteps": nsteps,
                     "DxMax": dx_max}

        if problem.hybridization:
            updates = {"HybridTraceSolve": trace_solve,
                       "HybridRHS": rhstime,
                       "HybridBreak": transfer,
                       "HybridReconstruction": recon_time,
                       "HybridProjection": projection,
                       "HybridFullRecovery": full_recon_time,
                       "HybridUpdate": update_time,
                       "HybridInit": inittime,
                       "PreHybridInit": prehybridinit_time,
                       "HybridFullSolveTime": full_solve,
                       "HybridKSPOther": other}

        else:
            updates = {"KSPSchur": schur_time,
                       "KSPF0": f0_time,
                       "KSPFSLow": ksplow_time,
                       "KSPother": other}

        time_data.update(updates)

        df_data = pd.DataFrame(data)
        df_data.to_csv(results_data, index=False,
                       mode="w", header=True)

        df_time = pd.DataFrame(time_data, index=[0])
        df_time.to_csv(results_timings, index=False,
                       mode="w", header=True)


W5Problem = module.W5Problem
method = args.method
model_degree = args.model_degree
mesh_degree = args.mesh_degree
refinements = args.refinements
hybridization = args.hybridization
Dt = args.dt

if args.profile:

    run_williamson5(problem_cls=W5Problem,
                    Dt=Dt,
                    refinements=refinements,
                    method=method,
                    model_degree=model_degree,
                    mesh_degree=mesh_degree,
                    nsteps=args.nsteps,
                    hybridization=hybridization,
                    write=False,
                    # Do a cold run to generate code
                    cold=True)

    # Now start the profiler
    run_williamson5(problem_cls=W5Problem,
                    Dt=Dt,
                    refinements=refinements,
                    method=method,
                    model_degree=model_degree,
                    mesh_degree=mesh_degree,
                    nsteps=args.nsteps,
                    hybridization=hybridization,
                    write=False,
                    cold=False)

else:
    run_williamson5(problem_cls=W5Problem,
                    Dt=Dt,
                    refinements=refinements,
                    method=method,
                    model_degree=model_degree,
                    mesh_degree=mesh_degree,
                    nsteps=args.nsteps,
                    hybridization=hybridization,
                    write=args.write,
                    cold=False)
