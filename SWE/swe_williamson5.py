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
                    default=900.0,
                    help="The time-step size.")

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

parser.add_argument("--nsteps",
                    action="store",
                    default=20,
                    type=int,
                    help="Number to time steps to take.")

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
                    choices=[3, 4, 5, 6, 7, 8],
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


PETSc.Log.begin()


def run_williamson5(problem_cls, Dt, refinements, method,
                    model_degree, nsteps,
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
                              model_degree=model_degree)
        problem.warmup()
        return

    problem = problem_cls(refinement_level=refinements,
                          R=R,
                          H=H,
                          Dt=Dt,
                          method=method,
                          hybridization=hybridization,
                          model_degree=model_degree)

    cfl = problem.courant
    dx_min = problem.dx_min
    dx_max = problem.dx_max

    PETSc.Sys.Print("""
Dt = %s,\n
Courant number (approximate): %s,\n
Dx (min): %s km,\n
Dx (max): %s km.
""" % (Dt, cfl, dx_min/1000, dx_max/1000))

    comm = problem.comm

    if args.profile:
        tmax = nsteps*Dt
    else:
        day = 24.*60.*60.
        tmax = 15*day
        PETSc.Sys.Print("Running 15 day simulation\n")

    PETSc.Sys.Print("Warm up with one-step.\n")
    with timed_stage("Warm up %s" % problem.name):
        problem.warmup()
        PETSc.Log.Stage("Warm up: Linear solve %s" % problem.name).push()
        prepcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pre_setup_time = comm.allreduce(prepcsetup["time"], op=MPI.SUM) / comm.size
        if problem.hybridization:
            prehybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()
            prehybridinit_time = comm.allreduce(prehybridinit["time"], op=MPI.SUM) / comm.size
        PETSc.Log.Stage("Warm up: Linear solve %s" % problem.name).pop()

    PETSc.Sys.Print("Warm up done. Profiling run for %d steps.\n" % nsteps)
    problem.initialize()
    problem.run_simulation(tmax, write=write, dumpfreq=args.dumpfreq)
    PETSc.Sys.Print("Simulation complete.\n")

    PETSc.Log.Stage("Linear solve %s" % problem.name).push()
    ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
    pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
    pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
    ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
    pc_setup_time = comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size
    pc_apply_time = comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size
    ref = problem.refinement_level

    num_cells = comm.allreduce(problem.num_cells, op=MPI.SUM)

    if problem.hybridization:
        results_data = "hybrid_%s%s_data_W5_ref%d_Dt%s_NS%d.csv" % (problem.method,
                                                                    problem.model_degree,
                                                                    ref,
                                                                    Dt,
                                                                    nsteps)
        results_timings = "hybrid_%s%s_profile_W5_ref%d_Dt%s_NS%d.csv" % (problem.method,
                                                                          problem.model_degree,
                                                                          ref,
                                                                          Dt,
                                                                          nsteps)

        RHS = PETSc.Log.Event("HybridRHS").getPerfInfo()
        trace = PETSc.Log.Event("HybridSolve").getPerfInfo()
        recover = PETSc.Log.Event("HybridRecover").getPerfInfo()
        recon = PETSc.Log.Event("HybridRecon").getPerfInfo()
        recon_scalar = PETSc.Log.Event("HybridReconScalarField").getPerfInfo()
        recon_flux = PETSc.Log.Event("HybridReconFluxField").getPerfInfo()
        hybridbreak = PETSc.Log.Event("HybridBreak").getPerfInfo()
        hybridupdate = PETSc.Log.Event("HybridUpdate").getPerfInfo()
        hybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()

        recon_time = comm.allreduce(recon["time"], op=MPI.SUM) / comm.size
        scalar_time = comm.allreduce(recon_scalar["time"], op=MPI.SUM) / comm.size
        flux_time = comm.allreduce(recon_flux["time"], op=MPI.SUM) / comm.size
        projection = comm.allreduce(recover["time"], op=MPI.SUM) / comm.size
        transfer = comm.allreduce(hybridbreak["time"], op=MPI.SUM) / comm.size
        full_recon = projection + recon_time
        update_time = comm.allreduce(hybridupdate["time"], op=MPI.SUM) / comm.size
        trace_solve = comm.allreduce(trace["time"], op=MPI.SUM) / comm.size
        rhstime = comm.allreduce(RHS["time"], op=MPI.SUM) / comm.size
        inittime = comm.allreduce(hybridinit["time"], op=MPI.SUM) / comm.size
        other = ksp_time - (trace_solve + transfer
                            + projection + recon_time + rhstime)
        full_solve = (transfer + trace_solve + rhstime
                      + recon_time + projection + update_time)
    else:
        results_data = "gmres_%s%s_data_W5_ref%d_Dt%s_NS%d.csv" % (problem.method,
                                                                   problem.model_degree,
                                                                   ref,
                                                                   Dt,
                                                                   nsteps)
        results_timings = "gmres_%s%s_profile_W5_ref%d_Dt%s_NS%d.csv" % (problem.method,
                                                                         problem.model_degree,
                                                                         ref,
                                                                         Dt,
                                                                         nsteps)

        KSPSchur = PETSc.Log.Event("KSPSolve_FS_Schu").getPerfInfo()
        KSPF0 = PETSc.Log.Event("KSPSolve_FS_0").getPerfInfo()
        KSPLow = PETSc.Log.Event("KSPSolve_FS_Low").getPerfInfo()
        KSPOrthog = PETSc.Log.Event("KSPGMRESOrthog").getPerfInfo()

        schur_time = comm.allreduce(KSPSchur["time"], op=MPI.SUM) / comm.size
        f0_time = comm.allreduce(KSPF0["time"], op=MPI.SUM) / comm.size
        ksplow_time = comm.allreduce(KSPLow["time"], op=MPI.SUM) / comm.size
        gmresortho = comm.allreduce(KSPOrthog["time"], op=MPI.SUM) / comm.size
        other = ksp_time - (schur_time + f0_time + ksplow_time + gmresortho)

    PETSc.Log.Stage("Linear solve %s" % problem.name).pop()
    if COMM_WORLD.rank == 0:
        data = {"OuterIters": problem.ksp_outer_its,
                "InnerIters": problem.ksp_inner_its,
                "PicardIters": problem.picard_seq,
                "SimTime": problem.sim_time,
                "ResidualReductions": problem.reductions}

        dofs = problem.DU.dof_dset.layout_vec.getSize()

        time_data = {"PETScLogKSPSolve": ksp_time,
                     "PETSCLogPCApply": pc_apply_time,
                     "PETSCLogPCSetup": pc_setup_time,
                     "PETSCLogPreSetup": pre_setup_time,
                     "num_processes": problem.comm.size,
                     "method": problem.method,
                     "model_degree": problem.model_degree,
                     "refinement_level": problem.refinement_level,
                     "total_dofs": dofs,
                     "num_cells": num_cells,
                     "Dt": Dt,
                     "CFL": cfl,
                     "DxMin": dx_min,
                     "DxMax": dx_max,
                     "DxAvg": problem.dx_avg}

        if problem.hybridization:
            updates = {"HybridTraceSolve": trace_solve,
                       "HybridRHS": rhstime,
                       "HybridBreak": transfer,
                       "HybridReconstruction": recon_time,
                       "HybridReconScalarField": scalar_time,
                       "HybridReconFluxField": flux_time,
                       "HybridProjection": projection,
                       "HybridFullRecovery": full_recon,
                       "HybridUpdate": update_time,
                       "HybridInit": inittime,
                       "PreHybridInit": prehybridinit_time,
                       "HybridFullSolveTime": full_solve,
                       "HybridKSPOther": other}

        else:
            updates = {"KSPSchur": schur_time,
                       "KSPF0": f0_time,
                       "KSPFSLow": ksplow_time,
                       "KSPGMRESOrthog": gmresortho,
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
hybridization = args.hybridization
nsteps = args.nsteps

if args.profile:
    ref_to_Dt = {5: 225,
                 6: 112.5,
                 7: 56.25,
                 8: 28.125}
    for refinements in [5, 6, 7, 8]:
        run_williamson5(problem_cls=W5Problem,
                        Dt=ref_to_Dt[refinements],
                        refinements=refinements,
                        method=method,
                        model_degree=model_degree,
                        nsteps=nsteps,
                        hybridization=hybridization,
                        write=False,
                        # Do a cold run to generate code
                        cold=True)

        # Now start the profile
        run_williamson5(problem_cls=W5Problem,
                        Dt=ref_to_Dt[refinements],
                        refinements=refinements,
                        method=method,
                        model_degree=model_degree,
                        nsteps=nsteps,
                        hybridization=hybridization,
                        write=False,
                        cold=False)

else:
    refinements = args.refinements
    Dt = args.dt
    run_williamson5(problem_cls=W5Problem,
                    Dt=Dt,
                    refinements=refinements,
                    method=method,
                    model_degree=model_degree,
                    nsteps=nsteps,
                    hybridization=hybridization,
                    write=args.write,
                    cold=False)
