"""
This script runs a linear compressible Boussinesq system describing
a simplified atmospheric model on an Earth-sized sphere mesh.
This model problem is designed from the Skamarock and Klemp
gravity wave test case. This is also the model for the DCMIP
test case 3-1.

We mimic the UK Met Office's approach by point-wise eliminating
the buoyancy variable and solve a coupled mixed system for the
velocity and pressure. Once that system is solve, the buoyancy
variable is reconstructed using the previously computed fields.
"""

from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, parameters
from argparse import ArgumentParser
from pyop2.profiling import timed_stage
from mpi4py import MPI
import pandas as pd
import sys
import os

import problem as module


parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Run the gravity wave test""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--X",
                    action="store",
                    default=125.0,
                    type=float,
                    help="Factor to scale the Earth's radius")

parser.add_argument("--H",
                    action="store",
                    default=1.0e4,
                    type=float,
                    help="Atmospheric lid")

parser.add_argument("--dt",
                    action="store",
                    type=float,
                    default=100.0,
                    help="The time-step size.")

parser.add_argument("--cfl",
                    action="store",
                    default=1.0,
                    type=float,
                    help="Horizontal CFL number")

parser.add_argument("--rtol",
                    action="store",
                    default=1.0e-5,
                    type=float,
                    help="Solver rtolerance for the u-p system.")

parser.add_argument("--use_dt_from_cfl",
                    action="store_true",
                    help="Overwrite Dt and deduce a value of Dt using CFL")

parser.add_argument("--model_degree",
                    action="store",
                    type=int,
                    default=1,
                    help="Degree of the finite element model.")

parser.add_argument("--method",
                    action="store",
                    default="RTCF",
                    choices=["RT", "RTCF", "BDFM"],
                    help="Mixed method type.")

parser.add_argument("--tmax",
                    action="store",
                    default=3600.0,
                    type=float,
                    help="Max time.")

parser.add_argument("--refinements",
                    action="store",
                    default=4,
                    type=int,
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--nlayers",
                    action="store",
                    default=20,
                    type=int,
                    help="Number of vertical layers.")

parser.add_argument("--write",
                    action="store_true",
                    help="Write output.")

parser.add_argument("--monitor",
                    action="store_true",
                    help="Turn on KSP monitors")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")


args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


PETSc.Log.begin()


def run_gravity_waves(problem_cls, Dt, cfl, refinements, nlayers, method,
                      model_degree, hybridization, write=False):

    # Max height (m)
    thickness = args.H

    problem = problem_cls(refinement_level=refinements,
                          nlayers=nlayers,
                          Dt=Dt,
                          method=method,
                          X=args.X,
                          thickness=thickness,
                          model_degree=model_degree,
                          rtol=args.rtol,
                          hybridization=hybridization,
                          cfl=cfl,
                          monitor=args.monitor,
                          use_dt_from_cfl=args.use_dt_from_cfl)

    Dt = problem.Dt
    cfl = problem.courant
    dx_max = problem.dx_max
    dz = problem.dz

    tmax = args.tmax

    PETSc.Sys.Print("""
Dt = %s,\n
Horizontal Courant number (approximate): %s,\n
Dx (max): %s km,
Dz: %s m,\n
tmax: %s s
""" % (Dt, cfl, dx_max/1000, dz, tmax))

    dumpfreq = 100 / Dt
    comm = problem.comm

    PETSc.Sys.Print("Warm up with one-step.\n")
    with timed_stage("Warm up"):
        problem.warmup()
        PETSc.Log.Stage("Warm up: Solver").push()
        PETSc.Log.Stage("UP Solver").push()
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

        PETSc.Log.Stage("UP Solver").pop()
        PETSc.Log.Stage("Warm up: Solver").pop()

    # Run the problem
    problem.run_simulation(tmax, write=write, dumpfreq=dumpfreq)

    PETSc.Sys.Print("Simulation complete.\n")

    PETSc.Log.Stage("UP Solver").push()

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
    nlayers = problem.nlayers
    num_cells = comm.allreduce(problem.num_cells, op=MPI.SUM)

    if problem.hybridization:
        results_data = "results/hybrid_%s_data_GW_ref%d_nlayers%d_CFL%d" % (
            problem.method,
            ref,
            nlayers,
            cfl
        )
        results_timings = "results/hybrid_%s_profile_GW_ref%d_nlayers%d_CFL%d" % (
            problem.method,
            ref,
            nlayers,
            cfl
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
        results_data = "results/gmres_%s_data_GW_ref%d_nlayers%d_CFL%d" % (
            problem.method,
            ref,
            nlayers,
            cfl
        )
        results_timings = "results/gmres_%s_profile_GW_ref%d_nlayers%d_CFL%d" % (
            problem.method,
            ref,
            nlayers,
            cfl
        )

        KSPSchur = PETSc.Log.Event("KSPSolve_FS_Schu").getPerfInfo()
        KSPF0 = PETSc.Log.Event("KSPSolve_FS_0").getPerfInfo()
        KSPLow = PETSc.Log.Event("KSPSolve_FS_Low").getPerfInfo()

        schur_time = comm.allreduce(KSPSchur["time"], op=MPI.SUM) / comm.size
        f0_time = comm.allreduce(KSPF0["time"], op=MPI.SUM) / comm.size
        ksplow_time = comm.allreduce(KSPLow["time"], op=MPI.SUM) / comm.size
        other = ksp_time - (schur_time + f0_time + ksplow_time)

    PETSc.Log.Stage("UP Solver").pop()

    results_data += ".csv"
    results_timings += ".csv"

    if COMM_WORLD.rank == 0:

        if not os.path.exists(os.path.dirname('results/')):
            os.makedirs(os.path.dirname('results/'))

        data = {"OuterIters": problem.ksp_outer_its,
                "InnerIters": problem.ksp_inner_its,
                "SimTime": problem.sim_time}

        _u, _p, _b = problem.state.split()
        up_dofs = (_u.dof_dset.layout_vec.getSize() +
                   _p.dof_dset.layout_vec.getSize())
        b_dofs = _b.dof_dset.layout_vec.getSize()
        dofs = b_dofs + up_dofs

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
                     "up_dofs": up_dofs,
                     "b_dofs": b_dofs,
                     "num_cells": num_cells,
                     "Dt": Dt,
                     "CFL": cfl,
                     "DxMax": dx_max,
                     "Dz": dz}

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


GWProblem = module.GravityWaveProblem
method = args.method
model_degree = args.model_degree
refinements = args.refinements
nlayers = args.nlayers
hybridization = args.hybridization
Dt = args.dt
cfl = args.cfl

run_gravity_waves(problem_cls=GWProblem,
                  Dt=Dt,
                  cfl=cfl,
                  refinements=refinements,
                  nlayers=nlayers,
                  method=method,
                  model_degree=model_degree,
                  hybridization=hybridization,
                  write=args.write)
