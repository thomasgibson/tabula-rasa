from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, parameters
from argparse import ArgumentParser
from pyop2.profiling import timed_stage
from mpi4py import MPI
import pandas as pd
import sys
import os

from profile_problem import ProfileGravityWaveSolver as Solver


parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Profile the gravity wave solver.""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--X",
                    action="store",
                    default=1.0,
                    type=float,
                    help="Factor to scale the Earth's radius")

parser.add_argument("--H",
                    action="store",
                    default=1.0e4,
                    type=float,
                    help="Atmospheric lid")

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

solver = Solver(refinement_level=args.refinements,
                nlayers=args.nlayers,
                model_degree=args.model_degree,
                method=args.method,
                X=args.X,
                H=args.H,
                rtol=args.rtol,
                hybridization=args.hybridization,
                cfl=args.cfl,
                monitor=args.monitor)

PETSc.Sys.Print("""Warming up solver with parameters:\n
Planet radius (m): %s,\n
Atmospheric lid (m) %s,\n
Mixed method: %s,\n
Model degree: %s,\n
Hybridization: %s,\n
Horizontal CFL: %s,\n
Dt (s): %s,\n
Dx (km): %s,\n
Dz (m): %s,\n
Solver rtol: %s,\n
KSP monitor: %s.
""" % (solver._R,
       solver.H,
       solver.method,
       solver.model_degree,
       solver.hybridization,
       solver.courant,
       solver.Dt,
       solver.dx_max / 1000,
       solver.dz,
       solver.rtol,
       solver.monitor))

with timed_stage("Warm up"):
    solver.warmup()

PETSc.Sys.Print("""Warm up complete. Profiling linear solver.""")

solver.run_profile()

PETSc.Sys.Print("""Run complete. Extracting data.""")

PETSc.Log.Stage("UP Solver").push()

comm = solver.comm

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

ref = solver.refinement_level
nlayers = solver.nlayers
num_cells = comm.allreduce(solver.num_cells, op=MPI.SUM)

if solver.hybridization:

    results_timings = "results/hybrid_%s%d_GW_ref%d_nlayers%d_CFL%d" % (
        solver.method,
        solver.model_degree,
        ref,
        nlayers,
        solver.courant
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

    results_timings = "results/gmres_%s%d_GW_ref%d_nlayers%d_CFL%d" % (
        solver.method,
        solver.model_degree,
        ref,
        nlayers,
        solver.courant
    )

    KSPSchur = PETSc.Log.Event("KSPSolve_FS_Schu").getPerfInfo()
    KSPF0 = PETSc.Log.Event("KSPSolve_FS_0").getPerfInfo()
    KSPLow = PETSc.Log.Event("KSPSolve_FS_Low").getPerfInfo()

    schur_time = comm.allreduce(KSPSchur["time"], op=MPI.SUM) / comm.size
    f0_time = comm.allreduce(KSPF0["time"], op=MPI.SUM) / comm.size
    ksplow_time = comm.allreduce(KSPLow["time"], op=MPI.SUM) / comm.size
    other = ksp_time - (schur_time + f0_time + ksplow_time)

PETSc.Log.Stage("UP Solver").pop()

results_timings += ".csv"

if COMM_WORLD.rank == 0:

    if not os.path.exists(os.path.dirname('results/')):
        os.makedirs(os.path.dirname('results/'))

    _u, _p, _b = solver.state.split()
    up_dofs = (_u.dof_dset.layout_vec.getSize() +
               _p.dof_dset.layout_vec.getSize())
    b_dofs = _b.dof_dset.layout_vec.getSize()
    dofs = b_dofs + up_dofs

    time_data = {"PETSCLogKSPSolve": ksp_time,
                 "PETSCLogPCApply": pc_apply_time,
                 "PETSCLogPCSetup": pc_setup_time,
                 "SNESSolve": snes_time,
                 "SNESFunctionEval": res_eval_time,
                 "SNESJacobianEval": jac_eval_time,
                 "num_processes": solver.comm.size,
                 "method": solver.method,
                 "model_degree": solver.model_degree,
                 "refinement_level": solver.refinement_level,
                 "total_dofs": dofs,
                 "up_dofs": up_dofs,
                 "b_dofs": b_dofs,
                 "num_cells": num_cells,
                 "Dt": solver.Dt,
                 "CFL": solver.courant,
                 "DxMax": solver.dx_max,
                 "Dz": solver.dz,
                 "OuterIters": solver.ksp_outer_its[0],
                 "InnerIters": solver.ksp_inner_its[0]}

    if args.hybridization:
        updates = {"HybridTraceSolve": trace_solve,
                   "HybridRHS": rhstime,
                   "HybridBreak": transfer,
                   "HybridReconstruction": recon_time,
                   "HybridProjection": projection,
                   "HybridFullRecovery": full_recon_time,
                   "HybridUpdate": update_time,
                   "HybridInit": inittime,
                   "HybridFullSolveTime": full_solve,
                   "HybridKSPOther": other}

    else:
        updates = {"KSPSchur": schur_time,
                   "KSPF0": f0_time,
                   "KSPFSLow": ksplow_time,
                   "KSPother": other}

    time_data.update(updates)

    df_time = pd.DataFrame(time_data, index=[0])
    df_time.to_csv(results_timings, index=False,
                   mode="w", header=True)
