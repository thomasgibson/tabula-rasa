from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import sys


parser = ArgumentParser(description=("""Convergence test for the 3D Helmholtz."""),
                        add_help=False)

parser.add_argument("--verify",
                    default=False,
                    type=bool,
                    action="store",
                    help=(
                        "Verify static condensation with GMRES "
                        "converges in one iteration."
                    ))

parser.add_argument("--degree",
                    default=None,
                    type=int,
                    action="store",
                    help="Degree of approximation.")

parser.add_argument("--write",
                    default=False,
                    action="store_true",
                    help="Write solution to output?")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    print("%s\n" % help)
    sys.exit(1)


def run_convergence_test(degree, write=False):

    name = "HelmholtzProblem"
    param_set = "scpc_hypre"
    params = {"snes_type": "ksponly",
              "pmat_type": "matfree",
              "ksp_type": "preonly",
              "pc_type": "python",
              "pc_python_type": "scpc.SCCG",
              # HYPRE on the reduced system
              "static_condensation": {"ksp_type": "cg",
                                      "ksp_rtol": 1e-16,
                                      "ksp_monitor": True,
                                      "pc_type": "hypre",
                                      "pc_hypre_type": "boomeramg",
                                      "pc_hypre_boomeramg_no_CF": True,
                                      "pc_hypre_boomeramg_coarsen_type": "HMIS",
                                      "pc_hypre_boomeramg_interp_type": "ext+i",
                                      "pc_hypre_boomeramg_P_max": 4,
                                      "pc_hypre_boomeramg_agg_nl": 1}}

    if args.verify:
        # Wrap PC in GMRES to monitor convergence.
        print("Wrapping solver options in GMRES outer loop.")
        params["ksp_type"] = "gmres"
        params["ksp_monitor"] = True

    r_params = range(0, 6)
    l2_errors = []
    gmres_its = []
    sc_ksp_its = []
    num_dofs = []
    num_cells = []
    reductions = []
    for r in r_params:
        mesh_size = 2 ** r
        mesh = UnitCubeMesh(mesh_size, mesh_size, mesh_size)
        V = FunctionSpace(mesh, "CG", degree)
        x, y, z = SpatialCoordinate(mesh)

        u = Function(V).assign(0.0)
        v = TestFunction(V)
        f = Function(V)
        f.interpolate((1 + 27*pi*pi)*cos(3*pi*x)*cos(3*pi*y)*cos(3*pi*z))
        F = inner(grad(v), grad(u))*dx + v*u*dx - inner(v, f)*dx
        r0 = assemble(F)
        problem = NonlinearVariationalProblem(F, u)
        solver = NonlinearVariationalSolver(problem,
                                            solver_parameters=params)

        output_file = "%s-d-%s-size-%s-params-%s" % (name,
                                                     degree,
                                                     r,
                                                     param_set)

        PETSc.Sys.Print("\nSolving 3D %s problem of degree %s, mesh size %s, "
                        "and parameter set %s\n" % (name,
                                                    degree,
                                                    r,
                                                    param_set))
        solver.solve()
        r1 = solver.snes.ksp.buildResidual()
        reductions.append(r1.norm()/r0.dat.norm)
        u_h = problem.u
        u_a = Function(FunctionSpace(mesh, "CG", degree + 1), name="analytic")
        sol = cos(3*pi*x)*cos(3*pi*y)*cos(3*pi*z)
        u_a.interpolate(sol)

        gmres_its.append(solver.snes.getLinearSolveIterations())
        sc_ksp_its.append(solver.snes.ksp.getPC().getPythonContext().sc_ksp.getIterationNumber())
        num_dofs.append(problem.u.dof_dset.layout_vec.getSize())
        num_cells.append(mesh.comm.allreduce(mesh.cell_set.size,
                                             op=MPI.SUM))

        PETSc.Sys.Print("\n Solver complete...\n")

        PETSc.Sys.Print("\n Computing L2 errors\n")
        l2_errors.append(errornorm(sol, u_h, norm_type="L2"))

        # Only write output for the final solve
        if write:
            if COMM_WORLD.rank == 0:
                if r == len(r_params):
                    PETSc.Sys.Print("\n Writing output to pvd file\n")
                    File(output_file + ".pvd").write(u_h, u_a)

    PETSc.Sys.Print("\nComputing convergence rates\n")
    l2_errors = np.array(l2_errors)
    rates = list(np.log2(l2_errors[:-1] / l2_errors[1:]))
    # Insert '---' in first slot, as there is rate to compute
    rates.insert(0, '---')
    degrees = [degree]*len(l2_errors)

    if COMM_WORLD.rank == 0:
        data = {"Mesh": r_params,
                "ResidualReductions": reductions,
                "L2Errors": l2_errors,
                "ConvRates": rates,
                "GMRESIterations": gmres_its,
                "SCPCIterations": sc_ksp_its,
                "NumDOFS": num_dofs,
                "NumCells": num_cells,
                "Degree": degrees}

        df = pd.DataFrame(data)
        result = "helmholtz_conv-d-%d.csv" % degree
        df.to_csv(result, index=False, mode="w")


if args.degree:
    run_convergence_test(degree=args.degree, write=args.write)
else:
    # If no degree provided, run full convergence test
    for degree in range(4, 8):
        run_convergence_test(degree=degree, write=args.write)
