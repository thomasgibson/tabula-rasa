from firedrake import *
from mpi4py import MPI
import numpy as np
import csv


def run_convergence_test(degree):

    name = "HelmholtzProblem"
    param_set = "scpc_hypre"
    params = {"snes_type": "ksponly",
              "mat_type": "matfree",
              # Wrap PC in GMRES to monitor convergence. This should
              # take only 1 GMRES iteration if the PC is working properly.
              "ksp_type": "gmres",
              "ksp_monitor": True,
              "pc_type": "python",
              "pc_python_type": "firedrake.CGStaticCondensationPC",
              "static_condensation": {"ksp_type": "cg",
                                      "ksp_rtol": 1e-10,
                                      "ksp_monitor": True,
                                      "pc_type": "hypre",
                                      "pc_hypre_type": "boomeramg",
                                      "pc_hypre_boomeramg_no_CF": True,
                                      "pc_hypre_boomeramg_coarsen_type": "HMIS",
                                      "pc_hypre_boomeramg_interp_type": "ext+i",
                                      "pc_hypre_boomeramg_P_max": 4,
                                      "pc_hypre_boomeramg_agg_nl": 1}}

    r_params = range(1, 6)
    l2_errors = []
    gmres_its = []
    sc_ksp_its = []
    num_dofs = []
    num_cells = []
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
        problem = NonlinearVariationalProblem(F, u)
        solver = NonlinearVariationalSolver(problem,
                                            solver_parameters=params)

        output_file = "%s-d-%s-size-%s-params-%s" % (name,
                                                     degree,
                                                     r,
                                                     param_set)

        print("\nSolving 3D %s problem of degree %s, mesh size %s, "
              "and parameter set %s\n" % (name,
                                          degree,
                                          r,
                                          param_set))
        solver.solve()
        u_h = problem.u
        u_a = Function(FunctionSpace(mesh, "CG", degree + 1), name="analytic")
        sol = cos(3*pi*x)*cos(3*pi*y)*cos(3*pi*z)
        u_a.interpolate(sol)

        gmres_its.append(solver.snes.getLinearSolveIterations())
        sc_ksp_its.append(solver.snes.ksp.getPC().getPythonContext().sc_ksp.getIterationNumber())
        num_dofs.append(problem.u.dof_dset.layout_vec.getSize())
        num_cells.append(mesh.comm.allreduce(mesh.cell_set.size,
                                             op=MPI.SUM))

        print("\n Solver complete...\n")

        print("\n Computing L2 errors\n")
        l2_errors.append(errornorm(sol, u_h, norm_type="L2"))

        # Only write output for the final solve
        if r == len(r_params):
            print("\n Writing output to pvd file\n")
            File(output_file + ".pvd").write(u_h, u_a)

    print("\nComputing convergence rates\n")
    l2_errors = np.array(l2_errors)
    rates = list(np.log2(l2_errors[:-1] / l2_errors[1:]))
    # Insert '---' in first slot, as there is rate to compute
    rates.insert(0, '---')

    fieldnames = ["Mesh",
                  "L2Errors",
                  "ConvRates",
                  "GMRESIterations",
                  "SCPCIterations",
                  "NumDOFS",
                  "NumCells"]
    data = [r_params,
            l2_errors,
            rates,
            gmres_its,
            sc_ksp_its,
            num_dofs,
            num_cells]

    csv_file = open("3D-Helmholtz-deg%d.csv" % degree, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(fieldnames)
    for d in zip(*data):
        csv_writer.writerow(d)

    csv_file.close()


for degree in range(4, 9):
    run_convergence_test(degree)
