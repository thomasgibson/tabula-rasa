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

from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, parameters
from pyop2.profiling import timed_stage
from argparse import ArgumentParser
from collections import defaultdict
import pandas as pd
import sys


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


warm = defaultdict(bool)


def run_williamson5(refinement_level=3, model_degree=2, method="BDM",
                    verbose=True, hybridization=False, write=False):

    if refinement_level not in ref_to_dt:
        raise ValueError("Refinement level must be one of "
                         "the following: [3, 4, 5, 6, 7]")

    Dt = ref_to_dt[refinement_level]
    R = 6371220.
    H = Constant(5960.)
    day = 24.*60.*60.

    # Earth-sized mesh
    if method == "RTCF":
        mesh = CubedSphereMesh(R, refinement_level,
                               degree=3)
    else:
        mesh = IcosahedralSphereMesh(R, refinement_level,
                                     degree=3)

    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    x = SpatialCoordinate(mesh)

    # Maximum amplitude of zonal winds (m/s)
    u_0 = 20.
    # Topography
    bexpr = Expression("2000*(1 - sqrt(fmin(pow(pi/9.0,2), pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2) + pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=R)

    if args.profile:
        tmax = 100*Dt
        PETSc.Sys.Print("Taking 100 time-steps\n")
    else:
        tmax = 15*day
        PETSc.Sys.Print("Running 15 day simulation\n")

    # Compatible FE spaces for velocity and depth
    if method == "RT":
        Vu = FunctionSpace(mesh, "RT", model_degree)
    elif method == "RTCF":
        Vu = FunctionSpace(mesh, "RTCF", model_degree)
    elif method == "BDM":
        Vu = FunctionSpace(mesh, "BDM", model_degree)
    else:
        raise ValueError("Unrecognized method '%s'" % method)

    VD = FunctionSpace(mesh, "DG", model_degree - 1)

    # State variables: velocity and depth
    un = Function(Vu, name="Velocity")
    Dn = Function(VD, name="Depth")

    outward_normals = CellNormal(mesh)

    def perp(u):
        return cross(outward_normals, u)

    # Initial conditions for velocity and depth (in geostrophic balance)
    u_max = Constant(u_0)
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    h0 = Constant(H)
    Omega = Constant(7.292e-5)
    g = Constant(9.810616)
    Dexpr = h0 - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
    Dn.interpolate(Dexpr)
    un.project(uexpr)
    b = Function(VD, name="Topography").interpolate(bexpr)
    Dn -= b

    # Coriolis expression (1/s)
    fexpr = 2*Omega*x[2]/R0
    Vm = FunctionSpace(mesh, "CG", 3)
    f = Function(Vm).interpolate(fexpr)

    # Build timestepping solver
    up = Function(Vu)
    Dp = Function(VD)
    dt = Constant(Dt)

    # Stage 1: Depth advection
    # DG upwinded advection for depth
    Dps = Function(VD)
    D = TrialFunction(VD)
    phi = TestFunction(VD)
    Dh = 0.5*(Dn + D)
    uh = 0.5*(un + up)
    n = FacetNormal(mesh)
    uup = 0.5*(dot(uh, n) + abs(dot(uh, n)))

    Deqn = (
        (D - Dn)*phi*dx - dt*inner(grad(phi), uh*Dh)*dx
        + dt*jump(phi)*(uup('+')*Dh('+')-uup('-')*Dh('-'))*dS
    )

    Dproblem = LinearVariationalProblem(lhs(Deqn), rhs(Deqn), Dps)
    Dsolver = LinearVariationalSolver(Dproblem,
                                      solver_parameters={'ksp_type': 'cg',
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu'},
                                      options_prefix="D-advection")

    # Stage 2: U update
    Ups = Function(Vu)
    u = TrialFunction(Vu)
    v = TestFunction(Vu)
    Dh = 0.5*(Dn + Dp)
    ubar = 0.5*(un + up)
    uup = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    uh = 0.5*(un + u)
    Upwind = 0.5*(sign(dot(ubar, n)) + 1)

    # Kinetic energy term (implicit midpoint)
    K = 0.5*(inner(0.5*(un + up), 0.5*(un + up)))

    both = lambda u: 2*avg(u)
    # u_t + gradperp.u + f)*perp(ubar) + grad(g*D + K)
    # <w, gradperp.u * perp(ubar)> = <perp(ubar).w, gradperp(u)>
    #                                = <-gradperp(w.perp(ubar))), u>
    #                                  +<< [[perp(n)(w.perp(ubar))]], u>>
    ueqn = (
        inner(u - un, v)*dx + dt*inner(perp(uh)*f, v)*dx
        - dt*inner(perp(grad(inner(v, perp(ubar)))), uh)*dx
        + dt*inner(both(perp(n)*inner(v, perp(ubar))), both(Upwind*uh))*dS
        - dt*div(v)*(g*(Dh + b) + K)*dx
    )

    Uproblem = LinearVariationalProblem(lhs(ueqn), rhs(ueqn), Ups)
    Usolver = LinearVariationalSolver(Uproblem,
                                      solver_parameters={'ksp_type': 'gmres',
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu'},
                                      options_prefix="U-advection")

    # Stage 3: Implicit linear solve for u, D increments
    W = MixedFunctionSpace((Vu, VD))
    DU = Function(W)
    w, phi = TestFunctions(W)
    du, dD = split(DU)

    uDlhs = (
        inner(w, du + 0.5*dt*f*perp(du)) - 0.5*dt*div(w)*g*dD +
        phi*(dD + 0.5*dt*H*div(du))
    )*dx
    Dh = 0.5*(Dp + Dn)
    uh = 0.5*(un + up)

    uDrhs = -(
        inner(w, up - Ups)*dx
        + phi*(Dp - Dps)*dx
    )

    FuD = uDlhs - uDrhs
    DUproblem = NonlinearVariationalProblem(FuD, DU)

    gamg_params = {'ksp_type': 'cg',
                   'pc_type': 'gamg',
                   'pc_gamg_sym_graph': True,
                   'ksp_rtol': 1e-8,
                   'mg_levels': {'ksp_type': 'chebyshev',
                                 'ksp_max_it': 2,
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}}
    if hybridization:
        parameters = {'snes_type': 'ksponly',
                      'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pmat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'scpc.HybridizationPC',
                      'hybridization': gamg_params}

    else:
        parameters = {'snes_type': 'ksponly',
                      'ksp_type': 'gmres',
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_type': 'gmres',
                      'ksp_rtol': 1e-8,
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': gamg_params}

    DUsolver = NonlinearVariationalSolver(DUproblem,
                                          solver_parameters=parameters,
                                          options_prefix="implicit-solve")
    deltau, deltaD = DU.split()

    dumpcount = args.dumpfreq
    dirname = "results/"
    if hybridization:
        dirname += "hybrid_%s_ref%d/" % (method, refinement_level)
    else:
        dirname += "gmres_%s_ref%d/" % (method, refinement_level)

    Dfile = File(dirname + "w5_" + str(refinement_level) + ".pvd")
    eta = Function(VD, name="Surface Height")

    def dump(dumpcount, dumpfreq):
        dumpcount += 1
        if(dumpcount > dumpfreq):
            if verbose:
                PETSc.Sys.Print("Writing output...\n")
            eta.assign(Dn+b)
            Dfile.write(un, Dn, eta, b)
            dumpcount -= dumpfreq
        return dumpcount

    # Initial output dump
    if write:
        dumpcount = dump(dumpcount, args.dumpfreq, count)

    t = 0.0
    ksp_outer_its = []
    ksp_inner_its = []
    sim_time = []
    picard_seq = []
    reductions = []

    if not warm[(refinement_level, model_degree, method)]:
        PETSc.Sys.Print("Warmup linear solver...\n")

        up.assign(un)
        Dp.assign(Dn)

        Dsolver.solve()
        Usolver.solve()
        DUsolver.solve()

        # Reset all functions for actual time-step run
        DU.assign(0.0)
        Dn.interpolate(Dexpr)
        un.project(uexpr)
        Dn -= b

        warm[(refinement_level, model_degree, method)] = True

    Dsolver.snes.setConvergenceHistory()
    Dsolver.snes.ksp.setConvergenceHistory()
    Usolver.snes.setConvergenceHistory()
    Usolver.snes.ksp.setConvergenceHistory()
    DUsolver.snes.setConvergenceHistory()
    DUsolver.snes.ksp.setConvergenceHistory()

    DUResidual_time = 0.0
    LinearSolve_time = 0.0
    time_assembling_residuals = 0.0
    time_writing_output = 0.0
    time_getting_ksp_info = 0.0

    PETSc.Sys.Print("Starting simulation...\n")
    start = PETSc.Log.getTime()
    while t < tmax - Dt/2:
        t += Dt

        # First guess for next timestep
        up.assign(un)
        Dp.assign(Dn)

        # Picard cycle
        for i in range(4):
            sim_time.append(t)
            picard_seq.append(i+1)

            with timed_stage("DU Residuals"):
                v1 = PETSc.Log.getTime()
                # Update layer depth
                Dsolver.solve()
                # Update velocity
                Usolver.solve()
                v2 = PETSc.Log.getTime()
                DUResidual_time += v2 - v1

            with timed_stage("Linear solve"):
                # Calculate increments for up, Dp
                v1 = PETSc.Log.getTime()
                DUsolver.solve()
                v2 = PETSc.Log.getTime()
                LinearSolve_time += v2 - v1
                if verbose:
                    PETSc.Sys.Print(
                        "Implicit solve finished for Picard iteration %s "
                        "at t=%s.\n" % (i + 1, t)
                    )

            # Here we collect the reductions in the linear residual.
            # We monitor the time taken to compute these and account
            # for it when determining the overall simulation time.
            r1 = PETSc.Log.getTime()
            # Get rhs from ksp
            r0 = DUsolver.snes.ksp.getRhs()
            # Assemble the problem residual (b - Ax)
            res = assemble(FuD, mat_type="aij")
            bnorm = r0.norm()
            rnorm = res.dat.norm
            r_factor = rnorm/bnorm
            reductions.append(r_factor)
            r2 = PETSc.Log.getTime()
            time_assembling_residuals += r2 - r1

            up += deltau
            Dp += deltaD

            kspt1 = PETSc.Log.getTime()
            outer_ksp = DUsolver.snes.ksp
            if hybridization:
                ctx = outer_ksp.getPC().getPythonContext()
                inner_ksp = ctx.trace_ksp
            else:
                ksps = outer_ksp.getPC().getFieldSplitSubKSP()
                _, inner_ksp = ksps
            kspt2 = PETSc.Log.getTime()

            # Collect ksp iterations
            ksp_outer_its.append(outer_ksp.getIterationNumber())
            ksp_inner_its.append(inner_ksp.getIterationNumber())
            time_getting_ksp_info += kspt2 - kspt1

        un.assign(up)
        Dn.assign(Dp)

        w1 = PETSc.Log.getTime()
        if write:
            with timed_stage("Dump output"):
                dumpcount = dump(dumpcount, args.dumpfreq)
        w2 = PETSc.Log.getTime()
        time_writing_output += w2 - w1

    end = PETSc.Log.getTime()
    elapsed_time = end - start
    PETSc.Sys.Print("Simulation complete (elapsed time: %s).\n" % elapsed_time)

    if COMM_WORLD.rank == 0:
        ref = refinement_level
        if hybridization:
            results_data = "hybrid_%s_data_W5_ref%s.csv" % (method, ref)
            results_timings = "hybrid_%s_profile_W5_ref%s.csv" % (method, ref)
        else:
            results_data = "gmres_%s_data_W5_ref%s.csv" % (method, ref)
            results_timings = "gmres_%s_profile_W5_ref%s.csv" % (method, ref)

        data = {"OuterIters": ksp_outer_its,
                "InnerIters": ksp_inner_its,
                "PicardIters": picard_seq,
                "SimTime": sim_time,
                "ResidualReductions": reductions}

        dofs = DUsolver._problem.u.dof_dset.layout_vec.getSize()

        # Time spent computing diagnostic information
        diagnostic_time = (time_assembling_residuals +
                           time_writing_output +
                           time_getting_ksp_info)

        time_data = {"num_processes": mesh.comm.size,
                     "method": method,
                     "model_degree": model_degree,
                     "refinement_level": refinement_level,
                     "total_dofs": dofs,
                     # Time spent in just the linear solver bit.
                     "LinearSolve": LinearSolve_time,
                     # Time spent setting up the stabilized residual RHS
                     # for the implicit linear system.
                     "DUResidual": DUResidual_time,
                     # Total time to run the problem, including time
                     # spent computing residuals.
                     "TotalRunTime": elapsed_time,
                     # To accurately record the time spend in the actual
                     # simulation, we remove the time spent computing
                     # residuals, writing output (if written), and
                     # time gathering KSP information.
                     "SimTime": elapsed_time - diagnostic_time,
                     # We still record these here for reference.
                     "ComputingResiduals": time_assembling_residuals,
                     "TimeWritingOutput": time_writing_output,
                     "TimeGatheringKSPInfo": time_getting_ksp_info}

        df_data = pd.DataFrame(data)
        df_data.to_csv(results_data, index=False,
                       mode="w", header=True)

        df_time = pd.DataFrame(time_data, index=[0])
        df_time.to_csv(results_timings, index=False,
                       mode="w", header=True)


if args.profile:
    params = [("RT", 1), ("RTCF", 1), ("BDM", 2)]
    for param in params:
        method, model_degree = param
        for ref_level in ref_to_dt:
            PETSc.Sys.Print("""
Running 100 timesteps for the Williamson 5 test with parameters:\n
method: %s,\n
model degree: %d,\n
refinements: %d,\n
hybridization: %s.
""" % (method, model_degree, ref_level, args.hybridization))
            run_williamson5(refinement_level=ref_level,
                            model_degree=model_degree,
                            method=method,
                            verbose=args.verbose,
                            hybridization=args.hybridization,
                            write=args.write)

else:
    run_williamson5(refinement_level=args.refinements,
                    model_degree=args.model_degree,
                    method=args.method,
                    verbose=args.verbose,
                    hybridization=args.hybridization,
                    write=args.write)
