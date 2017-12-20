from firedrake import *
from firedrake.petsc import PETSc
from pyop2.profiling import timed_stage
from argparse import ArgumentParser
import pandas as pd


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

parser.add_argument("--verification",
                    action="store_true",
                    help=("Turn verification mode on? "
                          "This computes residual reductions."))

parser.add_argument("--model_degree",
                    action="store",
                    type=int,
                    default=2,
                    help="Degree of the finite element model.")

parser.add_argument("--test",
                    action="store_true",
                    help=("Select 'True' or 'False' to enable a test run. "
                          "Default is False."))

parser.add_argument("--dumpfreq",
                    default=100,
                    type=int,
                    action="store",
                    help="Dump frequency of output.")

parser.add_argument("--refinements",
                    action="store",
                    default=3,
                    type=int,
                    choices=[3, 4, 5, 6, 7],
                    help="How many refinements to make to the sphere mesh.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn profiling on for 20 timesteps.")

parser.add_argument("--warmup",
                    action="store_true",
                    help="Turn off all output for warmup run.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.profile:
    # Ensures accurate timing of parallel loops
    parameters["pyop2_options"]["lazy_evaluation"] = False

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


def run_williamson5(refinement_level=3, dumpfreq=100, test=False,
                    profile=False, verbose=True,
                    model_degree=2, hybridization=False, verification=False):

    if refinement_level not in ref_to_dt:
        raise ValueError("Refinement level must be one of "
                         "the following: [3, 4, 5, 6, 7]")

    Dt = ref_to_dt[refinement_level]
    R = 6371220.
    H = Constant(5960.)
    day = 24.*60.*60.

    # Earth-sized mesh
    mesh_degree = 2
    mesh = OctahedralSphereMesh(R, refinement_level,
                                degree=mesh_degree,
                                hemisphere="both")

    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    x = SpatialCoordinate(mesh)

    # Maximum amplitude of zonal winds (m/s)
    u_0 = 20.
    # Topography
    bexpr = Expression("2000*(1 - sqrt(fmin(pow(pi/9.0,2), pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2) + pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=R)

    if test:
        tmax = 5*Dt
        PETSc.Sys.Print("Taking 5 time-steps\n")
    elif profile:
        tmax = 20*Dt
        PETSc.Sys.Print("Taking 20 time-steps\n")
    else:
        tmax = 15*day
        PETSc.Sys.Print("Running 15 day simulation\n")

    # Compatible FE spaces for velocity and depth
    Vu = FunctionSpace(mesh, "BDM", model_degree)
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
    Vm = FunctionSpace(mesh, "CG", mesh_degree)
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
    # K = 0.5*(inner(un, un)/3 + inner(un, up)/3 + inner(up, up)/3)
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

    if hybridization:
        parameters = {'snes_type': 'ksponly',
                      'ksp_type': 'preonly',
                      'pmat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'scpc.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'gamg',
                                        'ksp_rtol': 1e-8,
                                        'mg_levels': {'ksp_type': 'chebyshev',
                                                      'ksp_max_it': 2,
                                                      'pc_type': 'bjacobi',
                                                      'sub_pc_type': 'ilu'}}}

    else:
        parameters = {'snes_type': 'ksponly',
                      'ksp_type': 'gmres',
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_type': 'gmres',
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': {'ksp_type': 'cg',
                                       'pc_type': 'gamg',
                                       'ksp_rtol': 1e-8,
                                       'mg_levels': {'ksp_type': 'chebyshev',
                                                     'ksp_max_it': 2,
                                                     'pc_type': 'bjacobi',
                                                     'sub_pc_type': 'ilu'}}}

    DUsolver = NonlinearVariationalSolver(DUproblem,
                                          solver_parameters=parameters,
                                          options_prefix="implicit-solve")
    deltau, deltaD = DU.split()

    dumpcount = dumpfreq
    count = 0
    dirname = "results/"
    if hybridization:
        dirname += "hybrid/"
    Dfile = File(dirname + "w5_" + str(refinement_level) + ".pvd")
    eta = Function(VD, name="Surface Height")

    def dump(dumpcount, dumpfreq, count):
        dumpcount += 1
        if(dumpcount > dumpfreq):
            if verbose:
                PETSc.Sys.Print("Output: %s" % count)
            eta.assign(Dn+b)
            Dfile.write(un, Dn, eta, b)
            dumpcount -= dumpfreq
            count += 1
        return dumpcount

    # Initial output dump
    dumpcount = dump(dumpcount, dumpfreq, count)

    # Some diagnostics
    energy = []
    energy_t = assemble(0.5*inner(un, un)*Dn*dx +
                        0.5*g*(Dn + b)*(Dn + b)*dx)
    energy.append(energy_t)
    if verbose:
        PETSc.Sys.Print("Energy: %s" % energy_t)

    t = 0.0
    ksp_outer_its = []
    ksp_inner_its = []
    sim_time = [0.0]
    # At t=0, no solve has occured and therefore the reduction factor:
    # ||b - Ax*||_2 / ||b - Ax^0||_2 = ||b||_2/||b||_2 = 1.
    reductions = [1.0]
    while t < tmax - Dt/2:
        t += Dt

        # First guess for next timestep
        up.assign(un)
        Dp.assign(Dn)

        # Picard iteration
        res_reductions = []
        for i in range(4):
            with timed_stage("Advection"):
                # Update layer depth
                Dsolver.solve()
                # Update velocity
                Usolver.solve()

            with timed_stage("Linear solve"):
                # Calculate increments for up, Dp
                DUsolver.solve()
                PETSc.Sys.Print(
                    "Implicit solve finished for Picard iteration %s "
                    "at t=%s.\n" % (i + 1, t)
                )

            # x^0 == 0 (Preonly+hybrid)
            # => r0 == ||b||_2
            if verification:
                # Get rhs from ksp
                rhs = DUsolver.snes.ksp.getRhs()
                # Assemble the problem residual (b - Ax)
                res = assemble(FuD, mat_type="aij")
                bnorm = rhs.norm()
                rnorm = res.dat.norm
                r_factor = rnorm/bnorm
                res_reductions.append(r_factor)

            up += deltau
            Dp += deltaD

            outer_ksp = DUsolver.snes.ksp
            if hybridization:
                ctx = outer_ksp.getPC().getPythonContext()
                inner_ksp = ctx.trace_ksp
            else:
                ksps = outer_ksp.getPC().getFieldSplitSubKSP()
                _, inner_ksp = ksps

            # Collect ksp iterations
            ksp_outer_its.append(outer_ksp.getIterationNumber())
            ksp_inner_its.append(inner_ksp.getIterationNumber())

        un.assign(up)
        Dn.assign(Dp)

        with timed_stage("Dump output"):
            dumpcount = dump(dumpcount, dumpfreq, count)

        with timed_stage("Energy output"):
            energy_t = assemble(0.5*inner(un, un)*Dn*dx +
                                0.5*g*(Dn + b)*(Dn + b)*dx)
            energy.append(energy_t)
            if verbose:
                PETSc.Sys.Print("Energy: %s" % energy_t)

        # Simulation time (s)
        sim_time.append(t)

        # Add average residual reductions over picard iterations
        if verification:
            reductions.append(sum(res_reductions)/len(res_reductions))
        else:
            reductions.append('N/A')

    avg_outer_its = int(sum(ksp_outer_its)/len(ksp_outer_its))
    avg_inner_its = int(sum(ksp_inner_its)/len(ksp_inner_its))

    if not args.warmup:
        if COMM_WORLD.rank == 0:
            if hybridization:
                results_profile = "hybrid_profile_W5_ref%s.csv" % refinement_level
                results_diagnostics = "hybrid_diagnostics_W5_ref%s.csv" % refinement_level
            else:
                results_profile = "profile_W5_ref%s.csv" % refinement_level
                results_diagnostics = "diagnostics_W5_ref%s.csv" % refinement_level

            data_profile = {"AvgOuterIters": avg_outer_its,
                            "AvgInnerIters": avg_inner_its}

            data_diagnostics = {"SimTime": sim_time,
                                "ResidualReductions": reductions,
                                "Energy": energy}

            df_profile = pd.DataFrame(data_profile, index=[0])
            df_profile.to_csv(results_profile, index=False, mode="w", header=True)
            df_diagnostics = pd.DataFrame(data_diagnostics)
            df_diagnostics.to_csv(results_diagnostics, index=False, mode="w", header=True)


run_williamson5(refinement_level=args.refinements,
                dumpfreq=args.dumpfreq,
                test=args.test,
                profile=args.profile,
                verbose=args.verbose,
                model_degree=args.model_degree,
                hybridization=args.hybridization,
                verification=args.verification)
