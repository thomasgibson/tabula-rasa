from firedrake import *
import numpy as np
import pandas as pd


ref_to_dt = {3: 900.0,
             4: 450.0,
             5: 225.0,
             6: 112.5,
             7: 56.25}


def run_williamson5(refinement_level=3, dumpfreq=100,
                    tmax=None, verbose=True, model_degree=2,
                    hybridization=True):

    if refinement_level not in ref_to_dt:
        raise ValueError("Refinement level must be one of "
                         "the following: [3, 4, 5, 6, 7]")

    Dt = ref_to_dt[refinement_level]
    R = 6371220.
    H = 5960.
    day = 24.*60.*60.

    # Earth-sized mesh
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinement_level,
                                 degree=3)

    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    x = SpatialCoordinate(mesh)

    # Maximum amplitude of zonal winds (m/s)
    u_0 = 20.
    # Topography
    bexpr = Expression("2000*(1 - sqrt(fmin(pow(pi/9.0,2), pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2) + pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=R)

    # If none, run 15 simulation
    if tmax is None:
        tmax = 15*day

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

    # Coriolis expression (1/s)
    fexpr = 2*Omega*x[2]/R0
    Vm = FunctionSpace(mesh, "CG", 3)
    f = Function(Vm).interpolate(fexpr)
    f = Constant(0.0)
    b = Function(VD).interpolate(bexpr)
    Dn -= b

    # Build timestepping solver
    up = Function(Vu)
    Dp = Function(VD)
    dt = Constant(Dt)

    # Stage 1: Depth advection
    # DG upwinded advection for depth
    Dps = Function(VD)
    D = TrialFunction(VD)
    phi = TestFunction(VD)
    Dh = 0.5*(Dn+D)
    uh = 0.5*(un+up)
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
                                                         'sub_pc_type': 'ilu',
                                                         'ksp_monitor': True,
                                                         'ksp_rtol': 1e-8},
                                      options_prefix="D-advection")

    # Stage 2: U update
    Ups = Function(Vu)
    u = TrialFunction(Vu)
    v = TestFunction(Vu)
    Dh = 0.5*(Dn+Dp)
    ubar = 0.5*(un+up)
    uup = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    uh = 0.5*(un+u)
    Upwind = 0.5*(sign(dot(ubar, n)) + 1)
    # Kinetic energy term (implicit midpoint)
    # K = 0.5*(inner(0.5*(un+up), 0.5*(un+up)))
    K = 0.5*(inner(un, un)/3 + inner(un, up)/3 + inner(up, up)/3)
    both = lambda u: 2*avg(u)
    # u_t + gradperp.u + f)*perp(ubar) + grad(g*D + K)
    # <w, gradperp.u * perp(ubar)> = <perp(ubar).w, gradperp(u)>
    #                                = <-gradperp(w.perp(ubar))), u>
    #                                  +<< [[perp(n)(w.perp(ubar))]], u>>
    ueqn = (
        inner(u-un, v)*dx + dt*inner(perp(uh)*f, v)*dx
        - dt*inner(perp(grad(inner(v, perp(ubar)))), uh)*dx
        + dt*inner(both(perp(n)*inner(v, perp(ubar))), both(Upwind*uh))*dS
        - dt*div(v)*(g*(Dh+b) + K)*dx
    )

    Uproblem = LinearVariationalProblem(lhs(ueqn), rhs(ueqn), Ups)
    Usolver = LinearVariationalSolver(Uproblem,
                                      solver_parameters={'ksp_type': 'gmres',
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu',
                                                         'ksp_monitor': True,
                                                         'ksp_rtol': 1e-8},
                                      options_prefix="U-advection")

    # Stage 3: Implicit linear solve for u, D increments
    W = MixedFunctionSpace((Vu, VD))
    w, phi = TestFunctions(W)
    du, dD = TrialFunctions(W)

    uDlhs = (
        inner(w, du + 0.5*dt*f*perp(du)) - 0.5*dt*div(w)*g*dD +
        phi*(dD + 0.5*dt*H*div(du))
    )*dx
    Dh = 0.5*(Dp + Dn)
    uh = 0.5*(un + up)

    uDrhs = -(
        inner(w, up-Ups)*dx
        + phi*(Dp - Dps)*dx
    )

    DU = Function(W)
    DUproblem = LinearVariationalProblem(uDlhs, uDrhs, DU)

    if hybridization:
        parameters = {'ksp_type': 'gmres',
                      'ksp_monitor': True,
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'gamg',
                                        'ksp_monitor': True,
                                        'mg_levels_ksp_type': 'chebyshev',
                                        'mg_levels_ksp_max_it': 2,
                                        'mg_levels_pc_type': 'bjacobi',
                                        'mg_levels_sub_pc_type': 'ilu',
                                        'ksp_converged_reason': True,
                                        'hdiv_residual': {'ksp_type': 'cg',
                                                          'pc_type': 'bjacobi',
                                                          'sub_pc_type': 'ilu',
                                                          'ksp_rtol': 1e-16,
                                                          'ksp_monitor': True},
                                        'hdiv_projection': {'method': 'average'}}}

    else:
        parameters = {'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_type': 'gmres',
                      'ksp_monitor': True,
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': {'ksp_type': 'cg',
                                       'pc_type': 'gamg',
                                       'ksp_monitor': True,
                                       'mg_levels': {'ksp_type': 'chebyshev',
                                                     'ksp_max_it': 2,
                                                     'pc_type': 'bjacobi',
                                                     'sub_pc_type': 'ilu'}}}

    DUsolver = LinearVariationalSolver(DUproblem,
                                       solver_parameters=parameters,
                                       options_prefix="implicit-solve")
    deltau, deltaD = DU.split()

    dumpcount = dumpfreq
    Dfile = File("results/w5_"+str(refinement_level)+".pvd")
    eta = Function(VD, name="Surface Height")

    def dump(dumpcount, dumpfreq):
        dumpcount += 1
        print(dumpcount)
        if(dumpcount > dumpfreq):
            eta.assign(Dn+b)
            Dfile.write(un, Dn, eta)
            dumpcount -= dumpfreq
        return dumpcount

    dumpcount = dump(dumpcount, dumpfreq)

    # Some diagnostics
    energy = []
    energy_t = assemble(0.5*inner(un, un)*Dn*dx +
                        0.5*g*(Dn+b)*(Dn+b)*dx)
    energy.append(energy_t)
    if verbose:
        print(energy_t, 'Energy')

    t = 0.0
    while t < tmax - Dt/2:
        t += Dt

        # First guess for next timestep
        up.assign(un)
        Dp.assign(Dn)

        # Picard iteration
        for i in range(4):
            # Update layer depth
            Dsolver.solve()
            # Update velocity
            Usolver.solve()
            # Calculate increments for up, Dp
            DUsolver.solve()
            up += deltau
            Dp += deltaD

            un.assign(up)
            Dn.assign(Dp)

            dumpcount = dump(dumpcount, dumpfreq)
            energy_t = assemble(0.5*inner(un, un)*Dn*dx +
                                0.5*g*(Dn+b)*(Dn+b)*dx)
            energy.append(energy_t)
            if verbose:
                print(energy_t, 'Energy')


run_williamson5(refinement_level=3, dumpfreq=100,
                tmax=None, verbose=True, model_degree=2,
                hybridization=True)
