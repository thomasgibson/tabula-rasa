from firedrake import *
from firedrake.petsc import PETSc
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage


class W5Problem(object):
    """Williamson test case 5 Problem class."""

    def __init__(self, refinement_level, R,
                 H, Dt, method="BDM",
                 hybridization=False, model_degree=2):
        super(W5Problem, self).__init__()

        self.refinement_level = refinement_level
        self.dt = Constant(Dt)
        self.Dt = Dt
        self.method = method
        self.model_degree = model_degree
        self.hybridization = hybridization

        # Mesh radius
        self.R = R

        # Earth-sized mesh
        mesh_degree = 2
        if self.method == "RTCF":
            from firedrake import CubedSphereMesh
            mesh = CubedSphereMesh(self.R, self.refinement_level,
                                   degree=mesh_degree)
        else:
            from firedrake import OctahedralSphereMesh
            mesh = OctahedralSphereMesh(self.R, self.refinement_level,
                                        degree=mesh_degree,
                                        hemisphere="both")

        global_normal = Expression(("x[0]", "x[1]", "x[2]"))
        mesh.init_cell_orientations(global_normal)
        self.mesh = mesh

        # Get Dx information (min and max)
        cell_vs = interpolate(CellVolume(self.mesh),
                              FunctionSpace(self.mesh, "DG", 0))

        # Min and max dx (m)
        dx_min = sqrt(cell_vs.dat.data.min())
        dx_max = sqrt(cell_vs.dat.data.max())

        # gravity
        g = 9.810616

        wave_speed = sqrt(H*g)

        # Courant number (min and max)
        self.min_courant = (self.Dt / dx_max)*wave_speed
        self.max_courant = (self.Dt / dx_min)*wave_speed

        # Compatible FE spaces for velocity and depth
        if self.method == "RT":
            Vu = FunctionSpace(self.mesh, "RT", self.model_degree)
        elif self.method == "RTCF":
            Vu = FunctionSpace(self.mesh, "RTCF", self.model_degree)
        elif self.method == "BDM":
            Vu = FunctionSpace(self.mesh, "BDM", self.model_degree)
        else:
            raise ValueError("Unrecognized method '%s'" % self.method)

        VD = FunctionSpace(self.mesh, "DG", self.model_degree - 1)

        self.function_spaces = (Vu, VD)
        self.Vm = FunctionSpace(self.mesh, "CG", mesh_degree)

        # Mean depth
        self.H = Constant(H)

        # Build initial conditions and parameters
        self._build_initial_conditions()

        # Setup solvers
        self._build_DU_solvers()
        self._build_picard_solver()

        self.ksp_outer_its = []
        self.ksp_inner_its = []
        self.sim_time = []
        self.picard_seq = []
        self.reductions = []

    def _build_initial_conditions(self):

        x = SpatialCoordinate(self.mesh)
        _, VD = self.function_spaces

        # Initial conditions for velocity and depth (in geostrophic balance)
        u_0 = 20.0
        u_max = Constant(u_0)
        R0 = Constant(self.R)
        uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
        h0 = Constant(self.H)
        Omega = Constant(7.292e-5)
        g = Constant(9.810616)
        Dexpr = h0 - ((R0*Omega*u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g

        bexpr = Expression("2000*(1 - sqrt(fmin(pow(pi/9.0,2), pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2) + pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=self.R)
        self.b = Function(VD, name="Topography").interpolate(bexpr)

        # Coriolis expression (1/s)
        fexpr = 2*Omega*x[2]/R0
        self.f = Function(self.Vm).interpolate(fexpr)

        self.g = g
        self.uexpr = uexpr
        self.Dexpr = Dexpr

    def _build_DU_solvers(self):

        Vu, VD = self.function_spaces
        un, Dn = self.state
        up, Dp = self.updates
        dt = self.dt

        # Stage 1: Depth advection
        # DG upwinded advection for depth
        self.Dps = Function(VD, name="stabilized depth")
        D = TrialFunction(VD)
        phi = TestFunction(VD)
        Dh = 0.5*(Dn + D)
        uh = 0.5*(un + up)
        n = FacetNormal(self.Vm.mesh())
        uup = 0.5*(dot(uh, n) + abs(dot(uh, n)))

        Deqn = (
            (D - Dn)*phi*dx - dt*inner(grad(phi), uh*Dh)*dx
            + dt*jump(phi)*(uup('+')*Dh('+')-uup('-')*Dh('-'))*dS
        )

        Dparams = {'ksp_type': 'cg',
                   'pc_type': 'bjacobi',
                   'sub_pc_type': 'ilu'}
        Dproblem = LinearVariationalProblem(lhs(Deqn), rhs(Deqn),
                                            self.Dps)
        Dsolver = LinearVariationalSolver(Dproblem,
                                          solver_parameters=Dparams,
                                          options_prefix="D-advection")
        self.Dsolver = Dsolver

        # Stage 2: U update
        self.Ups = Function(Vu, name="stabilized velocity")
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
            inner(u - un, v)*dx + dt*inner(self.perp(uh)*self.f, v)*dx
            - dt*inner(self.perp(grad(inner(v, self.perp(ubar)))), uh)*dx
            + dt*inner(both(self.perp(n)*inner(v, self.perp(ubar))),
                       both(Upwind*uh))*dS
            - dt*div(v)*(self.g*(Dh + self.b) + K)*dx
        )

        uparams = {'ksp_type': 'gmres',
                   'pc_type': 'bjacobi',
                   'sub_pc_type': 'ilu'}
        Uproblem = LinearVariationalProblem(lhs(ueqn), rhs(ueqn),
                                            self.Ups)
        Usolver = LinearVariationalSolver(Uproblem,
                                          solver_parameters=uparams,
                                          options_prefix="U-advection")
        self.Usolver = Usolver

    def _build_picard_solver(self):
        Vu, VD = self.function_spaces
        un, Dn = self.state
        up, Dp = self.updates
        dt = self.dt
        f = self.f
        g = self.g
        H = self.H

        Dps = self.Dps
        Ups = self.Ups

        # Stage 3: Implicit linear solve for u, D increments
        W = MixedFunctionSpace((Vu, VD))
        self.DU = Function(W, name="linear updates")
        w, phi = TestFunctions(W)
        du, dD = TrialFunctions(W)

        uDlhs = (
            inner(w, du + 0.5*dt*f*self.perp(du)) - 0.5*dt*div(w)*g*dD +
            phi*(dD + 0.5*dt*H*div(du))
        )*dx

        uDrhs = -(
            inner(w, up - Ups)*dx
            + phi*(Dp - Dps)*dx
        )

        self.FuD = action(uDlhs, self.DU) - uDrhs
        DUproblem = LinearVariationalProblem(uDlhs, uDrhs, self.DU,
                                             constant_jacobian=True)

        gamg_params = {'ksp_type': 'cg',
                       'pc_type': 'gamg',
                       'pc_gamg_reuse_interpolation': True,
                       'ksp_rtol': 1e-8,
                       'mg_levels': {'ksp_type': 'chebyshev',
                                     'ksp_max_it': 2,
                                     'pc_type': 'bjacobi',
                                     'sub_pc_type': 'ilu'}}
        if self.hybridization:
            parameters = {'ksp_type': 'preonly',
                          'mat_type': 'matfree',
                          'pmat_type': 'matfree',
                          'pc_type': 'python',
                          'pc_python_type': 'scpc.HybridizationPC',
                          'hybridization': gamg_params}

        else:
            parameters = {'ksp_type': 'gmres',
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

        DUsolver = LinearVariationalSolver(DUproblem,
                                           solver_parameters=parameters,
                                           options_prefix="implicit-solve")
        self.DUsolver = DUsolver

    @cached_property
    def num_cells(self):
        return self.mesh.cell_set.size

    @cached_property
    def comm(self):
        return self.mesh.comm

    @cached_property
    def outward_normals(self):
        return CellNormal(self.mesh)

    def perp(self, u):
        return cross(self.outward_normals, u)

    @cached_property
    def eta(self):
        _, VD = self.function_spaces
        eta = Function(VD, name="Surface height")
        return eta

    @cached_property
    def state(self):
        Vu, VD = self.function_spaces
        un = Function(Vu, name="Velocity")
        Dn = Function(VD, name="Depth")
        return un, Dn

    @cached_property
    def updates(self):
        Vu, VD = self.function_spaces
        up = Function(Vu)
        Dp = Function(VD)
        return up, Dp

    @cached_property
    def output_file(self):
        dirname = "results/"
        if self.hybridization:
            dirname += "hybrid_%s_ref%d/" % (self.method,
                                             self.refinement_level)
        else:
            dirname += "gmres_%s_ref%d/" % (self.method,
                                            self.refinement_level)
        return File(dirname + "w5_" + str(self.refinement_level) + ".pvd")

    def write(self, dumpcount, dumpfreq):
        dumpcount += 1
        un, Dn = self.state
        if(dumpcount > dumpfreq):
            eta = self.eta
            eta.assign(Dn + self.b)
            self.output_file.write(un, Dn, eta, self.b)
            dumpcount -= dumpfreq
        return dumpcount

    def initialize(self):
        self.DU.assign(0.0)
        self.Ups.assign(0.0)
        self.Dps.assign(0.0)
        un, Dn = self.state
        un.project(self.uexpr)
        Dn.interpolate(self.Dexpr)
        Dn -= self.b

    def warmup(self):
        un, Dn = self.state
        up, Dp = self.updates
        up.assign(un)
        Dp.assign(Dn)

        with timed_stage("Warm up: DU Residuals"):
            self.Dsolver.solve()
            self.Usolver.solve()

        with timed_stage("Warm up: Linear solve"):
            self.DUsolver.solve()

    def run_simulation(self, tmax, write=False, dumpfreq=100):
        PETSc.Sys.Print("""
Running Williamson 5 test with parameters:\n
method: %s,\n
model degree: %d,\n
refinements: %d,\n
hybridization: %s,\n
tmax: %s
"""
                        % (self.method, self.model_degree,
                           self.refinement_level, self.hybridization, tmax))
        t = 0.0
        un, Dn = self.state
        up, Dp = self.updates
        deltau, deltaD = self.DU.split()
        dumpcount = dumpfreq
        if write:
            dumpcount = self.write(dumpcount, dumpfreq)

        self.Dsolver.snes.setConvergenceHistory()
        self.Dsolver.snes.ksp.setConvergenceHistory()
        self.Usolver.snes.setConvergenceHistory()
        self.Usolver.snes.ksp.setConvergenceHistory()
        self.DUsolver.snes.setConvergenceHistory()
        self.DUsolver.snes.ksp.setConvergenceHistory()

        while t < tmax - self.Dt/2:
            t += self.Dt

            up.assign(un)
            Dp.assign(Dn)

            for i in range(4):
                self.sim_time.append(t)
                self.picard_seq.append(i+1)

                with timed_stage("DU Residuals"):
                    self.Dsolver.solve()
                    self.Usolver.solve()

                with timed_stage("Linear solve"):
                    self.DUsolver.solve()

                # Here we collect the reductions in the linear residual.
                # Get rhs from ksp
                r0 = self.DUsolver.snes.ksp.getRhs()
                # Assemble the problem residual (b - Ax)
                res = assemble(self.FuD, mat_type="aij")
                bnorm = r0.norm()
                rnorm = res.dat.norm
                r_factor = rnorm/bnorm
                self.reductions.append(r_factor)

                up += deltau
                Dp += deltaD

                outer_ksp = self.DUsolver.snes.ksp
                if self.hybridization:
                    ctx = outer_ksp.getPC().getPythonContext()
                    inner_ksp = ctx.trace_ksp
                else:
                    ksps = outer_ksp.getPC().getFieldSplitSubKSP()
                    _, inner_ksp = ksps

                # Collect ksp iterations
                self.ksp_outer_its.append(outer_ksp.getIterationNumber())
                self.ksp_inner_its.append(inner_ksp.getIterationNumber())

            un.assign(up)
            Dn.assign(Dp)

            if write:
                with timed_stage("Dump output"):
                    dumpcount = self.write(dumpcount, dumpfreq)
