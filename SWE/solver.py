from firedrake import *
from firedrake.petsc import PETSc
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage
import numpy as np


def fmax(f):
    fmax = op2.Global(1, np.finfo(float).min, dtype=float)
    op2.par_loop(op2.Kernel("""
void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


def latlon_coords(mesh):

    x0, y0, z0 = SpatialCoordinate(mesh)
    unsafe = z0 / sqrt(x0*x0 + y0*y0 + z0*z0)
    safe = Min(Max(unsafe, -1.0), 1.0)
    theta = asin(safe)
    lamda = atan_2(y0, x0)

    return theta, lamda


class W5Problem(object):
    """Williamson test case 5 Problem class."""

    def __init__(self, refinement_level, R,
                 H, Dt, method="BDM",
                 hybridization=False, model_degree=2,
                 monitor=False):
        super(W5Problem, self).__init__()

        self.refinement_level = refinement_level
        self.method = method
        self.model_degree = model_degree
        self.hybridization = hybridization
        self.monitor = monitor

        # Mesh radius
        self.R = R

        # Earth-sized mesh
        mesh_degree = 2
        if self.method == "RTCF":
            mesh = CubedSphereMesh(self.R, self.refinement_level,
                                   degree=mesh_degree)
        else:
            mesh = OctahedralSphereMesh(self.R, self.refinement_level,
                                        degree=mesh_degree,
                                        hemisphere="both")

        x = SpatialCoordinate(mesh)
        global_normal = as_vector(x)
        mesh.init_cell_orientations(global_normal)
        self.mesh = mesh

        # Get Dx information (this is approximate).
        # We compute the area (m^2) of each cell in the mesh,
        # then take the square root to get the right units.
        cell_vs = interpolate(CellVolume(self.mesh),
                              FunctionSpace(self.mesh, "DG", 0))

        a_max = fmax(cell_vs)
        dx_max = sqrt(a_max)
        self.dx_max = dx_max

        # Wave speed for the shallow water system
        g = 9.810616
        wave_speed = sqrt(H*g)

        # Courant number
        self.courant = (Dt / dx_max) * wave_speed
        self.dt = Constant(Dt)
        self.Dt = Dt

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
        self.Vm = FunctionSpace(self.mesh, "CG", 1)

        # Mean depth
        self.H = Constant(H)

        # Acceleration due to gravity
        self.g = Constant(g)

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
        h0 = self.H
        Omega = Constant(7.292e-5)
        g = self.g
        Dexpr = h0 - ((R0*Omega*u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g

        theta, lamda = latlon_coords(self.mesh)
        Rpn = pi/9.
        R0sq = Rpn**2
        lamda_c = -pi/2.
        lsq = (lamda - lamda_c)**2
        theta_c = pi/6.
        thsq = (theta - theta_c)**2
        rsq = Min(R0sq, lsq + thsq)
        r = sqrt(rsq)

        bexpr = 2000 * (1 - r/Rpn)

        self.b = Function(VD, name="Topography").interpolate(bexpr)

        # Coriolis expression (1/s)
        fexpr = 2*Omega*x[2]/R0
        self.f = Function(self.Vm).interpolate(fexpr)

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

        if self.hybridization:
            parameters = {
                'ksp_type': 'preonly',
                'mat_type': 'matfree',
                'pmat_type': 'matfree',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',
                'hybridization': {
                    'ksp_type': 'gmres',
                    'ksp_max_it': 100,
                    'pc_type': 'gamg',
                    'pc_gamg_reuse_interpolation': None,
                    'pc_gamg_sym_graph': None,
                    'ksp_rtol': 1e-8,
                    'mg_levels': {
                        'ksp_type': 'richardson',
                        'ksp_max_it': 2,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    }
                }
            }

            if self.monitor:
                parameters['hybridization']['ksp_monitor_true_residual'] = None

        else:
            parameters = {
                'ksp_type': 'fgmres',
                'ksp_rtol': 1.0e-8,
                'ksp_max_it': 500,
                'ksp_gmres_restart': 50,
                'pc_type': 'fieldsplit',
                'pc_fieldsplit': {
                    'type': 'schur',
                    'schur_fact_type': 'full',
                    # Use Stilde = A11 - A10 Diag(A00).inv A01 as the
                    # preconditioner for the Schur-complement
                    'schur_precondition': 'selfp'
                },
                'fieldsplit_0': {
                    'ksp_type': 'preonly',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'
                },
                'fieldsplit_1': {
                    'ksp_type': 'gmres',
                    'pc_type': 'gamg',
                    'pc_gamg_reuse_interpolation': None,
                    'pc_gamg_sym_graph': None,
                    'mg_levels': {
                        'ksp_type': 'chebyshev',
                        'ksp_max_it': 2,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    }
                }
            }

            if self.monitor:
                parameters['ksp_monitor_true_residual'] = None
                parameters['fieldsplit_1']['ksp_monitor_true_residual'] = None

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
            dirname += "hybrid_%s%d_ref%d_Dt%s/" % (self.method,
                                                    self.model_degree,
                                                    self.refinement_level,
                                                    self.Dt)
        else:
            dirname += "gmres_%s%d_ref%d_Dt%s/" % (self.method,
                                                   self.model_degree,
                                                   self.refinement_level,
                                                   self.Dt)
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

        self.initialize()
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
