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


def compute_balanced_pressure(Vv, VDG, k, b0, p0, p_boundary, top=False):

    Vvd = FunctionSpace(Vv.mesh(), BrokenElement(Vv.ufl_element()))
    cell, _ = VDG.ufl_element().cell()._cells
    deg, _ = Vv.ufl_element().degree()
    DG = FiniteElement("DG", cell, deg)
    CG = FiniteElement("CG", interval, 1)
    Tv_ele = TensorProductElement(DG, CG)
    Tv = FunctionSpace(Vv.mesh(), Tv_ele)

    W = Vvd * VDG * Tv
    v, pp, lambdar = TrialFunctions(W)
    dv, dp, gammar = TestFunctions(W)

    n = FacetNormal(Vv.mesh())

    if top:
        bmeasure = ds_t
        tmeasure = ds_b
        tstring = "top"
    else:
        bmeasure = ds_b
        tmeasure = ds_t
        tstring = "bottom"

    arhs = -inner(dv, n)*p_boundary*bmeasure - b0*inner(dv, k)*dx
    alhs = (inner(v, dv)*dx -
            div(dv)*pp*dx +
            dp*div(v)*dx +
            lambdar('+')*jump(dv, n=n)*(dS_v + dS_h) +
            lambdar*dot(dv, n)*tmeasure +
            gammar('+')*jump(v, n=n)*(dS_v + dS_h) +
            gammar*dot(v, n)*tmeasure)

    w = Function(W)

    bcs = [DirichletBC(W.sub(2), Constant(0.0), tstring)]

    pproblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    params = {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': '0, 1',
        'condensed_field': {
            'ksp_type': 'cg',
            'pc_type': 'gamg',
            'ksp_rtol': 1e-13,
            'ksp_atol': 1e-13,
            'mg_levels': {
                'ksp_type': 'chebyshev',
                'ksp_chebyshev_esteig': None,
                'ksp_max_it': 3,
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }
    }

    psolver = LinearVariationalSolver(pproblem, solver_parameters=params)
    psolver.solve()
    _, p, _ = w.split()
    p0.assign(p)


class GravityWaveProblem(object):
    """Linear commpressible Boussinesq problem."""

    def __init__(self, refinement_level, nlayers, Dt, method="RTCF",
                 X=125.0, thickness=1.0E4, model_degree=1,
                 rtol=1.0E-6, hybridization=False, cfl=1.0, use_dt_from_cfl=False):

        super(GravityWaveProblem, self).__init__()

        self.refinement_level = refinement_level
        self.nlayers = nlayers
        self.thickness = thickness
        self.method = method
        self.model_degree = model_degree
        self.hybridization = hybridization
        self.rtol = rtol
        self.use_dt_from_cfl = use_dt_from_cfl

        # Scaled radius for gravity wave example
        self._X = X             # Factor to scale radius
        self._R = 6.371E6 / X
        self.R = Constant(self._R)
        self._c = 343.0         # speed of sound
        self._N = 0.01          # buoyancy frequency
        self._Omega = 7.292E-5  # Angular rotation rate

        self.c = Constant(self._c)
        self.N = Constant(self._N)
        self.Omega = Constant(self._Omega)

        self.mesh_degree = 3    # degree of the coordinate field

        if self.method == "RT":
            base = IcosahedralSphereMesh(self._R,
                                         refinement_level=self.refinement_level,
                                         degree=self.mesh_degree)
        elif self.method == "RTCF":
            base = CubedSphereMesh(self._R,
                                   refinement_level=self.refinement_level,
                                   degree=self.mesh_degree)
        else:
            raise ValueError("Unknown method %s" % self.method)

        x = SpatialCoordinate(base)
        global_normal = as_vector(x)
        base.init_cell_orientations(global_normal)

        mesh = ExtrudedMesh(base, extrusion_type="radial",
                            layers=self.nlayers,
                            layer_height=self.thickness/self.nlayers)

        self.mesh = mesh

        # Get Dx information (this is approximate).
        # We compute the area (m^2) of each cell in the mesh,
        # then take the square root to get the right units.
        cell_vs = interpolate(CellVolume(self.mesh),
                              FunctionSpace(self.mesh, "DG", 0))

        a_max = fmax(cell_vs)
        dx_max = sqrt(a_max)
        self.dx_max = dx_max
        self.dz = self.thickness / self.nlayers

        if use_dt_from_cfl:
            self.courant = cfl
            Dt = self.courant * dx_max / self._c
            self.Dt = Dt
            self.dt = Constant(self.Dt)
        else:
            self.Dt = Dt
            self.dt = Constant(self.Dt)
            self.courant = (Dt / dx_max) * self._c

        # Create tensor product complex:
        if self.method == "RT":
            U1 = FiniteElement('RT', triangle, self.model_degree)
            U2 = FiniteElement('DG', triangle, self.model_degree - 1)

        else:
            assert self.method == "RTCF"
            U1 = FiniteElement('RTCF', quadrilateral, self.model_degree)
            U2 = FiniteElement('DQ', quadrilateral, self.model_degree - 1)

        V0 = FiniteElement('CG', interval, self.model_degree)
        V1 = FiniteElement('DG', interval, self.model_degree - 1)

        # HDiv element
        W2_ele_h = HDiv(TensorProductElement(U1, V1))
        W2_ele_v = HDiv(TensorProductElement(U2, V0))
        W2_ele = W2_ele_h + W2_ele_v

        # L2 element
        W3_ele = TensorProductElement(U2, V1)

        # Charney-Phillips element
        Wb_ele = TensorProductElement(U2, V0)

        # Resulting function spaces
        self.W2 = FunctionSpace(mesh, W2_ele)
        self.W3 = FunctionSpace(mesh, W3_ele)
        self.Wb = FunctionSpace(mesh, Wb_ele)
        self.W2v = FunctionSpace(mesh, W2_ele_v)

        # Space for Coriolis term
        self.CG_family = "Q" if self.method == "RTCF" else "CG"
        self.Vm = FunctionSpace(mesh, self.CG_family, 1)

        x = SpatialCoordinate(self.mesh)
        self.khat = interpolate(x/sqrt(dot(x, x)),
                                self.mesh.coordinates.function_space())

        self._build_initial_conditions()
        self._build_mixed_solver()

        self.sim_time = []
        self.up_reductions = []
        self.ksp_outer_its = []
        self.ksp_inner_its = []

    def _build_initial_conditions(self):

        W2 = self.W2
        W3 = self.W3
        Wb = self.Wb
        R = self.R
        x = SpatialCoordinate(self.mesh)

        # Initial condition for velocity
        u0 = Function(W2)
        u_max = Constant(20.0)
        uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
        u0.project(uexpr)
        self.u0 = u0

        # Initial condition for the buoyancy perturbation
        lamda_c = 2.0*pi/3.0
        phi_c = 0.0
        W_CG1 = FunctionSpace(self.mesh, self.CG_family, 1)
        z = Function(W_CG1).interpolate(sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - R)
        lat = Function(W_CG1).interpolate(asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])))
        lon = Function(W_CG1).interpolate(atan_2(x[1], x[0]))
        b0 = Function(Wb)
        L_z = 20000.0
        d = 5000.0
        sin_tmp = sin(lat) * sin(phi_c)
        cos_tmp = cos(lat) * cos(phi_c)
        q = R*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
        s = (d**2)/(d**2 + q**2)
        bexpr = s*sin(2*pi*z/L_z)
        b0.interpolate(bexpr)
        self.b0 = b0

        # Initial condition for pressure
        p0 = Function(W3)

        pref = Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
        compute_balanced_pressure(self.W2v, self.W3, self.khat,
                                  self.b0, p0, p_boundary=pref)
        self.p0 = p0

    def _build_mixed_solver(self):
        from firedrake.assemble import create_assembly_callable

        W2 = self.W2
        W3 = self.W3
        Wb = self.Wb
        Vm = self.Vm
        x = SpatialCoordinate(self.mesh)

        # velocity-pressure mixed solver
        W = W2 * W3
        u, p = TrialFunctions(W)
        w, phi = TestFunctions(W)

        # Coriolis term
        fexpr = 2*self.Omega*x[2]/self.R
        f = interpolate(fexpr, Vm)

        # radial unit vector
        khat = self.khat
        dt = self.dt
        N = self.N
        c = self.c

        W23b = W2 * W3 * Wb
        self.qn = Function(W23b, name="State")
        u0, p0, b0 = self.qn.split()

        a_up = (dot(w, u) + 0.5*dt*dot(w, f*cross(khat, u))
                - 0.5*dt*p*div(w)
                # Appears after eliminating b
                + (0.5*dt*N)**2*dot(w, khat)*dot(u, khat)
                + phi*p + 0.5*dt*c**2*phi*div(u))*dx

        L_up = (dot(w, u0) + 0.5*dt*dot(w, f*cross(khat, u0))
                + 0.5*dt*dot(w, khat*b0)
                + phi*p0)*dx

        self.up = Function(W)

        self._up_residual = Function(W)
        self._assemble_up_residual = create_assembly_callable(
            action(a_up, self.up) - L_up,
            tensor=self._up_residual
        )
        self._up_b = Function(W)
        self._assemble_b = create_assembly_callable(
            L_up,
            tensor=self._up_b
        )

        # no-slip conditions
        bcs = [DirichletBC(W.sub(0), Constant(0.0), "bottom"),
               DirichletBC(W.sub(0), Constant(0.0), "top")]

        up_problem = LinearVariationalProblem(a_up, L_up, self.up, bcs=bcs)

        if self.hybridization:
            parameters = {
                'ksp_type': 'preonly',
                'mat_type': 'matfree',
                'pmat_type': 'matfree',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',
                'hybridization': {
                    'ksp_type': 'gmres',
                    'pc_type': 'gamg',
                    'ksp_max_it': 100,
                    'pc_gamg_sym_graph': None,
                    'ksp_monitor_true_residual': None,
                    'pc_gamg_reuse_interpolation': None,
                    'ksp_rtol': self.rtol,
                    'mg_levels': {
                        'ksp_type': 'richardson',
                        'ksp_max_it': 5,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    }
                }
            }

        else:
            parameters = {
                'mat_type': 'aij',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_monitor_true_residual': None,
                'ksp_type': 'gmres',
                'ksp_rtol': self.rtol,
                'ksp_max_it': 100,
                'pc_fieldsplit_schur_fact_type': 'FULL',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_0': {
                    'ksp_type': 'preonly',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'
                },
                'fieldsplit_1': {
                    'ksp_type': 'cg',
                    'ksp_max_it': 100,
                    'pc_type': 'gamg',
                    'pc_gamg_sym_graph': None,
                    'pc_gamg_reuse_interpolation': None,
                    'ksp_monitor_true_residual': None,
                    'mat_schur_complement_ainv_type': 'lump',
                    'mg_levels': {
                        'ksp_type': 'richardson',
                        'ksp_chebyshev_esteig': None,
                        'ksp_max_it': 5,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    }
                }
            }

        up_solver = LinearVariationalSolver(up_problem,
                                            solver_parameters=parameters,
                                            options_prefix="up-solver")
        self.up_solver = up_solver

        # Buoyancy solver
        gamma = TestFunction(Wb)
        b = TrialFunction(Wb)

        a_b = gamma*b*dx
        L_b = dot(gamma*khat, u0)*dx

        self.b_update = Function(Wb)
        b_problem = LinearVariationalProblem(a_b, L_b, self.b_update)
        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters={
                                               'ksp_type': 'cg',
                                               'pc_type': 'bjacobi',
                                               'sub_pc_type': 'ilu'
                                           },
                                           options_prefix="b-solver")
        self.b_solver = b_solver
        self.qn1 = Function(W23b)

    @cached_property
    def num_cells(self):
        return self.mesh.cell_set.size

    @cached_property
    def comm(self):
        return self.mesh.comm

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
        return File(dirname + "gw_" + str(self.refinement_level) + ".pvd")

    def write(self, dumpcount, dumpfreq):

        dumpcount += 1
        un, pn, bn = self.qn.split()
        if (dumpcount > dumpfreq):
            self.output_file.write(un, pn, bn)
            dumpcount -= dumpfreq

        return dumpcount

    def initialize(self):

        qn = self.qn
        un, pn, bn = qn.split()
        un.assign(self.u0)
        pn.assign(self.p0)
        bn.assign(self.b0)
        self.qn1.assign(0.0)
        self.b_update.assign(0.0)
        self.up.assign(0.0)

    def warmup(self):

        self.initialize()

        qn = self.qn
        qn1 = self.qn1
        b_update = self.b_update
        un, pn, bn = qn.split()
        un1, pn1, bn1 = qn1.split()
        u, p = self.up.split()

        with timed_stage("Warm up: UP Solver"):
            self.up_solver.solve()

        un1.assign(u)
        pn1.assign(p)
        un.assign(un1)
        pn.assign(pn1)

        with timed_stage("Warm up: B Solver"):
            self.b_solver.solve()

        with timed_stage("Warm up: B reconstruction"):
            bn1.assign(assemble(bn - 0.5*(self.dt*self.N**2)*b_update))

        bn.assign(bn1)

    def run_simulation(self, tmax, write=False, dumpfreq=100):

        PETSc.Sys.Print("""
Running Skamarock and Klemp Gravity wave problem:\n
method: %s,\n
model degree: %d,\n
refinements: %d,\n
hybridization: %s,\n
tmax: %s
"""
                        % (self.method, self.model_degree,
                           self.refinement_level, self.hybridization, tmax))

        t = 0.0
        un, pn, bn = self.qn.split()
        un1, pn1, bn1 = self.qn1.split()
        b_update = self.b_update
        u, p = self.up.split()

        dumpcount = dumpfreq
        if write:
            dumpcount = self.write(dumpcount, dumpfreq)

        with timed_stage("Initial conditions"):
            self.up_solver.solve()
            un1.assign(u)
            pn1.assign(p)
            un.assign(un1)
            pn.assign(pn1)
            self.b_solver.solve()
            bn1.assign(assemble(bn - 0.5*(self.dt*self.N**2)*b_update))
            bn.assign(bn1)
            t += self.Dt

        # Start profiling

        self.up_solver.snes.setConvergenceHistory()
        self.up_solver.snes.ksp.setConvergenceHistory()
        self.b_solver.snes.setConvergenceHistory()
        self.b_solver.snes.ksp.setConvergenceHistory()

        while t < tmax - self.Dt/2:
            t += self.Dt
            self.sim_time.append(t)

            with timed_stage("UP Solver"):
                self.up_solver.solve()

            # Here we collect the reductions in the u-p linear residual.
            self._assemble_up_residual()
            self._assemble_b()
            r_factor = self._up_residual.dat.norm / self._up_b.dat.norm
            print("reduction: %s" % r_factor)
            self.up_reductions.append(r_factor)

            un1.assign(u)
            pn1.assign(p)
            un.assign(un1)
            pn.assign(pn1)

            with timed_stage("B Solver"):
                self.b_solver.solve()

            with timed_stage("B reconstruction"):
                bn1.assign(assemble(bn - 0.5*(self.dt*self.N**2)*b_update))

            bn.assign(bn1)

            outer_ksp = self.up_solver.snes.ksp
            if self.hybridization:
                ctx = outer_ksp.getPC().getPythonContext()
                inner_ksp = ctx.trace_ksp
            else:
                ksps = outer_ksp.getPC().getFieldSplitSubKSP()
                _, inner_ksp = ksps

            # Collect ksp iterations
            self.ksp_outer_its.append(outer_ksp.getIterationNumber())
            self.ksp_inner_its.append(inner_ksp.getIterationNumber())

            if write:
                with timed_stage("Dump output"):
                    dumpcount = self.write(dumpcount, dumpfreq)
