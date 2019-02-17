from firedrake import *
from firedrake.petsc import PETSc
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage
import numpy as np
from balance_pressure import compute_balanced_pressure
from extruded_vertical_normal import VerticalNormal
from solver import GravityWaveSolver


def fmax(f):
    fmax = op2.Global(1, np.finfo(float).min, dtype=float)
    op2.par_loop(op2.Kernel("""
void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


class GravityWaveProblem(object):
    """Linear commpressible Boussinesq problem."""

    def __init__(self, refinement_level, nlayers, Dt, method="RT",
                 X=125.0, thickness=1.0E4, model_degree=1,
                 rtol=1.0E-6, hybridization=False,
                 coriolis=False,
                 cfl=1.0,
                 use_dt_from_cfl=False):

        super(GravityWaveProblem, self).__init__()

        self.refinement_level = refinement_level
        self.nlayers = nlayers
        self.thickness = thickness
        self.method = method
        self.model_degree = model_degree
        self.hybridization = hybridization
        self.rtol = rtol
        self.coriolis = coriolis
        self.use_dt_from_cfl = use_dt_from_cfl

        # Scaled radius for gravity wave example
        self._X = X             # Factor to scale radius
        self._R = 6.371E6 / X
        self.R = Constant(self._R)
        self._c = 343.0         # speed of sound
        self._N = 0.01          # buoyancy frequency
        self._Omega = 7.292E-5  # Angular rotation rate
        self.Omega = Constant(self._Omega)

        self.mesh_degree = 3    # degree of the coordinate field

        if self.method == "RT" or self.method == "BDFM":
            base = IcosahedralSphereMesh(self._R,
                                         refinement_level=self.refinement_level,
                                         degree=self.mesh_degree)
        elif self.method == "RTCF":
            base = CubedSphereMesh(self._R,
                                   refinement_level=self.refinement_level,
                                   degree=self.mesh_degree)
        else:
            raise ValueError("Unknown method %s" % self.method)

        global_normal = as_vector(SpatialCoordinate(base))
        base.init_cell_orientations(global_normal)

        mesh = ExtrudedMesh(base, extrusion_type="radial",
                            layers=self.nlayers,
                            layer_height=self.thickness/self.nlayers)
        self.mesh = mesh

        vert_norm = VerticalNormal(self.mesh)
        self.khat = vert_norm.khat

        # Get Dx information (this is approximate).
        # We compute the area (m^2) of each cell in the mesh,
        # then take the square root to get the right units.
        cell_vs = interpolate(CellVolume(base),
                              FunctionSpace(base, "DG", 0))

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

        elif self.method == "BDFM":
            U1 = FiniteElement('BDFM', triangle, 2)
            U2 = FiniteElement('DG', triangle, 1)
            # BDFM only supported for degree 2, so overwrite here
            self.model_degree = 2
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

        self.Wup = self.W2 * self.W3
        self.Wupb = self.W2 * self.W3 * self.Wb

        # Functions for the state and residual
        self.state = Function(self.Wupb)
        self.state0 = Function(self.Wupb)

        # Space for Coriolis term
        self.CG_family = "Q" if self.method == "RTCF" else "CG"
        self.Vm = FunctionSpace(mesh, self.CG_family, 1)
        x = SpatialCoordinate(mesh)
        fexpr = 2*self.Omega*x[2]/self.R
        f = interpolate(fexpr, self.Vm)
        self._fexpr = f

        self._build_initial_conditions()

        if self.coriolis:
            solver = GravityWaveSolver(W2=self.W2,
                                       W3=self.W3,
                                       Wb=self.Wb,
                                       dt=self.Dt,
                                       c=self._c,
                                       N=self._N,
                                       khat=self.khat,
                                       maxiter=100,
                                       tolerance=self.rtol,
                                       coriolis=self._fexpr,
                                       hybridization=self.hybridization)
        else:
            solver = GravityWaveSolver(W2=self.W2,
                                       W3=self.W3,
                                       Wb=self.Wb,
                                       dt=self.Dt,
                                       c=self._c,
                                       N=self._N,
                                       khat=self.khat,
                                       maxiter=100,
                                       tolerance=self.rtol,
                                       coriolis=None,
                                       hybridization=self.hybridization)

        self.gravity_wave_solver = solver
        self.ksp_inner_its = []
        self.ksp_outer_its = []
        self.sim_time = []

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
        compute_balanced_pressure(self.W2v, self.W3,
                                  self.khat,
                                  self.b0, p0, p_boundary=pref)
        self.p0 = p0

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
        un, pn, bn = self.state.split()
        if (dumpcount > dumpfreq):
            self.output_file.write(un, pn, bn)
            dumpcount -= dumpfreq

        return dumpcount

    def warmup(self):

        state = self.state

        un, pn, bn = state.split()
        un.assign(self.u0)
        pn.assign(self.p0)
        bn.assign(self.b0)

        with timed_stage("Warm up: Solver"):
            un1, pn1, bn1 = self.gravity_wave_solver.solve(un, pn, bn)

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

        state = self.state

        un, pn, bn = state.split()
        un.assign(self.u0)
        pn.assign(self.p0)
        bn.assign(self.b0)

        dumpcount = dumpfreq
        if write:
            dumpcount = self.write(dumpcount, dumpfreq)

        self.gravity_wave_solver._up_solver.snes.setConvergenceHistory()
        self.gravity_wave_solver._up_solver.snes.ksp.setConvergenceHistory()
        self.gravity_wave_solver._b_solver.snes.setConvergenceHistory()
        self.gravity_wave_solver._b_solver.snes.ksp.setConvergenceHistory()

        t = 0.0
        while t < tmax - self.Dt/2:
            t += self.Dt
            self.sim_time.append(t)

            un1, pn1, bn1 = self.gravity_wave_solver.solve(un, pn, bn)
            un.assign(un1)
            pn.assign(pn1)
            bn.assign(bn1)

            outer_ksp = self.gravity_wave_solver._up_solver.snes.ksp
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
