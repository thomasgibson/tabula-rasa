from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage
import numpy as np
from solver import GravityWaveSolver


def fmax(f):
    fmax = op2.Global(1, np.finfo(float).min, dtype=float)
    op2.par_loop(op2.Kernel("""
void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


class ProfileGravityWaveSolver(object):

    def __init__(self, refinement_level, nlayers, model_degree,
                 method="RTCF", X=1.0,
                 H=1E4, rtol=1.0E-5, hybridization=False, cfl=1,
                 monitor=False):

        super(ProfileGravityWaveSolver, self).__init__()

        self.refinement_level = refinement_level
        self.nlayers = nlayers
        self.H = H
        self.model_degree = model_degree
        self.method = method
        self.hybridization = hybridization
        self.rtol = rtol
        self.monitor = monitor
        self._X = X
        self._R = 6.371E6 / self._X
        self.R = Constant(self._R)
        self._c = 343.0
        self._N = 0.01
        self._Omega = 7.292E-5
        self.Omega = Constant(self._Omega)

        self.mesh_degree = 1
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
                            layer_height=self.H / self.nlayers)
        self.mesh = mesh

        # Get Dx information (this is approximate).
        # We compute the area (m^2) of each cell in the mesh,
        # then take the square root to get the right units.
        cell_vs = interpolate(CellVolume(base),
                              FunctionSpace(base, "DG", 0))

        a_max = fmax(cell_vs)
        dx_max = sqrt(a_max)
        self.dx_max = dx_max
        self.dz = self.H / self.nlayers
        self.courant = cfl
        Dt = self.courant * dx_max / self._c
        self.Dt = Dt
        self.dt = Constant(self.Dt)

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

        self.Wup = self.W2 * self.W3
        self.Wupb = self.W2 * self.W3 * self.Wb

        # Functions for the state and residual
        self.state = Function(self.Wupb)
        self.state0 = Function(self.Wupb)

        x = SpatialCoordinate(mesh)
        fexpr = 2*self.Omega*x[2]/self.R
        self._fexpr = fexpr

        xnorm = sqrt(inner(x, x))
        self.khat = interpolate(x/xnorm, mesh.coordinates.function_space())

        self._build_initial_conditions()

        solver = GravityWaveSolver(W2=self.W2,
                                   W3=self.W3,
                                   Wb=self.Wb,
                                   dt=self.Dt,
                                   c=self._c,
                                   N=self._N,
                                   khat=self.khat,
                                   maxiter=1000,
                                   tolerance=self.rtol,
                                   coriolis=self._fexpr,
                                   hybridization=self.hybridization,
                                   monitor=self.monitor)

        self.gravity_wave_solver = solver
        self.ksp_inner_its = []
        self.ksp_outer_its = []

    def _build_initial_conditions(self):

        W2 = self.W2
        W3 = self.W3
        Wb = self.Wb

        u0 = Function(W2)
        urand = Function(VectorFunctionSpace(self.mesh, "CG", 2))
        urand.dat.data[:] += np.random.randn(*urand.dat.data.shape)
        u0.project(urand)

        p0 = Function(W3)
        p0.dat.data[:] += np.random.randn(len(p0.dat.data))

        b0 = Function(Wb)
        b0.dat.data[:] += np.random.randn(len(b0.dat.data))

        self.u0 = u0
        self.p0 = p0
        self.b0 = b0

    @cached_property
    def num_cells(self):
        return self.mesh.cell_set.size

    @cached_property
    def comm(self):
        return self.mesh.comm

    def warmup(self):

        state = self.state

        un, pn, bn = state.split()
        un.assign(self.u0)
        pn.assign(self.p0)
        bn.assign(self.b0)

        with timed_stage("Warm up: Solver"):
            un1, pn1, bn1 = self.gravity_wave_solver.solve(un, pn, bn)

    def run_profile(self):

        state = self.state

        un, pn, bn = state.split()
        un.assign(self.u0)
        pn.assign(self.p0)
        bn.assign(self.b0)

        self.gravity_wave_solver._up_solver.snes.setConvergenceHistory()
        self.gravity_wave_solver._up_solver.snes.ksp.setConvergenceHistory()
        self.gravity_wave_solver._b_solver.snes.setConvergenceHistory()
        self.gravity_wave_solver._b_solver.snes.ksp.setConvergenceHistory()

        un1, pn1, bn1 = self.gravity_wave_solver.solve(un, pn, bn)

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
