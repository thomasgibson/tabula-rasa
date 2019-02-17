from firedrake import *
from pyop2.profiling import timed_stage


__all__ = ['GravityWaveSolver']


class GravityWaveSolver(object):

    def __init__(self, W2, W3, Wb, dt, c, N, khat,
                 maxiter=1000, tolerance=1.E-6,
                 coriolis=None,
                 hybridization=False):

        self._W2 = W2
        self._W3 = W3
        self._Wb = Wb
        self._dt = dt
        self._c = c
        self._N = N

        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._omega_N2 = Constant((0.5*dt*N)**2)

        self._maxiter = maxiter
        self._rtol = tolerance
        self._hybridization = hybridization

        self._coriolis = coriolis
        self._khat = khat

        self._Wmixed = W2 * W3
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._b = Function(self._Wb)
        self._ru = Function(self._W2)
        self._rp = Function(self._W3)
        self._rb = Function(self._Wb)
        self._up = Function(self._Wmixed)
        self._btmp = Function(self._Wb)

        self._solver_setup()

    def _solver_setup(self):

        utest, ptest = TestFunctions(self._Wmixed)
        utrial, ptrial = TrialFunctions(self._Wmixed)

        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]

        if self._hybridization:
            parameters = {
                'ksp_type': 'preonly',
                'mat_type': 'matfree',
                'pmat_type': 'matfree',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',
                'hybridization': {
                    'ksp_type': 'fgmres',
                    'ksp_monitor_true_residual': None,
                    'ksp_max_it': self._maxiter,
                    'ksp_rtol': self._rtol,
                    'pc_type': 'gamg',
                    'pc_mg_cycles': 'v',
                    'pc_gamg_reuse_interpolation': None,
                    'pc_gamg_sym_graph': None,
                    'mg_levels': {
                        'ksp_type': 'gmres',
                        'ksp_max_it': 5,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    }
                }
            }

        else:
            parameters = {
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_type': 'gmres',
                'ksp_norm_type': 'unpreconditioned',
                'ksp_max_it': self._maxiter,
                'ksp_rtol': self._rtol,
                'ksp_monitor_true_residual': None,
                'pc_fieldsplit_schur_fact_type': 'FULL',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_0': {
                    'ksp_type': 'preonly',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'
                },
                'fieldsplit_1': {
                    'ksp_type': 'preonly',
                    'pc_type': 'hypre',
                    'pc_hypre_type': 'boomeramg',
                    'pc_hypre_boomeramg_max_iter': 1,
                    'pc_hypre_boomeramg_agg_nl': 0,
                    'pc_hypre_boomeramg_coarsen_type': 'Falgout',
                    'pc_hypre_boomeramg_smooth_type': 'Euclid',
                    'pc_hypre_boomeramg_eu_bj': 1,
                    'pc_hypre_boomeramg_interptype': 'classical',
                    'pc_hypre_boomeramg_P_max': 0,
                    'pc_hypre_boomeramg_agg_nl': 0,
                    'pc_hypre_boomeramg_strong_threshold': 0.25,
                    'pc_hypre_boomeramg_max_levels': 5,
                    'pc_hypre_boomeramg_no_CF': False
                }
            }

        a_up = (ptest*ptrial
                + self._dt_half_c2*ptest*div(utrial)
                - self._dt_half*div(utest)*ptrial
                + (dot(utest, utrial)
                   + self._omega_N2*dot(utest, self._khat)*dot(utrial, self._khat)))*dx

        r_u = self._ru
        r_p = self._rp
        r_b = self._rb
        up = self._up

        L_up = (dot(utest, r_u)
                + self._dt_half*dot(utest, self._khat*r_b)
                + ptest*r_p)*dx

        if self._coriolis:
            f = self._coriolis
            a_up += self._dt_half*dot(utest, f*cross(self._khat, utrial))*dx
            L_up += self._dt_half*dot(utest, f*cross(self._khat, r_u))*dx

        up_problem = LinearVariationalProblem(a_up, L_up, up, bcs=bcs)
        up_solver = LinearVariationalSolver(up_problem,
                                            solver_parameters=parameters)

        self._up_solver = up_solver

        btest = TestFunction(self._Wb)
        L_b = dot(btest*self._khat, self._u)*dx
        a_b = btest*TrialFunction(self._Wb)*dx
        b_problem = LinearVariationalProblem(a_b, L_b, self._btmp)
        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters={
                                               'ksp_type': 'cg',
                                               'pc_type': 'bjacobi',
                                               'sub_pc_type': 'ilu'
                                           })
        self._b_solver = b_solver

    def solve(self, r_u, r_p, r_b):

        self._ru.assign(r_u)
        self._rp.assign(r_p)
        self._rb.assign(r_b)

        # Fields for solution
        self._u.assign(0.0)
        self._p.assign(0.0)
        self._b.assign(0.0)

        # initialize solver fields
        self._up.assign(0.0)
        self._btmp.assign(0.0)

        with timed_stage('UP Solver'):
            self._up_solver.solve()

        self._u.assign(self._up.sub(0))
        self._p.assign(self._up.sub(1))

        with timed_stage('B Solver'):
            self._b_solver.solve()
            self._b.assign(assemble(self._rb - self._dt_half_N2*self._btmp))

        return self._u, self._p, self._b
