from firedrake import *
from firedrake.utils import cached_property
from gusto.linear_solvers import TimesteppingSolver


class LinearizedShallowWaterSolver(TimesteppingSolver):
    """
    """

    def __init__(self, state, hybridization=False,
                 verification=False, profiling=False):

        self._hybridized = hybridization
        self._profiling = profiling
        self._verify = verification

        super(LinearizedShallowWaterSolver, self).__init__(state,
                                                           solver_parameters=self.solver_parameters,
                                                           overwrite_solver_parameters=False)

    @property
    def _hybrid_params(self):
        return {'ksp_type': 'preonly',
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
                                  # Construct broken residual
                                  'hdiv_residual': {'ksp_type': 'cg',
                                                    'pc_type': 'bjacobi',
                                                    'sub_pc_type': 'ilu',
                                                    'ksp_rtol': 1e-8,
                                                    'ksp_monitor': True},
                                  # Reconstruct HDiv vector field
                                  # via local averaging
                                  # (Alternatively, one could also use
                                  # a Galerkin projection onto the HDiv space)
                                  # 'hdiv_projection':{'ksp_type': 'cg',
                                  #                    'pc_type': 'bjacobi',
                                  #                    'sub_pc_type': 'ilu',
                                  #                    'ksp_rtol': 1e-8,
                                  #                    'ksp_monitor': True}}}
                                  'hdiv_projection': {'method': 'average'}}}

    @property
    def _approx_sc_params(self):
        return {'pc_type': 'fieldsplit',
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
                # This is the reduced system for the depth field.
                # We apply the same solver options as the hybridized
                # reduced system for the Lagrange multipliers.
                'fieldsplit_1': {'ksp_type': 'cg',
                                 'pc_type': 'gamg',
                                 'ksp_monitor': True,
                                 'mg_levels': {'ksp_type': 'chebyshev',
                                               'ksp_max_it': 2,
                                               'pc_type': 'bjacobi',
                                               'sub_pc_type': 'ilu'}}}

    @cached_property
    def solver_parameters(self):
        if self._hybridized:
            solver_parameters = self._hybrid_params
        else:
            solver_parameters = self._approx_sc_params
        if self._verify:
            # if verification mode is on, then wrap outer solve
            # in a GMRES loop and monitor the problem residual
            solver_parameters['ksp_monitor_true_residual'] = True
            solver_parameters['ksp_type'] = 'gmres'
            solver_parameters['ksp_monitor'] = True

        return solver_parameters

    def _setup_solver(self):
        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha

        # Split up the rhs vector (symbolically)
        u_in, D_in = split(state.xrhs)

        W = state.W
        w, phi = TestFunctions(W)
        u, D = TrialFunctions(W)

        # Linearized shallow water system in residual form
        eqn = (inner(w, u) - beta*g*div(w)*D
               - inner(w, u_in)
               + phi*D + beta*H*phi*div(u)
               - phi*D_in)*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u D solver
        self.uD = Function(W)

        # Solver for u, D
        uD_problem = LinearVariationalProblem(
            aeqn, Leqn, self.state.dy)

        if self._hybridized:
            prefix = 'SWImplicitHybridMixed'
        else:
            prefix = 'SWImplicit'

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=prefix)

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """
        self.uD_solver.solve()
