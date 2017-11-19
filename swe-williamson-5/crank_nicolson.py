from gusto.timeloop import BaseTimestepper
from gusto.configuration import logger
from pyop2.profiling import timed_stage
from firedrake.petsc import PETSc
from mpi4py import MPI


import pandas as pd


class CrankNicolsonStepper(BaseTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    NOTE: This is a slightly modified version of the CrankNicolson
    in the Gusto repository. This timestepper is modified to support
    extended profiling.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics`
        schemes.
    """

    def __init__(self, state, advected_fields, linear_solver, forcing,
                 diffused_fields=None, physics_list=None):

        super(CrankNicolsonStepper, self).__init__(state, advected_fields,
                                                   diffused_fields,
                                                   physics_list)
        self.linear_solver = linear_solver
        self.forcing = forcing

        self.incompressible = False

        if state.mu is not None:
            self.mu_alpha = [0., state.timestepping.dt]
        else:
            self.mu_alpha = [None, None]

        self.xstar_fields = {name: func for (name, func) in
                             zip(state.fieldlist, state.xstar.split())}
        self.xp_fields = {name: func for (name, func) in
                          zip(state.fieldlist, state.xp.split())}

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme)
                                 for name, scheme in advected_fields
                                 if name in state.fieldlist]

        state.xb.assign(state.xn)
        self.t_array = []
        self.solve_time_array = []
        self.ksp_iter_array = []
        self.inner_ksp_iter_array = []
        self.picard_iter_array = []

    @property
    def passive_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme) for name, scheme in
                self.advected_fields if name not in self.state.fieldlist]

    def semi_implicit_step(self):
        state = self.state
        dt = state.timestepping.dt
        alpha = state.timestepping.alpha

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               state.xstar, mu_alpha=self.mu_alpha[0])

        for k in range(state.timestepping.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and state.xnp1
                    advection.update_ubar(state.xn, state.xnp1, alpha)
                    # advects a field from xstar and puts result in xp
                    advection.apply(self.xstar_fields[name],
                                    self.xp_fields[name])
            # xrhs is the residual which goes in the linear solve
            state.xrhs.assign(0.0)

            for i in range(state.timestepping.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs, mu_alpha=self.mu_alpha[1],
                                       incompressible=self.incompressible)

                state.xrhs -= state.xnp1

                with timed_stage("Implicit solve"):
                    # solves linear system and places result in state.dy
                    if self.linear_solver._profiling:
                        tsec = float(state.t.dat.data)
                        self.t_array.append(tsec)
                        self.picard_iter_array.append(k)
                        PETSc.Sys.Print("Timing implicit solve.\n")
                        self.linear_solver.solve()
                        PETSc.Sys.Print(
                            "Implicit solve finished for t=%s.\n" % tsec
                        )

                        # Collect solver time
                        ksp_event = PETSc.Log.Event("KSPSolve").getPerfInfo()
                        comm = self.linear_solver.uD_solver._problem.u.comm
                        size = comm.size
                        ksp_time = comm.allreduce(ksp_event["time"],
                                                  op=MPI.SUM)/size
                        self.solve_time_array.append(ksp_time)

                        # Collect KSP iterations
                        outer_ksp = self.linear_solver.uD_solver.snes.ksp
                        if self.linear_solver._hybridized:
                            cxt = outer_ksp.getPC().getPythonContext()
                            inner_ksp = cxt.trace_ksp
                        else:
                            # Approx SC has two ksps (one for each field)
                            # We only need to compare with the second field
                            ksps = outer_ksp.getPC().getFieldSplitSubKSP()
                            _, inner_ksp = ksps

                        outer_its = outer_ksp.getIterationNumber()
                        inner_its = inner_ksp.getIterationNumber()
                        self.ksp_iter_array.append(outer_its)
                        self.inner_ksp_iter_array.append(inner_its)
                    else:
                        self.linear_solver.solve()

                state.xnp1 += state.dy

            self._apply_bcs()

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        t = self.setup_timeloop(t, tmax, pickup)

        state = self.state
        dt = state.timestepping.dt

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            state.xnp1.assign(state.xn)

            self.semi_implicit_step()

            for name, advection in self.passive_advection:
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn,
                                      state.xnp1,
                                      state.timestepping.alpha)
                # advects a field from xn and puts result in xnp1
                advection.apply(field, field)

            state.xb.assign(state.xn)
            state.xn.assign(state.xnp1)

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffused_fields:
                    field = getattr(state.fields, name)
                    diffusion.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))
