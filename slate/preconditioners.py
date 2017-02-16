"""This module provides custom python preconditioners utilizing
the Slate language.
"""

from __future__ import absolute_import, print_function, division

import ufl

from firedrake.formmanipulation import ArgumentReplacer, split_form
from firedrake.matrix_free.preconditioners import PCBase
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor


__all__ = ['HybridizationPC']


class HybridizationPC(PCBase):
    """A Slate-based python preconditioner that solves a
    mixed saddle-point problem using a hybridizable DG scheme.

    The forward eliminations and backwards reconstructions
    are performed element-local using the Slate language.
    """
    def initialize(self, pc):
        """Set up the problem context. Take the original
        mixed problem and reformulate the problem as a
        hybridized DG one.

        A KSP is created for the Lagrange multiplier system.
        """
        from ufl.algorithms.map_integrands import map_integrand_dags
        from firedrake import (FunctionSpace, TrialFunction,
                               TrialFunctions, TestFunction, Function,
                               BrokenElement, MixedElement,
                               FacetNormal, Constant, DirichletBC,
                               assemble, Projector)

        # Extract the problem context
        prefix = pc.getOptionsPrefix()
        _, P = pc.getOperators()
        context = P.getPythonContext()
        test, trial = context.a.arguments()

        # Break the function spaces and define fully discontinuous spaces
        self.V = test.function_space()
        if self.V.mesh().cell_set._extruded:
            raise NotImplementedError("Not implemented on extruded meshes.")

        broken_elements = [BrokenElement(Vi.ufl_element()) for Vi in self.V]
        elem = MixedElement(broken_elements)
        V_d = FunctionSpace(self.V.mesh(), elem)
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}

        # Replace the problems arguments with arguments defined
        # on the new discontinuous spaces
        replacer = ArgumentReplacer(arg_map)
        self.new_form = map_integrand_dags(replacer, context.a)

        split_forms = split_form(self.new_form)
        # Store these matrices for reconstruction
        self.A = Tensor([sf.form for sf in split_forms
                         if sf.indices == (0, 0)][0])
        self.B = Tensor([sf.form for sf in split_forms
                         if sf.indices == (1, 0)][0])
        self.C = Tensor([sf.form for sf in split_forms
                         if sf.indices == (1, 1)][0])

        # Create the space of approximate traces:
        # the space of Lagrange multipliers
        self.TraceSpace = FunctionSpace(self.V.mesh(), "HDiv Trace",
                                        V_d.ufl_element().degree() - 1)
        self.trace_condition = DirichletBC(self.TraceSpace, Constant(0.0),
                                           "on_boundary")

        # Broken flux and scalar terms (solution via reconstruction)
        self.broken_solution = Function(V_d)

        # Broken RHS of the fully discontinuous problem
        self.broken_rhs = Function(V_d)

        # Solution of the system for the Lagrange multipliers
        self.trace_solution = Function(self.TraceSpace)
        self.schur_rhs = Function(self.TraceSpace)

        # unbroken solutions and rhs
        self.unbroken_solution = Function(self.V)
        self.unbroken_rhs = Function(self.V)

        # Create the symbolic Schur-reduction of the discontinuous
        # problem in Slate. Weakly enforce continuity via the Lagrange
        # multipliers
        self.Atilde = Tensor(self.new_form)
        gammar = TestFunction(self.TraceSpace)
        self.n = FacetNormal(self.V.mesh())
        sigma, _ = TrialFunctions(V_d)
        self.K = Tensor(gammar('+') * ufl.dot(sigma, self.n) * ufl.dS)
        trial = TrialFunction(V_d[0])
        self.K_local = Tensor(gammar('+') * ufl.dot(trial, self.n) * ufl.dS)
        self.schur_comp = assemble(self.K * self.Atilde.inv * self.K.T,
                                   bcs=self.trace_condition)
        self.schur_comp.force_evaluation()

        # Set up the KSP for the system of Lagrange multipliers
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOperators(self.schur_comp.petscmat)
        ksp.setOptionsPrefix(prefix + "trace_")
        ksp.setTolerances(rtol=1e-8)
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp
        sigma_b, _ = self.broken_solution.split()
        sigma_u, _ = self.unbroken_solution.split()
        self.projector = Projector(sigma_b,
                                   sigma_u,
                                   solver_parameters={"ksp_type": "cg",
                                                      "ksp_rtol": 1e-13})

    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        from firedrake import assemble

        assemble(self.schur_comp, tensor=self.schur_comp,
                 bcs=self.trace_condition)
        assemble(self.schur_rhs, tensor=self.schur_rhs)

    def apply(self, pc, x, y):
        """We solve the forward eliminated problem for the
        approximate traces of the scalar solution (the multipliers)
        and reconstruct the "broken flux and scalar variable."

        Lastly, we project the broken solutions into the mimetic
        non-broken finite element space.
        """
        from firedrake import project, assemble

        # Transfer non-broken x into a firedrake function
        with self.unbroken_rhs.dat.vec as v:
            x.copy(v)

        # Transfer unbroken_rhs into broken_rhs
        field0, field1 = self.unbroken_rhs.split()
        bfield0, bfield1 = self.broken_rhs.split()

        # This updates broken_rhs
        project(field0, bfield0)
        field1.dat.copy(bfield1.dat)

        # Compute the rhs for the multiplier system
        assemble(self.K * self.Atilde.inv * self.broken_rhs,
                 tensor=self.schur_rhs)

        # Solve the system for the Lagrange multipliers
        with self.schur_rhs.dat.vec_ro as b:
            with self.trace_solution.dat.vec as x:
                self.ksp.solve(b, x)

        # Split the solution function and reconstruct
        # each bit separately
        sigma_h, u_h = self.broken_solution.split()
        _, f = self.broken_rhs.split()

        # Pressure reconstruction
        M = self.B * self.A.inv * self.B.T + self.C
        u_sol = M.inv * f + M.inv * (self.B * self.A.inv *
                                     self.K_local.T * self.trace_solution)
        assemble(u_sol, tensor=u_h)

        # Velocity reconstructions
        sigma_sol = self.A.inv * (self.B.T * u_h -
                                  self.K_local.T * self.trace_solution)
        assemble(sigma_sol, tensor=sigma_h)

        # Project the broken solution into non-broken spaces
        sigma_u, u_u = self.unbroken_solution.split()
        u_h.dat.copy(u_u.dat)
        self.projector.project()
        with self.unbroken_solution.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""
        raise NotImplementedError("Not implemented yet, sorry!")

    def view(self, pc, viewer=None):
        """View the KSP solver to monitor the Lagrange multiplier
        system.
        """
        super(HybridizationPC, self).view(pc, viewer)
        viewer.printfASCII("KSP solver for the Lagrange multipliers\n")
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()
