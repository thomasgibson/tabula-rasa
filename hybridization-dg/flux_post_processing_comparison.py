from firedrake import *
import numpy as np


def run_LDG_H_poisson_flux_PP(degree):
    """This function runs the exact same example as in the main run script
    'run-ldg-h-tests.py', but run over various parameters tau and gather
    all the post processed fluxes. We then compare differences between fluxes
    and divergences.

    Results here imply that while the fluxes are difference per problem, their
    divergences are all nearly identical.
    """
    # Set up problem domain
    res = 4
    mesh = UnitSquareMesh(2**res, 2**res, quadrilateral=False)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    def ejump(a):
        return 2*avg(a)

    # Run over various taus:
    h = FacetArea(mesh)
    pp_fluxes = []
    print("Solving the LDG-H system of degree %d over various "
          "choices of tau:\n" % degree)
    for tau in [Constant(1.0), h,  Constant(10.0)/h]:
        # Set up function spaces
        U = VectorFunctionSpace(mesh, "DG", degree)
        V = FunctionSpace(mesh, "DG", degree)
        T = FunctionSpace(mesh, "HDiv Trace", degree)

        # Mixed space and test/trial functions
        W = U * V * T
        q, u, uhat = TrialFunctions(W)
        v, w, mu = TestFunctions(W)

        Vh = FunctionSpace(mesh, "CG", degree + 2)
        f = Function(Vh).interpolate((2*pi*pi)*sin(x[0]*pi)*sin(x[1]*pi))

        a_flux = -grad(sin(x[0]*pi)*sin(x[1]*pi))

        # Numerical flux
        qhat = q + tau*(u - uhat)*n

        # Formulate the LDG-H method in UFL
        a = ((dot(v, q) - div(v)*u)*dx
             + ejump(uhat*inner(v, n))*dS
             + uhat*inner(v, n)*ds
             - dot(grad(w), q)*dx
             + ejump(inner(qhat, n)*w)*dS
             + inner(qhat, n)*w*ds
             # Transmission condition (interior only)
             + ejump(mu*inner(qhat, n))*dS
             # trace mass term for the boundary conditions
             # <uhat, mu>ds == <g, mu>ds where g=0 in this example
             + uhat*mu*ds)

        L = w*f*dx

        print("Solving the LDG-H system.")
        w = Function(W, name="solutions")
        params = {'mat_type': 'matfree',
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                  # Use the static condensation PC for hybridized problems
                  # and use a direct solve on the reduced system for u_hat
                  'pc_python_type': 'firedrake.HybridStaticCondensationPC',
                  'hybrid_sc': {'ksp_type': 'preonly',
                                'pc_type': 'lu'}}
        solve(a == L, w, solver_parameters=params)

        print("Solver finished.\n")
        # Computed flux, scalar, and trace
        q_h, u_h, uhat_h = w.split()

        print("Interior jump of unprocessed flux: %0.13f"
              % np.abs(assemble(jump(q_h, n=n)*dS)))

        # Post process the flux variable:
        qhat_h = q_h + tau*(u_h - uhat_h)*n
        RTd = FunctionSpace(mesh, "DRT", degree + 1)
        DGkn1 = VectorFunctionSpace(mesh, "DG", degree - 1)

        # Use the trace space already defined
        Npp = DGkn1 * T
        n_p = TrialFunction(RTd)
        vp, mu = TestFunctions(Npp)

        # Assemble the local system and invert using Slate
        A = Tensor(inner(n_p, vp)*dx +
                   jump(n_p, n=n)*mu*dS + dot(n_p, n)*mu*ds)
        B = Tensor(inner(q_h, vp)*dx +
                   jump(qhat_h, n=n)*mu*dS + dot(qhat_h, n)*mu*ds)

        print("Local post-processing of the flux.\n")
        q_pp = assemble(A.inv * B)

        print("Post-processing finished.\n")
        pp_fluxes.append(q_pp)

        print("Interior jump of processed flux: %0.13f"
              % np.abs(assemble(jump(q_pp, n=n)*dS)))

        err = sqrt(assemble(inner((q_pp - a_flux), (q_pp - a_flux))*dx))
        print("Error between post processed flux and true solution: %0.13f"
              % err)

    # Collected all fluxes.
    print("All post-processed fluxes are collected. "
          "Now we measure their difference.\n")

    q_l = pp_fluxes.pop(0)
    for i, q_s in enumerate(pp_fluxes):
        difference = q_l - q_s
        comp_error = sqrt(assemble(inner(difference, difference)*dx))
        print("Norm of the difference between the first "
              "post-processed flux and the flux q^*_%d is: "
              "%0.13f" % (i, comp_error))
        div_comp_err = sqrt(assemble(inner(div(q_l - q_s),
                                           div(q_l - q_s))*dx))
        print("The divergence of the difference between the first "
              "post-processed flux and the flux q^*_%d is: "
              "%0.13f" % (i, div_comp_err))
    return pp_fluxes.insert(0, q_l)


# Run comparisons over varying degrees
for d in range(1, 4):
    fluxes = run_LDG_H_poisson_flux_PP(d)
