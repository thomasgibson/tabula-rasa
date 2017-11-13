from firedrake import *
import numpy as np
import sys
import csv


def run_LDG_H_poisson(r, degree, tau_order="1", quads=False, write=False):
    """
    Solves the Dirichlet problem for the Poisson equation:

    -div(grad(u)) = f in [0, 1]^2, u = 0 on the domain boundary.

    The source function is chosen to be the smooth function:

    f(x, y) = (2*pi^2)*sin(x*pi)*sin(y*pi)

    which produces the smooth analytic function:

    u(x, y) = sin(x*pi)*sin(y*pi).

    This problem was crafted so that we can test the theoretical
    convergence rates for the hybridized DG method: LDG-H. This
    is accomplished by introducing the numerical fluxes:

    u_hat = lambda,
    q_hat = q + tau*(u - u_hat).

    The Slate DLS in Firedrake is used to perform the static condensation
    of the full LDG-H formulation of the Poisson problem to a single
    system for the trace u_hat (lambda) on the mesh skeleton:

    S * Lambda = E.

    The resulting linear system is solved via a direct method (LU) to
    ensure an accurate approximation to the trace variable. Once
    the trace is solved, the Slate DSL is used again to solve the
    elemental systems for the scalar solution u and its flux q.

    Post-processing of the scalar variable, as well as its flux, is
    performed using Slate to form and solve the elemental-systems for
    new approximations u*, q*. Depending on the choice of tau, these
    new solutions have superconvergent properties.

    The post-processed scalar u* superconverges at a rate of k+2 when
    two conditions are satisfied:

    (1) q converges at a rate of k+1, and
    (2) the cell average of u, ubar, superconverges at a rate of k+2.

    The choice of tau heavily influences these two conditions. For all
    tau > 0, the post-processed flux q* has enhanced convervation
    properties! The new solution q* has the following three properties:

    (1) q* converges at the same rate as q. However,
    (2) q* is in H(Div), meaning that the interior jump of q* is zero!
        And lastly,
    (3) div(q - q*) converges at a rate of k+1.

    The expected (theoretical) rates for the LDG-H method are
    summarized below for various orders of tau:

    -----------------------------------------------------------------
                          u     q    ubar    u*    q*     div(p*)
    -----------------------------------------------------------------
    tau = O(1) (k=0)      1     1      1     1     1         1
    tau = O(1) (k>0)     k+1   k+1    k+2   k+2   k+1       k+1
    tau = O(h) (k>0)      k    k+1    k+2   k+2   k+1       k+1
    tau = O(1/h) (k>0)   k+1    k     k+1   k+1    k        k+1
    -----------------------------------------------------------------

    Note that the post-processing used for the flux q only holds for
    simplices (triangles and tetrahedra). If someone knows of a local
    post-processing method valid for quadrilaterals, please contact me!
    For these numerical results, we chose the following values of tau:

    tau = O(1) -> tau = 1.0,
    tau = O(h) -> tau = sqrt(2)*h,
    tau = O(1/h) -> tau = 1.0/h,

    where h here denotes the facet area.

    This demo, as well as the primary author of the Slate DSL, was
    written by: Thomas H. Gibson (t.gibson15@imperial.ac.uk)
    """

    if tau_order not in ("1", "h", "1/h"):
        raise ValueError("Must specify tau to be of order '1', 'h' or '1/h'")

    # Set up problem domain
    res = r
    mesh = UnitSquareMesh(2**res, 2**res, quadrilateral=quads)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Set up function spaces
    U = VectorFunctionSpace(mesh, "DG", degree)
    V = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    # These spaces are for the smooth analytic expressions
    V_a = FunctionSpace(mesh, "DG", degree + 2)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 2)

    # Mixed space and test/trial functions
    W = U * V * T
    q, u, uhat = TrialFunctions(W)
    v, w, mu = TestFunctions(W)

    # Need smooth right-hand side for superconvergence magic
    Vh = FunctionSpace(mesh, "CG", degree + 2)
    f = Function(Vh).interpolate((2*pi*pi)*sin(x[0]*pi)*sin(x[1]*pi))

    # Determine stability parameter tau
    if tau_order == "1":
        tau = Constant(1.0)

    elif tau_order == "h":
        tau = sqrt(2)*FacetArea(mesh)

    else:
        assert tau_order == "1/h"
        tau = Constant(1.0)/FacetArea(mesh)

    # Numerical flux
    qhat = q + tau*(u - uhat)*n

    def ejump(a):
        return 2*avg(a)

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

    print("Solving using static condensation.\n")
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

    # Analytical solutions for u and q
    u_a = Function(V_a, name="Analytic Scalar")
    u_a.interpolate(sin(x[0]*pi)*sin(x[1]*pi))

    q_a = Function(U_a, name="Analytic Flux")
    q_a.project(-grad(sin(x[0]*pi)*sin(x[1]*pi)))

    # Now we compute the various metrics. First we
    # simply compute the L2 error between the analytic
    # solutions and the computed ones.
    scalar_error = errornorm(u_h, u_a, norm_type="L2")
    flux_error = errornorm(q_h, q_a, norm_type="L2")

    # We keep track of all metrics using a Python dictionary
    error_dictionary = {"scalar_error": scalar_error,
                        "flux_error": flux_error}

    # Now we use Slate to perform element-wise post-processing

    # Scalar post-processing:
    # This gives an approximation in DG(k+1) via solving for
    # the solution of the local Neumman data problem:
    #
    # (grad(u), grad(w))*dx = -(q_h, grad(w))*dx
    # m(u) = m(u_h) for all elements K, where
    #
    # m(v) := measure(K)^-1 * int_K v dx.

    # NOTE: It is currently not possible to correctly formulate this
    # in UFL. However, we can introduce a Lagrange multiplier and
    # transform the local problem above into a local mixed system:
    #
    # find (u, psi) in DG(k+1) * DG(0) such that:
    #
    # (grad(u), grad(w))*dx + (psi, grad(w))*dx = -(q_h, grad(w))*dx
    # (u, phi)*dx = (u_h, phi)*dx,
    #
    # for all w, phi in DG(k+1) * DG(0).
    DGk1 = FunctionSpace(mesh, "DG", degree + 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    Wpp = DGk1 * DG0

    up, psi = TrialFunctions(Wpp)
    wp, phi = TestFunctions(Wpp)

    # Create mixed tensors:
    K = Tensor((inner(grad(up), grad(wp)) +
                inner(psi, wp) +
                inner(up, phi))*dx)
    F = Tensor((-inner(q_h, grad(wp)) +
                inner(u_h, phi))*dx)

    print("Local post-processing of the scalar variable.\n")
    wpp = Function(Wpp, name="Post-processed scalar")
    assemble(K.inv * F, tensor=wpp,
             slac_parameters={"split_vector": 0})
    u_pp, _ = wpp.split()

    # Now we compute the error in the post-processed solution
    # and update our error dictionary
    scalar_pp_error = errornorm(u_pp, u_a, norm_type="L2")
    error_dictionary.update({"scalar_pp_error": scalar_pp_error})

    # Post processing of the flux:
    # This is a modification of the local Raviart-Thomas projector.
    # We solve the local problem: find 'nu' in RT(k+1)(K) such that
    #
    # (nu, v)*dx = 0,
    # (nu.n, gamma)*dS = ((qhat - q_h).n, gamma)*dS
    #
    # for all v, gamma in DG(k-1) * DG(k)|_{trace}. The post-processed
    # solution is defined as q_pp = q_h + nu. q_pp converges at the
    # same rate as q_h, but is HDiv-conforming. For all LDG-H methods,
    # div(q_pp) converges at the rate k+1. This is a way to obtain a
    # flux with better conservation properties. For tau of order 1/h,
    # div(q_pp) converges faster than q_h.

    # NOTE: You cannot use this post-processing for lowest order (k=0)
    # methods or quads
    if degree > 0 and not quads:
        qhat_h = q_h + tau*(u_h - uhat_h)*n
        RTd = FunctionSpace(mesh, "DRT", degree + 1)
        DGkn1 = VectorFunctionSpace(mesh, "DG", degree - 1)

        # Use the trace space already defined
        Npp = DGkn1 * T
        n_p = TrialFunction(RTd)
        vp, mu = TestFunctions(Npp)

        # Assemble the local system and invert using Slate
        A = Tensor(inner(n_p, vp)*dx +
                   jump(n_p, n=n)*mu('+')*dS +
                   dot(n_p, n)*mu*ds)
        B = Tensor(jump(qhat_h - q_h, n=n)*mu('+')*dS
                   + dot(qhat_h - q_h, n)*mu*ds)

        print("Local post-processing of the flux.\n")
        nu = assemble(A.inv * B)

        # Post-processed flux
        q_pp = nu + q_h

        # Compute error in the divergence of the new flux
        flux_pp_div_error = sqrt(assemble(div(q_pp - q_a) *
                                          div(q_pp - q_a) * dx))

        # And check the error in our new flux
        flux_pp_error = errornorm(q_pp, q_a, norm_type="L2")

        # To verify our new flux is HDiv conforming, we also
        # evaluate its jump over mesh interiors. This should be
        # approximately zero if everything worked correctly.
        flux_pp_jump = assemble(jump(q_pp, n=n)*dS)

        error_dictionary.update({"flux_pp_error": flux_pp_error})
        error_dictionary.update({"flux_pp_div_error": flux_pp_div_error})
        error_dictionary.update({"flux_pp_jump": np.abs(flux_pp_jump)})
    else:
        # For lowest order, just insert junk
        # (these should be ignored for lowest order)
        error_dictionary.update({"flux_pp_error": 10000})
        error_dictionary.update({"flux_pp_div_error": 10000})
        error_dictionary.update({"flux_pp_jump": 10000})

    print("Post-processing finished.\n")

    print("Finished test case for h=1/2^%d.\n" % r)

    # If write specified, then write output
    if write:
        if tau_order == "1/h":
            o = "h-1"
        else:
            o = tau_order

        File("LDGH_tauO%s_deg%d.pvd" %
             (o, degree)).write(q_a, u_a, q_h, u_h, u_pp)

    # Return all error metrics
    return error_dictionary


def compute_conv_rates(u):
    """Computes the convergence rate for this particular test case

    :arg u: a list of errors.

    Returns a list of convergence rates. Note the first element of
    the list will be empty, as there is no previous computation to
    compare with. A '---' will be inserted into the first component.
    """

    u_array = np.array(u)
    rates = list(np.log2(u_array[:-1] / u_array[1:]))
    rates.insert(0, None)
    return rates


if "--test-method" in sys.argv:
    # Run a quick test given a degree, tau order, and resolution
    # (provide those arguments in that order)

    if "--quads" in sys.argv:
        quads = True
    else:
        quads = False

    degree = int(sys.argv[1])
    tau_order = sys.argv[2]
    resolution_param = int(sys.argv[3])

    if quads:
        print("Running LDG-H method on quads of degree %d with tau=O('%s') "
              "and mesh parameter h=1/2^%d." %
              (degree, tau_order, resolution_param))
    else:
        print("Running LDG-H method (triangles) of degree %d with tau=O('%s') "
              "and mesh parameter h=1/2^%d." %
              (degree, tau_order, resolution_param))

    error_dict = run_LDG_H_poisson(r=resolution_param,
                                   degree=degree,
                                   tau_order=tau_order,
                                   quads=quads,
                                   write=True)

    print("Error in scalar: %0.8f" %
          error_dict["scalar_error"])
    print("Error in post-processed scalar: %0.8f" %
          error_dict["scalar_pp_error"])
    print("Error in flux: %0.8f" %
          error_dict["flux_error"])
    print("Error in post-processed flux: %0.8f" %
          error_dict["flux_pp_error"])
    print("Error in post-processed flux divergence: %0.8f" %
          error_dict["flux_pp_div_error"])
    print("Interior jump of post-processed flux: %0.8f" %
          np.abs(error_dict["flux_pp_jump"]))

elif "--run-convergence-test" in sys.argv:
    # Run a convergence test for a particular set
    # of parameters.

    if "--quads" in sys.argv:
        quads = True
    else:
        quads = False

    degree = int(sys.argv[1])
    tau_order = sys.argv[2]

    if quads:
        print("Running convergence test for LDG-H method (quads) "
              "of degree %d with tau order '%s'"
              % (degree, tau_order))
    else:
        print("Running convergence test for LDG-H method (triangles) "
              "of degree %d with tau order '%s'"
              % (degree, tau_order))

    # Create arrays to write to CSV file
    r_array = []
    scalar_errors = []
    scalar_pp_errors = []
    avg_scalar_errors = []
    flux_errors = []
    flux_pp_errors = []
    flux_pp_div_errors = []
    flux_pp_jumps = []

    # Run over mesh parameters and collect error metrics
    for r in range(1, 6):
        r_array.append(r)
        error_dict = run_LDG_H_poisson(r=r,
                                       degree=degree,
                                       tau_order=tau_order,
                                       quads=quads,
                                       write=False)

        # Extract errors and metrics
        scalar_errors.append(error_dict["scalar_error"])
        scalar_pp_errors.append(error_dict["scalar_pp_error"])
        flux_errors.append(error_dict["flux_error"])
        flux_pp_errors.append(error_dict["flux_pp_error"])
        flux_pp_div_errors.append(error_dict["flux_pp_div_error"])
        flux_pp_jumps.append(error_dict["flux_pp_jump"])

    # Now that all error metrics are collected, we can compute the rates:
    scalar_rates = compute_conv_rates(scalar_errors)
    scalar_pp_rates = compute_conv_rates(scalar_pp_errors)
    flux_rates = compute_conv_rates(flux_errors)
    flux_pp_rates = compute_conv_rates(flux_pp_errors)
    flux_pp_div_rates = compute_conv_rates(flux_pp_div_errors)

    print("Convergence rate for u - u_h: %0.3f" % scalar_rates[-1])
    print("Convergence rate for u - u_pp: %0.3f" % scalar_pp_rates[-1])
    print("Convergence rate for q - q_h: %0.3f" % flux_rates[-1])

    # Only applies to methods of degree > 0
    if degree > 0:
        print("Convergence rate for q - q_pp: %0.3f" %
              flux_pp_rates[-1])
        print("Convergence rate for div(q - q_pp): %0.3f" %
              flux_pp_div_rates[-1])

    print("Error in scalar: %0.13f" %
          scalar_errors[-1])
    print("Error in post-processed scalar: %0.13f" %
          scalar_pp_errors[-1])
    print("Error in flux: %0.13f" %
          flux_errors[-1])

    # Only applies to methods of degree > 0
    if degree > 0:
        print("Error in post-processed flux: %0.13f" %
              flux_pp_errors[-1])
        print("Error in post-processed flux divergence: %0.13f" %
              flux_pp_div_errors[-1])
        print("Interior jump of post-processed flux: %0.13f" %
              np.abs(flux_pp_jumps[-1]))

    # Write data to CSV file
    fieldnames = ["mesh",
                  "scalarerrors", "fluxerrors",
                  "ppscalarerrors", "ppfluxerrors",
                  "scalarrates", "fluxrates",
                  "ppscalarrates", "ppfluxrates",
                  "ppfluxdiverrors", "ppfluxdivrates",
                  "ppfluxjumps"]

    data = [r_array,
            scalar_errors, flux_errors,
            scalar_pp_errors, flux_pp_errors,
            scalar_rates, flux_rates,
            scalar_pp_rates, flux_pp_rates,
            flux_pp_div_errors, flux_pp_div_rates,
            flux_pp_jumps]

    if tau_order == "1/h":
            o = "h-1"
    else:
        o = tau_order

    if quads:
        csv_file = open("LDG-H-d%d-tau_order-%s-quads.csv" % (degree, o), "w")
    else:
        csv_file = open("LDG-H-d%d-tau_order-%s.csv" % (degree, o), "w")

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(fieldnames)
    for d in zip(*data):
        csv_writer.writerow(d)
    csv_file.close()

else:
    print("Please specify --test-method or --run-convergence-test")
