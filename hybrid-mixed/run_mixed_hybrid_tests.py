from firedrake import *
import numpy as np
import sys
import csv


def run_mixed_hybrid_poisson(r, degree, mixed_method="RT", write=False):
    """
    """

    if mixed_method not in ("RT", "BDM"):
        raise ValueError("Must specify a method of 'RT' or 'BDM'")

    # Set up problem domain
    res = r
    mesh = UnitSquareMesh(2**res, 2**res)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Set up function spaces
    if mixed_method == "RT":
        broken_element = BrokenElement(FiniteElement("RT",
                                                     triangle,
                                                     degree + 1))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DG", degree)
        T = FunctionSpace(mesh, "HDiv Trace", degree)
    else:
        assert mixed_method == "BDM"
        assert degree > 0, "Degree 0 is not valid for BDM method"
        broken_element = BrokenElement(FiniteElement("BDM",
                                                     triangle,
                                                     degree))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DG", degree - 1)
        T = FunctionSpace(mesh, "HDiv Trace", degree)

    W = U * V * T

    # These spaces are for the smooth analytic expressions
    V_a = FunctionSpace(mesh, "DG", degree + 2)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 2)

    # Mixed space and test/trial functions
    W = U * V * T
    q, u, lambdar = TrialFunctions(W)
    v, w, gammar = TestFunctions(W)

    # Need smooth right-hand side for superconvergence magic
    Vh = FunctionSpace(mesh, "CG", degree + 2)
    f = Function(Vh).interpolate((2*pi*pi)*sin(x[0]*pi)*sin(x[1]*pi))

    # This is the formulation described by the Feel++ folks where
    # the multipliers weakly enforce the Dirichlet condition on
    # the scalar unknown.
    adx = (dot(q, v) - div(v)*u + div(q)*w)*dx
    adS = (jump(q, n=n)*gammar('+') + jump(v, n=n)*lambdar('+'))*dS
    ads = (dot(v, n)*lambdar + lambdar*gammar)*ds
    a = adx + adS + ads

    L = w*f*dx

    print("Solving hybrid-mixed system using static condensation.")
    print("Slate is used to perform the local assembly of the Schur-complement"
          " and solving the local system to recover the relevant unknowns.\n")
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
    q_h, u_h, lambda_h = w.split()

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

    if degree % 2 == 0:
        print("Performing element-wise post-processing as described "
              "by Arnold & Brezzi (1985). Slate is used to solve "
              "the elemental systems.\n")
        print("NOTE: This approach only works for even-degrees "
              "(see the 1985 paper).\n")

        DG_pp = FunctionSpace(mesh, "DG", d + 1)
        u_pp_slate = Function(DG_pp,
                              name="Post-processed scalar (Arnold/Brezzi)")
        utilde = TrialFunction(DG_pp)
        if d == 0:
            gammar = TestFunction(T)
            K = inner(utilde, gammar)*(dS + ds)
            F = inner(lambdar_h, gammar)*(dS + ds)
        else:
            DG_n2 = FunctionSpace(mesh, "DG", d - 2)
            Wk = DG_n2 * T
            v, gammar = TestFunctions(Wk)
            K = inner(utilde, v)*dx + inner(utilde, gammar)*(dS + ds)
            F = inner(u_h, v)*dx + inner(lambdar_h, gammar)*(dS + ds)

        A = Tensor(K)
        B = Tensor(F)
        assemble(A.inv * B, tensor=u_pp_slate)
        error_dictionary.update(
            {"arnold_brezzi_pp": errornorm(u_pp_slate,
                                           u_a,
                                           norm_type="L2")}
        )
    else:
        # Odd degrees don't apply here. Insert nans to be ignored.
        error_dictionary.update({"arnold_brezzi_pp": np.nan})

    print("Using post-processing described by Cockburn (2010).\n")
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
    # transform the local problem above into a mixed system:
    #
    # find (u, psi) in DG(k+1) * DG(0) such that:
    #
    # (grad(u), grad(w))*dx + (psi, grad(w))*dx = -(q_h, grad(w))*dx
    # (u, phi)*dx = (u_h, phi)*dx,
    #
    # for all w, phi in DG(k+1) * DG(0).
    DGk1 = FunctionSpace(mesh, "DG", degree + 1)
    # Reuse the DG0 space from above
    Wpp = DGk1 * DG0

    up, psi = TrialFunctions(Wpp)
    wp, phi = TestFunctions(Wpp)

    # Create mixed system:
    K = (inner(grad(up), grad(wp)) +
         inner(psi, wp) +
         inner(up, phi))*dx
    F = (-inner(q_h, grad(wp)) +
         inner(u_h, phi))*dx

    wpp = Function(Wpp, name="Post-processed scalar (mixed method)")
    solve(K == F, wpp, solver_parameters={"ksp_type": "gmres",
                                          "ksp_rtol": 1e-14})
    u_pp, _ = wpp.split()

    # Now we compute the error in the post-processed solution
    # and update our error dictionary
    scalar_pp_error = errornorm(u_pp, u_a, norm_type="L2")
    error_dictionary.update({"scalar_pp_error": scalar_pp_error})

    print("Post-processing finished.\n")

    # To verify the hybrid-flux is HDiv conforming, we also
    # evaluate its jump over mesh interiors. This should be
    # approximately zero if everything worked correctly.
    flux_jump = assemble(jump(q_h, n=n)*dS)
    error_dictionary.update({"flux_jump": flux_jump})

    print("Finished test case for h=1/2^%d." % r)

    # If write specified, then write output
    if write:
        if degree % 2 == 0:
            File("Hybrid-%s_deg%d.pvd" %
                 (mixed_method, degree)).write(q_a, u_a,
                                               q_h, u_h,
                                               u_pp_slate,
                                               u_pp)
        else:
            File("Hybrid-%s_deg%d.pvd" %
                 (mixed_method, degree)).write(q_a, u_a,
                                               q_h, u_h,
                                               u_pp)

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
    rates.insert(0, "---")
    return rates


if "--test-method" in sys.argv:
    # Run a quick test given a degree, mixed method, and resolution
    # (provide those arguments in that order)
    degree = int(sys.argv[1])
    mixed_method = sys.argv[2]
    resolution_param = int(sys.argv[3])
    print("Running Hybrid-%s method of degree %d"
          "and mesh parameter h=1/2^%d." %
          (degree, mixed_method, resolution_param))

    error_dict = run_mixed_hybrid_poisson(r=resolution_param,
                                          degree=degree,
                                          mixed_method=mixed_method,
                                          write=True)

    print("Error in scalar: %0.8f" %
          error_dict["scalar_error"])
    print("Error in post-processed scalar (via mixed method): %0.8f" %
          error_dict["scalar_pp_error"])
    if degree % 2 == 0:
        print("Error in post-processed scalar (Arnold/Brezzi): %0.13f" %
              error_dict["arnold_brezzi_pp"])
    print("Error in flux: %0.8f" %
          error_dict["flux_error"])
    print("Interior jump of the hybrid flux: %0.8f" %
          np.abs(error_dict["flux_pp_jump"]))

elif "--run-convergence-test" in sys.argv:
    # Run a convergence test for a particular set
    # of parameters.
    degree = int(sys.argv[1])
    mixed_method = sys.argv[2]
    print("Running convergence test for the hybrid-%s method "
          "of degree %d"
          % (mixed_method, degree))

    # Create arrays to write to CSV file
    r_array = []
    scalar_errors = []
    scalar_pp_errors = []
    arnold_brezzi_pp_errors = []
    flux_errors = []
    flux_jumps = []

    # Run over mesh parameters and collect error metrics
    for r in range(1, 6):
        r_array.append(r)
        error_dict = run_LDG_H_poisson(r=r,
                                       degree=degree,
                                       tau_order=tau_order,
                                       write=False)

        # Extract errors and metrics
        scalar_errors.append(error_dict["scalar_error"])
        scalar_pp_errors.append(error_dict["scalar_pp_error"])
        avg_scalar_errors.append(error_dict["scalar_pp_error"])
        flux_errors.append(error_dict["flux_error"])
        flux_pp_errors.append(error_dict["flux_pp_error"])
        flux_pp_div_errors.append(error_dict["flux_pp_div_error"])
        flux_pp_jumps.append(error_dict["flux_pp_jump"])

    # Now that all error metrics are collected, we can compute the rates:
    scalar_rates = compute_conv_rates(scalar_errors)
    scalar_pp_rates = compute_conv_rates(scalar_pp_errors)
    avg_scalar_rates = compute_conv_rates(avg_scalar_errors)
    flux_rates = compute_conv_rates(flux_errors)
    flux_pp_rates = compute_conv_rates(flux_pp_errors)
    flux_pp_div_rates = compute_conv_rates(flux_pp_div_errors)

    print("Convergence rate for u - u_h: %0.2f" % scalar_rates[-1])
    print("Convergence rate for u - u_pp: %0.2f" % scalar_pp_rates[-1])
    print("Convergence rate for ubar - u_hbar: %0.2f" % avg_scalar_rates[-1])
    print("Convergence rate for q - q_h: %0.2f" % flux_rates[-1])

    # Only applies to methods of degree > 0
    if degree > 0:
        print("Convergence rate for q - q_pp: %0.2f" %
              flux_pp_rates[-1])
        print("Convergence rate for div(q - q_pp): %0.2f" %
              flux_pp_div_rates[-1])

    print("Error in scalar: %0.13f" %
          scalar_errors[-1])
    print("Error in post-processed scalar: %0.13f" %
          scalar_pp_errors[-1])
    print("Error in integral average of scalar: %0.13f" %
          avg_scalar_errors[-1])
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
    fieldnames = ["r",
                  "scalar_errors", "flux_errors",
                  "scalar_pp_errors", "flux_pp_errors",
                  "avg_scalar_errors",
                  "scalar_rates", "flux_rates",
                  "avg_scalar_rates",
                  "scalar_pp_rates", "flux_pp_rates",
                  "flux_pp_div_errors", "flux_pp_div_rates"]

    data = [r_array,
            scalar_errors, flux_errors,
            scalar_pp_errors, flux_pp_errors,
            avg_scalar_errors,
            scalar_rates, flux_rates,
            avg_scalar_rates,
            scalar_pp_rates, flux_pp_rates,
            flux_pp_div_errors, flux_pp_div_rates]

    if tau_order == "1/h":
            o = "h-1"
    else:
        o = tau_order

    csv_file = open("LDG-H-d%d-tau_order-%s.csv" % (degree, o), "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(fieldnames)
    for d in zip(*data):
        csv_writer.writerow(d)
    csv_file.close()

else:
    print("Please specify --test-method or --run-convergence-test")
