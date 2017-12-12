from firedrake import *
import numpy as np
import pandas as pd


def run_mixed_hybrid_poisson(r, degree, mixed_method, write=False):
    """
    Solves the Dirichlet problem for the Poisson equation:

    -div(grad(u)) = f in [0, 1]^2, u = 0 on the domain boundary.

    The source function is chosen to be the smooth function:

    f(x, y) = (2*pi^2)*sin(x*pi)*sin(y*pi)

    which produces the smooth analytic function:

    u(x, y) = sin(x*pi)*sin(y*pi).

    This problem was crafted so that we can test the theoretical
    convergence rates for the hybrid-mixed methods. This problem
    can be solved using the hybrid-RT/BDM method on simplices or
    the hybrid-RT method on quadrilaterals (hybrid-RTCF).

    The Slate DLS in Firedrake is used to perform the static condensation
    of the full hybrid-mixed formulation of the Poisson problem to a single
    system for the trace of u on the mesh skeleton:

    S * Lambda = E.

    The resulting linear system is solved via a direct method (LU) to
    ensure an accurate approximation to the trace variable. Once
    the trace is solved, the Slate DSL is used again to solve the
    elemental systems for the scalar solution u and its flux.

    Post-processing of the scalar variable is performed using Slate to
    form and solve the elemental-systems for a new approximation u*
    which superconverges at a rate of k+2.

    The expected (theoretical) rates for the hybrid-mixed methods are
    summarized below:

    -----------------------------------------
                  u     -grad(u)     u*
    -----------------------------------------
    H-RT-k       k+1       k+1      k+2
    H-BDM-k       k        k+1      k+1 (k=1) k+2 (k>1)
    H-RTCF-k     k+1       k+1      k+2
    -----------------------------------------

    This demo was written by: Thomas H. Gibson (t.gibson15@imperial.ac.uk)
    """

    if mixed_method is None or mixed_method not in ("RT", "BDM", "RTCF"):
        raise ValueError("Must specify a method of 'RT' 'RTCF' or 'BDM'")

    # Set up problem domain
    res = r

    # Set up function spaces
    if mixed_method == "RT":
        mesh = UnitSquareMesh(2**res, 2**res)
        broken_element = BrokenElement(FiniteElement("RT",
                                                     triangle,
                                                     degree + 1))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DG", degree)
        T = FunctionSpace(mesh, "HDiv Trace", degree)

    elif mixed_method == "RTCF":
        mesh = UnitSquareMesh(2**res, 2**res, quadrilateral=True)
        broken_element = BrokenElement(FiniteElement("RTCF",
                                                     quadrilateral,
                                                     degree + 1))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DQ", degree)
        T = FunctionSpace(mesh, "HDiv Trace", degree)
    else:
        assert mixed_method == "BDM"
        assert degree > 0, "Degree 0 is not valid for BDM method"
        mesh = UnitSquareMesh(2**res, 2**res)
        broken_element = BrokenElement(FiniteElement("BDM",
                                                     triangle,
                                                     degree))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DG", degree - 1)
        T = FunctionSpace(mesh, "HDiv Trace", degree)

    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

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

    print("Solving hybrid-mixed system using static condensation.\n")
    w = Function(W, name="solutions")
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              # Use the static condensation PC for hybridized problems
              # and use a direct solve on the reduced system for u_hat
              'pc_python_type': 'scpc.HybridSCPC',
              'hybrid_sc': {'ksp_type': 'preonly',
                            'pc_type': 'lu'}}
    solve(a == L, w, solver_parameters=params)

    print("Solver finished.\n")

    # Computed flux, scalar, and trace
    q_h, u_h, lambdar_h = w.split()

    # Analytical solutions for u and q
    u_a = Function(V_a, name="Analytic Scalar")
    u_a.interpolate(sin(x[0]*pi)*sin(x[1]*pi))

    q_a = Function(U_a, name="Analytic Flux")
    q_a.project(-grad(sin(x[0]*pi)*sin(x[1]*pi)))

    # Now we compute the various metrics. First we
    # simply compute the L2 error between the analytic
    # solutions and the computed ones.
    scalar_error = errornorm(u_a, u_h, norm_type="L2")
    flux_error = errornorm(q_a, q_h, norm_type="L2")

    # We keep track of all metrics using a Python dictionary
    error_dictionary = {"scalar_error": scalar_error,
                        "flux_error": flux_error}

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
    DG0 = FunctionSpace(mesh, "DG", 0)
    Wpp = DGk1 * DG0

    up, psi = TrialFunctions(Wpp)
    wp, phi = TestFunctions(Wpp)

    # Create mixed system:
    K = Tensor((inner(grad(up), grad(wp)) +
                inner(psi, wp) +
                inner(up, phi))*dx)
    F = Tensor((-inner(q_h, grad(wp)) +
                inner(u_h, phi))*dx)

    E = K.inv * F

    print("Local post-processing of the scalar variable.\n")
    u_pp = Function(DGk1, name="Post-processed scalar")
    assemble(E.block((0,)), tensor=u_pp)

    # Now we compute the error in the post-processed solution
    # and update our error dictionary
    scalar_pp_error = errornorm(u_a, u_pp, norm_type="L2")
    error_dictionary.update({"scalar_pp_error": scalar_pp_error})

    print("Post-processing finished.\n")

    # To verify the hybrid-flux is HDiv conforming, we also
    # evaluate its jump over mesh interiors. This should be
    # approximately zero if everything worked correctly.
    flux_jump = assemble(jump(q_h, n=n)*dS)
    error_dictionary.update({"flux_jump": np.abs(flux_jump)})

    print("Finished test case for h=1/2^%d.\n" % r)

    # If write specified, then write output
    if write:
        File("Hybrid-%s_deg%d.pvd" %
             (mixed_method, degree)).write(q_a, u_a,
                                           u_h, u_pp)

    # Return all error metrics
    return error_dictionary, mesh


def compute_conv_rates(u):
    """Computes the convergence rate for this particular test case

    :arg u: a list of errors.

    Returns a list of convergence rates. Note the first element of
    the list will be empty, as there is no previous computation to
    compare with. '---' will be inserted into the first component.
    """

    u_array = np.array(u)
    rates = list(np.log2(u_array[:-1] / u_array[1:]))
    rates.insert(0, '---')
    return rates


def run_single_test(r, degree, method):
    # Run a quick test given a degree, mixed method, and resolution

    print("Running Hybrid-%s method of degree %d"
          " and mesh parameter h=1/2^%d." %
          (method, degree, r))

    error_dict, _ = run_mixed_hybrid_poisson(r=r,
                                             degree=degree,
                                             mixed_method=method,
                                             write=True)

    print("Error in scalar: %0.8f" %
          error_dict["scalar_error"])
    print("Error in post-processed scalar: %0.8f" %
          error_dict["scalar_pp_error"])
    print("Error in flux: %0.8f" %
          error_dict["flux_error"])
    print("Interior jump of the hybrid flux: %0.8f" %
          np.abs(error_dict["flux_jump"]))


def run_mixed_hybrid_convergence(degree, method):

    print("Running convergence test for the hybrid-%s method "
          "of degree %d"
          % (method, degree))

    # Create arrays to write to CSV file
    r_array = []
    scalar_errors = []
    scalar_pp_errors = []
    flux_errors = []
    flux_jumps = []
    num_cells = []
    # Run over mesh parameters and collect error metrics
    for r in range(1, 6):
        r_array.append(r)
        error_dict, mesh = run_mixed_hybrid_poisson(r=r,
                                                    degree=degree,
                                                    mixed_method=method,
                                                    write=False)

        # Extract errors and metrics
        scalar_errors.append(error_dict["scalar_error"])
        scalar_pp_errors.append(error_dict["scalar_pp_error"])
        flux_errors.append(error_dict["flux_error"])
        flux_jumps.append(error_dict["flux_jump"])
        num_cells.append(mesh.num_cells())

    # Now that all error metrics are collected, we can compute the rates:
    scalar_rates = compute_conv_rates(scalar_errors)
    scalar_pp_rates = compute_conv_rates(scalar_pp_errors)
    flux_rates = compute_conv_rates(flux_errors)

    print("Error in scalar: %0.13f" % scalar_errors[-1])
    print("Error in post-processed scalar: %0.13f" % scalar_pp_errors[-1])
    print("Error in flux: %0.13f" % flux_errors[-1])
    print("Interior jump of computed flux: %0.13f" % flux_jumps[-1])

    degrees = [degree] * len(r_array)
    data = {"Mesh": r_array,
            "Degree": degrees,
            "NumCells": num_cells,
            "ScalarErrors": scalar_errors,
            "ScalarConvRates": scalar_rates,
            "FluxErrors": flux_errors,
            "FluxConvRates": flux_rates,
            "PostProcessedScalarErrors": scalar_pp_errors,
            "PostProcessedScalarRates": scalar_pp_rates}

    df = pd.DataFrame(data)
    result = "H-%s-degree-%d.csv" % (method, degree)
    df.to_csv(result, index=False, mode="w")
