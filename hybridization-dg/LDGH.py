from firedrake import *
from six import string_types
from decimal import Decimal
import numpy as np
import csv


def run_LDG_H_poisson(r, degree, tau_order, write=False):
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
    tau = O(1) (k>0)     k+1   k+1    k+2   k+2   k+1       k+1
    tau = O(h) (k>0)      k    k+1    k+2   k+2   k+1       k+1
    tau = O(1/h) (k>0)   k+1    k     k+1   k+1    k        k+1
    -----------------------------------------------------------------

    Note that the post-processing used for the flux q only holds for
    simplices (triangles and tetrahedra). If someone knows of a local
    post-processing method valid for quadrilaterals, please contact me!
    For these numerical results, we chose the following values of tau:

    tau = O(1) -> tau = 1,
    tau = O(h) -> tau = h,
    tau = O(1/h) -> tau = 1/h,

    where h here denotes the facet area.

    This demo was written by: Thomas H. Gibson (t.gibson15@imperial.ac.uk)
    """

    if tau_order is None or tau_order not in ("1", "h", "1/h"):
        raise ValueError(
            "Must specify tau to be of order '1', 'h' or '1/h'"
        )

    assert degree > 0, "Provide a degree >= 1"

    # Set up problem domain
    mesh = UnitSquareMesh(2**r, 2**r, quadrilateral=False)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Set up function spaces
    U = VectorFunctionSpace(mesh, "DG", degree)
    V = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    # Mixed space and test/trial functions
    W = U * V * T
    q, u, uhat = TrialFunctions(W)
    v, w, mu = TestFunctions(W)

    Vh = FunctionSpace(mesh, "DG", degree + 3)
    f = Function(Vh).interpolate((2*pi*pi)*sin(x[0]*pi)*sin(x[1]*pi))

    # Determine stability parameter tau
    if tau_order == "1":
        tau = Constant(1)

    elif tau_order == "h":
        tau = FacetArea(mesh)

    else:
        assert tau_order == "1/h"
        tau = 1/FacetArea(mesh)

    # Numerical flux
    qhat = q + tau*(u - uhat)*n

    def ejump(a):
        """UFL hack to get the right form."""
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
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_package': 'mumps'}}
    solve(a == L, w, solver_parameters=params)

    print("Solver finished.\n")
    # Computed flux, scalar, and trace
    q_h, u_h, uhat_h = w.split()

    # Analytical solutions for u and q
    V_a = FunctionSpace(mesh, "DG", degree + 3)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 3)
    Vdivrange = FunctionSpace(mesh, "DG", degree + 3)

    u_a = Function(V_a, name="Analytic Scalar")
    a_scalar = sin(x[0]*pi)*sin(x[1]*pi)
    u_a.interpolate(a_scalar)

    q_a = Function(U_a, name="Analytic Flux")
    a_flux = -grad(sin(x[0]*pi)*sin(x[1]*pi))
    q_a.project(a_flux)

    div_a = Function(Vdivrange, name="Analytic divergence")
    div_a.interpolate(div(q_a))

    # Now we compute the various metrics. First we
    # simply compute the L2 error between the analytic
    # solutions and the computed ones.
    scalar_error = errornorm(a_scalar, u_h, norm_type="L2")
    flux_error = errornorm(a_flux, q_h, norm_type="L2")

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
    scalar_pp_error = errornorm(a_scalar, u_pp, norm_type="L2")
    error_dictionary.update({"scalar_pp_error": scalar_pp_error})

    # Post processing of the flux:
    # This is a modification of the local Raviart-Thomas projector.
    # We solve the local problem: find 'q_pp' in RT(k+1)(K) such that
    #
    # (q_pp, v)*dx = (q_h, v)*dx,
    # (q_pp.n, gamma)*dS = (qhat.n, gamma)*dS
    #
    # for all v, gamma in DG(k-1) * DG(k)|_{trace}. The post-processed
    # solution q_pp converges at the same rate as q_h, but is HDiv
    # conforming. For all LDG-H methods,
    # div(q_pp) converges at the rate k+1. This is a way to obtain a
    # flux with better conservation properties. For tau of order 1/h,
    # div(q_pp) converges faster than q_h.
    qhat_h = q_h + tau*(u_h - uhat_h)*n
    local_RT = FiniteElement("RT", triangle, degree + 1)
    RTd = FunctionSpace(mesh, BrokenElement(local_RT))
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

    # Compute error in the divergence of the new flux
    flux_pp_div_error = sqrt(assemble((div(a_flux) - div(q_pp)) *
                                      (div(a_flux) - div(q_pp)) * dx))

    # And check the error in our new flux
    flux_pp_error = errornorm(a_flux, q_pp, norm_type="L2")

    # To verify our new flux is HDiv conforming, we also
    # evaluate its jump over mesh interiors. This should be
    # approximately zero if everything worked correctly.
    flux_pp_jump = assemble(jump(q_pp, n=n)*dS)

    error_dictionary.update({"flux_pp_error": flux_pp_error})
    error_dictionary.update({"flux_pp_div_error": flux_pp_div_error})
    error_dictionary.update({"flux_pp_jump": np.abs(flux_pp_jump)})

    print("Post-processing finished.\n")

    print("Finished test case for h=1/2^%d.\n" % r)

    # If write specified, then write output
    if write:
        if tau_order == "1/h":
            o = "hneg1"
        else:
            o = tau_order

        File("LDGH_tauO%s_deg%d.pvd" %
             (o, degree)).write(q_a, u_a, q_h, u_h, u_pp, div_a)

    # Return all error metrics
    return error_dictionary


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


def run_single_test(r, degree, tau_order):
    # Run a quick test given a degree, tau order, and resolution

    resolution_param = r
    print("Running LDG-H method (triangles) of degree %d with tau=O('%s') "
          "and mesh parameter h=1/2^%d." %
          (degree, tau_order, resolution_param))

    error_dict = run_LDG_H_poisson(r=resolution_param,
                                   degree=degree,
                                   tau_order=tau_order,
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


def run_LDG_H_convergence(degree, tau_order):

    print("Running convergence test for LDG-H method (triangles) "
          "of degree %d with tau order '%s'"
          % (degree, tau_order))

    # Create arrays to write to CSV file
    r_array = []
    scalar_errors = []
    scalar_pp_errors = []
    flux_errors = []
    flux_pp_errors = []
    flux_pp_div_errors = []
    flux_pp_jumps = []

    # Run over mesh parameters and collect error metrics
    for r in range(1, 7):
        r_array.append(r)
        error_dict = run_LDG_H_poisson(r=r,
                                       degree=degree,
                                       tau_order=tau_order,
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

    print("Error in scalar: %0.13f" %
          scalar_errors[-1])
    print("Error in post-processed scalar: %0.13f" %
          scalar_pp_errors[-1])
    print("Error in flux: %0.13f" %
          flux_errors[-1])
    print("Error in post-processed flux: %0.13f" %
          flux_pp_errors[-1])
    print("Error in post-processed flux divergence: %0.13f" %
          flux_pp_div_errors[-1])
    print("Interior jump of post-processed flux: %0.13f" %
          np.abs(flux_pp_jumps[-1]))

    # Write data to CSV file for table generation
    fieldnames = ["Mesh",
                  "ScalarErrors", "ScalarConvRates",
                  "FluxErrors", "FluxConvRates",
                  "PostProcessedScalarErrors", "PostProcessedScalarRates",
                  "PostProcessedFluxErrors", "PostProcessedFluxRates"]

    data = [r_array,
            scalar_errors, scalar_rates,
            flux_errors, flux_rates,
            scalar_pp_errors, scalar_pp_rates,
            flux_pp_errors, flux_pp_rates]

    if tau_order == "1/h":
            o = "hneg1"
    else:
        o = tau_order

    csv_file = open("LDG-H-d%d-tau_order-%s.csv" % (degree, o), "w")

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(fieldnames)
    for d in zip(*data):

        csv_writer.writerow([e if i == 0
                             else float2f(e) if i % 2 == 0
                             else format_si(e)
                             for i, e in enumerate(d)])
    csv_file.close()


def format_si(x):
    if not isinstance(x, string_types):
        o = '{:.2e}'.format(Decimal(x))
    else:
        o = x
    return o


def float2f(x):
    if not isinstance(x, string_types):
        o = '{:.2f}'.format(x)
    else:
        o = x
    return o
