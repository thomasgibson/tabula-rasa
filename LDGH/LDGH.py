"""
This module runs a convergence history for a hybridized-DG
discretization of a model elliptic problem (detailed in the main
function). The method used is the LDG-H method.
"""

from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
import pandas as pd


def run_LDG_H_helmholtz(r, degree, tau_order, write=False):
    """
    Solves the Dirichlet problem for the elliptic equation:

    -div(grad(u)) + u = f in [0, 1]^2, u = g on the domain boundary.

    The source function f and g are chosen such that the analytic
    solution is:

    u(x, y) = exp(sin(3*x*pi)*sin(3*y*pi)).

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

    if tau_order is None or tau_order not in ("1", "1/h", "h"):
        raise ValueError(
            "Must specify tau to be of order '1', '1/h', or 'h'"
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
    s = Function(W, name="solutions").assign(0.0)
    q, u, uhat = split(s)
    v, w, mu = TestFunctions(W)

    # Analytical solutions for u and q
    V_a = FunctionSpace(mesh, "CG", degree + 3)
    U_a = VectorFunctionSpace(mesh, "CG", degree + 3)

    u_a = Function(V_a, name="Analytic Scalar")
    a_scalar = exp(sin(3*pi*x[0])*sin(3*pi*x[1]))
    u_a.interpolate(a_scalar)

    q_a = Function(U_a, name="Analytic Flux")
    a_flux = -grad(a_scalar)
    q_a.project(a_flux)

    Vh = FunctionSpace(mesh, "DG", degree + 3)
    f = Function(Vh).interpolate(-div(grad(a_scalar))
                                 + a_scalar)

    # Determine stability parameter tau
    if tau_order == "1":
        tau = Constant(1)

    elif tau_order == "1/h":
        tau = 1/FacetArea(mesh)

    elif tau_order == "h":
        tau = FacetArea(mesh)

    else:
        raise ValueError("Invalid choice of tau")

    # Numerical flux
    qhat = q + tau*(u - uhat)*n

    # Formulate the LDG-H method in UFL
    a = ((dot(v, q) - div(v)*u)*dx
         + uhat('+')*jump(v, n=n)*dS
         + uhat*dot(v, n)*ds
         - dot(grad(w), q)*dx
         + jump(qhat, n=n)*w('+')*dS
         + dot(qhat, n)*w*ds
         + w*u*dx
         # Transmission condition
         + mu('+')*jump(qhat, n=n)*dS)

    L = w*f*dx
    F = a - L
    PETSc.Sys.Print("Solving using static condensation.\n")
    params = {'snes_type': 'ksponly',
              'mat_type': 'matfree',
              'pmat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              # Use the static condensation PC for hybridized problems
              # and use a direct solve on the reduced system for u_hat
              'pc_python_type': 'scpc.HybridSCPC',
              'hybrid_sc': {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_package': 'mumps'}}

    bcs = DirichletBC(W.sub(2), Constant(1.0), "on_boundary")
    problem = NonlinearVariationalProblem(F, s, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    PETSc.Sys.Print("Solver finished.\n")

    # Computed flux, scalar, and trace
    q_h, u_h, uhat_h = s.split()

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

    E = K.inv * F

    PETSc.Sys.Print("Local post-processing of the scalar variable.\n")
    u_pp = Function(DGk1, name="Post-processed scalar")
    assemble(E.block((0,)), tensor=u_pp)

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

    PETSc.Sys.Print("Local post-processing of the flux.\n")
    q_pp = assemble(A.inv * B)

    # And check the error in our new flux
    flux_pp_error = errornorm(a_flux, q_pp, norm_type="L2")

    # To verify our new flux is HDiv conforming, we also
    # evaluate its jump over mesh interiors. This should be
    # approximately zero if everything worked correctly.
    flux_pp_jump = assemble(jump(q_pp, n=n)*dS)

    error_dictionary.update({"flux_pp_error": flux_pp_error})
    error_dictionary.update({"flux_pp_jump": np.abs(flux_pp_jump)})

    PETSc.Sys.Print("Post-processing finished.\n")

    PETSc.Sys.Print("Finished test case for h=1/2^%d.\n" % r)

    # If write specified, then write output
    if write:
        if tau_order == "1/h":
            o = "hneg1"
        else:
            o = tau_order

        File("LDGH_tauO%s_deg%d.pvd" %
             (o, degree)).write(q_a, u_a, q_h, u_h, u_pp)

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


def run_single_test(r, degree, tau_order, write=False):
    # Run a quick test given a degree, tau order, and resolution

    resolution_param = r
    PETSc.Sys.Print("Running LDG-H method (triangles) of degree %d with tau=O('%s') "
                    "and mesh parameter h=1/2^%d." %
                    (degree, tau_order, resolution_param))

    error_dict, _ = run_LDG_H_helmholtz(r=resolution_param,
                                        degree=degree,
                                        tau_order=tau_order,
                                        write=write)

    PETSc.Sys.Print("Error in scalar: %0.8f" %
                    error_dict["scalar_error"])
    PETSc.Sys.Print("Error in post-processed scalar: %0.8f" %
                    error_dict["scalar_pp_error"])
    PETSc.Sys.Print("Error in flux: %0.8f" %
                    error_dict["flux_error"])
    PETSc.Sys.Print("Error in post-processed flux: %0.8f" %
                    error_dict["flux_pp_error"])
    PETSc.Sys.Print("Interior jump of post-processed flux: %0.8f" %
                    np.abs(error_dict["flux_pp_jump"]))


def run_LDG_H_convergence(degree, tau_order, start, end):

    PETSc.Sys.Print("Running convergence test for LDG-H method (triangles) "
                    "of degree %d with tau order '%s'"
                    % (degree, tau_order))

    # Create arrays to write to CSV file
    r_array = []
    scalar_errors = []
    scalar_pp_errors = []
    flux_errors = []
    flux_pp_errors = []
    flux_pp_jumps = []
    num_cells = []
    # Run over mesh parameters and collect error metrics
    for r in range(start, end + 1):
        r_array.append(r)
        error_dict, mesh = run_LDG_H_helmholtz(r=r,
                                               degree=degree,
                                               tau_order=tau_order,
                                               write=False)

        # Extract errors and metrics
        scalar_errors.append(error_dict["scalar_error"])
        scalar_pp_errors.append(error_dict["scalar_pp_error"])
        flux_errors.append(error_dict["flux_error"])
        flux_pp_errors.append(error_dict["flux_pp_error"])
        flux_pp_jumps.append(error_dict["flux_pp_jump"])
        num_cells.append(mesh.num_cells())

    # Now that all error metrics are collected, we can compute the rates:
    scalar_rates = compute_conv_rates(scalar_errors)
    scalar_pp_rates = compute_conv_rates(scalar_pp_errors)
    flux_rates = compute_conv_rates(flux_errors)
    flux_pp_rates = compute_conv_rates(flux_pp_errors)

    PETSc.Sys.Print("Error in scalar: %0.13f" %
                    scalar_errors[-1])
    PETSc.Sys.Print("Error in post-processed scalar: %0.13f" %
                    scalar_pp_errors[-1])
    PETSc.Sys.Print("Error in flux: %0.13f" %
                    flux_errors[-1])
    PETSc.Sys.Print("Error in post-processed flux: %0.13f" %
                    flux_pp_errors[-1])
    PETSc.Sys.Print("Interior jump of post-processed flux: %0.13f" %
                    np.abs(flux_pp_jumps[-1]))

    if COMM_WORLD.rank == 0:
        degrees = [degree] * len(r_array)
        data = {"Mesh": r_array,
                "Degree": degrees,
                "NumCells": num_cells,
                "ScalarErrors": scalar_errors,
                "ScalarConvRates": scalar_rates,
                "PostProcessedScalarErrors": scalar_pp_errors,
                "PostProcessedScalarRates": scalar_pp_rates,
                "FluxErrors": flux_errors,
                "FluxConvRates": flux_rates,
                "PostProcessedFluxErrors": flux_pp_errors,
                "PostProcessedFluxRates": flux_pp_rates}

        if tau_order == "1/h":
            o = "hneg1"
        else:
            o = tau_order

        df = pd.DataFrame(data)
        result = "LDG-H-d%d-tau_order-%s.csv" % (degree, o)
        df.to_csv(result, index=False, mode="w")
