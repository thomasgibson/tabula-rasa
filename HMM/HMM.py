"""
This module runs a convergence history for the mixed-hybrid methods
of a model elliptic problem (detailed in the main function).
"""

from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
import pandas as pd


def run_mixed_hybrid_helmholtz(r, degree, mixed_method, write=False):
    """
    Solves the Dirichlet problem for the elliptic equation:

    -div(grad(u)) + u = f in [0, 1]^2, u = g on the domain boundary.

    The source function f and g are chosen such that the analytic
    solution is:

    u(x, y) = exp(sin(3*x*pi)*sin(3*y*pi)).

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
    V_a = FunctionSpace(mesh, "CG", degree + 3)
    U_a = VectorFunctionSpace(mesh, "CG", degree + 3)

    # Mixed space and test/trial functions
    W = U * V * T
    s = Function(W, name="solutions").assign(0.0)
    q, u, lambdar = split(s)
    v, w, gammar = TestFunctions(W)

    u_a = Function(V_a, name="Analytic Scalar")
    a_scalar = exp(sin(3*pi*x[0])*sin(3*pi*x[1]))
    u_a.interpolate(a_scalar)

    q_a = Function(U_a, name="Analytic Flux")
    a_flux = -grad(a_scalar)
    q_a.project(a_flux)

    Vh = FunctionSpace(mesh, "DG", degree + 3)
    f = Function(Vh).interpolate(-div(grad(a_scalar))
                                 + a_scalar)

    adx = (dot(q, v) - div(v)*u + div(q)*w + w*u)*dx
    adS = (jump(q, n=n)*gammar('+') + jump(v, n=n)*lambdar('+'))*dS
    a = adx + adS

    L = w*f*dx - Constant(1.0)*dot(v, n)*ds
    F = a - L
    PETSc.Sys.Print("Solving hybrid-mixed system using static condensation.\n")
    bcs = DirichletBC(W.sub(2), 0.0, "on_boundary")
    params = {'snes_type': 'ksponly',
              'pmat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              # Use the static condensation PC for hybridized problems
              # and use a direct solve on the reduced system for lambdar
              'pc_python_type': 'scpc.HybridSCPC',
              'hybrid_sc': {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_package': 'mumps'}}
    problem = NonlinearVariationalProblem(F, s, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    PETSc.Sys.Print("Solver finished.\n")

    # Computed flux, scalar, and trace
    q_h, u_h, lambdar_h = s.split()

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

    PETSc.Sys.Print("Local post-processing of the scalar variable.\n")
    u_pp = Function(DGk1, name="Post-processed scalar")
    assemble(E.block((0,)), tensor=u_pp)

    # Now we compute the error in the post-processed solution
    # and update our error dictionary
    scalar_pp_error = errornorm(u_a, u_pp, norm_type="L2")
    error_dictionary.update({"scalar_pp_error": scalar_pp_error})

    PETSc.Sys.Print("Post-processing finished.\n")

    # To verify the hybrid-flux is HDiv conforming, we also
    # evaluate its jump over mesh interiors. This should be
    # approximately zero if everything worked correctly.
    flux_jump = assemble(jump(q_h, n=n)*dS)
    error_dictionary.update({"flux_jump": np.abs(flux_jump)})

    PETSc.Sys.Print("Finished test case for h=1/2^%d.\n" % r)

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

    PETSc.Sys.Print("Running Hybrid-%s method of degree %d"
                    " and mesh parameter h=1/2^%d." %
                    (method, degree, r))

    error_dict, _ = run_mixed_hybrid_helmholtz(r=r,
                                               degree=degree,
                                               mixed_method=method,
                                               write=True)

    PETSc.Sys.Print("Error in scalar: %0.8f" %
                    error_dict["scalar_error"])
    PETSc.Sys.Print("Error in post-processed scalar: %0.8f" %
                    error_dict["scalar_pp_error"])
    PETSc.Sys.Print("Error in flux: %0.8f" %
                    error_dict["flux_error"])
    PETSc.Sys.Print("Interior jump of the hybrid flux: %0.8f" %
                    np.abs(error_dict["flux_jump"]))


def run_mixed_hybrid_convergence(degree, method, start, end):

    PETSc.Sys.Print("Running convergence test for the hybrid-%s method "
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
    for r in range(start, end + 1):
        r_array.append(r)
        error_dict, mesh = run_mixed_hybrid_helmholtz(r=r,
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

    PETSc.Sys.Print("Error in scalar: %0.13f" % scalar_errors[-1])
    PETSc.Sys.Print("Error in post-processed scalar: %0.13f" % scalar_pp_errors[-1])
    PETSc.Sys.Print("Error in flux: %0.13f" % flux_errors[-1])
    PETSc.Sys.Print("Interior jump of computed flux: %0.13f" % flux_jumps[-1])

    if COMM_WORLD.rank == 0:
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
