from firedrake import *
import numpy as np


def run_hdg_poisson(r, d, write=False, post_process=False):
    """
    """
    degree = d
    res = r
    mesh = UnitSquareMesh(2**res, 2**res)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    U = VectorFunctionSpace(mesh, "DG", degree)
    V = FunctionSpace(mesh, "DG", degree)
    V_a = FunctionSpace(mesh, "DG", degree + 4)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 4)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    W = U * V * T
    q, u, uhat = TrialFunctions(W)
    v, w, mu = TestFunctions(W)

    # Need smooth right-hand side for superconvergence magic
    f = Function(V_a).interpolate((2*pi*pi)*sin(x[0]*pi)*sin(x[1]*pi))

    # For tau = O(1), q_h and u_h converge at order k+1, and
    # u_pp converges at order k+2.
    # For tau = O(1/h), q_h converges at order k, u_h AND
    # u_pp at order k+1 (post processing doesn't help).
    # In both cases, div(q - q_pp) converges at order k+1

    hk = CellVolume(mesh)
    tau = Constant(50)
    # tau = Constant(0.5)/hk
    qhat = q + tau*(u - uhat)*n

    def ejump(a):
        return 2*avg(a)

    a = (
        (dot(v, q) - div(v)*u)*dx
        + ejump(uhat*inner(v, n))*dS
        + uhat*inner(v, n)*ds
        - dot(grad(w), q)*dx
        + ejump(inner(qhat, n)*w)*dS
        + inner(qhat, n)*w*ds
        # Transmission condition (interior only)
        + ejump(mu*inner(qhat, n))*dS
        # trace mass term for the boundary conditions
        # <uhat, mu>ds == <g, mu>ds where g=0 in this example
        + uhat*mu*ds
    )

    L = w*f*dx
    w = Function(W, name="solutions")
    params = {'mat_type': 'matfree',
              'ksp_type': 'gmres',
              'pc_type': 'python',
              'ksp_monitor': True,
              'pc_python_type': 'firedrake.HybridStaticCondensationPC',
              'hybrid_sc': {'ksp_type': 'preonly',
                            'pc_type': 'lu'}}
    solve(a == L, w, solver_parameters=params)
    q_h, u_h, uhat_h = w.split()

    u_a = Function(V_a, name="Analytical Scalar")
    u_a.interpolate(sin(x[0]*pi)*sin(x[1]*pi))

    q_a = Function(U_a, name="Analytical vector")
    q_a.project(-grad(sin(x[0]*pi)*sin(x[1]*pi)))

    m = TestFunction(FunctionSpace(mesh, "DG", 0))
    int_avg_uh = assemble(Tensor((1/hk)*u_h*m*dx))
    int_avg_ua = assemble(Tensor((1/hk)*u_a*m*dx))

    error_dict = {"q_h": errornorm(q_h, q_a),
                  "u_h": errornorm(u_h, u_a),
                  "u_hbar": errornorm(int_avg_ua, int_avg_uh)}

    if post_process:
        # Post processing for scalar variable
        DGk1 = FunctionSpace(mesh, "DG", degree + 1)
        DG0 = FunctionSpace(mesh, "DG", 0)
        Wpp = DGk1 * DG0

        up, psi = TrialFunctions(Wpp)
        wp, phi = TestFunctions(Wpp)

        K = Tensor((inner(grad(up), grad(wp)) +
                    inner(psi, wp) +
                    inner(up, phi))*dx)
        F = Tensor((-inner(q_h, grad(wp)) +
                    inner(u_h, phi))*dx)

        wpp = Function(Wpp)
        assemble(K.inv * F, tensor=wpp)
        u_pp, _ = wpp.split()

        err_upp = sqrt(assemble((u_pp - u_a) * (u_pp - u_a) * dx))

        error_dict.update({"u_pp": err_upp})

        if d != 0:
            # Post processing of vector variable
            qhat_h = q_h + tau*(u_h - uhat_h)*n
            RT = FiniteElement("RT", triangle, degree + 1)
            RTd_element = BrokenElement(RT)
            RTd = FunctionSpace(mesh, RTd_element)
            nu = Function(RTd)
            DGkn1 = VectorFunctionSpace(mesh, "DG", degree - 1)
            Npp = DGkn1 * T
            n_p = TrialFunction(RTd)
            vp, mu = TestFunctions(Npp)

            A = Tensor(inner(n_p, vp)*dx +
                       jump(n_p, n=n)*mu('+')*dS +
                       dot(n_p, n)*mu*ds)
            B = Tensor(jump(qhat_h - q_h, n=n)*mu('+')*dS
                       + dot(qhat_h - q_h, n)*mu*ds)
            assemble(A.inv * B, tensor=nu)

            q_pp = nu + q_h

            diverr = sqrt(assemble(div(q_pp - q_a) *
                                   div(q_pp - q_a) * dx))
            qpp_err = errornorm(q_pp, q_a)

            error_dict.update({"q_pp": qpp_err})
            error_dict.update({"q_pp_div": diverr})
        else:
            error_dict.update({"q_pp": 0})
            error_dict.update({"q_pp_div": 0})

    if write:
        File("hdg-test.pvd").write(q_a, u_a, u_h)

    return error_dict

errs_u = []
errs_q = []
errs_ubar = []
errs_upp = []
errs_qpp = []
errs_qpp_div = []
d = 3
h_array = list(range(3, 9))
for r in h_array:
    errors = run_hdg_poisson(r, d, write=False, post_process=True)
    errs_u.append(errors["u_h"])
    errs_q.append(errors["q_h"])
    errs_ubar.append(errors["u_hbar"])
    errs_upp.append(errors["u_pp"])
    errs_qpp.append(errors["q_pp"])
    errs_qpp_div.append(errors["q_pp_div"])

errs_u = np.array(errs_u)
errs_q = np.array(errs_q)
errs_ubar = np.array(errs_ubar)
errs_upp = np.array(errs_upp)
errs_qpp = np.array(errs_qpp)
errs_qpp_div = np.array(errs_qpp_div)

conv_u = np.log2(errs_u[:-1] / errs_u[1:])[-1]
conv_q = np.log2(errs_q[:-1] / errs_q[1:])[-1]
conv_ubar = np.log2(errs_ubar[:-1] / errs_ubar[1:])[-1]
conv_upp = np.log2(errs_upp[:-1] / errs_upp[1:])[-1]
conv_qpp = np.log2(errs_qpp[:-1] / errs_qpp[1:])[-1]
conv_qpp_div = np.log2(errs_qpp_div[:-1] / errs_qpp_div[1:])[-1]

print("Convergence rate for u - u_h: %0.8f" % conv_u)
print("Convergence rate for u - u_pp: %0.8f" % conv_upp)
print("Convergence rate for ubar - u_hbar: %0.8f" % conv_ubar)
print("Convergence rate for q - q_h: %0.8f" % conv_q)
print("Convergence rate for q - q_pp: %0.8f" % conv_qpp)
print("Convergence rate for div(q - q_pp): %0.8f" % conv_qpp_div)
