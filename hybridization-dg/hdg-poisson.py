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
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    W = U * V * T
    q, u, uhat = TrialFunctions(W)
    v, w, mu = TestFunctions(W)

    # Lagrange multipliers enforce the Dirichlet condition
    bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

    f = Function(V).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    # $qhat\cdot n$
    tau = Constant(10)/CellVolume(mesh)
    qhat = q + tau*(u - uhat)*n

    def ejump(a):
        return 2*avg(a)

    a = (
        (dot(v, q) - div(v)*u)*dx
        + ejump(uhat*inner(v, n))*dS
        # + uhat*inner(v, n)*ds
        - dot(grad(w), q)*dx
        + ejump(inner(qhat, n)*w)*dS
        + inner(qhat, n)*w*ds
        + ejump(mu*inner(qhat, n))*dS
        # + mu*inner(qhat, n)*ds
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
    solve(a == L, w, bcs=bcs, solver_parameters=params)
    q_h, u_h, uhat_h = w.split()

    V_a = FunctionSpace(mesh, "DG", degree + 3)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 3)
    u_a = Function(V_a, name="Analytical Scalar")
    u_a.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    q_a = Function(U_a, name="Analytical vector")
    q_a.project(-grad(x[0]*(1-x[0])*x[1]*(1-x[1])))

    error_dict = {"q_h": errornorm(q_h, q_a),
                  "u_h": errornorm(u_h, u_a)}

    if post_process:
        # # Post processing for scalar variable
        Vk1 = FunctionSpace(mesh, "DG", degree + 1)
        u_pp = Function(Vk1, name="Post processed scalar")
        ustar = TrialFunction(Vk1)
        eta = TestFunction(Vk1)
        A = Tensor(inner(grad(ustar), grad(eta))*dx)
        B = Tensor((uhat_h - u_h)*jump(grad(eta), n=n)*dS +
                   inner(uhat_h - u_h, dot(grad(eta), n))*ds)

        assemble(A.inv * B, tensor=u_pp)
        uk1 = Function(Vk1).interpolate(u_h)
        u_pp += uk1

        error_dict.update({"u_pp": errornorm(u_a, u_pp)})

        # Post processing of vector variable
        RTd = FunctionSpace(mesh, "DRT", degree + 1)
        nu = Function(RTd)
        Un1 = VectorFunctionSpace(mesh, "DG", degree - 1)
        W = Un1 * T
        nu_h = TrialFunction(RTd)
        eta, gammar = TestFunctions(W)
        qhat_h = q_h + tau*(u_h - uhat_h)*n
        A = Tensor(inner(nu_h, eta)*dx +
                   jump(nu_h, n=n)*gammar*dS +
                   dot(nu_h, n)*gammar*ds)
        B = Tensor(jump(qhat_h - q_h, n=n)*gammar*dS +
                   dot(qhat_h - q_h, n)*gammar*ds)
        assemble(A.inv * B, tensor=nu)
        q_pp = Function(RTd, name="q_pp").project(q_h)
        q_pp += nu
        div_err = sqrt(assemble((div(q_a) - div(q_pp)) *
                                (div(q_a) - div(q_pp)) * dx))
        error_dict.update({"q_pp_div": div_err})

    if write:
        if post_process:
            File("hdg-test.pvd").write(q_a, u_a, u_h, q_pp)
        else:
            File("hdg-test.pvd").write(q_a, u_a, u_h)

    return error_dict


errs_u = []
errs_q = []
errs_upp = []
errs_qpp = []
d = 2
h_array = list(range(3, 7))
for r in h_array:
    errors = run_hdg_poisson(r, d, write=False, post_process=True)
    errs_u.append(errors["u_h"])
    errs_q.append(errors["q_h"])
    errs_upp.append(errors["u_pp"])
    errs_qpp.append(errors["q_pp_div"])

errs_u = np.array(errs_u)
errs_q = np.array(errs_q)
errs_upp = np.array(errs_upp)
errs_qpp = np.array(errs_qpp)

conv_u = np.log2(errs_u[:-1] / errs_u[1:])[-1]
conv_q = np.log2(errs_q[:-1] / errs_q[1:])[-1]
conv_upp = np.log2(errs_upp[:-1] / errs_upp[1:])[-1]
conv_qpp = np.log2(errs_qpp[:-1] / errs_qpp[1:])[-1]

print("Convergence rate for u_h: %0.8f" % conv_u)
print("Convergence rate for q_h: %0.8f" % conv_q)
print("Convergence rate for u_pp: %0.8f" % conv_upp)
print("Convergence rate for div(q_pp): %0.8f" % conv_qpp)
