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

    V_a = FunctionSpace(mesh, "DG", degree + 2)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 2)
    u_a = Function(V_a, name="Analytical Scalar")
    u_a.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    q_a = Function(U_a, name="Analytical vector")
    q_a.project(-grad(x[0]*(1-x[0])*x[1]*(1-x[1])))

    error_dict = {"q_h": errornorm(q_h, q_a),
                  "u_h": errornorm(u_h, u_a)}

    if post_process:
        # Post processing for scalar variable
        Vk1 = FunctionSpace(mesh, "DG", degree + 1)
        u_pp = Function(Vk1, name="Post processed scalar")
        nu = Function(Vk1)
        nu_h = TrialFunction(Vk1)
        eta = TestFunction(Vk1)
        K = inner(grad(nu_h), grad(eta))*dx
        F = -inner(grad(u_h) + q_h, grad(eta))*dx
        A = Tensor(K)
        B = Tensor(F)
        assemble(A.inv * B, tensor=nu)

        u_hk1 = Function(Vk1).interpolate(u_h)
        u_pp.assign(u_hk1 + nu)
        error_dict.update({"u_pp": errornorm(u_a, u_pp)})

    if write:
        if post_process:
            File("hdg-test.pvd").write(q_a, u_a, u_h, u_pp)
        else:
            File("hdg-test.pvd").write(q_a, u_a, u_h)

    return error_dict


errs_u = []
errs_q = []
errs_upp = []
d = 2
h_array = list(range(3, 7))
for r in h_array:
    errors = run_hdg_poisson(r, d, write=False, post_process=True)
    errs_u.append(errors["u_h"])
    errs_q.append(errors["q_h"])
    errs_upp.append(errors["u_pp"])

errs_u = np.array(errs_u)
errs_q = np.array(errs_q)
errs_upp = np.array(errs_upp)

conv_u = np.log2(errs_u[:-1] / errs_u[1:])[-1]
conv_q = np.log2(errs_q[:-1] / errs_q[1:])[-1]
conv_upp = np.log2(errs_upp[:-1] / errs_upp[1:])[-1]

print("Convergence rate for u_h: %0.8f" % conv_u)
print("Convergence rate for q_h: %0.8f" % conv_q)
print("Convergence rate for u_pp: %0.8f" % conv_upp)
