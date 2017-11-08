from firedrake import *
import numpy as np


def run_hybrid_mixed(r, d, write=False, post_process=False, mixed_method="RT"):
    """
    """
    if mixed_method == "RT":
        mesh = UnitSquareMesh(2**r, 2**r)
        broken_element = BrokenElement(FiniteElement("RT",
                                                     triangle,
                                                     d + 1))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DG", d)
        T = FunctionSpace(mesh, "HDiv Trace", d)
    elif mixed_method == "BDM":
        mesh = UnitSquareMesh(2**r, 2**r)
        broken_element = BrokenElement(FiniteElement("BDM",
                                                     triangle,
                                                     d))
        U = FunctionSpace(mesh, broken_element)
        V = FunctionSpace(mesh, "DG", d - 1)
        T = FunctionSpace(mesh, "HDiv Trace", d)
    else:
        raise ValueError

    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    V_a = FunctionSpace(mesh, "DG", d + 4)
    U_a = VectorFunctionSpace(mesh, "DG", d + 4)

    Wd = U * V * T

    sigma, u, lambdar = TrialFunctions(Wd)
    tau, v, gammar = TestFunctions(Wd)

    # This is the formulation described by the Feel++ folks where
    # the multipliers weakly enforce the Dirichlet condition on
    # the scalar unknown.
    adx = (dot(sigma, tau) - div(tau)*u + div(sigma)*v)*dx
    adS = (jump(sigma, n=n)*gammar('+') + jump(tau, n=n)*lambdar('+'))*dS
    ads = (dot(tau, n)*lambdar + lambdar*gammar)*ds
    a = adx + adS + ads

    f = Function(V_a)
    f.interpolate((2*pi*pi)*sin(x[0]*pi)*sin(x[1]*pi))
    L = v*f*dx

    w = Function(Wd, name="Approximate")
    params = {'mat_type': 'matfree',
              'ksp_type': 'gmres',
              'pc_type': 'python',
              'ksp_monitor': True,
              'pc_python_type': 'firedrake.HybridStaticCondensationPC',
              'hybrid_sc': {'ksp_type': 'preonly',
                            'pc_type': 'lu'}}
    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h, lambdar_h = w.split()

    u_a = Function(V_a, name="Analytic u")
    u_a.interpolate(sin(x[0]*pi)*sin(x[1]*pi))
    sigma_a = Function(U_a, name="Analytic sigma")
    sigma_a.project(-grad(sin(x[0]*pi)*sin(x[1]*pi)))

    error_dict = {"sigma": errornorm(sigma_a, sigma_h),
                  "u": errornorm(u_a, u_h)}

    if post_process:
        # Only works for even values of d
        assert d % 2 == 0
        DG_pp = FunctionSpace(mesh, "DG", d + 1)
        u_pp = Function(DG_pp, name="Post-processed u")
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
        assemble(A.inv * B, tensor=u_pp)
        error_dict.update({"u_pp": errornorm(u_a, u_pp)})

    if write:
        if post_process:
            File("hybrid-mixed-test.pvd").write(u_a, u_h, u_pp)
        else:
            File("hybrid-mixed-test.pvd").write(u_a, u_h)

    return error_dict

errs_u = []
errs_sigma = []
errs_upp = []
# Max d is 4
# Test cases: RT-H for k = 0, 2, 4 and BDM-H for k = 2, 4
d = 2
h_array = list(range(3, 7))
for r in h_array:
    errors = run_hybrid_mixed(r, d, write=False,
                              post_process=True,
                              mixed_method="BDM")
    errs_u.append(errors["u"])
    errs_sigma.append(errors["sigma"])
    errs_upp.append(errors["u_pp"])

errs_u = np.array(errs_u)
errs_sigma = np.array(errs_sigma)
errs_upp = np.array(errs_upp)

conv_u = np.log2(errs_u[:-1] / errs_u[1:])[-1]
conv_sigma = np.log2(errs_sigma[:-1] / errs_sigma[1:])[-1]
conv_upp = np.log2(errs_upp[:-1] / errs_upp[1:])[-1]

print("Convergence rate for u_h: %0.8f" % conv_u)
print("Convergence rate for sigma_h: %0.8f" % conv_sigma)
print("Convergence rate for u_pp: %0.8f" % conv_upp)
