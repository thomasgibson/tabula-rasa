from firedrake import *
import numpy as np


def run_hybrid_mixed(r, d, write=False, post_process=False):
    """
    """
    mesh = UnitSquareMesh(2**r, 2**r)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    RTd_element = BrokenElement(FiniteElement("RT", triangle, d + 1))
    RTd = FunctionSpace(mesh, RTd_element)
    DG = FunctionSpace(mesh, "DG", d)
    T = FunctionSpace(mesh, "HDiv Trace", d)

    Wd = RTd * DG * T

    sigma, u, lambdar = TrialFunctions(Wd)
    tau, v, gammar = TestFunctions(Wd)

    bcs = DirichletBC(Wd.sub(2), Constant(0.0), "on_boundary")

    adx = (dot(sigma, tau) - div(tau)*u + div(sigma)*v + u*v)*dx
    adS = (jump(sigma, n=n)*gammar('+') + jump(tau, n=n)*lambdar('+'))*dS
    a = adx + adS

    f = Function(DG)
    f.interpolate((1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
    L = v*f*dx

    w = Function(Wd, name="Approximate")
    params = {'mat_type': 'matfree',
              'ksp_type': 'gmres',
              'pc_type': 'python',
              'ksp_monitor': True,
              'pc_python_type': 'firedrake.HybridStaticCondensationPC',
              'hybrid_sc': {'ksp_type': 'preonly',
                            'pc_type': 'lu'}}
    solve(a == L, w, bcs=bcs, solver_parameters=params)
    sigma_h, u_h, lambdar_h = w.split()

    DG_a = FunctionSpace(mesh, "DG", d + 2)
    RTd_element_a = BrokenElement(FiniteElement("RT", triangle, d + 3))
    RTd_a = FunctionSpace(mesh, RTd_element_a)

    u_a = Function(DG_a, name="Analytic u")
    u_a.interpolate(sin(x[0]*pi*2)*sin(x[1]*pi*2))
    sigma_a = Function(RTd_a, name="Analytic sigma")
    sigma_a.project(-grad(sin(x[0]*pi*2)*sin(x[1]*pi*2)))

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
d = 2
h_array = list(range(3, 7))
for r in h_array:
    errors = run_hybrid_mixed(r, d, write=False, post_process=True)
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
