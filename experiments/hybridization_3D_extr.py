from __future__ import absolute_import, division

from firedrake import *


def run_hybrid_extr_helmholtz(degree, res, quads):
    nx = 2 ** res
    ny = 2 ** res
    nz = 2 ** (res - 1)
    h = 0.2 / nz
    base = UnitSquareMesh(nx, ny, quadrilateral=quads)
    mesh = ExtrudedMesh(base, layers=nz, layer_height=h)

    if quads:
        RT = FiniteElement("RTCF", quadrilateral, degree + 1)
        DG_v = FiniteElement("DG", interval, degree)
        DG_h = FiniteElement("DQ", quadrilateral, degree)
        CG = FiniteElement("CG", interval, degree + 1)

    else:
        RT = FiniteElement("RT", triangle, degree + 1)
        DG_v = FiniteElement("DG", interval, degree)
        DG_h = FiniteElement("DG", triangle, degree)
        CG = FiniteElement("CG", interval, degree + 1)

    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DG", degree)
    W = V * U

    x, y, z = SpatialCoordinate(mesh)
    f = Function(U)
    f.interpolate(Expression("(1+38*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5)"))
    exact = Function(U)
    exact.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5)"))
    exact.rename("exact")

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = dot(sigma, tau)*dx + u*v*dx + div(sigma)*v*dx - div(tau)*u*dx
    L = f*v*dx
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-14},
                                'use_reconstructor': True}}
    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h = w.split()
    sigma_h.rename("flux")
    u_h.rename("pressure")

    print errornorm(u_h, exact)

    File("3D-hybrid.pvd").write(sigma_h, u_h, exact)

run_hybrid_extr_helmholtz(0, 5, quads=False)
