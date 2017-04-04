from __future__ import absolute_import, print_function, division

from firedrake import *


def run_hybrid_extr_helmholtz(degree, res, quads):
    nx = 2 ** res
    ny = 2 ** res
    nz = 2 ** res
    base = UnitSquareMesh(nx, ny, quadrilateral=quads)
    mesh = ExtrudedMesh(base, layers=nz, layer_height=1.0/nz)

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
    expr = (1+12*pi*pi)*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z)
    f.interpolate(expr)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = dot(sigma, tau)*dx + u*v*dx + div(sigma)*v*dx - div(tau)*u*dx
    L = f*v*dx
    w = Function(W)
    params = {'mat_type': 'matfree',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization_ksp_type': 'cg',
              'hybridization_projector_tolerance': 1e-14}
    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h = w.split()

    nh_w = Function(W)
    nh_params = {'pc_type': 'fieldsplit',
                 'pc_fieldsplit_type': 'schur',
                 'ksp_type': 'cg',
                 'ksp_rtol': 1e-14,
                 'pc_fieldsplit_schur_fact_type': 'FULL',
                 'fieldsplit_0_ksp_type': 'cg',
                 'fieldsplit_1_ksp_type': 'cg'}
    solve(a == L, nh_w, solver_parameters=nh_params)
    s_nh, u_nh = nh_w.split()

    File("3D-hybrid.pvd").write(sigma_h, u_h, s_nh, u_nh)

run_hybrid_extr_helmholtz(0, 5, quads=False)
