from __future__ import absolute_import, division

from firedrake import *
import numpy as np
import csv


def l2_error(a, b):
    return sqrt(assemble(inner(a - b, a - b)*dx))


def run_hybrid_extr_helmholtz(degree, res, quads=False, write=False):
    nx = 2 ** res
    ny = 2 ** res
    nz = 2 ** res
    l = 1/nx
    w = 1/ny
    h = 0.2 / nz
    base = UnitSquareMesh(nx, ny, quadrilateral=quads)
    mesh = ExtrudedMesh(base, layers=nz, layer_height=h)

    if quads:
        RT = FiniteElement("RTCF", quadrilateral, degree + 1)
        DG_v = FiniteElement("DG", interval, degree)
        DG_h = FiniteElement("DQ", quadrilateral, degree)
        CG = FiniteElement("CG", interval, degree + 1)
        vol = l * w * h

    else:
        RT = FiniteElement("RT", triangle, degree + 1)
        DG_v = FiniteElement("DG", interval, degree)
        DG_h = FiniteElement("DG", triangle, degree)
        CG = FiniteElement("CG", interval, degree + 1)
        vol = 0.5 * l * w * h

    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DG", degree)
    W = V * U

    x = SpatialCoordinate(mesh)
    f = Function(U)
    f.interpolate((1+38*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5))
    exact = sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5)
    exact_flux = -grad(sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5))

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = dot(sigma, tau)*dx + u*v*dx + div(sigma)*v*dx - div(tau)*u*dx
    L = f*v*dx
    w = Function(W)
    params = {'ksp_type': 'preonly',
              # 'ksp_monitor_true_residual': True,
              'mat_type': 'matfree',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'hypre',
                                'pc_hypre_type': 'boomeramg',
                                'ksp_rtol': 1e-14,
                                'ksp_monitor': True,
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu',
                                                  'ksp_monitor': True},
                                'use_reconstructor': True}}
    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h = w.split()
    sigma_h.rename("flux")
    u_h.rename("pressure")

    err_s = l2_error(u_h, exact)
    err_f = l2_error(sigma_h, exact_flux)

    if write:
        File("3D-hybrid.pvd").write(sigma_h, u_h, exact, exact_flux)
        return
    else:
        return (err_s, err_f), vol

# Max local res is 6 for both LO and NLO
ref_levels = range(1, 6)
degree = 1
errRT_u = []
errRT_sigma = []
errRTCF_u = []
errRTCF_sigma = []
volRT = []
volRTCF = []
for i in ref_levels:
    e, vol_tri = run_hybrid_extr_helmholtz(degree=degree,
                                           res=i, quads=False)
    rt_err_s, rt_err_f = e

    ertc, vol_quad = run_hybrid_extr_helmholtz(degree=degree,
                                               res=i, quads=True)
    rtcf_err_s, rtcf_err_f = ertc

    volRT.append(vol_tri)
    volRTCF.append(vol_quad)
    errRT_u.append(rt_err_s)
    errRT_sigma.append(rt_err_f)
    errRTCF_u.append(rtcf_err_s)
    errRTCF_sigma.append(rtcf_err_f)

errRT_u = np.asarray(errRT_u)
errRT_sigma = np.asarray(errRT_sigma)
errRTCF_u = np.asarray(errRTCF_u)
errRTCF_sigma = np.asarray(errRTCF_sigma)

fieldnames = ['cell_vol_rt',
              'cell_vol_rtcf',
              'RT_u_err',
              'RT_sigma_err',
              'RTCF_u_err',
              'RTCF_sigma_err',
              'dh_rt',
              'dh_rtcf']
res = [1/(2 ** r) for r in ref_levels]
dh = np.array(res)
k = degree + 1
dh = dh ** k
dh_arry_rt = 2 * k * dh
dh_arry_rtcf = 2 * k * dh
data = [volRT,
        volRTCF,
        errRT_u,
        errRT_sigma,
        errRTCF_u,
        errRTCF_sigma,
        dh_arry_rt,
        dh_arry_rtcf]
csv_file = open('3d_helmholtz_d%d.csv' % k, 'w')
csvwriter = csv.writer(csv_file)
csvwriter.writerow(fieldnames)
for data in zip(*data):
    csvwriter.writerow(data)
csv_file.close()
