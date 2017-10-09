from __future__ import absolute_import, division
from firedrake import *
import numpy as np
import csv


def l2_error(a, b):
    return sqrt(assemble(inner(a - b, a - b)*dx))


def poisson_sphere(MeshClass, refinement, hdiv_space, degree):
    """Test hybridizing lowest order mixed methods on a sphere."""
    mesh = MeshClass(refinement_level=refinement, degree=degree + 3)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, hdiv_space, degree + 1)
    U = FunctionSpace(mesh, "DG", degree)
    W = U * V

    f = Function(U)
    f.interpolate(x*y*z)

    u_exact = Function(FunctionSpace(mesh, "DG", degree + 2)).interpolate(x*y*z/12.0)
    sigma_exact = Function(VectorFunctionSpace(mesh, "CG", degree + 4)).project(-grad(u_exact))

    u, sigma = TrialFunctions(W)
    v, tau = TestFunctions(W)

    a = (dot(sigma, tau) - div(tau)*u + v*div(sigma))*dx
    L = f*v*dx
    w = Function(W)

    nullsp = MixedVectorSpaceBasis(W, [VectorSpaceBasis(constant=True), W[1]])
    params = {'mat_type': 'matfree',
              'ksp_type': 'gmres',
              'ksp_monitor_true_residual': True,
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'gmres',
                                'pc_type': 'lu',
                                'pc_factor_mat_solver_package': 'mumps',
                                'ksp_monitor': True,
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu',
                                                  'ksp_monitor': True},
                                'use_reconstructor': True}}
    solve(a == L, w, nullspace=nullsp, solver_parameters=params)
    u_h, sigma_h = w.split()
    error_u = l2_error(u_exact, u_h)
    error_sigma = l2_error(sigma_exact, sigma_h)
    return error_u, error_sigma


degree = 0
mesh = {"BDM": UnitIcosahedralSphereMesh,
        "RT": UnitIcosahedralSphereMesh,
        "RTCF": UnitCubedSphereMesh}
rterr_u = []
rterr_sigma = []
bdmerr_u = []
bdmerr_sigma = []
rtcferr_u = []
rtcferr_sigma = []
volTri = []
volQuad = []

# Local max is 7
ref_levels = range(2, 7)
for i in ref_levels:
    volTri.append(sqrt((4.0*pi)/mesh["RT"](i).topology.num_cells()))
    volQuad.append(sqrt((4.0*pi)/mesh["RTCF"](i).topology.num_cells()))
    rt_u_err, rt_s_err = poisson_sphere(mesh["RT"], i, "RT", degree)
    bdm_u_err, bdm_s_err = poisson_sphere(mesh["BDM"], i, "BDM", degree)
    rtcf_u_err, rtcf_s_err = poisson_sphere(mesh["RTCF"], i, "RTCF", degree)

    rterr_u.append(rt_u_err)
    rterr_sigma.append(rt_s_err)
    bdmerr_u.append(bdm_u_err)
    bdmerr_sigma.append(bdm_s_err)
    rtcferr_u.append(rtcf_u_err)
    rtcferr_sigma.append(rtcf_s_err)

rterr_u = np.asarray(rterr_u)
rterr_sigma = np.asarray(rterr_sigma)
bdmerr_u = np.asarray(bdmerr_u)
bdmerr_sigma = np.asarray(bdmerr_sigma)
rtcferr_u = np.asarray(rtcferr_u)
rtcferr_sigma = np.asarray(rtcferr_sigma)

k = degree + 1
res = [1/(2 ** r) for r in ref_levels]
dhk = np.array(res) ** k
dhk1 = np.array(res) ** (k + 1)
dh_arry_k = (0.1 ** k) * dhk
dh_arry_k1 = (0.1 ** k) * dhk1

fieldnames = ['cell_vol_tri',
              'cell_vol_quad',
              'RT_u_err',
              'BDM_u_err',
              'RTCF_u_err',
              'RT_sigma_err',
              'BDM_sigma_err',
              'RTCF_sigma_err',
              'dh_k',
              'dh_k1']
data = [volTri,
        volQuad,
        rterr_u,
        bdmerr_u,
        rtcferr_u,
        rterr_sigma,
        bdmerr_sigma,
        rtcferr_sigma,
        dh_arry_k,
        dh_arry_k1]
csv_file = open('2d_poisson_sphere_d%d.csv' % k, 'w')
csvwriter = csv.writer(csv_file)
csvwriter.writerow(fieldnames)
for data in zip(*data):
    csvwriter.writerow(data)
csv_file.close()
