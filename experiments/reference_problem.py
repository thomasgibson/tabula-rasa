from __future__ import absolute_import, print_function, division

from firedrake import *

qflag = False
degree = 1
res = 32

mesh = UnitSquareMesh(res, res, quadrilateral=qflag)
n = FacetNormal(mesh)

if qflag:
    RT = FiniteElement("RTCF", quadrilateral, degree)
    DG = FiniteElement("DQ", quadrilateral, degree - 1)

else:
    RT = FiniteElement("RT", triangle, degree)
    DG = FiniteElement("DG", triangle, degree - 1)

V = FunctionSpace(mesh, RT)
U = FunctionSpace(mesh, DG)

W = V * U

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = (dot(u, v) + div(v)*p + q*div(u) + p*q)*dx

x = SpatialCoordinate(mesh)
f = Function(U).assign(0)

L = -f*q*dx + 42*dot(v, n)*ds

usol = []
psol = []
for hybrid in [False, True]:
    if hybrid:
        params = {"ksp_type": "preonly",
                  "mat_type": "matfree",
                  "pc_type": "python",
                  "pc_python_type": "firedrake.HybridizationPC",
                  "hybridization_ksp_type": "preonly",
                  "hybridization_pc_type": "lu",
                  "hybridization_projector_tolerance": 1e-10}
        suffix = "-hybrid"
    else:
        params = {"ksp_type": "gmres",
                  "ksp_rtol": 1e-10}
        suffix = ""

    w = Function(W)
    solve(a == L, w, solver_parameters=params)
    udat, pdat = w.split()
    uh = Function(V, name="velocity"+suffix).assign(udat)
    ph = Function(U, name="pressure"+suffix).assign(pdat)
    usol.append(uh)
    psol.append(ph)

File("natural_bc_test.pvd").write(usol[0], usol[1], psol[0], psol[1])
