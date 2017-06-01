from __future__ import absolute_import, print_function, division

from firedrake import *

qflag = True
degree = 1
extrusion = True
degree = 1

if extrusion:
    base = UnitSquareMesh(2, 2, quadrilateral=qflag)
    mesh = ExtrudedMesh(base, layers=2, layer_height=0.5)
    n = FacetNormal(mesh)

    if qflag:
        RT = FiniteElement("RTCF", quadrilateral, degree)
        DG_v = FiniteElement("DG", interval, degree - 1)
        DG_h = FiniteElement("DQ", quadrilateral, degree - 1)
        CG = FiniteElement("CG", interval, degree)

    else:
        RT = FiniteElement("RT", triangle, degree)
        DG_v = FiniteElement("DG", interval, degree - 1)
        DG_h = FiniteElement("DG", triangle, degree - 1)
        CG = FiniteElement("CG", interval, degree)

    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, BrokenElement(HDiv_ele))
    U = FunctionSpace(mesh, "DG", degree - 1)
    T = FunctionSpace(mesh, "HDiv Trace", (degree - 1, degree - 1))
    W = V * U

    x, y, z = SpatialCoordinate(mesh)
    t = Function(T)
    expr = (1+12*pi*pi)*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z)
    t.interpolate(expr)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    gammar = TestFunction(T)

    a_dx = (dot(u, v) + div(v)*p + q*div(u) + p*q)*dx
    a_dS = t*dot(u*q, n)*(dS_v + dS_h)
    a = a_dx + a_dS

    tr = gammar('+')*dot(u, n)*(dS_v + dS_h)

    A = Tensor(a)
    K = Tensor(tr)

    S = assemble(K * A.inv * K.T)
    S.force_evaluation()

else:
    mesh = UnitSquareMesh(2, 2, quadrilateral=qflag)
    n = FacetNormal(mesh)

    if qflag:
        RT = FiniteElement("RTCF", quadrilateral, degree)
        DG = FiniteElement("DQ", quadrilateral, degree - 1)
        Te = FiniteElement("HDiv Trace", quadrilateral, degree - 1)

    else:
        RT = FiniteElement("RT", triangle, degree)
        DG = FiniteElement("DG", triangle, degree - 1)
        Te = FiniteElement("HDiv Trace", triangle, degree - 1)

    V = FunctionSpace(mesh, BrokenElement(RT))
    U = FunctionSpace(mesh, DG)
    T = FunctionSpace(mesh, Te)

    W = V * U

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    gammar = TestFunction(T)

    t = Function(T).assign(3.14)

    a_dx = (dot(u, v) + div(v)*p + q*div(u) + p*q)*dx
    a_dS = t*dot(u*q, n)*dS
    a = a_dx + a_dS

    tr = gammar('+')*dot(u, n)*dS

    A = Tensor(a)
    K = Tensor(tr)

    S = assemble(K * A.inv * K.T)
    S.force_evaluation()
