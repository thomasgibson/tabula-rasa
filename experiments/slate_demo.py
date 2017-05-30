from firedrake import *
from matplotlib.pyplot import figure, show, title

res = 1
nx = 2 ** res
ny = 2 ** res
nz = 2 ** res

quads = False
broken = False

base = UnitSquareMesh(nx, ny, quadrilateral=quads)
mesh = ExtrudedMesh(base, layers=nz, layer_height=1.0/nz)

degree = 0

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
if broken:
    V = FunctionSpace(mesh, BrokenElement(HDiv_ele))
    U = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace",
                      HDiv_ele.degree())
    W = V * U * T
    u, p, lambdar = TrialFunctions(W)
    w, v, gammar = TestFunctions(W)
    n = FacetNormal(mesh)

    a_dx = dot(w, u)*dx - div(w)*p*dx + v*div(u)*dx + p*v*dx
    a_dS = jump(w, n=n)*lambdar('+')*dS_h + jump(w, n=n)*lambdar('+')*dS_v + jump(u, n=n)*gammar('+')*dS_h + jump(u, n=n)*gammar('+')*dS_v

    form = a_dx + a_dS

else:
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DG", degree)
    W = V * U
    u, p = TrialFunctions(W)
    w, v = TestFunctions(W)
    n = FacetNormal(mesh)

    form = dot(w, u)*dx - div(w)*p*dx + v*div(u)*dx # + p*v*dx

M = assemble(form, mat_type="aij")
fig = figure()
ax1 = fig.add_subplot(111)
ax1.spy(M.M.values, markersize=2, precision=0.0001)

show()
