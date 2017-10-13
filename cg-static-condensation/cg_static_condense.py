from firedrake import *

from firedrake.parloops import par_loop, READ, INC

import numpy as np


def bilinear_form(test, trial):
    """
    """
    return (dot(grad(test), grad(trial)) + test*trial) * dx


def linear_form(test, f):
    """
    """
    return f * test * dx


def run_sc_helmholtz(r, d, write=False):
    mesh = UnitSquareMesh(2**r, 2**r)

    CG = FiniteElement("Lagrange", triangle, d)
    int_ele = CG["interior"]
    facet_ele = CG["facet"]

    V_o = FunctionSpace(mesh, int_ele)
    V_d = FunctionSpace(mesh, facet_ele)
    V = FunctionSpace(mesh, "CG", d)
    V_ho = FunctionSpace(mesh, "CG", d+2)
    f = Function(V)
    f.interpolate(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))

    A00 = Tensor(bilinear_form(TestFunction(V_o), TrialFunction(V_o)))
    A01 = Tensor(bilinear_form(TestFunction(V_o), TrialFunction(V_d)))
    A10 = Tensor(bilinear_form(TestFunction(V_d), TrialFunction(V_o)))
    A11 = Tensor(bilinear_form(TestFunction(V_d), TrialFunction(V_d)))
    F0 = Tensor(linear_form(TestFunction(V_o), f))
    F1 = Tensor(linear_form(TestFunction(V_d), f))

    u_ext = Function(V_d)

    S = A11 - A10 * A00.inv * A01
    E = F1 - A10 * A00.inv * F0

    Mat = assemble(S)
    Mat.force_evaluation()
    vec = assemble(E)

    solve(Mat, u_ext, vec)

    u_int = Function(V_o)
    assemble(A00.inv * (F0 - A01 * u_ext), tensor=u_int)

    u_h = Function(V, name="Approximate")

    # Extract first node index in the cell interior
    offset = V.finat_element.entity_dofs()[2][0][0]

    args = (V_o.finat_element.space_dimension(), np.prod(V_o.shape),
            offset,
            V_d.finat_element.space_dimension(), np.prod(V_d.shape))

    # Offset for interior dof mapping is determined by inspecting the entity
    # dofs of V (original FE space) and the dofs of V_o. For example,
    # degree 5 CG element has entity dofs:
    #
    # {0: {0: [0], 1: [1], 2: [2]}, 1: {0: [3, 4, 5, 6], 1: [7, 8, 9, 10],
    #  2: [11, 12, 13, 14]}, 2: {0: [15, 16, 17, 18, 19, 20]}}.
    #
    # Looking at the cell dofs, we have a starting dof index of 15. The
    # interior element has dofs:
    #
    # {0: {0: [], 1: [], 2: []}, 1: {0: [], 1:[], 2:[]},
    #  2: {0: [0, 1, 2, 3, 4, 5]}}
    #
    # with a starting dof index of 0. So the par_loop will need to be adjusted
    # by the difference: i + 15. The skeleton dofs do no need any offsets.
    kernel = """
    for (int i=0; i<%d; ++i){
        for (int j=0; j<%d; ++j) {
            u_h[i+%d][j] = u_o[i][j];
    }}

    for (int i=0; i<%d; ++i){
        for (int j=0; j<%d; ++j) {
            u_h[i][j] = u_d[i][j];
    }}""" % args

    par_loop(kernel, dx, {"u_h": (u_h, INC),
                          "u_o": (u_int, READ),
                          "u_d": (u_ext, READ)})

    u_t = Function(V_ho, name="Analytic")
    u_t.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    if write:
        File("cgsc.pvd").write(u_h, u_t)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = bilinear_form(v, u)
    L = linear_form(v, f)
    uh = Function(V)

    solve(a == L, uh)
    return errornorm(u_t, u_h), errornorm(u_h, uh)


degree = 3
error, error_comp = run_sc_helmholtz(3, degree)
print(error)
print(error_comp)
