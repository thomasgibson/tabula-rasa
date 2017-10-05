from firedrake import *

from firedrake.formmanipulation import split_form
from firedrake.parloops import par_loop, READ, INC

import numpy as np

mesh = UnitSquareMesh(32, 32)

CG = FiniteElement("Lagrange", triangle, 5)
int_ele = CG["interior"]
facet_ele = CG["facet"]

V_o = FunctionSpace(mesh, int_ele)
V_d = FunctionSpace(mesh, facet_ele)
V = FunctionSpace(mesh, "CG", 5)
f = Function(V)
f.interpolate(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))

W = V_o * V_d

u_in, u_d = TrialFunctions(W)
v_in, v_d = TestFunctions(W)

a = (dot(grad(v_in + v_d), grad(u_in + u_d))
     + (v_in + v_d) * (u_in + u_d)) * dx
L = f * (v_in + v_d) * dx

A = dict(split_form(a))
F = dict(split_form(L))

A00 = Tensor(A[(0, 0)])
A01 = Tensor(A[(0, 1)])
A10 = Tensor(A[(1, 0)])
A11 = Tensor(A[(1, 1)])
F0 = Tensor(F[(0,)])
F1 = Tensor(F[(1,)])

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

shapes = (V_o.finat_element.space_dimension(), np.prod(V_o.shape),
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
        u_h[i+15][j] = u_o[i][j];
}}

for (int i=0; i<%d; ++i){
    for (int j=0; j<%d; ++j) {
        u_h[i][j] = u_d[i][j];
}}""" % shapes

par_loop(kernel, dx, {"u_h": (u_h, INC),
                      "u_o": (u_int, READ),
                      "u_d": (u_ext, READ)})

u_t = Function(V, name="Analytic")
u_t.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
print(errornorm(u_t, u_h))
File("cgsc.pvd").write(u_h, u_t)
