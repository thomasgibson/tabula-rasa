from firedrake import *
from matplotlib.pyplot import figure, show, title
import numpy

mesh = UnitSquareMesh(2, 2)
degree = 2
RT = FiniteElement("RT", triangle, degree)
V = FunctionSpace(mesh, BrokenElement(RT))
U = FunctionSpace(mesh, "DG", degree - 1)
T = FunctionSpace(mesh, "HDiv Trace", degree - 1)
W = V * U
n = FacetNormal(mesh)

u, p = TrialFunctions(W)
w, v = TestFunctions(W)
gammar = TestFunction(T)

form = dot(w, u)*dx - div(w)*p*dx +\
       v*div(u)*dx + p*v*dx
local_trace = gammar('+')*dot(u, n)*dS
l = v*dx

A = Tensor(form)
K = Tensor(local_trace)
F = Tensor(l)
S = K * A.inv * K.T
E = K * A.inv * F

sigma = TrialFunction(V)
tau = TestFunction(V)
q = TrialFunction(U)
r = TestFunction(U)
M = assemble(S)
fig = figure()
ax1 = fig.add_subplot(111)
ax1.spy(M.M.values, markersize=15, precision=0.0001)
title("Schur-complement matrix for $\lambda$: HDivTrace1")

show()
