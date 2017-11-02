from firedrake import *
from firedrake.formmanipulation import ExtractSubBlock
from pyop2.base import MixedDat
import numpy as np
import matplotlib.pyplot as plt


def generate_schur_matrix_method_A():
    mesh = UnitSquareMesh(2, 2)
    n = FacetNormal(mesh)
    RTd = FunctionSpace(mesh, BrokenElement(FiniteElement("RT", triangle, 1)))
    DG = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)

    Wd = RTd * DG
    sigma, u = TrialFunctions(Wd)
    tau, v = TestFunctions(Wd)
    gammar = TestFunction(T)

    bcs = DirichletBC(T, Constant(0.0), "on_boundary")

    Atilde = Tensor((dot(sigma, tau) - div(tau)*u +
                     div(sigma)*v + u*v)*dx)
    K = Tensor(jump(sigma, n=n)*gammar('+')*dS)
    S = -K * Atilde.inv * K.T
    Smat = assemble(S, bcs=bcs)
    Smat.force_evaluation()

    return Smat


def generate_schur_matrix_method_B():
    mesh = UnitSquareMesh(2, 2)
    n = FacetNormal(mesh)
    RTd = FunctionSpace(mesh, BrokenElement(FiniteElement("RT", triangle, 1)))
    DG = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)

    Wd = RTd * DG * T

    sigma, u, lambdar = TrialFunctions(Wd)
    tau, v, gammar = TestFunctions(Wd)

    bcs = DirichletBC(T, Constant(0.0), "on_boundary")

    adx = (dot(sigma, tau) - div(tau)*u + div(sigma)*v + u*v)*dx
    adS = (jump(sigma, n=n)*gammar('+') + jump(tau, n=n)*lambdar('+'))*dS
    a = adx + adS

    splitter = ExtractSubBlock()

    M = Tensor(splitter.split(a, ((0, 1), (0, 1))))

    K = Tensor(splitter.split(a, ((0, 1), (2,))))

    L = Tensor(splitter.split(a, ((2,), (0, 1))))

    J = Tensor(splitter.split(a, (2, 2)))

    # Schur complement for traces
    S = J - L * M.inv * K
    Smat = assemble(S, bcs=bcs)
    Smat.force_evaluation()

    return Smat


def generate_schur_rhs():
    mesh = UnitSquareMesh(2, 2)
    n = FacetNormal(mesh)
    RTd = FunctionSpace(mesh, BrokenElement(FiniteElement("RT", triangle, 1)))
    DG = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)

    Wd = RTd * DG * T

    sigma, u, lambdar = TrialFunctions(Wd)
    tau, v, gammar = TestFunctions(Wd)

    adx = (dot(sigma, tau) - div(tau)*u + div(sigma)*v + u*v)*dx
    adS = (jump(sigma, n=n)*gammar('+') + jump(tau, n=n)*lambdar('+'))*dS
    a = adx + adS

    f = Function(DG)
    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    L = v*f*dx
    v = assemble(L)

    splitter = ExtractSubBlock()

    M = Tensor(splitter.split(a, ((0, 1), (0, 1))))
    L = Tensor(splitter.split(a, ((2,), (0, 1))))

    r = Function(T)
    v1, v2, v3 = l.split()
    VV = W[0] * W[1]
    mdat = MixedDat([v1.dat, v2.dat])
    v1v2 = Function(VV, val=mdat.data)

    r_lambda = -L * M.inv * AssembledVector(v1v2)
    r_thunk = assemble(r_lambda)
    r.assign(v3 + r_thunk)

    return r

SmatA = generate_schur_matrix_method_A()
SmatB = generate_schur_matrix_method_B()
va, sa, qa = np.linalg.svd(SmatA.M.values)
vb, sb, qb = np.linalg.svd(SmatB.M.values)
