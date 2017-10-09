from __future__ import absolute_import, division
from firedrake import *
import numpy as np


def helmholtz_mixed(x, V1, V2):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**x, 2**x)
    V1 = FunctionSpace(mesh, *V1, name="V")
    V2 = FunctionSpace(mesh, *V2, name="P")
    W = V1 * V2

    # Define variational problem
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(V2)

    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    a = (p*q - q*div(u) + lmbda*inner(v, u) + div(v)*p) * dx
    L = f*q*dx

    # Compute solution
    x = Function(W)

    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-14},
                                'use_reconstructor': True}}
    solve(a == L, x, solver_parameters=params)

    # Analytical solution
    f.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    u, p = x.split()
    err = sqrt(assemble(dot(p - f, p - f) * dx))
    return x, err

V1 = ('RT', 1)
V2 = ('DG', 0)
x, err = helmholtz_mixed(8, V1, V2)

print err
File("helmholtz_mixed.pvd").write(x.split()[0], x.split()[1])

l2errs = []
for i in range(1, 9):
    l2errs.append(helmholtz_mixed(i, V1, V2)[1])

l2errs = np.array(l2errs)
conv = np.log2(l2errs[:-1] / l2errs[1:])[-1]
print conv
