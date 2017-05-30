from __future__ import absolute_import, print_function, division

from firedrake import *
from firedrake.slate.preconditioners import create_schur_nullspace
import numpy as np

mesh = UnitIcosahedralSphereMesh(2)
mesh.init_cell_orientations(SpatialCoordinate(mesh))

n = FacetNormal(mesh)

V = FunctionSpace(mesh, "RT", 1)
Q = FunctionSpace(mesh, "DG", 0)

W = V*Q
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
a = (inner(sigma, tau) + div(sigma)*v + div(tau)*u)*dx

W_d = FunctionSpace(mesh,
                    MixedElement([BrokenElement(Vi.ufl_element())
                                  for Vi in W]))

atilde = Tensor(replace(a, dict(zip(a.arguments(),
                                    (TestFunction(W_d),
                                     TrialFunction(W_d))))))

Vt = FunctionSpace(mesh, "HDiv Trace", 0)
gamma = TestFunction(Vt)

sigma, _ = TrialFunctions(W_d)

K = Tensor(gamma('+') * dot(sigma, n) * dS)
k = assemble(K*K.T)
print(k.M.values)

A = assemble(a, mat_type="aij")
nullspace = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
nullspace._build_monolithic_basis()
A.petscmat.setNullSpace(nullspace._nullspace)

Snullsp = create_schur_nullspace(A.petscmat, -(K * K.T).inv * K * atilde,
                                 W, W_d, Vt,
                                 COMM_WORLD)

v = Snullsp.getVecs()[0]
print(
    "Computed nullspace of S (min, max, norm)", v.array_r.min(), v.array_r.max(), v.norm()
)

S = K * atilde.inv * K.T

u, s, v = np.linalg.svd(assemble(S, mat_type="aij").M.values)

singular_vector = v[-1]
print(
    "Actual nullspace of S (min, max, norm)", singular_vector.min(), singular_vector.max(), np.linalg.norm(singular_vector)
)

u, s, v = np.linalg.svd(A.M.handle[:, :])

offset = V.dof_dset.size

singular_vector = v[-1][offset:]
print(
    "Nullspace of original operator (min, max, norm)", singular_vector.min(), singular_vector.max(), np.linalg.norm(singular_vector)
)
