from __future__ import absolute_import, print_function, division
import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return m


def test_facet_interior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh)
    f = project(as_vector([x, y]), DG)

    form = dot(f[0]*f[1]*u, n)*dS

    A = assemble(Tensor(form)).dat.data
    ref = assemble(jump(f[0]*f[1]*u, n=n)*dS).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh)
    f = project(as_vector([x, y]), DG)

    form = dot(f[0]*f[1]*u, n)*ds

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_total_facet_int_ext(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh)
    f = project(as_vector([x, y]), DG)
    g = project(as_vector([-y, -x]), DG)

    l_ds = dot(f[0]*f[1]*u, n)*ds
    l_dS = dot(g[0]*g[1]*u, n)*dS
    ref_form = jump(g[0]*g[1]*u, n=n)*dS + dot(f[0]*f[1]*u, n)*ds

    A = assemble(Tensor(l_ds + l_dS)).dat.data
    ref = assemble(ref_form).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_trace_coefficient(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)
    T = FunctionSpace(mesh, "HDiv Trace", 1)

    x, y = SpatialCoordinate(mesh)
    f = interpolate(2*x + 2*y, T)
    A = assemble(Tensor(f('+')*dot(u, n)*dS)).dat.data
    ref = assemble(jump(f*u, n)*dS).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
