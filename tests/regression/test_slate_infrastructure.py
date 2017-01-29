from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *


@pytest.fixture(scope='module', params=[interval, triangle, quadrilateral])
def mesh(request):
    """Generate a mesh according to the cell provided."""
    cell = request.param
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell not recognized" % cell)


@pytest.fixture(scope='module', params=['cg1', 'cg2', 'dg0', 'dg1'])
def function_space(request, mesh):
    """Generates function spaces for testing SLATE tensor assembly."""
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg1': cg1,
            'cg2': cg2,
            'dg0': dg0,
            'dg1': dg1}[request.param]


@pytest.fixture
def mass(function_space):
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return Tensor(u * v * dx)


@pytest.fixture
def stiffness(function_space):
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return Tensor(inner(grad(u), grad(v)) * dx)


@pytest.fixture
def load(function_space):
    f = Function(function_space)
    f.interpolate(Expression("cos(x[0]*pi*2)"))
    v = TestFunction(function_space)
    return Tensor(f * v * dx)


@pytest.fixture
def boundary_load(function_space):
    f = Function(function_space)
    f.interpolate(Expression("cos(x[1]*pi*2)"))
    v = TestFunction(function_space)
    return Tensor(f * v * ds)


@pytest.fixture
def zero_rank_tensor(function_space):
    c = Function(function_space)
    c.interpolate(Expression("x[0]*x[1]"))
    return Tensor(c * dx)


def test_arguments(mass, stiffness, load, boundary_load, zero_rank_tensor):
    S = zero_rank_tensor
    M = mass
    N = stiffness
    F = load
    G = boundary_load

    c, = S.form.coefficients()
    f, = F.form.coefficients()
    g, = G.form.coefficients()
    v, u = M.form.arguments()

    assert len(N.arguments()) == N.rank
    assert len(M.arguments()) == M.rank
    assert N.arguments() == (v, u)
    assert len(F.arguments()) == F.rank
    assert len(G.arguments()) == G.rank
    assert F.arguments() == (v,)
    assert G.arguments() == F.arguments()
    assert len(S.arguments()) == S.rank
    assert S.arguments() == ()
    assert (M.T).arguments() == (u, v)
    assert (N.inv).arguments() == (u, v)
    assert (N.T + M.inv).arguments() == (u, v)
    assert (F.T).arguments() == (v,)
    assert (F.T + G.T).arguments() == (v,)
    assert (M*F).arguments() == (v,)
    assert (N*G).arguments() == (v,)
    assert ((M + N) * (F - G)).arguments() == (v,)

    assert Tensor(v * dx).arguments() == (v,)
    assert (Tensor(v * dx) + Tensor(f * v * ds)).arguments() == (v,)
    assert (M + N).arguments() == (v, u)
    assert (Tensor((f * v) * u * dx) + Tensor((u * 3) * (v / 2) * dx)).arguments() == (v, u)
    assert (G - F).arguments() == (v,)


def test_coefficients(mass, stiffness, load, boundary_load, zero_rank_tensor):
    S = zero_rank_tensor
    M = mass
    N = stiffness
    F = load
    G = boundary_load

    c, = S.form.coefficients()
    f, = F.form.coefficients()
    g, = G.form.coefficients()
    v, u = M.form.arguments()

    assert S.coefficients() == (c,)
    assert F.coefficients() == (f,)
    assert G.coefficients() == (g,)
    assert (M*F).coefficients() == (f,)
    assert (N*G).coefficients() == (g,)
    assert (N*F + M*G).coefficients() == (f, g)
    assert (M.T).coefficients() == ()
    assert (M.inv).coefficients() == ()
    assert (M.T + N.inv).coefficients() == ()
    assert (F.T).coefficients() == (f,)
    assert (G.T).coefficients() == (g,)
    assert (F + G).coefficients() == (f, g)
    assert (F.T - G.T).coefficients() == (f, g)

    assert Tensor(f * dx).coefficients() == (f,)
    assert (Tensor(f * dx) + Tensor(f * ds)).coefficients() == (f,)
    assert (Tensor(f * dx) + Tensor(g * dS)).coefficients() == (f, g)
    assert Tensor(f * v * dx).coefficients() == (f,)
    assert (Tensor(f * v * ds) + Tensor(f * v * dS)).coefficients() == (f,)
    assert (Tensor(f * v * dx) + Tensor(g * v * ds)).coefficients() == (f, g)
    assert Tensor(f * u * v * dx).coefficients() == (f,)
    assert (Tensor(f * u * v * dx) + Tensor(f * inner(grad(u), grad(v)) * dx)).coefficients() == (f,)
    assert (Tensor(f * u * v * dx) + Tensor(g * inner(grad(u), grad(v)) * dx)).coefficients() == (f, g)


def test_integral_information(mass, stiffness, load, boundary_load, zero_rank_tensor):
    S = zero_rank_tensor
    M = mass
    N = stiffness
    F = load
    G = boundary_load

    # Checks the generated information of the tensor agrees with the original
    # data directly in its associated `ufl.Form` object
    assert S.ufl_domain() == S.form.ufl_domain()
    assert M.ufl_domain() == M.form.ufl_domain()
    assert N.ufl_domain() == N.form.ufl_domain()
    assert F.ufl_domain() == F.form.ufl_domain()
    assert G.ufl_domain() == G.form.ufl_domain()
    assert M.inv.ufl_domain() == M.form.ufl_domain()
    assert M.T.ufl_domain() == M.form.ufl_domain()
    assert (-N).ufl_domain() == N.form.ufl_domain()
    assert (F + G).ufl_domain() == (F.form + G.form).ufl_domain()
    assert (M + N).ufl_domain() == (M.form + N.form).ufl_domain()

    assert S.subdomain_data() == S.form.subdomain_data()
    assert N.subdomain_data() == N.form.subdomain_data()
    assert M.subdomain_data() == M.form.subdomain_data()
    assert F.subdomain_data() == F.form.subdomain_data()
    assert N.inv.subdomain_data() == N.form.subdomain_data()
    assert (-M).subdomain_data() == M.form.subdomain_data()
    assert (M + N).T.subdomain_data() == (M.form + N.form).subdomain_data()
    assert (F + G).subdomain_data() == (F.form + G.form).subdomain_data()


def test_equality_relations(function_space):
    # Small test to check hash functions
    V = function_space
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)

    A = Tensor(u * v * dx)
    B = Tensor(inner(grad(u), grad(v)) * dx)

    assert A == Tensor(u * v * dx)
    assert B != A
    assert B * f != A * f
    assert A + B == Tensor(u * v * dx) + Tensor(inner(grad(u), grad(v)) * dx)
    assert A*B != B*A
    assert B.T != B.inv
    assert A != -A


def test_illegal_add_sub():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = Tensor(u * v * dx)
    b = Tensor(v * dx)
    c = Function(V)
    c.interpolate(Expression("1"))
    s = Tensor(c * dx)

    with pytest.raises(ValueError):
        A + b

    with pytest.raises(ValueError):
        s + b

    with pytest.raises(ValueError):
        b - A

    with pytest.raises(ValueError):
        A - s


def test_ops_NotImplementedError():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    A = Tensor(u * v * dx)

    with pytest.raises(NotImplementedError):
        A + f

    with pytest.raises(NotImplementedError):
        f + A

    with pytest.raises(NotImplementedError):
        A - f

    with pytest.raises(NotImplementedError):
        f - A

    with pytest.raises(NotImplementedError):
        f * A


def test_illegal_mul():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    W = FunctionSpace(mesh, "CG", 2)
    w = TrialFunction(W)
    x = TestFunction(W)

    A = Tensor(u * v * dx)
    B = Tensor(w * x * dx)

    with pytest.raises(ValueError):
        B * A

    with pytest.raises(ValueError):
        A * B


def test_illegal_inverse():
    mesh = UnitSquareMesh(1, 1)
    RT = FunctionSpace(mesh, "RT", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    sigma = TrialFunction(RT)
    v = TestFunction(DG)
    A = Tensor(v * div(sigma) * dx)
    with pytest.raises(AssertionError):
        A.inv


def test_illegal_compile():
    from firedrake.slate.slac import compile_expression as compile_slate
    V = FunctionSpace(UnitSquareMesh(1, 1), "CG", 1)
    v = TestFunction(V)
    form = v * dx
    with pytest.raises(ValueError):
        compile_slate(form)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
