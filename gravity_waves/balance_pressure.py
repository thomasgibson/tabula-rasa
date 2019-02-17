from firedrake import *


__all__ = ['compute_balanced_pressure']


def compute_balanced_pressure(Vv, VDG, k, b0, p0, p_boundary, top=False):

    Vvd = FunctionSpace(Vv.mesh(), BrokenElement(Vv.ufl_element()))
    cell, _ = VDG.ufl_element().cell()._cells
    deg, _ = Vv.ufl_element().degree()
    DG = FiniteElement("DG", cell, deg)
    CG = FiniteElement("CG", interval, 1)
    Tv_ele = TensorProductElement(DG, CG)
    Tv = FunctionSpace(Vv.mesh(), Tv_ele)

    W = Vvd * VDG * Tv
    v, pp, lambdar = TrialFunctions(W)
    dv, dp, gammar = TestFunctions(W)

    n = FacetNormal(Vv.mesh())

    if top:
        bmeasure = ds_t
        tmeasure = ds_b
        tstring = "top"
    else:
        bmeasure = ds_b
        tmeasure = ds_t
        tstring = "bottom"

    arhs = -inner(dv, n)*p_boundary*bmeasure - b0*inner(dv, k)*dx
    alhs = (inner(v, dv)*dx -
            div(dv)*pp*dx +
            dp*div(v)*dx +
            lambdar('+')*jump(dv, n=n)*(dS_v + dS_h) +
            lambdar*dot(dv, n)*tmeasure +
            gammar('+')*jump(v, n=n)*(dS_v + dS_h) +
            gammar*dot(v, n)*tmeasure)

    w = Function(W)

    bcs = [DirichletBC(W.sub(2), Constant(0.0), tstring)]

    pproblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    params = {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': '0, 1',
        'condensed_field': {
            'ksp_type': 'cg',
            'pc_type': 'gamg',
            'ksp_rtol': 1e-13,
            'ksp_atol': 1e-13,
            'mg_levels': {
                'ksp_type': 'chebyshev',
                'ksp_chebyshev_esteig': None,
                'ksp_max_it': 3,
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }
    }

    psolver = LinearVariationalSolver(pproblem, solver_parameters=params)
    psolver.solve()
    _, p, _ = w.split()
    p0.assign(p)
