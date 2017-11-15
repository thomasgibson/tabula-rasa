from firedrake import *
from firedrake.utils import cached_property


class HelmholtzProblem(object):

    name = "Helmholtz"

    parameter_names = ["scpc_hypre", "hypre"]

    def __init__(self, mesh_size=None, degree=None):

        super(object, self).__init__()
        self.degree = degree
        self.mesh_size = mesh_size

    def re_initialize(self, degree=None, mesh_size=None):
        if degree is None:
            degree = self.degree
        if mesh_size is None:
            mesh_size = self.mesh_size

        degree_changed = degree != self.degree
        mesh_changed = mesh_size != self.mesh_size

        if not (degree_changed or mesh_changed):
            return

        for attr in ["function_space", "source",
                     "u", "F", "bcs", "Jp", "output"]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

        if mesh_changed:
            try:
                delattr(self, "mesh")
            except AttributeError:
                pass
        self.degree = degree
        self.mesh_size = mesh_size

    @property
    def hypre(self):
        return {"snes_type": "ksponly",
                "ksp_type": "cg",
                "ksp_rtol": 1e-8,
                "ksp_monitor": True,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "pc_hypre_boomeramg_no_CF": True,
                "pc_hypre_boomeramg_coarsen_type": "HMIS",
                "pc_hypre_boomeramg_interp_type": "ext+i",
                "pc_hypre_boomeramg_P_max": 4,
                "pc_hypre_boomeramg_agg_nl": 1}

    @property
    def scpc_hypre(self):
        return {"snes_type": "ksponly",
                "mat_type": "matfree",
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.CGStaticCondensationPC",
                "static_condensation": {"ksp_type": "cg",
                                        "ksp_rtol": 1e-8,
                                        "ksp_monitor": True,
                                        "pc_type": "hypre",
                                        "pc_hypre_type": "boomeramg",
                                        "pc_hypre_boomeramg_no_CF": True,
                                        "pc_hypre_boomeramg_coarsen_type": "HMIS",
                                        "pc_hypre_boomeramg_interp_type": "ext+i",
                                        "pc_hypre_boomeramg_P_max": 4,
                                        "pc_hypre_boomeramg_agg_nl": 1}}

    @cached_property
    def mesh(self):
        return UnitCubeMesh(self.mesh_size,
                            self.mesh_size,
                            self.mesh_size)

    @property
    def comm(self):
        return self.mesh.comm

    @cached_property
    def function_space(self):
        return FunctionSpace(self.mesh, "CG", self.degree)

    @cached_property
    def source(self):
        x, y, z = SpatialCoordinate(self.mesh)
        f = (1 + 108*pi*pi)*cos(6*pi*x)*cos(6*pi*y)*cos(6*pi*z)
        source = Function(self.function_space, name="source")
        return source.interpolate(f)

    @cached_property
    def u(self):
        return Function(self.function_space, name="solution")

    @cached_property
    def F(self):
        v = TestFunction(self.function_space)
        f = self.source
        a = inner(grad(v), grad(self.u))*dx + v*self.u*dx
        L = inner(v, f)*dx
        return a - L

    @cached_property
    def bcs(self):
        return None

    @cached_property
    def Jp(self):
        return None

    def solver(self, parameters=None):
        problem = NonlinearVariationalProblem(self.F, self.u, bcs=self.bcs,
                                              Jp=self.Jp)
        solver = NonlinearVariationalSolver(problem,
                                            solver_parameters=parameters)
        return solver

    @cached_property
    def output(self):
        return (self.u,)
