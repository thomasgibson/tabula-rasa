from firedrake import *
from firedrake.utils import cached_property

from abc import ABCMeta, abstractproperty


class Problem(object):

    __metaclass__ = ABCMeta

    def __init__(self, N=None, degree=None, dimension=None,
                 quadrilaterals=False):
        super(Problem, self).__init__()

        self.degree = degree
        self.N = N
        self.dim = dimension
        self.quads = quadrilaterals

    @property
    def comm(self):
        return self.mesh.comm

    @cached_property
    def mesh(self):
        if self.dim == 2:
            return UnitSquareMesh(self.N, self.N, quadrilateral=self.quads)
        else:
            assert self.dim == 3
            if self.quads:
                base = UnitSquareMesh(self.N, self.N, quadrilateral=self.quads)
                return ExtrudedMesh(base, self.N, layer_height=1.0/self.N)
            else:
                return UnitCubeMesh(self.N, self.N, self.N)

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def function_space(self):
        pass

    @abstractproperty
    def u(self):
        pass

    @abstractproperty
    def a(self):
        pass

    @abstractproperty
    def L(self):
        pass

    @abstractproperty
    def bcs(self):
        pass

    @cached_property
    def analytic_solution(self):
        x = SpatialCoordinate(self.mesh)
        if self.dim == 2:
            return exp(sin(pi*x[0])*sin(pi*x[1]))
        else:
            assert self.dim == 3
            return exp(sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2]))

    def solver(self, parameters=None):

        # For the rebuilding of the Jacobian to record assembly time
        problem = LinearVariationalProblem(self.a, self.L, self.u,
                                           bcs=self.bcs,
                                           constant_jacobian=False)
        solver = LinearVariationalSolver(problem, solver_parameters=parameters)

        return solver

    @abstractproperty
    def output(self):
        pass

    @abstractproperty
    def err(self):
        pass

    @abstractproperty
    def true_err(self):
        pass

    @abstractproperty
    def sol(self):
        pass
