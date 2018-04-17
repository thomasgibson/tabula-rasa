from firedrake import *
from firedrake.utils import cached_property

from abc import ABCMeta, abstractproperty


class Problem(object):

    __metaclass__ = ABCMeta

    def __init__(self, N=None, degree=None):
        super(Problem, self).__init__()

        self.degree = degree
        self.N = N

    @property
    def comm(self):
        return self.mesh.comm

    @cached_property
    def mesh(self):
        return UnitCubeMesh(self.N, self.N, self.N)

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def function_space(self):
        pass

    @cached_property
    def u(self):
        return Function(self.function_space, name="Solution")

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
        x, y, z = SpatialCoordinate(self.mesh)
        return exp(sin(pi*x)*sin(pi*y)*sin(pi*z))

    def solver(self, parameters=None):

        problem = LinearVariationalProblem(self.a, self.L, self.u,
                                           bcs=self.bcs)
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
