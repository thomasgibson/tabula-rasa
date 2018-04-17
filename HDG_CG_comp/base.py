from firedrake import *
from firedrake.utils import cached_property
from firedrake.petsc import PETSc

from abc import ABCMeta, abstractproperty, abstractmethod


class Problem(object):

    __metaclass__ = ABCMeta

    def __init__(self, N=None, degree=None):
        super(Problem, self).__init__()

        args, _ = self.argparser().parse_known_args()
        if args.help:
            import sys
            self.argparser().print_help()
            sys.exit(0)

        self.degree = degree or args.degree
        self.N = N or args.size
        self.args = args

    def reinitialize(self, degree=None, size=None):
        if degree is None:
            degree = self.degree

        if size is None:
            size = self.N

        degree_changed = degree != self.degree
        mesh_changed = size != self.N

        if not (degree_changed or mesh_changed):
            return

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

    @abstractmethod
    def argparser():
        pass

    @abstractproperty
    def output(self):
        pass

    @abstractproperty
    def err(self):
        pass
