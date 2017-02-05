"""A generic abstract node class for the Slate language."""

from __future__ import absolute_import, print_function, division
from six import with_metaclass

from abc import ABCMeta, abstractproperty, abstractmethod

from firedrake.utils import cached_property


class TensorBase(with_metaclass(ABCMeta)):
    """An abstract Slate node class.

       This class is not meant to be modified unless fundamental
       changes are being made to the Slate language.
    """

    id = 0

    def __init__(self):
        """Constructor for the TensorBase abstract class."""
        self._kernels = None
        self.id = TensorBase.id
        TensorBase.id += 1

    @abstractmethod
    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""

    @cached_property
    def shapes(self):
        """Computes the internal shape information of its components.
        This is particularly useful to know if the tensor comes from a
        mixed form.
        """
        shapes = {}
        for i, arg in enumerate(self.arguments()):
            shapes[i] = tuple(fs.fiat_element.space_dimension() * fs.dim
                              for fs in arg.function_space())
        return shapes

    @cached_property
    def shape(self):
        """Computes the shape information of the local tensor."""
        return tuple(sum(shapelist) for shapelist in self.shapes.values())

    @cached_property
    def rank(self):
        """Returns the rank information of the tensor object."""
        return len(self.arguments())

    @abstractmethod
    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""

    @abstractmethod
    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found.
        """
        domains = self.ufl_domains()
        assert all(domain == domains[0] for domain in domains), (
            "All integrals must share the same domain of integration."
        )
        return domains[0]

    @abstractmethod
    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """

    @abstractmethod
    def _output_string(self):
        """Creates an expression-aware string representation of the tensor.

        This is used when calling the `__str__` method on
        TensorBase objects. This function facilitates pretty printing.
        """

    def __str__(self):
        """Returns a string representation."""
        return self._output_string(self.prec)

    @abstractproperty
    def _key(self):
        """Returns a key for hash and equality.

        This is used to generate a unique id associated with the
        TensorBase object.
        """

    def __eq__(self, other):
        """Determines whether two TensorBase objects are equal using their
        associated keys.
        """
        return self._key == other._key

    def __ne__(self, other):
        return not self.__eq__(other)

    @cached_property
    def _hash_id(self):
        """Returns a hash id."""
        return hash(self._key)

    def __hash__(self):
        """Generates a hash for the TensorBase object."""
        return self._hash_id
