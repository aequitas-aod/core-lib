from aequitas import isinstance
from aequitas.core import Scalar
import numpy
import typing
from dataclasses import dataclass


class Condition(typing.Callable[[numpy.array], numpy.array]):
    """A type for functions that check a condition on a vector

    This type is used to represent conditions on vectors, e.g. `x > 0` or `x == 1`. The condition is checked by calling
    the object, passing the vector to check as an argument, and obtaining a boolean vector as a result.

    Also see: https://docs.python.org/3/library/typing.html#typing.Callable
    """
    def __call__(self, x: numpy.array) -> numpy.array:
        """Checks the condition on a given vector

        :param x: the vector on which the condition is checked
        :return: a boolean vector, where `True` values indicate that the condition is satisfied
        """
        raise NotImplementedError()

    def negate(self) -> 'Condition':
        """Returns another condition that is the negation of the current one"""
        raise NotImplementedError()

    @staticmethod
    def ensure(value: typing.Any) -> 'Condition':
        if isinstance(value, Condition):
            return value
        elif isinstance(value, Scalar):
            return Condition.constant(value)
        elif callable(value):
            return Condition.function(value)
        elif isinstance(value, tuple):
            return Condition.range(*value)
        else:
            raise TypeError("The value must be a scalar, a function, or a tuple")

    @staticmethod
    def constant(value: Scalar) -> 'Constant':
        if not isinstance(value, Scalar):
            raise TypeError("The value must be a scalar")
        return Constant(value)

    @staticmethod
    def function(function: typing.Callable[[numpy.array], numpy.array]) -> 'Function':
        if not callable(function):
            raise TypeError("The function must be callable")
        return Function(function)

    @staticmethod
    def range(lower: Scalar, upper: Scalar, include_lower: bool = True, include_upper: bool = True) -> 'Range':
        if not isinstance(lower, Scalar) or not isinstance(upper, Scalar):
            raise TypeError("The lower and upper bounds must be scalars")
        return Range(lower, upper, include_lower, include_upper)


ConditionLike = typing.Union[Condition, Scalar, typing.Callable[[numpy.array], numpy.array], typing.Tuple]


@dataclass(frozen=True)
class Function(Condition):
    function: typing.Callable[[numpy.array], numpy.array]

    def __call__(self, x: numpy.array) -> numpy.array:
        result = self.function(x)
        if not isinstance(result, numpy.ndarray) or not result.dtype == bool:
            raise TypeError("The function must return a NumPy array or booleans")
        return result

    def negate(self) -> Condition:
        return Function(lambda x: not self.function(x))


@dataclass(frozen=True)
class Constant(Condition):
    value: Scalar

    def __call__(self, x: numpy.array) -> numpy.array:
        return x == self.value

    def negate(self) -> Condition:
        return Function(lambda x: x != self.value)


@dataclass(frozen=True)
class Range(Condition):
    lower: Scalar
    upper: Scalar
    include_lower: bool = True
    include_upper: bool = True

    def inside_lower_bound(self, x: numpy.array) -> numpy.array:
        f = numpy.greater_equal if self.include_lower else numpy.greater
        return f(x, self.lower)

    def inside_upper_bound(self, x: numpy.array) -> numpy.array:
        f = numpy.less_equal if self.include_upper else numpy.less
        return f(x, self.upper)

    # def outside_lower_bound(self, x: numpy.array) -> numpy.array:
    #     f = numpy.greater if self.include_lower else numpy.greater_equal
    #     return f(x, self.lower)
    #
    # def outside_upper_bound(self, x: numpy.array) -> numpy.array:
    #     f = numpy.less if self.include_upper else numpy.less_equal
    #     return f(x, self.upper)

    def __call__(self, x: numpy.array) -> numpy.array:
        return numpy.logical_and(
            self.inside_upper_bound(x), self.inside_lower_bound(x)
        )

    def negate(self) -> Condition:
        return Function(lambda x: numpy.logical_not(self(x)))
