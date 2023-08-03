import aequitas
import typing
import numpy


# keep this line at the top of this file
__all__ = ['Scalar', 'DEFAULT_EPSILON', 'is_zero']


Scalar = typing.Union[int, float, bool, complex, str, numpy.generic]
"""General type for scalars (numbers, strings, or NumPy scalars)

Also see: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""


DEFAULT_EPSILON: float = 1e-9
"""Default threshold for floating-point comparisons"""


def is_zero(x: Scalar, epsilon: float = DEFAULT_EPSILON) -> bool:
    """Checks whether `x` is zero, up to a given threshold

    :param x: the scalar to check
    :param epsilon: the threshold for the comparison (defaults to `DEFAULT_EPSILON`)
    """

    return abs(x) < epsilon


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
