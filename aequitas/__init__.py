import logging
import types
import typing
import numpy


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aequitas')

Scalar = typing.Union[int, float, bool, complex, str, numpy.generic]


EPSILON: float = 1e-9


def is_zero(x: Scalar) -> bool:
    return abs(x) < EPSILON


__py_isinstance = isinstance


def isinstance(obj, cls):
    if hasattr(cls, '__args__') and __py_isinstance(cls.__args__, tuple):
        return any(__py_isinstance(obj, t) for t in cls.__args__)
    return __py_isinstance(obj, cls)


# let this be the last line of this file
logger.debug("Module %s correctly loaded", __name__)
