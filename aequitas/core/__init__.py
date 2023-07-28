import aequitas
import eagerpy
import eagerpy.types
import numpy
import typing
from ._eagerpy_ext import *


Tensor = typing.Union[
    typing.Iterable,
    eagerpy.Tensor,
    eagerpy.types.NativeTensor
]


def ensure_tensor(x: Tensor) -> eagerpy.Tensor:
    if isinstance(x, set):
        x = list(x)
    try:
        return eagerpy.astensor(x)
    except ValueError:
        return eagerpy.astensor(numpy.array(x))


def ensure_same_tensor_type(*xs: Tensor):
    types = {type(x) for x in xs}
    if len(types) > 1:
        raise TypeError("All tensors must have the same type")


def returning_tensor(f: typing.Callable[..., eagerpy.Tensor]) -> typing.Callable[..., Tensor]:
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> Tensor:
        return ensure_tensor(f(*args, **kwargs)).raw
    return wrapper


# let this be the last line of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
