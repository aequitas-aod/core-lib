import eagerpy
import eagerpy.types


def scalar_of_same_type(x: eagerpy.Tensor, value) -> eagerpy.Tensor:
    scalar_of_type(type(x), value)


def scalar_of_type(x: eagerpy.TensorType, value) -> eagerpy.Tensor:
    return eagerpy.full(x, (1,), value)


def unique(x: eagerpy.Tensor) -> eagerpy.Tensor:
    type_of_x = type(x)
    distinct_values = {scalar_of_type(type_of_x, value) for value in x.flatten()}
    return eagerpy.concatenate(distinct_values, axis=0)
