import numpy as np
import numpy.random as random
import typing


SEED = 1
random.seed(SEED)


def uniform_binary_dataset(rows: int = 1000, columns: int = 2) -> np.array:
    return random.uniform(0, 1, size=(rows, columns)).round().astype(int)


def bernoulli(p: float, size: typing.Tuple[int, int] = (1,)) -> np.array:
    assert 0 <= p <= 1, "p must be in [0, 1]"
    return (random.uniform(0, 1, size=size) < p).astype(int)


def skewed_binary_dataset(rows: int = 1000, columns: int = 2, p: float = 0.8) -> np.array:
    xs = uniform_binary_dataset(rows, 1)
    cols = []
    for _ in range(columns - 1):
        ys = np.array([bernoulli(p)[0] * x for x in xs])
        cols.append(ys)
    return np.concatenate([xs] + cols, axis=1)
