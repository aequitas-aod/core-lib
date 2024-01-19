import numpy as np
import numpy.random as random
import typing

SEED = 1
random.seed(SEED)


def uniform_binary_dataset(rows: int = 1000, columns: int = 2) -> np.array:
    return random.uniform(0, 1, size=(rows, columns)).round().astype(int)


def uniform_binary_dataset_gt(rows: int = 1000, columns: int = 2) -> np.array:
    xs = uniform_binary_dataset(rows, 1)
    labels = uniform_binary_dataset(rows, 1)
    noise = random.choice([0, 1], p=[0.8, 0.2], size=(rows, 1))
    preds = abs(labels - noise)
    if columns > 2:
        data = []
        for _ in range(columns - 2):
            data.append(uniform_binary_dataset(rows, 1))
        data = np.concatenate(data, axis=1)
        return np.concatenate((data, xs, labels, preds), axis=1)
    else:
        return np.concatenate((xs, labels, preds), axis=1)


def skewed_binary_dataset_gt(rows: int = 1000, columns: int = 2, p: float = 0.8) -> np.array:
    xs = uniform_binary_dataset(rows, 1)
    preds = np.array([bernoulli(p)[0] * x for x in xs])
    labels = uniform_binary_dataset(rows, 1)
    if columns > 2:
        data = []
        for _ in range(columns - 2):
            data.append(uniform_binary_dataset(rows, 1))
        data = np.concatenate(data, axis=1)
        return np.concatenate((data, xs, labels, preds), axis=1)
    else:
        return np.concatenate((xs, labels, preds), axis=1)


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


def uniform_binary_dataset_probs(rows: int = 1000, columns: int = 2) -> np.array:
    probs = random.uniform(0, 1, (rows, 1))
    xs = uniform_binary_dataset(rows, columns)
    labels = uniform_binary_dataset(rows, 1)
    return np.concatenate((xs, probs, labels), axis=1)


def skewed_binary_dataset_probs(rows: int = 1000, columns: int = 2, p: float =
                               0.8) -> np.array:
    features = uniform_binary_dataset(rows, columns)
    xs = uniform_binary_dataset(rows, 1)

    probs = np.array([random.normal(0.3, 0.1, 1) if x == 0 else
                      random.normal(0.7, 0.1, 1) for x in xs])
    # labels = (probs >= 0.5).astype(int)
    labels = uniform_binary_dataset(rows, 1)

    return np.concatenate((features, xs, probs, labels), axis=1)
