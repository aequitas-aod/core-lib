import typing
import numpy as np
import numpy.random as random
import pandas as pd
from sklearn.preprocessing import minmax_scale

SEED = 1
random.seed(SEED)


def bernoulli(p: float, size: typing.Tuple[int, int] = (1,)) -> np.array:
    assert 0 <= p <= 1, "p must be in [0, 1]"
    return (random.uniform(0, 1, size=size) < p).astype(int)


def generate_binary_label_dataframe(rows: int = 1000, num_features: int = 2) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    labels = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_skewed_binary_label_dataframe(rows: int = 1000, num_features: int = 2, p: float = 0.8) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    labels = np.array([bernoulli(p)[0] * x for x in prot_attr]).round().astype(int)
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_binary_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_skewed_binary_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2,
                                                       p: float = 0.8) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = np.array([x * bernoulli(p)[0] for x in prot_attr]) + random.uniform(0, 1, size=(rows, 1))
    # normalise scores
    scores = minmax_scale(scores)
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_multi_label_dataframe(rows: int = 1000, num_features: int = 2) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    labels = np.random.randint(5, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_multi_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    labels = np.random.randint(5, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_skewed_multi_label_dataframe(rows: int = 1000, num_features: int = 2,
                                          p: float = 0.8) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    labels = np.array([random.choice([0., 1., 2.]) if x*bernoulli(p)[0] else random.choice([3., 4.]) for x in prot_attr])
    labels = np.expand_dims(labels, axis=1)
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_skewed_multi_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2,
                                                      p: float = 0.8) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = np.array([x * bernoulli(p)[0] for x in prot_attr]) + random.uniform(0, 1, size=(rows, 1))
    # normalise scores
    scores = minmax_scale(scores)
    labels = np.array([random.choice([0., 1., 2.0]) if x >= 0.5 else random.choice([3., 4.]) for x in scores])
    labels = np.expand_dims(labels, axis=1)
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])
