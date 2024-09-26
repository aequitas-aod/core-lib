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


def get_mask(shape, nan_perc):
    mask = np.ones(shape[0] * shape[1], dtype=int)
    num_masked = int(shape[0] * shape[1] * nan_perc)
    mask[:shape[0] * shape[1] - num_masked] = 0
    np.random.shuffle(mask)
    mask = mask.astype(bool)
    mask = np.reshape(mask, shape)
    return mask


def add_nans(data_to_mask, nan_perc=0.1):
    mask = get_mask(data_to_mask.shape, nan_perc)
    masked_arr = np.ma.masked_array(data=data_to_mask, mask=mask)
    masked_arr = masked_arr.filled(np.nan)
    return masked_arr


def generate_dataframe_with_preds(df: pd.DataFrame, p=0.7):
    # utility function to call when testing the ClassificationMetric class
    new_df = df.copy()
    new_df["label"] = np.array([x * bernoulli(p) for x in df["label"]])
    return new_df


def generate_binary_label_dataframe(rows: int = 1000, num_features: int = 2, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr], axis=1)
    if nans:
        data = add_nans(data)
    labels = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([data] + [labels], axis=1)

    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))

    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_skewed_binary_label_dataframe(rows: int = 1000, num_features: int = 2, p: float = 0.8,
                                           nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr], axis=1)
    if nans:
        data = add_nans(data)
    labels = np.array([bernoulli(p)[0] * x for x in prot_attr]).round().astype(int)
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_binary_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [scores], axis=1)
    if nans:
        data = add_nans(data)
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_skewed_binary_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2,
                                                       p: float = 0.8, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = np.array([x * bernoulli(p)[0] for x in prot_attr]) + random.uniform(0, 1, size=(rows, 1))
    # normalise scores
    scores = minmax_scale(scores)
    data = np.concatenate([features] + [prot_attr] + [scores], axis=1)
    if nans:
        data = add_nans(data)
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_multi_label_dataframe(rows: int = 1000, num_features: int = 2, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr], axis=1)
    if nans:
        data = add_nans(data)
    labels = np.random.randint(5, size=(rows, 1))
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_multi_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [scores], axis=1)
    if nans:
        data = add_nans(data)
    labels = np.random.randint(5, size=(rows, 1))
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_skewed_multi_label_dataframe(rows: int = 1000, num_features: int = 2,
                                          p: float = 0.8, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr], axis=1)
    if nans:
        data = add_nans(data)
    labels = np.array(
        [random.choice([0., 1., 2.]) if x * bernoulli(p)[0] else random.choice([3., 4.]) for x in prot_attr])
    labels = np.expand_dims(labels, axis=1)
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'label'])


def generate_skewed_multi_label_dataframe_with_scores(rows: int = 1000, num_features: int = 2,
                                                      p: float = 0.8, nans=False) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, num_features))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = np.array([x * bernoulli(p)[0] for x in prot_attr]) + random.uniform(0, 1, size=(rows, 1))
    # normalise scores
    scores = minmax_scale(scores)
    data = np.concatenate([features] + [prot_attr] + [scores], axis=1)
    if nans:
        data = add_nans(data)
    labels = np.array([random.choice([0., 1., 2.0]) if x >= 0.5 else random.choice([3., 4.]) for x in scores])
    labels = np.expand_dims(labels, axis=1)
    data = np.concatenate([data] + [labels], axis=1)
    feature_names = []
    for i in range(num_features):
        feature_names.append("feat_" + str(i + 1))
    return pd.DataFrame(data, columns=feature_names + ['prot_attr', 'score', 'label'])


def generate_skewed_regression_dataset() -> pd.DataFrame:
    # taken from reranking example in the aif360 code
    return pd.DataFrame(
        [['r', 100], ['r', 90], ['r', 85], ['r', 70], ['b', 70], ['b', 60], ['b', 50], ['b', 40], ['b', 30], ['r', 20]],
        columns=['color', 'score'])
