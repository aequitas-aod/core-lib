import numpy as np
import numpy.random as random
import pandas as pd


def generate_binary_label_dataframe(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    prot_attr = np.random.randint(2, size=(rows, 1))
    labels = np.random.randint(2, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'prot_attr', 'label'])


def generate_binary_label_dataframe_with_scores(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'prot_attr', 'score', 'label'])


def generate_multi_label_dataframe(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    prot_attr = np.random.randint(2, size=(rows, 1))
    labels = np.random.randint(5, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    res = pd.DataFrame(data, columns=['feat', 'prot_attr', 'label'])
    return res


def generate_multi_label_dataframe_with_scores(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    prot_attr = np.random.randint(2, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    labels = np.random.randint(5, size=(rows, 1))
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'prot_attr', 'score', 'label'])
