import numpy as np
import numpy.random as random
import pandas as pd


def generate_binary_label_dataframe(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    prot_attr = random.uniform(0, 1, size=(rows, 1)).astype(int)
    labels = random.uniform(0, 1, size=(rows, 1)).astype(int)
    data = np.concatenate([features] + [prot_attr] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'prot_attr', 'label'])

def generate_binary_label_dataframe_with_scores(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    prot_attr = random.uniform(0, 1, size=(rows, 1)).astype(int)
    scores = random.uniform(0, 1, size=(rows, 1))
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([features] + [prot_attr] + [scores] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'prot_attr', 'scores', 'label'])

