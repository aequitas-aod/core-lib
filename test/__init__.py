import numpy as np
import numpy.random as random
import pandas as pd


def generate_binary_label_dataframe(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    labels = random.uniform(0, 1, size=(rows, 1)).astype(int)
    data = np.concatenate([features] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'label'])

def generate_binary_label_dataframe_with_scores(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    scores = random.uniform(0, 1, size=(rows, 1))
    labels = (scores > 0.5).astype(int)
    data = np.concatenate([features] + [scores] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'scores', 'label'])

