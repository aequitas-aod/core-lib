import numpy as np
import numpy.random as random
import pandas as pd


def binary_label_dataset(rows: int = 1000) -> pd.DataFrame:
    features = random.uniform(0, 1, size=(rows, 1))
    labels = random.uniform(0, 1, size=(rows, 1)).astype(int)
    data = np.concatenate([features] + [labels], axis=1)
    return pd.DataFrame(data, columns=['feat', 'label'])
