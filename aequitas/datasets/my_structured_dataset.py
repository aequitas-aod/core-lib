from aequitas.datasets.structured_dataset import StructuredDataset
from aequitas.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
import pandas as pd


class MyStructuredDataset(StructuredDataset):
    @property
    def metrics(self):
        print("DEBUG:instantiate specific metric class")
        return 1

    def __init__(self,
                 **kwargs):
        super(MyStructuredDataset, self).__init__(**kwargs)
