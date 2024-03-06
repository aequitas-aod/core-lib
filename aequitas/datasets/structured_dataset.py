import aif360.datasets as datasets
import pandas as pd
from aequitas.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from abc import ABC, abstractmethod


class StructuredDataset(datasets.StructuredDataset, ABC):
    def __init__(self,
                 imputation_strategy: MissingValuesImputationStrategy,
                 **kwargs: object,
                 ):
        self.__strategy = imputation_strategy
        self._df = kwargs.get('df')
        self._df = self.__strategy.custom_preprocessing(df=self._df)
        kwargs["df"] = self._df
        super(StructuredDataset, self).__init__(**kwargs)

    @property
    def strategy(self):
        return self.__strategy

    @property
    @abstractmethod
    def metrics(self):
        raise NotImplementedError

    def __custom_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Applying custom preprocessing")
        return self.strategy.custom_preprocessing(df)
