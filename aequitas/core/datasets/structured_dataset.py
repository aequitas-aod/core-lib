import aif360.datasets as datasets
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
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
    @abstractmethod
    def metrics(self):
        raise NotImplementedError
