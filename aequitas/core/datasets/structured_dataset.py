import aif360.datasets as datasets
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from abc import ABC, abstractmethod


class StructuredDataset(datasets.StructuredDataset, ABC):
    def __init__(self,
                 imputation_strategy: MissingValuesImputationStrategy,
                 **kwargs):
        df = kwargs.get('df')
        df = imputation_strategy.custom_preprocessing(df=df)
        kwargs["df"] = df
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def metrics(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def scores_metrics(self):
        raise NotImplementedError
