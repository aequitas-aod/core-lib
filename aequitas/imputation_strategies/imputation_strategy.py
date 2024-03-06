from abc import ABC, abstractmethod
import pandas
import pandas as pd


class MissingValuesImputationStrategy(ABC):
    @abstractmethod
    def custom_preprocessing(self, df: pandas.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
