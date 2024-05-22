from abc import ABC, abstractmethod
import aequitas
import pandas
import pandas as pd


class MissingValuesImputationStrategy(ABC):
    def __call__(self, df: pandas.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implement this method in a subclass")


class MeanImputationStrategy(MissingValuesImputationStrategy):
    def __call__(self, df: pandas.DataFrame) -> pd.DataFrame:
        return df.fillna(df.mean())
    

class DoNothingImputationStrategy(MissingValuesImputationStrategy):
    def __call__(self, df: pandas.DataFrame) -> pd.DataFrame:
        return df


class MCMCImputationStrategy(MissingValuesImputationStrategy):
    def __call__(self, df: pandas.DataFrame) -> pd.DataFrame:
        import warnings
        warnings.warn("Unimplemented MCMC imputation strategy: doing nothig")
        return df


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)