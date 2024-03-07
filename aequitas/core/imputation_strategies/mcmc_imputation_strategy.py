from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
import pandas as pd


class MCMCImputationStrategy(MissingValuesImputationStrategy):
    def __init__(self):
        super(MCMCImputationStrategy, self).__init__()
    
    def custom_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
