import unittest
from test import binary_label_dataset
from aequitas.datasets.my_structured_dataset import MyStructuredDataset
from aequitas.imputation_strategies.mcmc_imputation_strategy import MCMCImputationStrategy


class TestClasses(unittest.TestCase):

    def test_strategy_pattern(self):
        df = binary_label_dataset()
        strategy = MCMCImputationStrategy()
        ds = MyStructuredDataset(imputation_strategy=strategy, df=df, label_names=['label'],
                                 protected_attribute_names=['feat'])
        self.assertTrue(ds is not None)


if __name__ == '__main__':
    unittest.main()
