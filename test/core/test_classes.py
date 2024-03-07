import unittest
from test import binary_label_dataset
from test import binary_label_dataset_with_scores

from aequitas.core.datasets.create_dataset import CreateDataset
from aequitas.core.imputation_strategies.mcmc_imputation_strategy import MCMCImputationStrategy


class TestClasses(unittest.TestCase):

    def test_factory_pattern_bld(self):
        df = binary_label_dataset()
        strategy = MCMCImputationStrategy()
        cd = CreateDataset(dataset_type="binary label")
        ds = cd.create_dataset(imputation_strategy=strategy, df=df, label_names=['label'],
                               protected_attribute_names=['feat'])
        self.assertTrue(ds is not None)

    def test_factory_pattern_bld_scores(self):
        df = binary_label_dataset_with_scores()
        strategy = MCMCImputationStrategy()
        cd = CreateDataset(dataset_type="binary label")
        ds = cd.create_dataset(imputation_strategy=strategy, df=df, label_names=['label'],
                               protected_attribute_names=['feat'], scores_names=["scores"])
        self.assertTrue(ds is not None)


if __name__ == '__main__':
    unittest.main()
