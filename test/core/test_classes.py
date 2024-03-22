import unittest
from test import generate_binary_label_dataframe
from test import generate_binary_label_dataframe_with_scores

from aequitas.core.datasets import create_dataset, BinaryLabelDataset
from aequitas.core.imputation_strategies import MCMCImputationStrategy
from aequitas.core.metrics import BinaryLabelDatasetScoresMetric


class TestBinaryLabelDataset(unittest.TestCase):

    def test_dataset_creation_via_factory(self):
        ds = create_dataset("binary label", 
                            imputation_strategy=MCMCImputationStrategy(), 
                            df=generate_binary_label_dataframe(), 
                            label_names=['label'],
                            protected_attribute_names=['feat'])
        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset("binary label",
                            imputation_strategy=MCMCImputationStrategy(), 
                            df=generate_binary_label_dataframe_with_scores(), 
                            label_names=['label'],
                            protected_attribute_names=['feat'], 
                            scores_names=["scores"])
        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

    def test_metrics_on_dataset(self):
        ds = create_dataset("binary label",
                            imputation_strategy=MCMCImputationStrategy(), 
                            df=generate_binary_label_dataframe_with_scores(), 
                            label_names=['label'],
                            protected_attribute_names=['feat'], 
                            scores_names=["scores"])
        self.assertIsInstance(ds.metrics, BinaryLabelDatasetScoresMetric)
        self.assertIsNotNone(ds)
        score = ds.metrics.new_fancy_metric()
        self.assertIsNotNone(score)
        score = ds.metrics.disparate_impact()
        self.assertIsNotNone(score)


if __name__ == '__main__':
    unittest.main()
