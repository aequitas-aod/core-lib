import unittest
import pandas as pd
from test import generate_binary_label_dataframe
from test import generate_binary_label_dataframe_with_scores

from aequitas.core.datasets import create_dataset, BinaryLabelDataset
from aequitas.core.imputation_strategies import MCMCImputationStrategy
from aequitas.core.metrics import BinaryLabelDatasetScoresMetric


class TestBinaryLabelDataset(unittest.TestCase):

    def test_dataset_creation_via_factory(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0
                            )
        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names=['scores'],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0)
        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

    def test_metrics_on_dataset(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names=['score'],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0)
        self.assertIsInstance(ds.scores_metrics, BinaryLabelDatasetScoresMetric)
        self.assertIsNotNone(ds)
        score = ds.metrics.disparate_impact()
        self.assertIsNotNone(score)
        score = ds.scores_metrics.new_fancy_metric()
        self.assertIsNotNone(score)


if __name__ == '__main__':
    unittest.main()
