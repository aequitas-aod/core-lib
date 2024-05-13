import unittest
import pandas as pd
from test import (
    generate_binary_label_dataframe,
    generate_binary_label_dataframe_with_scores,
    generate_multi_label_dataframe,
    generate_multi_label_dataframe_with_scores
)

from aequitas.core.datasets import (
    create_dataset,
    BinaryLabelDataset,
    MulticlassLabelDataset
)

from aequitas.core.imputation_strategies import MCMCImputationStrategy
from aequitas.core.metrics import BinaryLabelDatasetScoresMetric


class TestBinaryLabelDataset(unittest.TestCase):

    def test_dataset_creation_via_factory(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0,
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr']
                            )
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0,
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names="score"
                            )
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

    def test_metrics_on_dataset(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0,
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names="score"
                            )
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
        self.assertIsInstance(ds.scores_metrics, BinaryLabelDatasetScoresMetric)
        self.assertIsNotNone(ds)

        ### METRICS USING LABELS ###

        # Disparate Impact
        score = ds.metrics.disparate_impact()
        print(f"Disparate impact: {score}")
        self.assertIsNotNone(score)

        # Statistical Parity
        score = ds.metrics.statistical_parity_difference()
        print(f"Statistical Parity: {score}")
        self.assertIsNotNone(score)

        # Dirichlet-smoothed base rates
        score = ds.metrics._smoothed_base_rates(ds.labels)
        print(f"Dirichlet-smoothed base rates: {score}")
        self.assertIsNotNone(score)

        # Smoothed EDF
        score = ds.metrics.smoothed_empirical_differential_fairness()
        print(f"Smoothed EDF: {score}")
        self.assertIsNotNone(score)

        # Consistency
        score = ds.metrics.consistency()
        print(f"Consistency: {score}")
        self.assertIsNotNone(score)

        ### METRICS USING SCORES ###

        score = ds.scores_metrics.new_fancy_metric()
        self.assertIsNotNone(score)


class TestMulticlassLabelDataset(unittest.TestCase):
    def test_dataset_creation_via_factory(self):
        ds = create_dataset("multi class",
                            # parameters of aequitas.MulticlassLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.MulticlassLabelDataset init
                            favorable_label=[0, 1., 2.],
                            unfavorable_label=[3., 4.],
                            # parameters of aif360.StructuredDataset init
                            df=generate_multi_label_dataframe(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr']
                            )
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
        self.assertIsInstance(ds, MulticlassLabelDataset)
        self.assertIsNotNone(ds)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset("multi class",
                            # parameters of aequitas.MulticlassLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.MulticlassLabelDataset init
                            favorable_label=[0, 1., 2.],
                            unfavorable_label=[3., 4.],
                            # parameters of aif360.StructuredDataset init
                            df=generate_multi_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names="score"
                            )
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
        self.assertIsInstance(ds, MulticlassLabelDataset)
        self.assertIsNotNone(ds)


if __name__ == '__main__':
    unittest.main()
