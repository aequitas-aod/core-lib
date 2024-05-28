import unittest

from test import (
    generate_binary_label_dataframe,
    generate_skewed_binary_label_dataframe,
    generate_binary_label_dataframe_with_scores,
    generate_skewed_binary_label_dataframe_with_scores,
    generate_multi_label_dataframe,
    generate_skewed_multi_label_dataframe,
    generate_multi_label_dataframe_with_scores,
    generate_skewed_multi_label_dataframe_with_scores
)
from test.core import AbstractMetricTestCase

from aequitas.core.imputation import *
from aequitas.core.datasets.zoo import *

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class TestBinaryLabelDataset(AbstractMetricTestCase):
    def test_dataset_creation_via_factory(self):
        ds = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.BinaryLabelDataset init
            favorable_label=1.,
            unfavorable_label=0.,
            # parameters of aif360.StructuredDataset init
            df=generate_binary_label_dataframe(),
            label_names=['label'],
            protected_attribute_names=['prot_attr']
        )
        self.assertBinaryLabelDataset(ds)

        ds_skewed = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.BinaryLabelDataset init
            favorable_label=1,
            unfavorable_label=0,
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_binary_label_dataframe(),
            label_names=['label'],
            protected_attribute_names=['prot_attr']
        )
        self.assertBinaryLabelDataset(ds_skewed)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.BinaryLabelDataset init
            favorable_label=1,
            unfavorable_label=0,
            # parameters of aif360.StructuredDataset init
            df=generate_binary_label_dataframe_with_scores(),
            label_names=['label'],
            protected_attribute_names=['prot_attr'],
            scores_names="score"
        )
        self.assertBinaryLabelDataset(ds)

        ds_skewed = create_dataset(
            "binary label",
           # parameters of aequitas.BinaryLabelDataset init
           unprivileged_groups=[{'prot_attr': 0}],
           privileged_groups=[{'prot_attr': 1}],
           # parameters of aequitas.StructuredDataset init
           imputation_strategy=MeanImputationStrategy(),
           # parameters of aif360.BinaryLabelDataset init
           favorable_label=1,
           unfavorable_label=0,
           # parameters of aif360.StructuredDataset init
           df=generate_skewed_binary_label_dataframe_with_scores(),
           label_names=['label'],
           protected_attribute_names=['prot_attr']
        )
        self.assertBinaryLabelDataset(ds_skewed)

    def test_metrics_on_dataset(self):
        ds = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.BinaryLabelDataset init
            favorable_label=1,
            unfavorable_label=0,
            # parameters of aif360.StructuredDataset init
            df=generate_binary_label_dataframe_with_scores(),
            label_names=['label'],
            protected_attribute_names=['prot_attr'],
            scores_names="score"
        )
        self.assertBinaryLabelDataset(ds)

        ds_skewed = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.BinaryLabelDataset init
            favorable_label=1,
            unfavorable_label=0,
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_binary_label_dataframe_with_scores(),
            label_names=['label'],
            protected_attribute_names=['prot_attr'],
            scores_names="score"
        )
        self.assertBinaryLabelDataset(ds_skewed)

        ### METRICS USING LABELS ###

        # Disparate Impact
        self.assertDI(ds, ds_skewed)
        self.assertSP(ds, ds_skewed)
        self.assertEDF(ds, ds_skewed)
        self.assertConsistency(ds, ds_skewed)


class TestMulticlassLabelDataset(AbstractMetricTestCase):
    def test_dataset_creation_via_factory(self):
        ds = create_dataset(
            "multi class",
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
        self.assertMultiLabelDataset(ds)

        ds_skewed = create_dataset(
            "multi class",
            # parameters of aequitas.MulticlassLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MCMCImputationStrategy(),
            # parameters of aif360.MulticlassLabelDataset init
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_multi_label_dataframe(),
            label_names=['label'],
            protected_attribute_names=['prot_attr']
        )
        self.assertMultiLabelDataset(ds)

        self.assertDI(ds, ds_skewed)
        self.assertSP(ds, ds_skewed)
        self.assertConsistency(ds, ds_skewed)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset(
            "multi class",
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
        self.assertMultiLabelDataset(ds)

        ds_skewed = create_dataset(
            "multi class",
            # parameters of aequitas.MulticlassLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MCMCImputationStrategy(),
            # parameters of aif360.MulticlassLabelDataset init
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_multi_label_dataframe_with_scores(),
            label_names=['label'],
            protected_attribute_names=['prot_attr'],
            scores_names="score"
        )
        self.assertMultiLabelDataset(ds_skewed)


if __name__ == '__main__':
    unittest.main()

# TODO: test ClassificationMetric, test RegressionMetric
