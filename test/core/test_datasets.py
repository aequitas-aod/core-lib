import unittest
import numpy as np
import pandas as pd

from aequitas.core.datasets import create_metric, create_regression_dataset

from test import (
    generate_binary_label_dataframe,
    generate_skewed_binary_label_dataframe,
    generate_binary_label_dataframe_with_scores,
    generate_skewed_binary_label_dataframe_with_scores,
    generate_multi_label_dataframe,
    generate_skewed_multi_label_dataframe,
    generate_multi_label_dataframe_with_scores,
    generate_skewed_multi_label_dataframe_with_scores,
    generate_skewed_regression_dataset,
    generate_dataframe_with_preds
)
from test.core import AbstractMetricTestCase
from aif360.metrics import ClassificationMetric
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
            imputation_strategy=MeanImputationStrategy(),
            df=generate_binary_label_dataframe()
        )
        self.assertBinaryLabelDataset(ds)

        ds_skewed = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            df=generate_skewed_binary_label_dataframe()
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
            # parameters of aif360.StructuredDataset init
            df=generate_binary_label_dataframe_with_scores(),
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
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_binary_label_dataframe_with_scores(),
            scores_names="score"
        )
        self.assertBinaryLabelDataset(ds_skewed)

    def test_metrics_on_dataset(self):
        df = generate_binary_label_dataframe_with_scores()
        df_skewed = generate_skewed_binary_label_dataframe_with_scores()
        ds = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.StructuredDataset init
            df=df,
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
            # parameters of aif360.StructuredDataset init
            df=df_skewed,
            scores_names="score"
        )
        self.assertBinaryLabelDataset(ds_skewed)

        ### METRICS USING LABELS ###

        # Disparate Impact
        self.assertDI(ds, ds_skewed)
        self.assertSP(ds, ds_skewed)
        # self.assertEDF(ds, ds_skewed)
        self.assertConsistency(ds, ds_skewed)

        ds_preds = create_dataset(
            "binary label",
            # parameters of aequitas.BinaryLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            imputation_strategy=MeanImputationStrategy(),
            df=generate_dataframe_with_preds(df),
            scores_names="score"
        )
        self.assertBinaryLabelDataset(ds_preds)

        cm = create_metric(
            metric="classification",
            dataset1=ds,
            dataset2=ds_preds,
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}]
        )

        self.assertIsInstance(cm, ClassificationMetric)


class TestMulticlassLabelDataset(AbstractMetricTestCase):
    def test_dataset_creation_via_factory(self):
        ds = create_dataset(
            "multi class",
            # parameters of aequitas.MulticlassLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=DoNothingImputationStrategy(),
            # parameters of aif360.MulticlassLabelDataset init
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            # parameters of aif360.StructuredDataset init
            df=generate_multi_label_dataframe()
        )
        self.assertMultiLabelDataset(ds)

        ds_skewed = create_dataset(
            "multi class",
            # parameters of aequitas.MulticlassLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=DoNothingImputationStrategy(),
            # parameters of aif360.MulticlassLabelDataset init
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_multi_label_dataframe()
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
            imputation_strategy=DoNothingImputationStrategy(),
            # parameters of aif360.MulticlassLabelDataset init
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            # parameters of aif360.StructuredDataset init
            df=generate_multi_label_dataframe_with_scores(),
            scores_names="score"
        )
        self.assertMultiLabelDataset(ds)

        ds_skewed = create_dataset(
            "multi class",
            # parameters of aequitas.MulticlassLabelDataset init
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=DoNothingImputationStrategy(),
            # parameters of aif360.MulticlassLabelDataset init
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            # parameters of aif360.StructuredDataset init
            df=generate_skewed_multi_label_dataframe_with_scores(),
            scores_names="score"
        )
        self.assertMultiLabelDataset(ds_skewed)


class TestRegressionDataset(AbstractMetricTestCase):
    def test_regression_dataset_creation_via_factory(self):
        ds = create_regression_dataset(  # parameters of aequitas.DatasetWithRegressionMetrics init
            unprivileged_groups=[{'color': 'b'}],
            privileged_groups=[{'color': 'r'}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=MeanImputationStrategy(),
            # parameters of aif360.RegressionDataset init
            df=generate_skewed_regression_dataset(),
            dep_var_name='score'
        )
        self.assertRegressionDataset(ds)


class TestDataframeCreationFunctions(AbstractMetricTestCase):

    def setUp(self):
        self.nan_perc = 0.1

    def test_binary_dataframe_creation_with_nans(self):
        df = generate_binary_label_dataframe(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_skewed_binary_dataframe_creation_with_nans(self):
        df = generate_skewed_binary_label_dataframe(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_binary_dataframe_creation_with_nans_and_scores(self):
        df = generate_binary_label_dataframe_with_scores(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_skewed_binary_dataframe_creation_with_nans_and_scores(self):
        df = generate_skewed_binary_label_dataframe_with_scores(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_multi_label_dataframe_creation_with_nans(self):
        df = generate_multi_label_dataframe(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_skewed_multi_label_dataframe_creation_with_nans(self):
        df = generate_skewed_multi_label_dataframe(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_multi_label_dataframe_creation_with_nans_and_scores(self):
        df = generate_multi_label_dataframe(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")

    def test_skewed_multi_label_dataframe_creation_with_nans_and_scores(self):
        df = generate_skewed_multi_label_dataframe(nans=True)
        self.assertEqual(df.isna().sum().sum(), int(df.shape[0] * (df.shape[1] - 1) * self.nan_perc),
                         msg="nan counts do not correspond")


if __name__ == '__main__':
    unittest.main()
