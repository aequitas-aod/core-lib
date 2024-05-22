import unittest
import numpy as np
import math

from aequitas.core.datasets import *
from aequitas.core.metrics import *

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


DEFAULT_TOLERANCE = 0.01


class AbstractMetricTestCase(unittest.TestCase):
    def assertInRange(self, value, lower, upper, msg=None, tolerance=DEFAULT_TOLERANCE):
        self.assertTrue(
            lower - tolerance <= value <= upper + tolerance,
            msg=msg or f"Value {value} should be in range [{lower}, {upper}]"
        )

    def assertNotInRange(self, value, lower, upper, msg=None, tolerance=DEFAULT_TOLERANCE):
        self.assertTrue(
            value < lower - tolerance or value > upper + tolerance,
            msg=msg or f"Value {value} should not be in range [{lower}, {upper}]"
        )

    def assertEDF(self, dataset, dataset_skewed, epsilon=2.337, tolerance=DEFAULT_TOLERANCE):
        score = dataset.metrics.smoothed_empirical_differential_fairness()
        self.assertFalse(math.isnan(score), msg="EDF should not be nan")
        self.assertIsNotNone(score, msg="EDF should not be None")
        self.assertInRange(score, lower=np.exp(-epsilon), upper=np.exp(epsilon), tolerance=tolerance)

        score = dataset_skewed.metrics.smoothed_empirical_differential_fairness()
        self.assertFalse(math.isnan(score), msg="EDF should not be nan")
        self.assertIsNotNone(score, msg="EDF should not be None")
        self.assertInRange(score, lower=np.exp(-epsilon), upper=np.exp(epsilon), tolerance=tolerance)

    def assertSP(self, dataset, dataset_skewed, bound=0.1, tolerance=DEFAULT_TOLERANCE):
        score = dataset_skewed.metrics.statistical_parity_difference()
        self.assertFalse(math.isnan(score), msg="SP should not be nan")
        self.assertIsNotNone(score, msg="SP should not be None")
        self.assertNotInRange(score, lower=-bound, upper=bound, tolerance=tolerance, msg="score should not be close to 0 for a biased dataset")

        score = dataset.metrics.statistical_parity_difference()
        self.assertFalse(math.isnan(score), msg="SP should not be nan")
        self.assertIsNotNone(score, msg="SP should not be None")
        self.assertInRange(score, -bound, bound, tolerance)

    def assertDI(self, dataset, dataset_skewed, bound=0.8, tolerance=DEFAULT_TOLERANCE):
        score = dataset.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="DI should not be nan")
        self.assertIsNotNone(score, msg="DI should not be None")
        self.assertGreaterEqual(score, bound - tolerance, f"DI should be >= {bound} for a nonbiased dataset")

        score = dataset_skewed.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="DI should not be nan")
        self.assertIsNotNone(score, msg="DI should not be None")
        self.assertLess(score, bound + tolerance, msg=f"DI is expected to be below {bound} for biased dataset")

    def assertConsistency(self, dataset, dataset_skewed, bound=0.8, tolerance=DEFAULT_TOLERANCE):
        score = dataset.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="Consistency should not be nan")
        self.assertIsNotNone(score, msg="Consistency should not be None")
        self.assertGreaterEqual(score, bound - tolerance, f"Consistency should be >= {bound} for a nonbiased dataset")

        score = dataset_skewed.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="Consistency should not be nan")
        self.assertIsNotNone(score, msg="Consistency should not be None")
        self.assertLess(score, bound + tolerance, msg=f"Consistency is expected to be < {bound} for biased dataset")

    def assertMeanDifference(self, dataset, dataset_skewed, bound=0.1, tolerance=DEFAULT_TOLERANCE):
        score = dataset.metrics.mean_difference()
        self.assertFalse(
            math.isnan(score),
            msg="Difference in mean outcomes between unprivileged and privileged groups should not be nan"
        )
        self.assertIsNotNone(
            score,
            msg="Difference in mean outcomes between unprivileged and privileged groups should not be None"
        )
        self.assertInRange(score, lower=-bound, upper=bound, tolerance=tolerance)

        score = dataset_skewed.metrics.mean_difference()
        self.assertFalse(
            math.isnan(score),
            msg="Difference in mean outcomes between unprivileged and privileged groups should not be nan"
        )
        self.assertIsNotNone(
            score,
            msg="Difference in mean outcomes between unprivileged and privileged groups should not be None"
        )
        self.assertLess(
            score,
            bound + tolerance,
            msg=f"Difference in mean outcomes between unprivileged and privileged should be < {-bound} in biased dataset"
        )

    def assertBinaryLabelDataset(self, dataset):
        self.assertIsInstance(dataset, DatasetWithMetrics)
        self.assertIsInstance(dataset._delegate, BinaryLabelDataset)
        self.assertIsInstance(dataset.metrics, BinaryLabelDatasetMetric)

    def assertMultiLabelDataset(self, dataset):
        self.assertIsInstance(dataset, DatasetWithMetrics)
        self.assertIsInstance(dataset._delegate, MulticlassLabelDataset)
        self.assertIsInstance(dataset.metrics, BinaryLabelDatasetMetric)
