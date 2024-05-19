from aequitas.core.datasets import DatasetWithBinaryLabelMetrics
from aequitas.core.datasets.zoo import adult
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import unittest


class TestBinaryLabelDataset(unittest.TestCase):
    def setUp(self) -> None:
        protected = 'sex'
        self._dataset = adult(
            unprivileged_groups=[{protected: 0}],
            privileged_groups=[{protected: 1}],
            protected_attribute_names=[protected],
            privileged_classes=[['Male']],
        )

    def test_types(self):
        self.assertIsInstance(self._dataset, DatasetWithBinaryLabelMetrics)
        self.assertIsInstance(self._dataset._delegate, BinaryLabelDataset)

    def test_metrics(self):
        self.assertIsInstance(self._dataset.metrics, BinaryLabelDatasetMetric)
