from aequitas.core.datasets import DatasetWithBinaryLabelMetrics, BinaryLabelDataset
from aequitas.core.datasets.zoo import adult
from aequitas.core.metrics import BinaryLabelDatasetMetric
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

    def test_copy_preserves_type(self):
        copied = self._dataset.copy()
        self.assertIsInstance(copied, DatasetWithBinaryLabelMetrics)
        self.assertIsInstance(copied._delegate, BinaryLabelDataset)
        self.assertIsInstance(copied.metrics, BinaryLabelDatasetMetric)

    def test_split_preserves_type(self):
        splitted = self._dataset.split([0.5])
        for split in splitted:
            self.assertIsInstance(split, DatasetWithBinaryLabelMetrics)
            self.assertIsInstance(split._delegate, BinaryLabelDataset)
            self.assertIsInstance(split.metrics, BinaryLabelDatasetMetric)
    