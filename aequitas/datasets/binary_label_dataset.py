from aequitas.datasets.structured_dataset import StructuredDataset
from aequitas.metrics.binary_label_dataset_metric import BinaryLabelDatasetScoresMetric


class BinaryLabelDataset(StructuredDataset):

    def __init__(self, favorable_label=1., unfavorable_label=0., **kwargs):
        self._favorable_label = float(favorable_label)
        self._unfavorable_label = float(unfavorable_label)

        super(BinaryLabelDataset, self).__init__(**kwargs)

    @classmethod
    def new_instance(cls):
        return cls

    @property
    def metrics(self, **kwargs):
        return BinaryLabelDatasetScoresMetric(**kwargs)

