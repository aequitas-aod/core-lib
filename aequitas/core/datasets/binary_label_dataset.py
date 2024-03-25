from aequitas.core.datasets.structured_dataset import StructuredDataset
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from aequitas.core.metrics.binary_label_dataset_scores_metric import BinaryLabelDatasetScoresMetric
import aif360.datasets as datasets


class BinaryLabelDataset(StructuredDataset, datasets.BinaryLabelDataset):

    def __init__(self, **kwargs):
        self.params = kwargs
        super(BinaryLabelDataset, self).__init__(**kwargs)

    @property
    def metrics(self):
        dataset = BinaryLabelDataset(**self.params)
        return BinaryLabelDatasetScoresMetric(dataset=dataset)
