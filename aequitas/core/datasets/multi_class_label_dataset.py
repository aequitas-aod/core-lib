from aequitas.core.datasets.structured_dataset import StructuredDataset
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from aequitas.core.metrics.binary_label_dataset_scores_metric import BinaryLabelDatasetScoresMetric
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset


class MulticlassLabelDataset(StructuredDataset, MulticlassLabelDataset):

    def __init__(self, imputation_strategy: MissingValuesImputationStrategy,
                 favorable_label, unfavorable_label, **kwargs):

        super(MulticlassLabelDataset, self).__init__(imputation_strategy=imputation_strategy, favorable_label=favorable_label,
                                                 unfavorable_label=unfavorable_label, **kwargs)

    @property
    def metrics(self, **kwargs):
        return BinaryLabelDatasetScoresMetric(self, **kwargs)
