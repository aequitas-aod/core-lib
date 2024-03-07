from aequitas.core.datasets.structured_dataset import StructuredDataset
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from aequitas.core.metrics.binary_label_dataset_scores_metric import BinaryLabelDatasetScoresMetric
import aif360.datasets as datasets


class BinaryLabelDataset(StructuredDataset, datasets.BinaryLabelDataset):

    def __init__(self, imputation_strategy: MissingValuesImputationStrategy,
                 favorable_label, unfavorable_label, label_names, protected_attribute_names, **kwargs):

        super(BinaryLabelDataset, self).__init__(imputation_strategy=imputation_strategy, favorable_label=favorable_label,
                                                 unfavorable_label=unfavorable_label, label_names=label_names,
                                                 protected_attribute_names=protected_attribute_names, **kwargs)

    @property
    def metrics(self, **kwargs):
        return BinaryLabelDatasetScoresMetric(**kwargs)
