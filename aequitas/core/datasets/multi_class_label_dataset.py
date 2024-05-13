from aequitas.core.datasets.structured_dataset import StructuredDataset
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from aequitas.core.metrics.binary_label_dataset_scores_metric import BinaryLabelDatasetScoresMetric
from aif360.metrics import BinaryLabelDatasetMetric

from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset


class MulticlassLabelDataset(StructuredDataset, MulticlassLabelDataset):

    def __init__(self, unprivileged_groups, privileged_groups, **kwargs):
        self.kwargs = kwargs
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        super(MulticlassLabelDataset, self).__init__(**kwargs)

    @property
    def metrics(self, **kwargs):
        return BinaryLabelDatasetMetric(dataset=self,
                                        unprivileged_groups=self.unprivileged_groups,
                                        privileged_groups=self.privileged_groups)

    @property
    def scores_metrics(self, **kwargs):
        return BinaryLabelDatasetScoresMetric(dataset=self,
                                              unprivileged_groups=self.unprivileged_groups,
                                              privileged_groups=self.privileged_groups)
