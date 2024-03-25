from aequitas.core.datasets.structured_dataset import StructuredDataset
from aequitas.core.imputation_strategies.imputation_strategy import MissingValuesImputationStrategy
from aequitas.core.metrics.binary_label_dataset_scores_metric import BinaryLabelDatasetScoresMetric
import aif360.datasets as datasets


class BinaryLabelDataset(StructuredDataset, datasets.BinaryLabelDataset):

    def __init__(self, unprivileged_groups, privileged_groups, **kwargs):
        self.kwargs = kwargs
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        super(BinaryLabelDataset, self).__init__(**kwargs)

    @property
    def metrics(self):
        dataset = BinaryLabelDataset(unprivileged_groups=self.unprivileged_groups,
                                     privileged_groups=self.privileged_groups,
                                     **self.kwargs)
        return BinaryLabelDatasetScoresMetric(dataset=dataset,
                                              unprivileged_groups=self.unprivileged_groups,
                                              privileged_groups=self.privileged_groups)
