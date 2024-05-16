import aif360.datasets as datasets
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aequitas.core.metrics import BinaryLabelDatasetScoresMetric


class AdultDataset(datasets.AdultDataset):
    def __init__(self, unprivileged_groups, privileged_groups, **kwargs):
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        super(AdultDataset, self).__init__(**kwargs)

    @property
    def metrics(self):
        return BinaryLabelDatasetMetric(dataset=self, unprivileged_groups=self.unprivileged_groups,
                                        privileged_groups=self.privileged_groups)

    @property
    def score_metrics(self):
        return BinaryLabelDatasetScoresMetric(dataset=self, unprivileged_groups=self.unprivileged_groups,
                                              privileged_groups=self.privileged_groups)
