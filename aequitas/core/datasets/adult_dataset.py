import aif360.datasets as datasets
from aif360.metrics import BinaryLabelDatasetMetric


class AdultDataset(datasets.AdultDataset):
    def __init__(self, **kwargs):
        super(AdultDataset, self).__init__(**kwargs)

        self.unprivileged_groups = [{attr_name: 0} for attr_name in self.protected_attribute_names]
        self.privileged_groups = [{attr_name: 1} for attr_name in self.protected_attribute_names]

    def metrics(self):
        return BinaryLabelDatasetMetric(dataset=self, unprivileged_groups=self.unprivileged_groups,
                                        privileged_groups=self.privileged_groups)
