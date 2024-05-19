import aequitas

from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
import aif360.metrics as aif360metrics
import aequitas.core.imputation as imputations


class DatasetWithMetrics(aequitas.Decorator):
    def __init__(self, decorated, metrics_type: type):
        super().__init__(decorated)
        self._metrics_type = metrics_type

    def _new_metrics(self):
        return self._metrics_type(dataset=self._delegate)

    @property
    def metrics(self):
        return self._new_metrics()


class DatasetWithBinaryLabelMetrics(DatasetWithMetrics):
    def __init__(self, dataset, unprivileged_groups, privileged_groups):
        super().__init__(dataset, aif360metrics.BinaryLabelDatasetMetric)
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        
    def _new_metrics(self):
        return self._metrics_type(
            dataset=self._delegate,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )


_DATASET_TYPES = {
    "binary label": BinaryLabelDataset,
    "multi class": MulticlassLabelDataset,
    "binary": BinaryLabelDataset,
    "multiclass": MulticlassLabelDataset,
    "multi": MulticlassLabelDataset,
}


def create_dataset(dataset_type, 
                   unprivileged_groups, 
                   privileged_groups,
                   **kwargs):
    dataset_type = dataset_type.lower()
    imputation_strategy = kwargs.get('imputation_strategy', imputations.DoNothingImputationStrategy())
    if 'wrap' in kwargs:
        dataset = kwargs['wrap']
    else:
        if dataset_type not in _DATASET_TYPES:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        dataset = _DATASET_TYPES[dataset_type](**kwargs)
    dataset = imputation_strategy(dataset)
    return DatasetWithBinaryLabelMetrics(dataset, unprivileged_groups, privileged_groups)


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
