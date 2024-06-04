# noinspection PyUnresolvedReferences
from aif360.datasets import (
    Dataset,
    StructuredDataset,
    RegressionDataset,
    BinaryLabelDataset,
    StandardDataset
)

# noinspection PyUnresolvedReferences
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset

import aequitas
import aequitas.core.metrics as _metrics
import aequitas.core.imputation as _imputations


class DatasetWithMetrics(aequitas.Decorator):
    def __init__(self, decorated: StructuredDataset, metrics_type: type):
        if not isinstance(decorated, StructuredDataset):
            raise ValueError("Only structured datasets can be decorated via this class")
        super().__init__(decorated)
        self._metrics_type = metrics_type

    def _new_metrics(self) -> _metrics.Metric:
        return self._metrics_type(dataset=self._delegate)

    @property
    def metrics(self) -> _metrics.Metric:
        return self._new_metrics()

    def _attributes_to_propagate_when_wrapping(self):
        return []

    def wrap_another_dataset(self, dataset: StructuredDataset, **kwargs) -> 'DatasetWithMetrics':
        propagated = {k: getattr(self, k) for k in self._attributes_to_propagate_when_wrapping()}
        kwargs.update(propagated)
        return self.__class__(dataset, **kwargs)

    def split(self, *args, **kwargs):
        return tuple(
            self.wrap_another_dataset(splitted)
            for splitted in self._delegate.split(*args, **kwargs)
        )

    def copy(self, *args, **kwargs):
        return self.wrap_another_dataset(self._delegate.copy(*args, **kwargs))


class DatasetWithBinaryLabelMetrics(DatasetWithMetrics):
    def __init__(self, dataset, unprivileged_groups, privileged_groups):
        super().__init__(dataset, _metrics.BinaryLabelDatasetMetric)
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups

    def _new_metrics(self) -> _metrics.Metric:
        return self._metrics_type(
            dataset=self._delegate,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )

    def _attributes_to_propagate_when_wrapping(self):
        return ['unprivileged_groups', 'privileged_groups']


class DatasetWithRegressionMetrics(DatasetWithMetrics):
    def __init__(self, dataset, unprivileged_groups, privileged_groups):
        super().__init__(dataset, _metrics.RegressionDatasetMetric)
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups

    def _new_metrics(self) -> _metrics.Metric:
        return self._metrics_type(
            dataset=self._delegate,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )

    def _attributes_to_propagate_when_wrapping(self):
        return ['unprivileged_groups', 'privileged_groups']


_DATASET_TYPES = {
    "binary label": BinaryLabelDataset,
    "multi class": MulticlassLabelDataset,
    "binary": BinaryLabelDataset,
    "multiclass": MulticlassLabelDataset,
    "multi": MulticlassLabelDataset,
    "regression": RegressionDataset
}


def create_dataset(dataset_type,
                   unprivileged_groups,
                   privileged_groups,
                   **kwargs):
    dataset_type = dataset_type.lower()
    imputation_strategy = kwargs.pop('imputation_strategy', _imputations.DoNothingImputationStrategy())
    if 'wrap' in kwargs:
        dataset = kwargs.pop('wrap')
    else:
        if dataset_type not in _DATASET_TYPES:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        imputed_df = imputation_strategy(kwargs['df'])
        kwargs['df'] = imputed_df
        dataset = _DATASET_TYPES[dataset_type](**kwargs)

    if dataset_type == "binary label":
        return DatasetWithBinaryLabelMetrics(dataset, unprivileged_groups, privileged_groups)
    elif dataset_type == "regression":
        return DatasetWithRegressionMetrics(dataset, unprivileged_groups, privileged_groups)


__all__ = [
    'Dataset',
    'StructuredDataset',
    'RegressionDataset',
    'BinaryLabelDataset',
    'StandardDataset',
    'MulticlassLabelDataset',
    'DatasetWithMetrics',
    'DatasetWithBinaryLabelMetrics',
    'create_dataset',
]

# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
