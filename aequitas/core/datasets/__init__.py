import aequitas

from .structured_dataset import StructuredDataset
from .binary_label_dataset import BinaryLabelDataset
from .multi_class_label_dataset import MultiClassLabelDataset


_DATASET_TYPES = {
    "binary label": BinaryLabelDataset,
    "multi class": MultiClassLabelDataset,
    "binary": BinaryLabelDataset,
    "multiclass": MultiClassLabelDataset,
}


def create_dataset(dataset_type, **kwargs):
    dataset_type = dataset_type.lower()
    if dataset_type not in _DATASET_TYPES:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return _DATASET_TYPES[dataset_type](**kwargs)


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)