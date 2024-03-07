from aequitas.core.datasets.structured_dataset import StructuredDataset
from aequitas.core.datasets.concrete_dataset_factories import BinaryLabelDatasetFactory


class CreateDataset:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type

    def create_dataset(self, **kwargs):
        if self.dataset_type == "binary label":
            bldf = BinaryLabelDatasetFactory()
            return bldf.create_dataset(**kwargs)
