from aequitas.core.datasets.dataset_factory import DatasetFactory
from aequitas.core.datasets.binary_label_dataset import BinaryLabelDataset


class BinaryLabelDatasetFactory(DatasetFactory):
    def __init__(self):
        pass

    def create_dataset(self, **kwargs) -> BinaryLabelDataset:
        ds = BinaryLabelDataset(favorable_label=1., unfavorable_label=0., **kwargs)
        return ds
