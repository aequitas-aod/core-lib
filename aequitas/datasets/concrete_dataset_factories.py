from aequitas.datasets.dataset_factory import DatasetFactory
from aequitas.datasets.binary_label_dataset import BinaryLabelDataset


class BinaryLabelDatasetFactory(DatasetFactory):
    def __init__(self):
        pass

    def create_dataset(self, **kwargs) -> BinaryLabelDataset:
        print("DEBUG:calling BinaryLabelDataset constructor")
        ds = BinaryLabelDataset(favorable_label=1., unfavorable_label=0., **kwargs)
        return ds
