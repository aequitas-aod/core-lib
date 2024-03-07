from abc import ABC, abstractmethod
from aequitas.core.datasets.structured_dataset import StructuredDataset


class DatasetFactory(ABC):
    @abstractmethod
    def create_dataset(self, **kwargs) -> StructuredDataset:
        raise NotImplementedError


