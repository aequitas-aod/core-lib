import aequitas

from aequitas import Decorator
from aequitas.core.datasets import DatasetWithMetrics

from aif360.algorithms.transformer import Transformer

# from aif360.algorithms.inprocessing import (
#     AdversarialDebiasing,
#     ExponentiatedGradientReduction
# )
#
# from aif360.algorithms.preprocessing import (
#     DisparateImpactRemover,
#     LFR,
#     Reweighing
# )
#
# _ALGORITHM_TYPES = {
#     "adversarial debiasing": AdversarialDebiasing,
#     "egr": ExponentiatedGradientReduction,
#     "exponentiated gradient reduction": ExponentiatedGradientReduction,
#     "dir": DisparateImpactRemover,
#     "disparate impact remover": DisparateImpactRemover,
#     "lfr": LFR,
#     "reweighing": Reweighing
# }
#
#
# def create_algorithm(algorithm_type, **kwargs):
#     algorithm_type = algorithm_type.lower()
#     if algorithm_type not in _ALGORITHM_TYPES:
#         raise ValueError(f"Unknown algorithm type: {algorithm_type} \n Check among these: {_ALGORITHM_TYPES}")
#     return _ALGORITHM_TYPES[algorithm_type](**kwargs)


class AequitasTransformer(Decorator):
    def __init__(self, type, **kwargs):
        if not issubclass(type, Transformer):
            raise ValueError("Only AIF360 transformers can be decorated via this class")
        super().__init__(type(**kwargs))

    def _wrap_result(self, input: DatasetWithMetrics, result):
        if isinstance(result, DatasetWithMetrics):
            return result
        return input.wrap_another_dataset(result)

    def transform(self, dataset):
        return self._wrap_result(dataset, self._delegate.transform(dataset))

    def fit_transform(self, dataset):
        return self._wrap_result(dataset, self._delegate.fit_transform(dataset))


__all__ = [
    'AequitasTransformer',
]


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)