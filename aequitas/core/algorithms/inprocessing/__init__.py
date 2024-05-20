import aequitas
from aequitas.core.algorithms import AequitasTransformer

from aif360.algorithms.inprocessing import (
    AdversarialDebiasing as _Aif360AdversarialDebiasing,
    ExponentiatedGradientReduction as _Aif360ExponentiatedGradientReduction,
    GridSearchReduction as _Aif360GridSearchReduction,
    PrejudiceRemover as _Aif360PrejudiceRemover,
    ARTClassifier as _Aif360ARTClassifier,
    GerryFairClassifier as _Aif360GerryFairClassifier,
    MetaFairClassifier as _Aif360MetaFairClassifier,
)


class AdversarialDebiasing(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360AdversarialDebiasing, **kwargs)


class ExponentiatedGradientReduction(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360ExponentiatedGradientReduction, **kwargs)


class GridSearchReduction(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360GridSearchReduction, **kwargs)


class PrejudiceRemover(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360PrejudiceRemover, **kwargs)


class ARTClassifier(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360ARTClassifier, **kwargs)


class GerryFairClassifier(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360GerryFairClassifier, **kwargs)


class MetaFairClassifier(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360MetaFairClassifier, **kwargs)


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
