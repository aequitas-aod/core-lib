import aequitas
from aequitas.core.algorithms import AequitasTransformer

from aif360.algorithms.postprocessing import (
    DeterministicReranking as _Aif360DeterministicReranking,
    EqOddsPostprocessing as _Aif360EqOddsPostprocessing,
    RejectOptionClassification as _Aif360RejectOptionClassification,
    CalibratedEqOddsPostprocessing as _Aif360CalibratedEqOddsPostprocessing,
)


class DeterministicReranking(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360DeterministicReranking, **kwargs)


class EqOddsPostprocessing(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360EqOddsPostprocessing, **kwargs)


class RejectOptionClassification(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360RejectOptionClassification, **kwargs)


class CalibratedEqOddsPostprocessing(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360CalibratedEqOddsPostprocessing, **kwargs)


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
