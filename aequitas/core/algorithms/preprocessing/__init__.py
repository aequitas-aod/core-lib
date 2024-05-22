import aequitas
from aequitas.core.algorithms import AequitasTransformer

from aif360.algorithms.preprocessing import (
    DisparateImpactRemover as _Aif360DisparateImpactRemover,
    LFR as _Aif360LFR,
    Reweighing as _Aif360Reweighing,
    OptimPreproc as _Aif360OptimPreproc,
)


class DisparateImpactRemover(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360DisparateImpactRemover, **kwargs)


class LFR(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360LFR, **kwargs)


class Reweighing(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360Reweighing, **kwargs)


class OptimPreproc(AequitasTransformer):
    def __init__(self, **kwargs):
        super().__init__(_Aif360OptimPreproc, **kwargs)


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
