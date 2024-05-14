import aequitas
from aif360.algorithms.inprocessing import (
    AdversarialDebiasing,
    ExponentiatedGradientReduction
    )

from aif360.algorithms.preprocessing import (
    DisparateImpactRemover,
    LFR,
    Reweighing
    )

_ALGORITHM_TYPES = {
    "adversarial debiasing": AdversarialDebiasing,
    "egr": ExponentiatedGradientReduction,
    "exponentiated gradient reduction": ExponentiatedGradientReduction,
    "dir": DisparateImpactRemover,
    "disparate impact remover": DisparateImpactRemover,
    "lfr": LFR,
    "reweighing": Reweighing

}


def create_algorithm(algorithm_type, **kwargs):
    algorithm_type = algorithm_type.lower()
    if algorithm_type not in _ALGORITHM_TYPES:
        raise ValueError(f"Unknown algorithm type: {algorithm_type} \n Check among these: {_ALGORITHM_TYPES}")
    return _ALGORITHM_TYPES[algorithm_type](**kwargs)


# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)
# keep this line at the bottom of this file
aequitas.logger.debug("Module %s correctly loaded", __name__)