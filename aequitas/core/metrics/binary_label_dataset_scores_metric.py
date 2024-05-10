import aif360.metrics as metrics
import numpy as np


class BinaryLabelDatasetScoresMetric(metrics.BinaryLabelDatasetMetric):
    def __init__(self, **kwargs):
        super(BinaryLabelDatasetScoresMetric, self).__init__(**kwargs)

    def new_fancy_metric(self) -> float:
        # TODO: change name and behaviour
        return np.max(self.dataset.scores)