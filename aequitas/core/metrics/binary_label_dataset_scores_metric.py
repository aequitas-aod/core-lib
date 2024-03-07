import aif360.metrics as metrics
import numpy as np


class BinaryLabelDatasetScoresMetric(metrics.BinaryLabelDatasetMetric):
    def __init__(self, dataset, **kwargs):
        self._scores = dataset.scores
        if self._scores is None:
            raise TypeError("Must provide a numpy array representing the score associated with each sample")
        super(BinaryLabelDatasetScoresMetric, self).__init__(dataset, **kwargs)

    def scores_metric(self):
        # TODO: change name and behaviour
        return self._scores.mean()
