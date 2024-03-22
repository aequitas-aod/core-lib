import aif360.metrics as metrics
import numpy as np


class BinaryLabelDatasetScoresMetric(metrics.BinaryLabelDatasetMetric):
    def __init__(self, dataset, **kwargs):
        super(BinaryLabelDatasetScoresMetric, self).__init__(dataset, **kwargs)

    def new_fancy_metric(self):
        # TODO: change name and behaviour
        return self.disparate_impact()
