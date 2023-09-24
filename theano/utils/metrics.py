import numpy as np

class Metric:

    def __init__(self):
        self._name = None

    def __call__(self, pred, real):
        pass

    def __repr__(self):
        return self._name


class AccuracyMetric(Metric):

    def __init__(self):
        self._name = "accuracy"

    def __call__(self, pred, real):
        return np.sum(pred == real)/len(pred)

