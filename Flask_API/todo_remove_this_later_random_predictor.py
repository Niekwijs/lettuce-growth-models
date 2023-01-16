# TODO: Remove this predictor when a real predictor becomes available
import random

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class RandomYPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, min,max, distrib=None, random_state=None):
        self.min = min
        self.max = max
        self.distrib = distrib
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return [random.randint(self.min, self.max)]