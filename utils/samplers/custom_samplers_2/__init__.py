import numpy as np
from ..base import AbstractSampler, IndexSampler

class ShapleySampler(AbstractSampler):
    def __init__(self, X_test, y_test, num_samples=100):
        self.num_samples = num_samples
        self.X_test = X_test
        self.y_test = y_test
        self.index_sampler = IndexSampler()
        self.sh_values = None

    def __call__(self, X, y, sh_values, weight):

        self.sh_values = sh_values

        indices = np.argsort(sh_values.values)[-self.num_samples:]

        return self.index_sampler(X, y, weight, index=indices)